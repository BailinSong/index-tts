"""
MLX implementation of Conv2d Subsampling for Conformer
Matches PyTorch Conv2dSubsampling2 behavior
"""

import mlx.core as mx
import mlx.nn as nn
import math


class MLXConv2d(nn.Module):
    """
    2D Convolution for MLX
    Simplified version for subsampling (stride=2, kernel=3)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding  # PyTorch default is 0
        
        # Weight: (out_channels, in_channels, kernel_h, kernel_w)
        # Initialize with small random values
        scale = math.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weight = mx.random.normal(
            (out_channels, in_channels, kernel_size, kernel_size)
        ) * scale
        self.bias = mx.zeros((out_channels,))
    
    def __call__(self, x):
        """
        Forward pass for Conv2d with stride
        
        Args:
            x: (batch, height, width, in_channels) - MLX format
        
        Returns:
            out: (batch, height', width', out_channels)
                 where height' = height // stride
        """
        batch, height, width, in_channels = x.shape
        
        # Apply padding
        if self.padding > 0:
            # Pad: (left, right, top, bottom)
            pad_config = [
                (0, 0),  # batch
                (self.padding, self.padding),  # height
                (self.padding, self.padding),  # width
                (0, 0),  # channels
            ]
            x = mx.pad(x, pad_config)
            height += 2 * self.padding
            width += 2 * self.padding
        
        # Calculate output dimensions
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        
        # Sliding window convolution with stride
        outputs = []
        for i in range(0, height - self.kernel_size + 1, self.stride):
            row_outputs = []
            for j in range(0, width - self.kernel_size + 1, self.stride):
                # Extract window
                window = x[:, i:i+self.kernel_size, j:j+self.kernel_size, :]
                # (batch, kernel_h, kernel_w, in_channels)
                
                # Reshape for matmul
                window_flat = window.reshape(batch, -1)  # (batch, kernel_h*kernel_w*in_channels)
                weight_flat = self.weight.reshape(self.out_channels, -1).T  # (kernel_h*kernel_w*in_channels, out_channels)
                
                # Compute convolution for this position
                out = mx.matmul(window_flat, weight_flat)  # (batch, out_channels)
                row_outputs.append(out)
            
            outputs.append(mx.stack(row_outputs, axis=1))  # (batch, width', out_channels)
        
        out = mx.stack(outputs, axis=1)  # (batch, height', width', out_channels)
        
        # Add bias
        out = out + self.bias
        
        return out


class MLXConv2dSubsampling2(nn.Module):
    """
    Conv2d Subsampling with factor 2 (matches PyTorch Conv2dSubsampling2)
    
    Architecture:
        Conv2d(1→odim, kernel=3, stride=2) + ReLU
        → Linear projection
        → Positional encoding
    
    Input:  (batch, time, idim=1024)
    Output: (batch, time//2, odim=512)
    """
    
    def __init__(self, idim, odim):
        super().__init__()
        self.idim = idim
        self.odim = odim
        
        # Conv2d: (batch, 1, time, idim) → (batch, odim, time//2, idim//2)
        self.conv = MLXConv2d(
            in_channels=1,
            out_channels=odim,
            kernel_size=3,
            stride=2
        )
        
        # Calculate projection size after conv
        # After conv: time//2, idim//2 (approximately)
        # Flattened: odim * (idim//2)
        # Need to project to odim
        # For idim=1024: after conv we get 512 features per channel
        # Total: odim * 512 → odim
        self.out = nn.Linear(odim * (idim // 2), odim)
    
    def __call__(self, x):
        """
        Args:
            x: (batch, time, idim) e.g. (1, 121, 1024)
        
        Returns:
            x: (batch, time//2, odim) e.g. (1, 60, 512)
        """
        batch, time, idim = x.shape
        
        # Reshape for Conv2d: (batch, time, idim) → (batch, time, idim, 1)
        x = x.reshape(batch, time, idim, 1)
        
        # Conv2d expects (batch, height, width, channels)
        # Treat (time, idim) as (height, width)
        x = self.conv(x)  # (batch, time//2, idim//2, odim)
        
        # Reshape: (batch, time', idim', odim) → (batch, time', odim*idim')
        batch, time_new, idim_new, odim = x.shape
        x = x.reshape(batch, time_new, odim * idim_new)
        
        # Project to odim
        x = self.out(x)  # (batch, time//2, odim)
        
        return x


class MLXConv2dSubsampling2Fixed(nn.Module):
    """
    Fixed version that matches PyTorch Conv2dSubsampling2 exactly
    
    PyTorch does:
    1. unsqueeze to (b, c=1, t, f)
    2. Conv2d(1→odim, kernel=3, stride=2)
    3. Result: (b, odim, t', f')
    4. Transpose and flatten: (b, t', odim*f')
    5. Linear(odim*f' → odim)
    """
    
    def __init__(self, idim, odim):
        super().__init__()
        self.idim = idim
        self.odim = odim
        
        # Conv layer
        self.conv = MLXConv2d(
            in_channels=1,
            out_channels=odim,
            kernel_size=3,
            stride=2
        )
        
        # After conv with stride=2, kernel=3, padding=0:
        # time: (121 - 3) // 2 + 1 = 60
        # freq: (1024 - 3) // 2 + 1 = 511
        # So output is (batch, 60, 511, odim=512)
        # After transpose and flatten: (batch, 60, 512*511=261632)
        # Then project 261632 → 512
        
        # Calculate exact size: odim * ((idim - 3) // 2 + 1)
        idim_after_conv = (idim - 3) // 2 + 1
        self.out = nn.Linear(odim * idim_after_conv, odim)
    
    def __call__(self, x):
        """
        Args:
            x: (batch, time, idim) in MLX format
        
        Returns:
            x: (batch, time//2, odim)
        """
        batch, time, idim = x.shape
        
        # Reshape to (batch, time, idim, 1) for Conv2d
        # MLX Conv2d expects (batch, height, width, channels)
        x = x.reshape(batch, time, idim, 1)
        
        # Apply Conv2d: (batch, time, idim, 1) → (batch, time', idim', odim)
        x = self.conv(x)
        x = nn.relu(x)
        
        # PyTorch does: transpose(1,2) then flatten
        # MLX: (batch, time', idim', odim) → (batch, time', odim*idim')
        batch, time_new, idim_new, odim_channels = x.shape
        
        # Reshape: move odim (last dim) before idim', then flatten
        # (batch, time', idim', odim) → (batch, time', odim, idim') → (batch, time', odim*idim')
        x = x.transpose(0, 1, 3, 2)  # (batch, time', odim, idim')
        x = x.reshape(batch, time_new, odim_channels * idim_new)
        
        # Linear projection: (odim * idim') → odim
        x = self.out(x)
        
        return x


# Positional Encoding (simple version, will be replaced by loaded weights)
class MLXPositionalEncoding(nn.Module):
    """Positional encoding for Conformer"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        # Will be replaced by loaded weights from PyTorch
        self.pe = mx.zeros((max_len, d_model))
    
    def __call__(self, x):
        """Add positional encoding"""
        seq_len = x.shape[1]
        if seq_len <= self.pe.shape[0]:
            return x + self.pe[:seq_len]
        else:
            return x

