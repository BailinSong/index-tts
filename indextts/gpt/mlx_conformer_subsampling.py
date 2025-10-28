"""
MLX implementation of Conv2d Subsampling for Conformer
Matches PyTorch Conv2dSubsampling2 behavior

Uses MLX native Conv2d for numerical accuracy.
"""

import mlx.core as mx
import mlx.nn as nn
import math


class MLXConv2dNative(nn.Module):
    """
    Native MLX Conv2d wrapper that matches PyTorch behavior
    
    PyTorch Conv2d format: (batch, in_channels, height, width)
    MLX Conv2d format: (batch, height, width, in_channels)
    
    This wrapper handles the format conversion.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Use MLX's native Conv2d
        # MLX Conv2d kernel format: (out_channels, kernel_h, kernel_w, in_channels)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
    
    @property
    def weight(self):
        """
        Expose weight in PyTorch format for loading
        PyTorch: (out_channels, in_channels, kernel_h, kernel_w)
        MLX: (out_channels, kernel_h, kernel_w, in_channels)
        """
        # Return in MLX format, conversion will be handled by loader
        return self.conv.weight
    
    @weight.setter
    def weight(self, value):
        """Set weight, handling format conversion if needed"""
        self.conv.weight = value
    
    @property
    def bias(self):
        return self.conv.bias
    
    @bias.setter
    def bias(self, value):
        self.conv.bias = value
    
    def __call__(self, x):
        """
        Forward pass matching PyTorch behavior
        
        Args:
            x: (batch, height, width, in_channels) - MLX format
        
        Returns:
            out: (batch, height', width', out_channels) - MLX format
        """
        # MLX Conv2d expects (batch, height, width, in_channels)
        # which is what we have
        return self.conv(x)


class MLXConv2dSubsampling2(nn.Module):
    """
    Conv2d Subsampling with factor 2 (matches PyTorch Conv2dSubsampling2)
    
    Uses native MLX Conv2d for numerical accuracy.
    
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
        
        # Use native MLX Conv2d
        self.conv = MLXConv2dNative(
            in_channels=1,
            out_channels=odim,
            kernel_size=3,
            stride=2,
            padding=0
        )
        
        # Calculate projection size after conv
        # After conv with kernel=3, stride=2, padding=0:
        # out_dim = (in_dim - 3) // 2 + 1
        # For idim=1024: (1024 - 3) // 2 + 1 = 511
        # Total: odim * 511 → odim
        self.out = nn.Linear(odim * ((idim - 3) // 2 + 1), odim)
    
    def __call__(self, x):
        """
        Args:
            x: (batch, time, idim) e.g. (1, 50, 1024)
        
        Returns:
            x: (batch, time//2, odim) e.g. (1, 24, 512)
        """
        batch, time, idim = x.shape
        
        # Reshape for Conv2d: (batch, time, idim) → (batch, time, idim, 1)
        # MLX Conv2d expects (batch, height, width, in_channels)
        # We treat time as height, idim as width, 1 as in_channels
        x = x.reshape(batch, time, idim, 1)
        
        # Apply Conv2d + ReLU
        x = self.conv(x)  # (batch, time', idim', odim)
        x = nn.relu(x)
        
        # Transpose to match PyTorch's ordering
        # PyTorch does: (b, c, t, f) → transpose(1,2) → (b, t, c, f) → flatten to (b, t, c*f)
        # MLX: (b, t', f', c) → transpose → (b, t', c, f') → flatten to (b, t', c*f')
        batch, time_new, idim_new, odim_channels = x.shape
        x = x.transpose(0, 1, 3, 2)  # (batch, time', odim, idim')
        x = x.reshape(batch, time_new, odim_channels * idim_new)
        
        # Linear projection
        x = self.out(x)  # (batch, time//2, odim)
        
        return x


class MLXConv2dSubsampling2Fixed(nn.Module):
    """
    Fixed version that matches PyTorch Conv2dSubsampling2 exactly
    
    Uses native MLX Conv2d for numerical accuracy.
    
    PyTorch does:
    1. unsqueeze to (b, c=1, t, f)
    2. Conv2d(1→odim, kernel=3, stride=2) + ReLU
    3. Result: (b, odim, t', f')
    4. Transpose and flatten: (b, t', odim*f')
    5. Linear(odim*f' → odim)
    """
    
    def __init__(self, idim, odim):
        super().__init__()
        self.idim = idim
        self.odim = odim
        
        # Use native MLX Conv2d (same as MLXConv2dSubsampling2)
        self.conv = MLXConv2dNative(
            in_channels=1,
            out_channels=odim,
            kernel_size=3,
            stride=2,
            padding=0
        )
        
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
        
        # Reshape for Conv2d: (batch, time, idim) → (batch, time, idim, 1)
        x = x.reshape(batch, time, idim, 1)
        
        # Apply Conv2d + ReLU
        x = self.conv(x)  # (batch, time', idim', odim)
        x = nn.relu(x)
        
        # Transpose to match PyTorch's ordering
        # PyTorch: (b, c, t, f) → transpose(1,2) → (b, t, c, f) → flatten to (b, t, c*f)
        # MLX: (b, t', f', c) → transpose → (b, t', c, f') → flatten to (b, t', c*f')
        batch, time_new, idim_new, odim_channels = x.shape
        x = x.transpose(0, 1, 3, 2)  # (batch, time', odim, idim')
        x = x.reshape(batch, time_new, odim_channels * idim_new)
        
        # Linear projection
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

