"""
改进的MLX WaveNet实现
尝试更接近PyTorch的SConv1d行为
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional
import math


def mlx_pad_reflect_1d(x, padding_left, padding_right):
    """
    MLX实现的reflect padding for 1D
    x: (batch, seq_len, channels)
    """
    if padding_left == 0 and padding_right == 0:
        return x
    
    batch, seq_len, channels = x.shape
    
    # Reflect padding
    if padding_left > 0:
        # Mirror the left side
        left_pad = x[:, 1:padding_left+1, :]  # Reflect first few elements
        left_pad = left_pad[:, ::-1, :]  # Reverse
        x = mx.concatenate([left_pad, x], axis=1)
    
    if padding_right > 0:
        # Mirror the right side
        right_pad = x[:, -(padding_right+1):-1, :]  # Reflect last few elements
        right_pad = right_pad[:, ::-1, :]  # Reverse
        x = mx.concatenate([x, right_pad], axis=1)
    
    return x


class MLXSConv1dLayer(nn.Module):
    """
    MLX implementation of a single SConv1d layer
    Attempts to mimic PyTorch's SConv1d padding behavior
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        causal: bool = False,
        pad_mode: str = 'reflect'
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.causal = causal
        self.pad_mode = pad_mode
        
        # MLX Conv1d with NO padding (we'll handle padding ourselves)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # No padding - we handle it manually
            bias=bias
        )
    
    def __call__(self, x):
        """
        x: (batch, seq_len, in_channels) - MLX format
        Returns: (batch, seq_len_out, out_channels)
        """
        batch, seq_len, channels = x.shape
        
        # Calculate effective kernel size with dilation
        effective_kernel_size = (self.kernel_size - 1) * self.dilation + 1
        padding_total = effective_kernel_size - self.stride
        
        # Calculate extra padding for alignment
        n_frames = (seq_len - effective_kernel_size + padding_total) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + effective_kernel_size - padding_total
        extra_padding = max(0, ideal_length - seq_len)
        
        if self.causal:
            # Causal: all padding on left
            padding_left = padding_total
            padding_right = extra_padding
        else:
            # Non-causal: symmetric padding
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            padding_right += extra_padding
        
        # Apply padding (using reflect mode if possible, zeros as fallback)
        if self.pad_mode == 'reflect' and padding_left < seq_len and padding_right < seq_len:
            x = mlx_pad_reflect_1d(x, padding_left, padding_right)
        else:
            # Fallback to zero padding
            if padding_left > 0 or padding_right > 0:
                pad_shape = (batch, padding_left, channels)
                left_pad = mx.zeros(pad_shape)
                if padding_right > 0:
                    right_pad = mx.zeros((batch, padding_right, channels))
                    x = mx.concatenate([left_pad, x, right_pad], axis=1)
                else:
                    x = mx.concatenate([left_pad, x], axis=1)
        
        # Apply convolution
        out = self.conv(x)
        
        return out


class MLXWaveNetImproved(nn.Module):
    """
    改进的MLX WaveNet实现
    使用接近SConv1d的padding策略
    """
    
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
        causal: bool = False
    ):
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.causal = causal
        
        # Input layers (dilated convolutions with SConv1d-like behavior)
        self.in_layers = []
        for i in range(n_layers):
            dilation = dilation_rate ** i
            
            layer = MLXSConv1dLayer(
                hidden_channels,
                2 * hidden_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                bias=True,
                causal=causal,
                pad_mode='reflect'
            )
            self.in_layers.append(layer)
        
        # Residual/skip layers (1x1 conv, no special padding needed)
        self.res_skip_layers = []
        for i in range(n_layers):
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels
            
            layer = nn.Conv1d(
                hidden_channels,
                res_skip_channels,
                kernel_size=1,
                padding=0,
                bias=True
            )
            self.res_skip_layers.append(layer)
        
        # Conditioning layer (if using global conditioning)
        if gin_channels != 0:
            self.cond_layer = nn.Conv1d(
                gin_channels,
                2 * hidden_channels * n_layers,
                kernel_size=1,
                padding=0,
                bias=True
            )
        else:
            self.cond_layer = None
        
        # Dropout
        self.dropout = nn.Dropout(p_dropout)
        
        print(f">> MLX WaveNet (Improved) initialized:")
        print(f"   Layers: {n_layers}, Channels: {hidden_channels}")
        print(f"   Kernel: {kernel_size}, Dilation rate: {dilation_rate}")
        print(f"   Using SConv1d-like padding (reflect mode)")
    
    def __call__(self, x, x_mask, g=None):
        """
        x: (batch, seq_len, hidden_channels) - MLX format
        x_mask: (batch, 1, seq_len) - mask (True = keep)
        g: (batch, seq_len, gin_channels) - global conditioning
        
        Returns:
            output: (batch, seq_len, hidden_channels)
        """
        # Reshape mask for MLX broadcast: (batch, 1, seq_len) -> (batch, seq_len, 1)
        x_mask_mlx = x_mask.transpose(0, 2, 1)  # (batch, seq_len, 1)
        
        output = mx.zeros_like(x)
        
        # Process global conditioning
        if g is not None and self.cond_layer is not None:
            g_cond = self.cond_layer(g)  # (batch, seq_len, 2*hidden_channels*n_layers)
        else:
            g_cond = None
        
        for i in range(self.n_layers):
            # Apply mask and dilated conv
            x_masked = x * x_mask_mlx
            x_in = self.in_layers[i](x_masked)  # (batch, seq_len, 2*hidden_channels)
            
            # Add global conditioning
            if g_cond is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g_cond[:, :, cond_offset:cond_offset + 2 * self.hidden_channels]
            else:
                g_l = mx.zeros_like(x_in)
            
            # Fused gated activation: tanh(a) * sigmoid(b)
            in_act = x_in + g_l
            t_act = mx.tanh(in_act[:, :, :self.hidden_channels])
            s_act = mx.sigmoid(in_act[:, :, self.hidden_channels:])
            acts = t_act * s_act
            
            # Dropout
            acts = self.dropout(acts)
            
            # Residual/skip connection
            res_skip_acts = self.res_skip_layers[i](acts)
            
            if i < self.n_layers - 1:
                # Split into residual and skip
                res_acts = res_skip_acts[:, :, :self.hidden_channels]
                skip_acts = res_skip_acts[:, :, self.hidden_channels:]
                
                x = (x + res_acts) * x_mask_mlx
                output = output + skip_acts
            else:
                # Last layer: only skip
                output = output + res_skip_acts
        
        return output * x_mask_mlx
    
    def load_weights_from_pytorch(self, state_dict, prefix="wavenet."):
        """
        Load weights from PyTorch WaveNet (with weight_norm removed)
        
        Args:
            state_dict: PyTorch state dict (numpy arrays)
            prefix: Key prefix
        """
        loaded = 0
        
        def convert_conv1d_weight(w):
            """PyTorch (O, I, K) -> MLX (O, K, I)"""
            return w.transpose(0, 2, 1)
        
        # Load in_layers
        for i in range(self.n_layers):
            # Conv weights
            w_key = f"{prefix}in_layers.{i}.weight"
            b_key = f"{prefix}in_layers.{i}.bias"
            
            if w_key in state_dict:
                w = state_dict[w_key]
                self.in_layers[i].conv.weight = mx.array(convert_conv1d_weight(w))
                loaded += 1
            if b_key in state_dict:
                self.in_layers[i].conv.bias = mx.array(state_dict[b_key])
                loaded += 1
        
        # Load res_skip_layers
        for i in range(self.n_layers):
            w_key = f"{prefix}res_skip_layers.{i}.weight"
            b_key = f"{prefix}res_skip_layers.{i}.bias"
            
            if w_key in state_dict:
                w = state_dict[w_key]
                self.res_skip_layers[i].weight = mx.array(convert_conv1d_weight(w))
                loaded += 1
            if b_key in state_dict:
                self.res_skip_layers[i].bias = mx.array(state_dict[b_key])
                loaded += 1
        
        # Load cond_layer
        if self.gin_channels != 0 and self.cond_layer is not None:
            w_key = f"{prefix}cond_layer.weight"
            b_key = f"{prefix}cond_layer.bias"
            
            if w_key in state_dict:
                w = state_dict[w_key]
                self.cond_layer.weight = mx.array(convert_conv1d_weight(w))
                loaded += 1
            if b_key in state_dict:
                self.cond_layer.bias = mx.array(state_dict[b_key])
                loaded += 1
        
        print(f">> MLX WaveNet (Improved) loaded {loaded} weights")
        return loaded

