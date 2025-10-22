"""
MLX Implementation of WaveNet for DiT Final Layer
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional


def mlx_pad_reflect_1d(x, padding_left, padding_right):
    """
    MLX实现的reflect padding for 1D  
    x: (batch, seq_len, channels)
    
    PyTorch reflect: 镜像反射但不包括边界本身
    例如: [1,2,3,4,5] pad(2,2) -> [3,2, 1,2,3,4,5, 4,3]
    """
    if padding_left == 0 and padding_right == 0:
        return x
    
    batch, seq_len, channels = x.shape
    
    # 左侧reflect padding
    if padding_left > 0:
        pad_size = min(padding_left, seq_len - 1)
        if pad_size > 0:
            # PyTorch reflect: 取索引[1, 2, ..., pad_size]并反转
            # 对于[1,2,3,4,5] pad_size=2: 取索引[1,2]的值[2,3]反转得[3,2]
            left_slice = x[:, 1:pad_size+1, :]  # Shape: (batch, pad_size, channels)
            # 手动反转：构建反转索引 [pad_size-1, pad_size-2, ..., 0]
            reverse_indices = mx.arange(pad_size - 1, -1, -1)
            left_pad = left_slice[:, reverse_indices, :]
            
            if pad_size < padding_left:
                # 不够的用edge
                extra = padding_left - pad_size
                edge_pad = mx.broadcast_to(x[:, 0:1, :], (batch, extra, channels))
                left_pad = mx.concatenate([edge_pad, left_pad], axis=1)
        else:
            left_pad = mx.broadcast_to(x[:, 0:1, :], (batch, padding_left, channels))
        
        x = mx.concatenate([left_pad, x], axis=1)
    
    # 右侧reflect padding
    if padding_right > 0:
        # 注意：此时x已经被左padding扩展了
        # 原始序列的最后一个索引
        original_end_idx = seq_len + (padding_left if padding_left > 0 else 0) - 1
        
        pad_size = min(padding_right, seq_len - 1)
        if pad_size > 0:
            # PyTorch reflect: 取倒数第[2, 3, ..., pad_size+1]个元素并反转
            # 对于[..., 8,9,10] pad_size=2: 取索引[-2,-3]相对于original_end的值[9,8]反转得[9,8]...不对
            # 应该是: 取[original_end-pad_size:original_end]即[8,9]反转得[9,8]
            right_slice = x[:, original_end_idx-pad_size:original_end_idx, :]
            # 手动反转
            reverse_indices = mx.arange(pad_size - 1, -1, -1)
            right_pad = right_slice[:, reverse_indices, :]
            
            if pad_size < padding_right:
                extra = padding_right - pad_size
                edge_pad = mx.broadcast_to(x[:, original_end_idx:original_end_idx+1, :], (batch, extra, channels))
                right_pad = mx.concatenate([right_pad, edge_pad], axis=1)
        else:
            right_pad = mx.broadcast_to(x[:, original_end_idx:original_end_idx+1, :], (batch, padding_right, channels))
        
        x = mx.concatenate([x, right_pad], axis=1)
    
    return x


def fused_add_tanh_sigmoid_multiply_mlx(input_a, input_b, n_channels):
    """
    MLX implementation of fused gated activation.
    
    Args:
        input_a: (batch, 2*n_channels, seq_len)
        input_b: (batch, 2*n_channels, seq_len)
        n_channels: int
    
    Returns:
        output: (batch, n_channels, seq_len)
    """
    # Note: MLX uses (batch, seq_len, channels) format
    # So we need to adapt
    
    in_act = input_a + input_b
    
    # Split channels
    t_act = mx.tanh(in_act[:, :, :n_channels])
    s_act = mx.sigmoid(in_act[:, :, n_channels:])
    
    return t_act * s_act


class MLXWaveNet(nn.Module):
    """
    MLX implementation of WaveNet.
    Matches PyTorch SConv1d behavior with reflect padding.
    """
    
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0
    ):
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        
        # Input layers (dilated convolutions)
        # Note: PyTorch SConv1d uses padding=0 in Conv1d and adds padding manually
        self.in_layers = []
        for i in range(n_layers):
            dilation = dilation_rate ** i
            
            # SConv1d uses padding=0 and handles padding in forward
            layer = nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=0,  # No padding - we'll add it manually
                bias=True
            )
            self.in_layers.append(layer)
        
        # Residual/skip layers
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
        
        print(f">> MLX WaveNet initialized:")
        print(f"   Layers: {n_layers}, Channels: {hidden_channels}")
        print(f"   Kernel: {kernel_size}, Dilation rate: {dilation_rate}")
    
    def __call__(self, x, x_mask, g=None):
        """
        x: (batch, seq_len, hidden_channels) - MLX format
        x_mask: (batch, 1, seq_len) - mask (True = keep)
        g: (batch, seq_len, gin_channels) - global conditioning
        
        Returns:
            output: (batch, seq_len, hidden_channels)
        """
        # Note: MLX Conv1d expects (batch, seq_len, channels)
        # PyTorch WaveNet uses (batch, channels, seq_len)
        # We keep MLX format internally
        
        # Reshape mask for MLX broadcast: (batch, 1, seq_len) -> (batch, seq_len, 1)
        x_mask_mlx = x_mask.transpose(0, 2, 1)  # (batch, seq_len, 1)
        
        output = mx.zeros_like(x)
        
        # Process global conditioning
        if g is not None and self.cond_layer is not None:
            g_cond = self.cond_layer(g)  # (batch, seq_len, 2*hidden_channels*n_layers)
        else:
            g_cond = None
        
        for i in range(self.n_layers):
            # Apply mask
            x_masked = x * x_mask_mlx
            
            # Apply reflect padding to match SConv1d behavior
            # SConv1d calculates: padding_total = kernel_size - stride
            # For stride=1: padding_total = kernel_size - 1
            dilation = self.dilation_rate ** i
            effective_kernel_size = (self.kernel_size - 1) * dilation + 1
            padding_total = effective_kernel_size - 1  # stride=1
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            
            # Apply reflect padding
            x_padded = mlx_pad_reflect_1d(x_masked, padding_left, padding_right)
            
            # Apply convolution (padding=0 since we manually padded)
            x_in = self.in_layers[i](x_padded)  # (batch, seq_len, 2*hidden_channels)
            
            # Add global conditioning
            if g_cond is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g_cond[:, :, cond_offset:cond_offset + 2 * self.hidden_channels]
            else:
                g_l = mx.zeros_like(x_in)
            
            # Fused gated activation
            acts = fused_add_tanh_sigmoid_multiply_mlx(x_in, g_l, self.hidden_channels)
            
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

