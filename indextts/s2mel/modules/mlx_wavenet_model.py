"""
MLX Implementation of WaveNet for DiT Final Layer
使用经过测试验证的SConv1d实现
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any
import typing as tp


def get_extra_padding_for_conv1d_mlx(x: mx.array, kernel_size: int, stride: int,
                                      padding_total: int = 0) -> int:
    """MLX版本的get_extra_padding_for_conv1d，完全复刻PyTorch版本"""
    length = x.shape[1]  # MLX格式: (batch, seq_len, channels)
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return int(ideal_length - length)

def pad1d_mlx(x: mx.array, paddings: tp.Tuple[int, int], mode: str = 'zero', value: float = 0.):
    """MLX版本的pad1d，完全复刻PyTorch F.pad的行为，包括截断操作"""
    import torch
    
    length = x.shape[1]  # MLX格式: (batch, seq_len, channels)
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = mx.pad(x, ((0, 0), (0, extra_pad), (0, 0)), mode='constant', constant_values=0)
        
        # 使用PyTorch的reflect padding逻辑，完全复刻
        # 转换为PyTorch格式进行padding，然后转换回MLX格式
        x_torch = torch.from_numpy(np.array(x).transpose(0, 2, 1))  # MLX -> PyTorch格式
        x_padded_torch = torch.nn.functional.pad(x_torch, (padding_left, padding_right), mode='reflect')
        x_padded_mlx = mx.array(x_padded_torch.numpy().transpose(0, 2, 1))  # PyTorch -> MLX格式
        
        # 关键修复：截断到原始长度，完全复刻PyTorch的pad1d行为
        end = x_padded_mlx.shape[1] - extra_pad
        return x_padded_mlx[:, :end, :]
    else:
        return mx.pad(x, ((0, 0), paddings, (0, 0)), mode='constant', constant_values=value)

def mlx_pad_reflect_1d(x, padding_left, padding_right):
    """
    MLX实现的reflect padding for 1D，完全复刻PyTorch F.pad的行为
    x: (batch, seq_len, channels)
    """
    return pad1d_mlx(x, (padding_left, padding_right), mode='reflect')


class MLXNormConv1d(nn.Module):
    """MLX版本的NormConv1d，完全复刻PyTorch版本"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, groups: int = 1, bias: bool = True,
                 causal: bool = False, norm: str = 'none',
                 norm_kwargs: tp.Dict[str, tp.Any] = {}):
        super().__init__()
        self.causal = causal
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                             dilation=dilation, groups=groups, bias=bias)

    def __call__(self, x):
        return self.conv(x)

class MLXSConv1d(nn.Module):
    """MLX版本的SConv1d，完全复刻PyTorch版本 - 经过测试验证的实现"""
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, dilation: int = 1,
                 groups: int = 1, bias: bool = True, causal: bool = False,
                 norm: str = 'none', norm_kwargs: tp.Dict[str, tp.Any] = {},
                 pad_mode: str = 'reflect', **kwargs):
        super().__init__()
        self.kernel_size_init = kernel_size  # Store original kernel_size
        self.stride_init = stride
        self.dilation_init = dilation
        self.causal = causal
        self.pad_mode = pad_mode
        
        # MLX NormConv1d equivalent
        self.conv = MLXNormConv1d(in_channels, out_channels, kernel_size, stride,
                                  dilation=dilation, groups=groups, bias=bias, causal=causal,
                                  norm=norm, norm_kwargs=norm_kwargs)

    def get_extra_padding_for_conv1d(self, x: mx.array, kernel_size: int, stride: int,
                                     padding_total: int = 0) -> int:
        """MLX版本的get_extra_padding_for_conv1d"""
        return get_extra_padding_for_conv1d_mlx(x, kernel_size, stride, padding_total)

    def mlx_pad1d(self, x: mx.array, paddings: tp.Tuple[int, int], mode: str = 'zero', value: float = 0.):
        """MLX版本的pad1d，模拟PyTorch F.pad的行为"""
        return pad1d_mlx(x, paddings, mode, value)

    def __call__(self, x):
        # PyTorch: B, C, T = x.shape
        # MLX: batch, channels, seq_len = x.shape  # MLX格式: (batch, seq_len, channels)
        batch, seq_len, channels = x.shape
    
        # PyTorch: kernel_size = self.conv.conv.kernel_size[0]
        # MLX: kernel_size = self.conv.conv.weight.shape[1]  # MLX Conv1d的kernel_size在weight的第二个维度
        kernel_size = self.conv.conv.weight.shape[1]
        
        # PyTorch: stride = self.conv.conv.stride[0]
        # MLX: stride = self.conv.conv.stride
        stride = self.conv.conv.stride
        
        # PyTorch: dilation = self.conv.conv.dilation[0]
        # MLX: dilation = self.conv.conv.dilation
        dilation = self.conv.conv.dilation
        
        # PyTorch: kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size with dilations
        # MLX: 完全相同
        kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size with dilations
        
        # PyTorch: padding_total = kernel_size - stride
        # MLX: 完全相同
        padding_total = kernel_size - stride
        
        # PyTorch: extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        # MLX: 使用MLX版本
        extra_padding = self.get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        
        # PyTorch: if self.causal: x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        # MLX: 使用MLX版本
        if self.causal:
            # Left padding for causal
            x_padded = self.mlx_pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        # PyTorch: else: padding_right = padding_total // 2; padding_left = padding_total - padding_right; x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        # MLX: 使用MLX版本
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x_padded = self.mlx_pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        
        # PyTorch: return self.conv(x)
        # MLX: 完全相同
        output = self.conv(x_padded)
    
        return output

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
    使用经过测试验证的SConv1d实现，完全复刻PyTorch行为
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
        
        # Input layers (dilated convolutions with tested SConv1d)
        self.in_layers = []
        for i in range(n_layers):
            dilation = dilation_rate ** i
            
            layer = MLXSConv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                bias=True,
                causal=False,
                pad_mode='reflect'
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
        
        print(f">> MLX WaveNet initialized (with tested SConv1d):")
        print(f"   Layers: {n_layers}, Channels: {hidden_channels}")
        print(f"   Kernel: {kernel_size}, Dilation rate: {dilation_rate}")
        print(f"   Using tested SConv1d implementation")
    
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
        
        # 检查mask的形状并适当处理
        if len(x_mask.shape) == 3 and x_mask.shape[1] == 1:
            # 输入是 (batch, 1, seq_len)，需要转换为 (batch, seq_len, 1)
            x_mask_mlx = x_mask.transpose((0, 2, 1))  # (batch, seq_len, 1)
        else:
            # 输入已经是 (batch, seq_len, 1) 格式
            x_mask_mlx = x_mask
        
        output = mx.zeros_like(x)
        
        # Process global conditioning
        if g is not None and self.cond_layer is not None:
            # 修复g维度处理：处理不同的g形状
            if len(g.shape) == 4:  # (batch, 1, 1, gin_channels)
                g = g.squeeze(1).squeeze(1)  # (batch, gin_channels)
            elif len(g.shape) == 3 and g.shape[1] == 1:  # (batch, 1, gin_channels)
                g = g.squeeze(1)  # (batch, gin_channels)
            elif len(g.shape) == 3 and g.shape[1] > 1:  # (batch, seq_len, gin_channels)
                # 已经是正确格式，直接使用
                pass
            elif len(g.shape) == 2:  # (batch, gin_channels)
                # 已经是正确格式，直接使用
                pass
            
            # 如果g是(batch, gin_channels)，需要广播到所有时间步
            if len(g.shape) == 2:
                g = mx.broadcast_to(g.reshape(g.shape[0], 1, -1), (g.shape[0], x.shape[1], g.shape[-1]))
            
            # 确保g的形状正确：(batch, seq_len, gin_channels)
            if len(g.shape) == 3 and g.shape[2] == 1:
                # 如果g是(batch, gin_channels, 1)，需要转置并广播到所有时间步
                g = g.transpose(0, 2, 1)  # (batch, 1, gin_channels)
                g = mx.broadcast_to(g, (g.shape[0], x.shape[1], g.shape[2]))  # (batch, seq_len, gin_channels)
            elif len(g.shape) == 3 and g.shape[1] == 1:
                # 如果g是(batch, 1, gin_channels)，广播到所有时间步
                g = mx.broadcast_to(g, (g.shape[0], x.shape[1], g.shape[2]))
            
            g_cond = self.cond_layer(g)  # (batch, seq_len, 2*hidden_channels*n_layers)
        else:
            g_cond = None
        
        for i in range(self.n_layers):
            # Apply mask and dilated conv (SConv1d handles padding automatically)
            x_masked = x * x_mask_mlx
            x_in = self.in_layers[i](x_masked)  # (batch, seq_len, 2*hidden_channels)
            
            # Add global conditioning
            if g_cond is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g_cond[:, :, cond_offset:cond_offset + 2 * self.hidden_channels]
            else:
                g_l = mx.zeros_like(x_in)
            
            # Fused gated activation
            acts = fused_add_tanh_sigmoid_multiply_mlx(x_in, g_l, self.hidden_channels)
            
            # Dropout (disabled to match PyTorch eval behavior)
            acts = acts
            
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

