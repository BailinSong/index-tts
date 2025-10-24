#!/usr/bin/env python3
"""
调试MLX SConv1d与PyTorch SConv1d的数值差异
"""

import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import math
from typing import Tuple, Dict, Any
import typing as tp

def get_extra_padding_for_conv1d_mlx(x: mx.array, kernel_size: int, stride: int,
                                      padding_total: int = 0) -> int:
    """MLX版本的get_extra_padding_for_conv1d"""
    length = x.shape[1]  # MLX格式: (batch, seq_len, channels)
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return int(ideal_length - length)

def pad1d_mlx(x: mx.array, paddings: tp.Tuple[int, int], mode: str = 'zero', value: float = 0.):
    """MLX版本的pad1d，完全复刻PyTorch F.pad的行为"""
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
        
        return x_padded_mlx
    else:
        return mx.pad(x, ((0, 0), paddings, (0, 0)), mode='constant', constant_values=value)

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
    """MLX版本的SConv1d，完全复刻PyTorch版本"""
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
        
        print(f"调试信息:")
        print(f"  输入长度: {x.shape[1]}")
        print(f"  原始kernel_size: {self.conv.conv.weight.shape[1]}")
        print(f"  有效kernel_size: {kernel_size}")
        print(f"  stride: {stride}")
        print(f"  dilation: {dilation}")
        print(f"  padding_total: {padding_total}")
        print(f"  extra_padding: {extra_padding}")
        
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
        
        print(f"  padded长度: {x_padded.shape[1]}")
        
        # PyTorch: return self.conv(x)
        # MLX: 完全相同
        output = self.conv(x_padded)
        
        print(f"  conv输出长度: {output.shape[1]}")
        
        return output

def debug_weight_differences():
    """调试权重差异"""
    print("=== 调试权重差异 ===")
    
    # 创建PyTorch SConv1d
    from indextts.s2mel.modules.encodec import SConv1d
    pytorch_sconv = SConv1d(512, 1024, 5, dilation=1, bias=True)
    
    # 创建MLX SConv1d
    mlx_sconv = MLXSConv1d(512, 1024, 5, dilation=1, bias=True)
    
    # 获取PyTorch权重
    pytorch_weight = pytorch_sconv.conv.conv.weight.detach().cpu().numpy()
    pytorch_bias = pytorch_sconv.conv.conv.bias.detach().cpu().numpy()
    
    print(f"PyTorch权重形状: {pytorch_weight.shape}")
    print(f"PyTorch偏置形状: {pytorch_bias.shape}")
    print(f"PyTorch权重范围: [{pytorch_weight.min():.6f}, {pytorch_weight.max():.6f}]")
    print(f"PyTorch偏置范围: [{pytorch_bias.min():.6f}, {pytorch_bias.max():.6f}]")
    
    # 转换权重格式
    mlx_weight = pytorch_weight.transpose(0, 2, 1)
    mlx_bias = pytorch_bias
    
    print(f"MLX权重形状: {mlx_weight.shape}")
    print(f"MLX偏置形状: {mlx_bias.shape}")
    print(f"MLX权重范围: [{mlx_weight.min():.6f}, {mlx_weight.max():.6f}]")
    print(f"MLX偏置范围: [{mlx_bias.min():.6f}, {mlx_bias.max():.6f}]")
    
    # 检查权重是否完全一致
    weight_diff = np.abs(pytorch_weight - mlx_weight.transpose(0, 2, 1))
    bias_diff = np.abs(pytorch_bias - mlx_bias)
    
    print(f"权重差异: max={weight_diff.max():.10f}, mean={weight_diff.mean():.10f}")
    print(f"偏置差异: max={bias_diff.max():.10f}, mean={bias_diff.mean():.10f}")
    
    # 设置MLX权重
    mlx_sconv.conv.conv.weight = mx.array(mlx_weight)
    mlx_sconv.conv.conv.bias = mx.array(mlx_bias)
    
    return pytorch_sconv, mlx_sconv

def debug_padding_differences():
    """调试padding差异"""
    print("\n=== 调试padding差异 ===")
    
    # 创建测试数据
    x_torch = torch.randn(1, 512, 100)
    x_mlx = mx.array(x_torch.numpy().transpose(0, 2, 1))
    
    print(f"输入形状: PyTorch={x_torch.shape}, MLX={x_mlx.shape}")
    
    # 创建模型
    pytorch_sconv, mlx_sconv = debug_weight_differences()
    
    # 调试PyTorch的padding过程
    print("\n--- PyTorch padding过程 ---")
    B, C, T = x_torch.shape
    kernel_size = pytorch_sconv.conv.conv.kernel_size[0]
    stride = pytorch_sconv.conv.conv.stride[0]
    dilation = pytorch_sconv.conv.conv.dilation[0]
    kernel_size = (kernel_size - 1) * dilation + 1
    padding_total = kernel_size - stride
    
    from indextts.s2mel.modules.encodec import get_extra_padding_for_conv1d
    extra_padding = get_extra_padding_for_conv1d(x_torch, kernel_size, stride, padding_total)
    
    print(f"PyTorch kernel_size: {kernel_size}")
    print(f"PyTorch padding_total: {padding_total}")
    print(f"PyTorch extra_padding: {extra_padding}")
    
    # 应用padding
    padding_right = padding_total // 2
    padding_left = padding_total - padding_right
    
    from indextts.s2mel.modules.encodec import pad1d
    x_padded_torch = pad1d(x_torch, (padding_left, padding_right + extra_padding), mode='reflect')
    print(f"PyTorch padded形状: {x_padded_torch.shape}")
    
    # 调试MLX的padding过程
    print("\n--- MLX padding过程 ---")
    batch, seq_len, channels = x_mlx.shape
    kernel_size_mlx = mlx_sconv.conv.conv.weight.shape[1]
    stride_mlx = mlx_sconv.conv.conv.stride
    dilation_mlx = mlx_sconv.conv.conv.dilation
    kernel_size_mlx = (kernel_size_mlx - 1) * dilation_mlx + 1
    padding_total_mlx = kernel_size_mlx - stride_mlx
    
    extra_padding_mlx = mlx_sconv.get_extra_padding_for_conv1d(x_mlx, kernel_size_mlx, stride_mlx, padding_total_mlx)
    
    print(f"MLX kernel_size: {kernel_size_mlx}")
    print(f"MLX padding_total: {padding_total_mlx}")
    print(f"MLX extra_padding: {extra_padding_mlx}")
    
    # 应用padding
    padding_right_mlx = padding_total_mlx // 2
    padding_left_mlx = padding_total_mlx - padding_right_mlx
    
    x_padded_mlx = mlx_sconv.mlx_pad1d(x_mlx, (padding_left_mlx, padding_right_mlx + extra_padding_mlx), mode='reflect')
    print(f"MLX padded形状: {x_padded_mlx.shape}")
    
    # 比较padded结果
    x_padded_torch_np = x_padded_torch.numpy().transpose(0, 2, 1)
    x_padded_mlx_np = np.array(x_padded_mlx)
    
    padded_diff = np.abs(x_padded_torch_np - x_padded_mlx_np)
    print(f"Padded差异: max={padded_diff.max():.10f}, mean={padded_diff.mean():.10f}")
    
    return x_padded_torch, x_padded_mlx

def debug_conv_differences():
    """调试卷积差异"""
    print("\n=== 调试卷积差异 ===")
    
    x_padded_torch, x_padded_mlx = debug_padding_differences()
    
    # 创建模型
    pytorch_sconv, mlx_sconv = debug_weight_differences()
    
    # 应用卷积
    pytorch_output = pytorch_sconv.conv(x_padded_torch)
    mlx_output = mlx_sconv.conv(x_padded_mlx)
    
    print(f"PyTorch conv输出形状: {pytorch_output.shape}")
    print(f"MLX conv输出形状: {mlx_output.shape}")
    
    # 转换MLX输出格式进行比较
    mlx_output_torch_format = mx.transpose(mlx_output, (0, 2, 1))
    mlx_output_np = np.array(mlx_output_torch_format)
    pytorch_output_np = pytorch_output.detach().cpu().numpy()
    
    conv_diff = np.abs(pytorch_output_np - mlx_output_np)
    print(f"Conv差异: max={conv_diff.max():.10f}, mean={conv_diff.mean():.10f}")
    
    return pytorch_output, mlx_output

if __name__ == "__main__":
    print("=== 调试MLX SConv1d与PyTorch SConv1d的数值差异 ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    mx.random.seed(42)
    
    # 调试权重差异
    debug_weight_differences()
    
    # 调试padding差异
    debug_padding_differences()
    
    # 调试卷积差异
    debug_conv_differences()
