#!/usr/bin/env python3
"""
逐行翻译PyTorch SConv1d到MLX版本 - 完全一致性测试
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
        return self.conv(x_padded)

def test_mlx_sconv1d_consistency():
    """测试MLX SConv1d与PyTorch SConv1d的完全一致性"""
    print("=== 测试逐行翻译的MLX SConv1d与PyTorch SConv1d的完全一致性 ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    mx.random.seed(42)
    
    # 创建PyTorch SConv1d
    from indextts.s2mel.modules.encodec import SConv1d
    pytorch_sconv = SConv1d(512, 1024, 5, dilation=1, bias=True)
    
    # 创建MLX SConv1d
    mlx_sconv = MLXSConv1d(512, 1024, 5, dilation=1, bias=True)
    
    # 同步权重：将PyTorch权重复制到MLX
    pytorch_weight = pytorch_sconv.conv.conv.weight.detach().cpu().numpy()
    pytorch_bias = pytorch_sconv.conv.conv.bias.detach().cpu().numpy()
    
    # 转换权重格式：PyTorch (out_channels, in_channels, kernel_size) -> MLX (out_channels, kernel_size, in_channels)
    mlx_weight = mx.array(pytorch_weight.transpose(0, 2, 1))
    mlx_bias = mx.array(pytorch_bias)
    
    # 设置MLX权重
    mlx_sconv.conv.conv.weight = mlx_weight
    mlx_sconv.conv.conv.bias = mlx_bias
    
    # 创建测试数据
    batch_size = 1
    seq_len = 100
    channels = 512
    
    # PyTorch格式: (batch, channels, seq_len)
    x_torch = torch.randn(batch_size, channels, seq_len)
    
    # MLX格式: (batch, seq_len, channels)
    x_mlx = mx.array(x_torch.numpy().transpose(0, 2, 1))
    
    print(f"输入形状: PyTorch={x_torch.shape}, MLX={x_mlx.shape}")
    
    # 前向传播
    pytorch_output = pytorch_sconv(x_torch)
    mlx_output = mlx_sconv(x_mlx)
    
    # 转换MLX输出格式进行比较
    mlx_output_torch_format = mx.transpose(mlx_output, (0, 2, 1))
    mlx_output_np = np.array(mlx_output_torch_format)
    pytorch_output_np = pytorch_output.detach().cpu().numpy()
    
    print(f"PyTorch输出: {pytorch_output.shape}")
    print(f"MLX输出: {mlx_output.shape}")
    print(f"MLX输出(转换后): {mlx_output_torch_format.shape}")
    
    # 检查输出长度一致性
    if pytorch_output.shape[2] == mlx_output_torch_format.shape[2]:
        print("✅ 输出长度一致")
    else:
        print("❌ 输出长度不一致")
        return
    
    # 计算数值差异
    diff = np.abs(pytorch_output_np - mlx_output_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"数值差异: max={max_diff:.6f}, mean={mean_diff:.6f}")
    
    # 计算差异分位数
    diff_flat = diff.flatten()
    percentiles = [50, 90, 95, 99]
    print("差异分位数:", end="")
    for p in percentiles:
        val = np.percentile(diff_flat, p)
        print(f" {p}%={val:.6f}", end="")
    print()
    
    # 测试不同的dilation值
    test_cases = [
        (1, "Dilation=1"),
        (2, "Dilation=2"),
        (4, "Dilation=4"),
        (8, "Dilation=8")
    ]
    
    for dilation, name in test_cases:
        print(f"\n--- {name} ---")
        
        # 创建新的模型
        pytorch_sconv = SConv1d(512, 1024, 5, dilation=dilation, bias=True)
        mlx_sconv = MLXSConv1d(512, 1024, 5, dilation=dilation, bias=True)
        
        # 同步权重：将PyTorch权重复制到MLX
        pytorch_weight = pytorch_sconv.conv.conv.weight.detach().cpu().numpy()
        pytorch_bias = pytorch_sconv.conv.conv.bias.detach().cpu().numpy()
        
        # 转换权重格式：PyTorch (out_channels, in_channels, kernel_size) -> MLX (out_channels, kernel_size, in_channels)
        mlx_weight = mx.array(pytorch_weight.transpose(0, 2, 1))
        mlx_bias = mx.array(pytorch_bias)
        
        # 设置MLX权重
        mlx_sconv.conv.conv.weight = mlx_weight
        mlx_sconv.conv.conv.bias = mlx_bias
        
        # 前向传播
        pytorch_output = pytorch_sconv(x_torch)
        mlx_output = mlx_sconv(x_mlx)
        
        # 转换MLX输出格式进行比较
        mlx_output_torch_format = mx.transpose(mlx_output, (0, 2, 1))
        mlx_output_np = np.array(mlx_output_torch_format)
        pytorch_output_np = pytorch_output.detach().cpu().numpy()
        
        print(f"PyTorch输出: {pytorch_output.shape}")
        print(f"MLX输出: {mlx_output.shape}")
        print(f"MLX输出(转换后): {mlx_output_torch_format.shape}")
        
        # 检查输出长度一致性
        if pytorch_output.shape[2] == mlx_output_torch_format.shape[2]:
            print("✅ 输出长度一致")
        else:
            print("❌ 输出长度不一致")
            continue
        
        # 计算数值差异
        diff = np.abs(pytorch_output_np - mlx_output_np)
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        print(f"数值差异: max={max_diff:.6f}, mean={mean_diff:.6f}")
        
        # 计算差异分位数
        diff_flat = diff.flatten()
        percentiles = [50, 90, 95, 99]
        print("差异分位数:", end="")
        for p in percentiles:
            val = np.percentile(diff_flat, p)
            print(f" {p}%={val:.6f}", end="")
        print()

if __name__ == "__main__":
    test_mlx_sconv1d_consistency()
