#!/usr/bin/env python3
"""
修复MLX WaveNet的权重格式问题
"""

import sys
sys.path.append('.')
import torch
import mlx.core as mx
import mlx.nn as nn
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def test_mlx_conv1d_direct():
    """直接测试MLX Conv1d"""
    print("=== 直接测试MLX Conv1d ===")
    
    # 设置随机种子
    mx.random.seed(42)
    
    # 创建MLX Conv1d
    mlx_conv = nn.Conv1d(512, 1024, 5, dilation=1, bias=True)
    
    print(f"MLX Conv1d权重形状: {mlx_conv.weight.shape}")
    
    # 创建测试输入
    batch_size = 1
    seq_len = 100
    x_mlx = mx.random.normal((batch_size, seq_len, 512))
    
    print(f"输入形状: {x_mlx.shape}")
    
    # 前向传播
    try:
        mlx_output = mlx_conv(x_mlx)
        print(f"输出形状: {mlx_output.shape}")
        print(f"输出范围: [{mx.min(mlx_output):.6f}, {mx.max(mlx_output):.6f}]")
    except Exception as e:
        print(f"错误: {e}")

def test_weight_format_compatibility():
    """测试权重格式兼容性"""
    print("=== 测试MLX Conv1d权重格式兼容性 ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    mx.random.seed(42)
    
    # 创建PyTorch Conv1d
    pytorch_conv = torch.nn.Conv1d(512, 1024, 5, dilation=1, bias=True)
    
    # 创建MLX兼容Conv1d
    mlx_conv = MLXConv1dCompat(512, 1024, 5, dilation=1, bias=True)
    
    print(f"PyTorch权重形状: {pytorch_conv.weight.shape}")
    print(f"MLX权重形状: {mlx_conv.weight.shape}")
    
    # 检查权重格式
    pytorch_weight = pytorch_conv.weight  # (1024, 512, 5)
    mlx_weight = mlx_conv.weight  # (1024, 5, 512)
    
    # 转换MLX权重到PyTorch格式进行比较
    mlx_weight_torch = mlx_to_torch(mlx_weight)
    # MLX格式: (out_channels, kernel_size, in_channels)
    # PyTorch格式: (out_channels, in_channels, kernel_size)
    # 需要转置: (out_channels, kernel_size, in_channels) -> (out_channels, in_channels, kernel_size)
    mlx_weight_torch = mlx_weight_torch.transpose(1, 2)
    
    print(f"权重差异: max={torch.max(torch.abs(pytorch_weight - mlx_weight_torch)):.6f}")
    
    # 测试前向传播
    batch_size = 1
    seq_len = 100
    
    # PyTorch输入: (batch, channels, seq_len)
    x_torch = torch.randn(batch_size, 512, seq_len)
    
    # MLX输入: (batch, seq_len, channels)
    x_mlx = torch_to_mlx(x_torch.transpose(1, 2))
    
    # 前向传播
    pytorch_output = pytorch_conv(x_torch)
    mlx_output = mlx_conv(x_mlx)
    
    # 转换MLX输出到PyTorch格式
    mlx_output_torch = mlx_to_torch(mlx_output).transpose(1, 2)
    
    print(f"输出形状:")
    print(f"  PyTorch: {pytorch_output.shape}")
    print(f"  MLX: {mlx_output_torch.shape}")
    
    # 计算输出差异
    diff = torch.abs(pytorch_output - mlx_output_torch)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    print(f"输出差异: max={max_diff:.6f}, mean={mean_diff:.6f}")

if __name__ == "__main__":
    test_mlx_conv1d_direct()
