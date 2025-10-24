#!/usr/bin/env python3
"""
深入分析PyTorch reflect padding与MLX edge padding的差异
"""

import sys
sys.path.append('.')
import torch
import mlx.core as mx
import numpy as np
from indextts.s2mel.modules.encodec import pad1d
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def analyze_padding_differences():
    """分析padding差异"""
    print("=== 深入分析Padding差异 ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    mx.random.seed(42)
    
    # 创建测试数据
    batch_size = 1
    seq_len = 10
    channels = 3
    
    # PyTorch格式: (batch, channels, seq_len)
    x_torch = torch.randn(batch_size, channels, seq_len)
    print(f"原始数据: {x_torch.shape}")
    print(f"数据内容: {x_torch[0, 0, :]}")
    
    # 测试不同的padding情况
    test_cases = [
        (2, 2),  # 对称padding
        (3, 1),  # 非对称padding
        (5, 0),  # 只有左padding
        (0, 4),  # 只有右padding
        (8, 8),  # 大padding
    ]
    
    for padding_left, padding_right in test_cases:
        print(f"\n--- Padding: left={padding_left}, right={padding_right} ---")
        
        # PyTorch reflect padding
        try:
            pytorch_padded = pad1d(x_torch, (padding_left, padding_right), mode='reflect')
            print(f"PyTorch reflect: {pytorch_padded.shape}")
            print(f"PyTorch内容: {pytorch_padded[0, 0, :]}")
        except Exception as e:
            print(f"PyTorch reflect失败: {e}")
            continue
        
        # MLX edge padding
        x_mlx = torch_to_mlx(x_torch.transpose(1, 2))  # 转换为MLX格式
        try:
            mlx_padded = mx.pad(x_mlx, ((0, 0), (padding_left, padding_right), (0, 0)), mode='edge')
            print(f"MLX edge: {mlx_padded.shape}")
            print(f"MLX内容: {mlx_padded[0, :, 0]}")
        except Exception as e:
            print(f"MLX edge失败: {e}")
            continue
        
        # 转换MLX输出到PyTorch格式进行对比
        mlx_padded_torch = mlx_to_torch(mlx_padded).transpose(1, 2)
        
        # 计算差异
        diff = torch.abs(pytorch_padded - mlx_padded_torch)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"差异: max={max_diff:.6f}, mean={mean_diff:.6f}")
        
        # 详细分析边界值
        if padding_left > 0:
            print(f"左边界对比:")
            print(f"  PyTorch左边界: {pytorch_padded[0, 0, :padding_left]}")
            print(f"  MLX左边界: {mlx_padded_torch[0, 0, :padding_left]}")
        
        if padding_right > 0:
            print(f"右边界对比:")
            print(f"  PyTorch右边界: {pytorch_padded[0, 0, -padding_right:]}")
            print(f"  MLX右边界: {mlx_padded_torch[0, 0, -padding_right:]}")

def implement_mlx_reflect_padding():
    """实现MLX版本的reflect padding"""
    print("\n=== 实现MLX Reflect Padding ===")
    
    def mlx_pad_reflect_1d(x, padding_left, padding_right):
        """
        MLX版本的reflect padding
        x: (batch, seq_len, channels)
        """
        batch, seq_len, channels = x.shape
        
        if padding_left == 0 and padding_right == 0:
            return x
        
        # 处理reflect padding的特殊情况
        if seq_len <= max(padding_left, padding_right):
            # 如果输入长度小于padding，需要先扩展
            extra_pad = max(padding_left, padding_right) - seq_len + 1
            x = mx.pad(x, ((0, 0), (0, extra_pad), (0, 0)), mode='constant', constant_values=0)
            seq_len = x.shape[1]
        
        # 左padding: 使用切片反转前padding_left个元素
        if padding_left > 0:
            left_reflect = x[:, padding_left-1::-1, :]  # 反转前padding_left个元素
            x = mx.concatenate([left_reflect, x], axis=1)
        
        # 右padding: 使用切片反转后padding_right个元素
        if padding_right > 0:
            right_reflect = x[:, -1:-padding_right-1:-1, :]  # 反转后padding_right个元素
            x = mx.concatenate([x, right_reflect], axis=1)
        
        return x
    
    # 测试MLX reflect padding
    torch.manual_seed(42)
    mx.random.seed(42)
    
    x_torch = torch.randn(1, 3, 10)
    x_mlx = torch_to_mlx(x_torch.transpose(1, 2))
    
    print(f"原始数据: {x_torch[0, 0, :]}")
    
    # 测试不同padding
    for padding_left, padding_right in [(2, 2), (3, 1), (5, 0)]:
        print(f"\nPadding: left={padding_left}, right={padding_right}")
        
        # PyTorch reflect
        pytorch_padded = pad1d(x_torch, (padding_left, padding_right), mode='reflect')
        print(f"PyTorch: {pytorch_padded[0, 0, :]}")
        
        # MLX reflect
        mlx_padded = mlx_pad_reflect_1d(x_mlx, padding_left, padding_right)
        mlx_padded_torch = mlx_to_torch(mlx_padded).transpose(1, 2)
        print(f"MLX: {mlx_padded_torch[0, 0, :]}")
        
        # 计算差异
        diff = torch.abs(pytorch_padded - mlx_padded_torch)
        max_diff = torch.max(diff).item()
        print(f"差异: max={max_diff:.6f}")

def test_wavenet_padding_impact():
    """测试padding差异对WaveNet的影响"""
    print("\n=== 测试Padding差异对WaveNet的影响 ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    mx.random.seed(42)
    
    # 创建简单的测试数据
    batch_size = 1
    seq_len = 20
    channels = 512
    
    x_torch = torch.randn(batch_size, channels, seq_len)
    x_mlx = torch_to_mlx(x_torch.transpose(1, 2))
    
    # 测试不同kernel size和dilation的padding
    test_cases = [
        (5, 1),   # kernel=5, dilation=1
        (5, 2),   # kernel=5, dilation=2
        (5, 4),   # kernel=5, dilation=4
        (5, 8),   # kernel=5, dilation=8
    ]
    
    for kernel_size, dilation in test_cases:
        print(f"\n--- Kernel={kernel_size}, Dilation={dilation} ---")
        
        # 计算padding
        effective_kernel_size = (kernel_size - 1) * dilation + 1
        padding_total = effective_kernel_size - 1
        padding_left = padding_total // 2
        padding_right = padding_total - padding_left
        
        print(f"Effective kernel: {effective_kernel_size}")
        print(f"Padding: left={padding_left}, right={padding_right}")
        
        # PyTorch reflect padding
        pytorch_padded = pad1d(x_torch, (padding_left, padding_right), mode='reflect')
        
        # MLX edge padding
        mlx_padded = mx.pad(x_mlx, ((0, 0), (padding_left, padding_right), (0, 0)), mode='edge')
        
        # 转换MLX输出
        mlx_padded_torch = mlx_to_torch(mlx_padded).transpose(1, 2)
        
        # 计算差异
        diff = torch.abs(pytorch_padded - mlx_padded_torch)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"Padding差异: max={max_diff:.6f}, mean={mean_diff:.6f}")
        
        # 分析差异的分布
        diff_percentiles = torch.quantile(diff.flatten(), torch.tensor([0.5, 0.9, 0.95, 0.99]))
        print(f"差异分位数: 50%={diff_percentiles[0]:.6f}, 90%={diff_percentiles[1]:.6f}, 95%={diff_percentiles[2]:.6f}, 99%={diff_percentiles[3]:.6f}")

if __name__ == "__main__":
    analyze_padding_differences()
    implement_mlx_reflect_padding()
    test_wavenet_padding_impact()
