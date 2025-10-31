#!/usr/bin/env python3
"""
Random Noise生成差异分析工具
分析MLX和PyTorch版本在random noise生成中的差异
"""

import torch
import mlx.core as mx
import numpy as np
import math
from typing import Optional

def analyze_random_noise_differences():
    """分析random noise生成的差异"""
    
    print("🔍 Random Noise生成差异分析")
    print("=" * 60)
    
    # 设置相同的随机种子
    torch.manual_seed(42)
    mx.random.seed(42)
    np.random.seed(42)
    
    # 测试参数
    batch_size = 2
    in_channels = 80
    seq_len = 415
    temperature = 1.0
    
    print(f"测试参数: batch={batch_size}, in_channels={in_channels}, seq_len={seq_len}, temperature={temperature}")
    
    # 1. 分析PyTorch noise生成
    print("\n📊 1. PyTorch Random Noise生成")
    print("-" * 40)
    
    # PyTorch方法1: 使用torch.randn
    torch.manual_seed(42)
    noise_pytorch_1 = torch.randn([batch_size, in_channels, seq_len]) * temperature
    
    # PyTorch方法2: 使用torch.randn + device
    torch.manual_seed(42)
    noise_pytorch_2 = torch.randn([batch_size, in_channels, seq_len], device='cpu') * temperature
    
    print(f"PyTorch方法1 - shape: {noise_pytorch_1.shape}")
    print(f"PyTorch方法1 - min: {noise_pytorch_1.min().item():.6f}, max: {noise_pytorch_1.max().item():.6f}")
    print(f"PyTorch方法1 - mean: {noise_pytorch_1.mean().item():.6f}, std: {noise_pytorch_1.std().item():.6f}")
    
    print(f"PyTorch方法2 - shape: {noise_pytorch_2.shape}")
    print(f"PyTorch方法2 - min: {noise_pytorch_2.min().item():.6f}, max: {noise_pytorch_2.max().item():.6f}")
    print(f"PyTorch方法2 - mean: {noise_pytorch_2.mean().item():.6f}, std: {noise_pytorch_2.std().item():.6f}")
    
    # 2. 分析MLX noise生成
    print("\n📊 2. MLX Random Noise生成")
    print("-" * 40)
    
    # MLX方法1: 使用mx.random.normal
    mx.random.seed(42)
    noise_mlx_1 = mx.random.normal((batch_size, in_channels, seq_len)) * temperature
    
    # MLX方法2: 使用numpy + mx.array
    np.random.seed(42)
    noise_numpy = np.random.normal(0, 1, (batch_size, in_channels, seq_len)) * temperature
    noise_mlx_2 = mx.array(noise_numpy)
    
    print(f"MLX方法1 - shape: {noise_mlx_1.shape}")
    print(f"MLX方法1 - min: {noise_mlx_1.min():.6f}, max: {noise_mlx_1.max():.6f}")
    print(f"MLX方法1 - mean: {noise_mlx_1.mean():.6f}, std: {noise_mlx_1.std():.6f}")
    
    print(f"MLX方法2 - shape: {noise_mlx_2.shape}")
    print(f"MLX方法2 - min: {noise_mlx_2.min():.6f}, max: {noise_mlx_2.max():.6f}")
    print(f"MLX方法2 - mean: {noise_mlx_2.mean():.6f}, std: {noise_mlx_2.std():.6f}")
    
    # 3. 比较PyTorch和MLX的差异
    print("\n📊 3. PyTorch vs MLX Noise差异")
    print("-" * 40)
    
    # 转换MLX到PyTorch进行比较
    noise_mlx_1_torch = torch.from_numpy(np.array(noise_mlx_1))
    noise_mlx_2_torch = torch.from_numpy(np.array(noise_mlx_2))
    
    diff_1 = torch.abs(noise_pytorch_1 - noise_mlx_1_torch).max().item()
    diff_2 = torch.abs(noise_pytorch_1 - noise_mlx_2_torch).max().item()
    
    print(f"PyTorch vs MLX方法1最大差异: {diff_1:.10f}")
    print(f"PyTorch vs MLX方法2最大差异: {diff_2:.10f}")
    
    # 4. 分析随机数生成器的差异
    print("\n📊 4. 随机数生成器差异分析")
    print("-" * 40)
    
    # 生成多个样本进行比较
    torch.manual_seed(42)
    mx.random.seed(42)
    np.random.seed(42)
    
    samples_pytorch = []
    samples_mlx = []
    samples_numpy = []
    
    for i in range(5):
        torch.manual_seed(42 + i)
        mx.random.seed(42 + i)
        np.random.seed(42 + i)
        
        sample_pytorch = torch.randn(1, 10, 10)
        sample_mlx = mx.random.normal((1, 10, 10))
        sample_numpy = np.random.normal(0, 1, (1, 10, 10))
        
        samples_pytorch.append(sample_pytorch)
        samples_mlx.append(sample_mlx)
        samples_numpy.append(sample_numpy)
    
    print("随机数生成器一致性测试:")
    for i in range(5):
        diff_pytorch_mlx = torch.abs(samples_pytorch[i] - torch.from_numpy(np.array(samples_mlx[i]))).max().item()
        diff_pytorch_numpy = torch.abs(samples_pytorch[i] - torch.from_numpy(samples_numpy[i])).max().item()
        print(f"  样本{i+1}: PyTorch-MLX={diff_pytorch_mlx:.10f}, PyTorch-NumPy={diff_pytorch_numpy:.10f}")
    
    # 5. 分析数值精度差异
    print("\n📊 5. 数值精度差异分析")
    print("-" * 40)
    
    # 测试不同数据类型的噪声生成
    torch.manual_seed(42)
    mx.random.seed(42)
    
    # Float32
    noise_pytorch_f32 = torch.randn(2, 10, 10, dtype=torch.float32)
    noise_mlx_f32 = mx.random.normal((2, 10, 10), dtype=mx.float32)
    
    # Float16
    noise_pytorch_f16 = torch.randn(2, 10, 10, dtype=torch.float16)
    noise_mlx_f16 = mx.random.normal((2, 10, 10), dtype=mx.float16)
    
    diff_f32 = torch.abs(noise_pytorch_f32 - torch.from_numpy(np.array(noise_mlx_f32))).max().item()
    diff_f16 = torch.abs(noise_pytorch_f16 - torch.from_numpy(np.array(noise_mlx_f16))).max().item()
    
    print(f"Float32噪声差异: {diff_f32:.10f}")
    print(f"Float16噪声差异: {diff_f16:.10f}")
    
    # 6. 分析噪声对模型输出的影响
    print("\n📊 6. 噪声对模型输出的影响分析")
    print("-" * 40)
    
    # 使用相同的噪声测试模型
    torch.manual_seed(42)
    mx.random.seed(42)
    
    # 生成相同的噪声
    noise_pytorch = torch.randn(batch_size, in_channels, seq_len)
    noise_mlx = mx.random.normal((batch_size, in_channels, seq_len))
    
    # 计算噪声差异
    noise_diff = torch.abs(noise_pytorch - torch.from_numpy(np.array(noise_mlx))).max().item()
    print(f"噪声本身的最大差异: {noise_diff:.10f}")
    
    # 分析噪声的统计特性
    print(f"PyTorch噪声统计:")
    print(f"  min: {noise_pytorch.min().item():.6f}")
    print(f"  max: {noise_pytorch.max().item():.6f}")
    print(f"  mean: {noise_pytorch.mean().item():.6f}")
    print(f"  std: {noise_pytorch.std().item():.6f}")
    
    print(f"MLX噪声统计:")
    print(f"  min: {noise_mlx.min():.6f}")
    print(f"  max: {noise_mlx.max():.6f}")
    print(f"  mean: {noise_mlx.mean():.6f}")
    print(f"  std: {noise_mlx.std():.6f}")
    
    # 7. 分析噪声传播的影响
    print("\n📊 7. 噪声传播影响分析")
    print("-" * 40)
    
    # 模拟噪声在模型中的传播
    # 简单的线性变换测试
    torch.manual_seed(42)
    mx.random.seed(42)
    
    # 生成噪声
    noise_pytorch = torch.randn(2, 10, 10)
    noise_mlx = mx.random.normal((2, 10, 10))
    
    # 应用线性变换
    weight = torch.randn(10, 10)
    weight_mlx = mx.array(weight.numpy())
    
    # PyTorch变换
    output_pytorch = torch.matmul(noise_pytorch, weight)
    
    # MLX变换
    output_mlx = noise_mlx @ weight_mlx
    
    # 计算输出差异
    output_diff = torch.abs(output_pytorch - torch.from_numpy(np.array(output_mlx))).max().item()
    print(f"噪声经过线性变换后的输出差异: {output_diff:.10f}")
    
    # 8. 总结分析
    print("\n📊 8. 噪声差异总结分析")
    print("-" * 40)
    
    all_noise_diffs = [diff_1, diff_2, diff_f32, diff_f16, noise_diff, output_diff]
    max_noise_diff = max(all_noise_diffs)
    min_noise_diff = min(all_noise_diffs)
    avg_noise_diff = sum(all_noise_diffs) / len(all_noise_diffs)
    
    print(f"噪声相关最大差异: {max_noise_diff:.10f}")
    print(f"噪声相关最小差异: {min_noise_diff:.10f}")
    print(f"噪声相关平均差异: {avg_noise_diff:.10f}")
    
    # 判断噪声差异是否影响模型输出
    if max_noise_diff < 1e-6:
        print("✅ 噪声差异极小，不会影响模型输出")
    elif max_noise_diff < 1e-4:
        print("⚠️  噪声差异较小，可能轻微影响模型输出")
    elif max_noise_diff < 1e-2:
        print("❌ 噪声差异较大，可能显著影响模型输出")
    else:
        print("❌ 噪声差异过大，严重影响模型输出")
    
    print("\n" + "=" * 60)
    print("🎯 噪声差异分析结论:")
    print("1. PyTorch和MLX的随机数生成器实现不同")
    print("2. 噪声生成算法可能使用不同的底层实现")
    print("3. 数值精度和数据类型处理可能不同")
    print("4. 噪声差异会通过模型传播，影响最终输出")
    print("5. 如果噪声差异较大，可能是导致模型输出差异的主要原因")
    
    return True

if __name__ == "__main__":
    analyze_random_noise_differences()

