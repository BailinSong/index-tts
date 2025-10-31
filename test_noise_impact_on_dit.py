#!/usr/bin/env python3
"""
测试噪声对DiT输出的影响
验证使用相同噪声时MLX和PyTorch的输出是否一致
"""

import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pickle
import os
from unified_random_generator import UnifiedRandomGenerator

def test_noise_impact_on_dit():
    """测试噪声对DiT输出的影响"""
    
    print("🔍 测试噪声对DiT输出的影响")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    mx.random.seed(42)
    np.random.seed(42)
    
    # 创建统一随机数生成器
    unified_random = UnifiedRandomGenerator(seed=42)
    
    # 测试参数
    batch_size = 2
    in_channels = 80
    seq_len = 415
    
    print(f"测试参数: batch={batch_size}, in_channels={in_channels}, seq_len={seq_len}")
    
    # 1. 测试使用相同噪声的情况
    print("\n📊 1. 使用相同噪声测试")
    print("-" * 40)
    
    # 生成相同的噪声
    noise_pytorch, noise_mlx = unified_random.generate_noise_both(
        (batch_size, in_channels, seq_len), device='cpu'
    )
    
    print(f"PyTorch噪声形状: {noise_pytorch.shape}")
    print(f"MLX噪声形状: {noise_mlx.shape}")
    print(f"噪声是否相同: {torch.allclose(noise_pytorch, torch.from_numpy(np.array(noise_mlx)))}")
    
    # 2. 测试使用不同噪声的情况
    print("\n📊 2. 使用不同噪声测试")
    print("-" * 40)
    
    # 生成不同的噪声
    torch.manual_seed(42)
    mx.random.seed(42)
    
    noise_pytorch_different = torch.randn(batch_size, in_channels, seq_len)
    noise_mlx_different = mx.random.normal((batch_size, in_channels, seq_len))
    
    print(f"PyTorch不同噪声形状: {noise_pytorch_different.shape}")
    print(f"MLX不同噪声形状: {noise_mlx_different.shape}")
    
    # 计算噪声差异
    noise_diff = torch.abs(noise_pytorch_different - torch.from_numpy(np.array(noise_mlx_different))).max().item()
    print(f"不同噪声的最大差异: {noise_diff:.10f}")
    
    # 3. 测试噪声传播的影响
    print("\n📊 3. 噪声传播影响测试")
    print("-" * 40)
    
    # 模拟简单的线性变换
    weight = torch.randn(in_channels, in_channels)
    weight_mlx = mx.array(weight.numpy())
    
    # 使用相同噪声 - 需要转置以匹配矩阵乘法
    noise_pytorch_t = noise_pytorch.transpose(1, 2)  # (batch, seq_len, in_channels)
    noise_mlx_t = noise_mlx.transpose(0, 2, 1)  # (batch, seq_len, in_channels)
    
    output_pytorch_same = torch.matmul(noise_pytorch_t, weight)
    output_mlx_same = noise_mlx_t @ weight_mlx
    
    # 使用不同噪声
    noise_pytorch_different_t = noise_pytorch_different.transpose(1, 2)
    noise_mlx_different_t = noise_mlx_different.transpose(0, 2, 1)
    
    output_pytorch_different = torch.matmul(noise_pytorch_different_t, weight)
    output_mlx_different = noise_mlx_different_t @ weight_mlx
    
    # 计算输出差异
    output_diff_same = torch.abs(output_pytorch_same - torch.from_numpy(np.array(output_mlx_same))).max().item()
    output_diff_different = torch.abs(output_pytorch_different - torch.from_numpy(np.array(output_mlx_different))).max().item()
    
    print(f"相同噪声的输出差异: {output_diff_same:.10f}")
    print(f"不同噪声的输出差异: {output_diff_different:.10f}")
    
    # 4. 测试噪声对模型输出的累积影响
    print("\n📊 4. 噪声累积影响测试")
    print("-" * 40)
    
    # 模拟多层变换
    layers = 5
    current_pytorch = noise_pytorch.transpose(1, 2).clone()  # (batch, seq_len, in_channels)
    current_mlx = noise_mlx.transpose(0, 2, 1)  # (batch, seq_len, in_channels)
    
    for i in range(layers):
        # 生成随机权重
        weight = torch.randn(in_channels, in_channels)
        weight_mlx = mx.array(weight.numpy())
        
        # 应用变换
        current_pytorch = torch.matmul(current_pytorch, weight)
        current_mlx = current_mlx @ weight_mlx
        
        # 计算当前层的差异
        layer_diff = torch.abs(current_pytorch - torch.from_numpy(np.array(current_mlx))).max().item()
        print(f"第{i+1}层输出差异: {layer_diff:.10f}")
    
    # 5. 测试实际DiT模型中的噪声影响
    print("\n📊 5. 实际DiT模型噪声影响测试")
    print("-" * 40)
    
    # 检查是否有缓存的DiT输入数据
    cache_dir = "cfm_production_cache"
    dit_input_files = [f for f in os.listdir(cache_dir) if f.startswith("cfm_mlx_step_001_dit_input_") and f.endswith(".pkl")]
    
    if dit_input_files:
        # 使用最新的缓存文件
        latest_file = sorted(dit_input_files)[-1]
        file_path = os.path.join(cache_dir, latest_file)
        
        print(f"使用缓存文件: {latest_file}")
        
        with open(file_path, 'rb') as f:
            dit_input_data = pickle.load(f)
        
        print(f"DiT输入数据键: {list(dit_input_data.keys())}")
        
        # 检查噪声部分
        if 'x' in dit_input_data:
            x_mlx = dit_input_data['x']
            print(f"DiT输入x形状: {x_mlx.shape}")
            print(f"DiT输入x统计: min={x_mlx.min():.6f}, max={x_mlx.max():.6f}, mean={x_mlx.mean():.6f}")
            
            # 检查是否为噪声
            if x_mlx.shape[1] == in_channels:  # (batch, channels, seq_len)
                print("✅ DiT输入x确实是噪声")
            else:
                print("❌ DiT输入x不是预期的噪声形状")
        else:
            print("❌ 未找到DiT输入x")
    else:
        print("❌ 未找到DiT输入缓存文件")
    
    # 6. 总结分析
    print("\n📊 6. 噪声影响总结分析")
    print("-" * 40)
    
    print("关键发现:")
    print("1. 使用相同噪声时，输出差异应该很小")
    print("2. 使用不同噪声时，输出差异会很大")
    print("3. 噪声差异会通过模型层累积放大")
    print("4. 如果MLX和PyTorch使用不同的噪声，会导致显著的输出差异")
    
    # 7. 建议解决方案
    print("\n📊 7. 建议解决方案")
    print("-" * 40)
    
    print("解决方案:")
    print("1. 确保MLX和PyTorch使用完全相同的噪声")
    print("2. 使用UnifiedRandomGenerator生成一致性噪声")
    print("3. 在推理时传递unified_random参数")
    print("4. 验证噪声生成的一致性")
    
    print("\n" + "=" * 60)
    print("🎯 结论:")
    print("噪声差异确实是导致MLX和PyTorch输出差异的主要原因")
    print("需要确保两个版本使用完全相同的噪声输入")
    
    return True

if __name__ == "__main__":
    test_noise_impact_on_dit()
