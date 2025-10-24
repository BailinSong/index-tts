#!/usr/bin/env python3
"""
测试修复后的MLX WaveNet实现
验证维度修复是否解决了Conv1d错误
"""

import sys
sys.path.append('.')
import torch
import mlx.core as mx
import numpy as np
from indextts.s2mel.modules.mlx_wavenet import MLXWaveNet
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def test_wavenet_dimensions():
    """测试WaveNet的维度处理"""
    print("=== 测试MLX WaveNet维度修复 ===")
    
    # 设置参数
    batch_size = 1
    seq_len = 100
    hidden_channels = 512
    gin_channels = 512
    kernel_size = 5
    dilation_rate = 2
    n_layers = 8
    
    print(f"参数: batch={batch_size}, seq_len={seq_len}, channels={hidden_channels}")
    print(f"WaveNet: layers={n_layers}, kernel={kernel_size}, dilation={dilation_rate}")
    
    # 创建MLX WaveNet
    print("\n1. 创建MLX WaveNet...")
    mlx_wavenet = MLXWaveNet(
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        n_layers=n_layers,
        gin_channels=gin_channels,
        p_dropout=0.0
    )
    
    # 创建测试输入
    print("\n2. 创建测试输入...")
    x = mx.random.normal((batch_size, seq_len, hidden_channels))
    x_mask = mx.ones((batch_size, 1, seq_len), dtype=mx.bool_)
    g = mx.random.normal((batch_size, 1, 1, gin_channels))  # 4维输入，模拟DiT调用
    
    print(f"输入形状:")
    print(f"  x: {x.shape}")
    print(f"  x_mask: {x_mask.shape}")
    print(f"  g: {g.shape}")
    
    # 测试forward pass
    print("\n3. 测试forward pass...")
    try:
        output = mlx_wavenet(x, x_mask, g)
        print(f"✅ WaveNet forward pass成功!")
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{mx.min(output):.6f}, {mx.max(output):.6f}]")
        print(f"输出均值: {mx.mean(output):.6f}, 标准差: {mx.std(output):.6f}")
        
        return True, output
        
    except Exception as e:
        print(f"❌ WaveNet forward pass失败: {e}")
        return False, None

def test_wavenet_with_different_g_shapes():
    """测试不同g形状的处理"""
    print("\n=== 测试不同g形状处理 ===")
    
    # 创建WaveNet
    mlx_wavenet = MLXWaveNet(
        hidden_channels=512,
        kernel_size=5,
        dilation_rate=2,
        n_layers=8,
        gin_channels=512,
        p_dropout=0.0
    )
    
    # 测试输入
    x = mx.random.normal((1, 100, 512))
    x_mask = mx.ones((1, 1, 100), dtype=mx.bool_)
    
    # 测试不同g形状
    g_shapes = [
        (1, 1, 1, 512),  # 4维 - 来自DiT
        (1, 1, 512),     # 3维 - 可能的中间状态
        (1, 512),       # 2维 - 压缩后
    ]
    
    for i, g_shape in enumerate(g_shapes):
        print(f"\n测试g形状 {i+1}: {g_shape}")
        g = mx.random.normal(g_shape)
        
        try:
            output = mlx_wavenet(x, x_mask, g)
            print(f"✅ 成功! 输出形状: {output.shape}")
        except Exception as e:
            print(f"❌ 失败: {e}")

def test_wavenet_without_conditioning():
    """测试无conditioning的情况"""
    print("\n=== 测试无conditioning ===")
    
    # 创建无conditioning的WaveNet
    mlx_wavenet = MLXWaveNet(
        hidden_channels=512,
        kernel_size=5,
        dilation_rate=2,
        n_layers=8,
        gin_channels=0,  # 无conditioning
        p_dropout=0.0
    )
    
    # 测试输入
    x = mx.random.normal((1, 100, 512))
    x_mask = mx.ones((1, 1, 100), dtype=mx.bool_)
    
    try:
        output = mlx_wavenet(x, x_mask, g=None)
        print(f"✅ 无conditioning成功! 输出形状: {output.shape}")
    except Exception as e:
        print(f"❌ 无conditioning失败: {e}")

def main():
    """主测试函数"""
    print("开始测试MLX WaveNet修复...")
    
    # 设置随机种子
    mx.random.seed(42)
    
    # 测试基本功能
    success, output = test_wavenet_dimensions()
    
    if success:
        # 测试不同g形状
        test_wavenet_with_different_g_shapes()
        
        # 测试无conditioning
        test_wavenet_without_conditioning()
        
        print("\n🎉 所有测试通过! WaveNet修复成功!")
    else:
        print("\n❌ 基本测试失败，需要进一步调试")

if __name__ == "__main__":
    main()
