#!/usr/bin/env python3
"""
测试PyTorch和MLX WaveNet的权重一致性
"""

import sys
sys.path.append('.')
import torch
import mlx.core as mx
import numpy as np
from indextts.s2mel.modules.wavenet import WN as PyTorchWaveNet
from indextts.s2mel.modules.mlx_wavenet import MLXWaveNet
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def test_wavenet_weights():
    """测试WaveNet权重的一致性"""
    print("=== 测试WaveNet权重一致性 ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    mx.random.seed(42)
    
    # 创建WaveNet实例
    hidden_channels = 512
    kernel_size = 5
    dilation_rate = 2
    n_layers = 8
    gin_channels = 512
    
    pytorch_wavenet = PyTorchWaveNet(
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        n_layers=n_layers,
        gin_channels=gin_channels,
        p_dropout=0.0
    )
    
    mlx_wavenet = MLXWaveNet(
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        n_layers=n_layers,
        gin_channels=gin_channels,
        p_dropout=0.0
    )
    
    print(f"PyTorch WaveNet参数数量: {sum(p.numel() for p in pytorch_wavenet.parameters())}")
    print(f"MLX WaveNet参数数量: 计算中...")
    
    # 对比权重
    print(f"\n=== 权重对比 ===")
    
    # 对比in_layers
    print(f"\n--- in_layers 对比 ---")
    for i in range(n_layers):
        # PyTorch SConv1d的权重在conv.conv中
        pytorch_weight = pytorch_wavenet.in_layers[i].conv.conv.weight
        mlx_weight = mlx_wavenet.in_layers[i].weight
        
        # 转换MLX权重到PyTorch格式
        mlx_weight_torch = mlx_to_torch(mlx_weight)
        
        print(f"Layer {i} 权重形状:")
        print(f"  PyTorch: {pytorch_weight.shape}")
        print(f"  MLX: {mlx_weight_torch.shape}")
        
        # 检查形状是否匹配
        if pytorch_weight.shape == mlx_weight_torch.shape:
            # 计算差异
            diff = torch.abs(pytorch_weight - mlx_weight_torch)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            
            print(f"  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        else:
            print(f"  形状不匹配，跳过差异计算")
        
        # 检查bias
        if hasattr(pytorch_wavenet.in_layers[i].conv.conv, 'bias') and pytorch_wavenet.in_layers[i].conv.conv.bias is not None:
            pytorch_bias = pytorch_wavenet.in_layers[i].conv.conv.bias
            mlx_bias = mlx_wavenet.in_layers[i].bias
            mlx_bias_torch = mlx_to_torch(mlx_bias)
            
            bias_diff = torch.abs(pytorch_bias - mlx_bias_torch)
            bias_max_diff = torch.max(bias_diff).item()
            bias_mean_diff = torch.mean(bias_diff).item()
            
            print(f"  Bias: max_diff={bias_max_diff:.6f}, mean_diff={bias_mean_diff:.6f}")
    
    # 对比res_skip_layers
    print(f"\n--- res_skip_layers 对比 ---")
    for i in range(n_layers):
        # PyTorch SConv1d的权重在conv.conv中
        pytorch_weight = pytorch_wavenet.res_skip_layers[i].conv.conv.weight
        mlx_weight = mlx_wavenet.res_skip_layers[i].weight
        
        # 转换MLX权重到PyTorch格式
        mlx_weight_torch = mlx_to_torch(mlx_weight)
        
        # 计算差异
        diff = torch.abs(pytorch_weight - mlx_weight_torch)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"Layer {i}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        
        # 检查bias
        if hasattr(pytorch_wavenet.res_skip_layers[i].conv.conv, 'bias') and pytorch_wavenet.res_skip_layers[i].conv.conv.bias is not None:
            pytorch_bias = pytorch_wavenet.res_skip_layers[i].conv.conv.bias
            mlx_bias = mlx_wavenet.res_skip_layers[i].bias
            mlx_bias_torch = mlx_to_torch(mlx_bias)
            
            bias_diff = torch.abs(pytorch_bias - mlx_bias_torch)
            bias_max_diff = torch.max(bias_diff).item()
            bias_mean_diff = torch.mean(bias_diff).item()
            
            print(f"  Bias: max_diff={bias_max_diff:.6f}, mean_diff={bias_mean_diff:.6f}")
    
    # 对比cond_layer
    print(f"\n--- cond_layer 对比 ---")
    if hasattr(pytorch_wavenet, 'cond_layer') and pytorch_wavenet.cond_layer is not None:
        # PyTorch SConv1d的权重在conv.conv中
        pytorch_weight = pytorch_wavenet.cond_layer.conv.conv.weight
        mlx_weight = mlx_wavenet.cond_layer.weight
        
        # 转换MLX权重到PyTorch格式
        mlx_weight_torch = mlx_to_torch(mlx_weight)
        
        # 计算差异
        diff = torch.abs(pytorch_weight - mlx_weight_torch)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"Weight: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        
        # 检查bias
        if hasattr(pytorch_wavenet.cond_layer.conv.conv, 'bias') and pytorch_wavenet.cond_layer.conv.conv.bias is not None:
            pytorch_bias = pytorch_wavenet.cond_layer.conv.conv.bias
            mlx_bias = mlx_wavenet.cond_layer.bias
            mlx_bias_torch = mlx_to_torch(mlx_bias)
            
            bias_diff = torch.abs(pytorch_bias - mlx_bias_torch)
            bias_max_diff = torch.max(bias_diff).item()
            bias_mean_diff = torch.mean(bias_diff).item()
            
            print(f"Bias: max_diff={bias_max_diff:.6f}, mean_diff={bias_mean_diff:.6f}")
    
    # 测试权重初始化的一致性
    print(f"\n=== 权重初始化一致性测试 ===")
    
    # 重新创建实例（相同种子）
    torch.manual_seed(42)
    mx.random.seed(42)
    
    pytorch_wavenet2 = PyTorchWaveNet(
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        n_layers=n_layers,
        gin_channels=gin_channels,
        p_dropout=0.0
    )
    
    mlx_wavenet2 = MLXWaveNet(
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        n_layers=n_layers,
        gin_channels=gin_channels,
        p_dropout=0.0
    )
    
    # 对比第一个in_layer的权重
    pytorch_weight1 = pytorch_wavenet.in_layers[0].conv.conv.weight
    pytorch_weight2 = pytorch_wavenet2.in_layers[0].conv.conv.weight
    mlx_weight1 = mlx_wavenet.in_layers[0].weight
    mlx_weight2 = mlx_wavenet2.in_layers[0].weight
    
    # 检查PyTorch权重是否一致
    pytorch_consistency = torch.allclose(pytorch_weight1, pytorch_weight2)
    print(f"PyTorch权重一致性: {pytorch_consistency}")
    
    # 检查MLX权重是否一致
    mlx_consistency = mx.allclose(mlx_weight1, mlx_weight2)
    print(f"MLX权重一致性: {mlx_consistency}")
    
    # 检查PyTorch和MLX权重是否一致
    mlx_weight1_torch = mlx_to_torch(mlx_weight1)
    pytorch_mlx_consistency = torch.allclose(pytorch_weight1, mlx_weight1_torch)
    print(f"PyTorch-MLX权重一致性: {pytorch_mlx_consistency}")

if __name__ == "__main__":
    test_wavenet_weights()
