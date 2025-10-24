#!/usr/bin/env python3
"""
最终修复MLX WaveNet的权重格式问题
"""

import sys
sys.path.append('.')
import torch
import mlx.core as mx
import mlx.nn as nn
from indextts.s2mel.modules.wavenet import WN as PyTorchWaveNet
from indextts.s2mel.modules.mlx_wavenet import MLXWaveNet
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def fix_mlx_wavenet_weights():
    """修复MLX WaveNet的权重初始化"""
    print("=== 修复MLX WaveNet权重初始化 ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    mx.random.seed(42)
    
    # 创建PyTorch WaveNet
    pytorch_wavenet = PyTorchWaveNet(
        hidden_channels=512,
        kernel_size=5,
        dilation_rate=2,
        n_layers=8,
        gin_channels=512,
        p_dropout=0.0
    )
    
    # 创建MLX WaveNet
    mlx_wavenet = MLXWaveNet(
        hidden_channels=512,
        kernel_size=5,
        dilation_rate=2,
        n_layers=8,
        gin_channels=512,
        p_dropout=0.0
    )
    
    print("修复前权重对比:")
    for i in range(8):
        pytorch_weight = pytorch_wavenet.in_layers[i].conv.conv.weight  # (1024, 512, 5)
        mlx_weight = mlx_wavenet.in_layers[i].weight  # (1024, 5, 512)
        
        print(f"Layer {i}:")
        print(f"  PyTorch: {pytorch_weight.shape}")
        print(f"  MLX: {mlx_weight.shape}")
        
        # 转换MLX权重到PyTorch格式
        mlx_weight_torch = mlx_to_torch(mlx_weight).transpose(1, 2)
        
        # 计算差异
        diff = torch.abs(pytorch_weight - mlx_weight_torch)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"  差异: max={max_diff:.6f}, mean={mean_diff:.6f}")
    
    # 修复MLX权重格式
    print("\n修复MLX权重格式...")
    for i in range(8):
        pytorch_weight = pytorch_wavenet.in_layers[i].conv.conv.weight  # (1024, 512, 5)
        
        # 将PyTorch权重转换为MLX格式
        # PyTorch: (out_channels, in_channels, kernel_size)
        # MLX: (out_channels, kernel_size, in_channels)
        pytorch_weight_np = pytorch_weight.detach().cpu().numpy()
        mlx_weight_fixed = mx.array(pytorch_weight_np.transpose(0, 2, 1))
        
        # 更新MLX权重
        mlx_wavenet.in_layers[i].weight = mlx_weight_fixed
        
        print(f"Layer {i} 权重已修复")
    
    # 修复bias
    for i in range(8):
        if hasattr(pytorch_wavenet.in_layers[i].conv.conv, 'bias') and pytorch_wavenet.in_layers[i].conv.conv.bias is not None:
            pytorch_bias = pytorch_wavenet.in_layers[i].conv.conv.bias
            mlx_bias_fixed = torch_to_mlx(pytorch_bias)
            mlx_wavenet.in_layers[i].bias = mlx_bias_fixed
            print(f"Layer {i} bias已修复")
    
    # 修复res_skip_layers权重
    for i in range(8):
        pytorch_weight = pytorch_wavenet.res_skip_layers[i].conv.conv.weight
        pytorch_weight_np = pytorch_weight.detach().cpu().numpy()
        mlx_weight_fixed = mx.array(pytorch_weight_np.transpose(0, 2, 1))
        mlx_wavenet.res_skip_layers[i].weight = mlx_weight_fixed
        
        if hasattr(pytorch_wavenet.res_skip_layers[i].conv.conv, 'bias') and pytorch_wavenet.res_skip_layers[i].conv.conv.bias is not None:
            pytorch_bias = pytorch_wavenet.res_skip_layers[i].conv.conv.bias
            mlx_bias_fixed = torch_to_mlx(pytorch_bias)
            mlx_wavenet.res_skip_layers[i].bias = mlx_bias_fixed
        
        print(f"ResSkip Layer {i} 权重已修复")
    
    # 修复cond_layer权重
    if hasattr(pytorch_wavenet, 'cond_layer') and pytorch_wavenet.cond_layer is not None:
        pytorch_weight = pytorch_wavenet.cond_layer.conv.conv.weight
        pytorch_weight_np = pytorch_weight.detach().cpu().numpy()
        mlx_weight_fixed = mx.array(pytorch_weight_np.transpose(0, 2, 1))
        mlx_wavenet.cond_layer.weight = mlx_weight_fixed
        
        if hasattr(pytorch_wavenet.cond_layer.conv.conv, 'bias') and pytorch_wavenet.cond_layer.conv.conv.bias is not None:
            pytorch_bias = pytorch_wavenet.cond_layer.conv.conv.bias
            mlx_bias_fixed = torch_to_mlx(pytorch_bias)
            mlx_wavenet.cond_layer.bias = mlx_bias_fixed
        
        print("Cond layer权重已修复")
    
    # 验证修复结果
    print("\n修复后权重对比:")
    for i in range(8):
        pytorch_weight = pytorch_wavenet.in_layers[i].conv.conv.weight  # (1024, 512, 5)
        mlx_weight = mlx_wavenet.in_layers[i].weight  # (1024, 5, 512)
        
        # 转换MLX权重到PyTorch格式
        mlx_weight_torch = mlx_to_torch(mlx_weight).transpose(1, 2)
        
        # 计算差异
        diff = torch.abs(pytorch_weight - mlx_weight_torch)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"Layer {i}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    
    # 测试前向传播
    print("\n测试前向传播...")
    batch_size = 1
    seq_len = 100
    
    # PyTorch输入
    x_torch = torch.randn(batch_size, 512, seq_len)
    x_mask_torch = torch.ones(batch_size, 1, seq_len, dtype=torch.bool)
    g_torch = torch.randn(batch_size, 512, 1)
    
    # MLX输入
    x_mlx = torch_to_mlx(x_torch.transpose(1, 2))
    x_mask_mlx = torch_to_mlx(x_mask_torch.transpose(1, 2))
    g_mlx = torch_to_mlx(g_torch.transpose(1, 2))
    
    # 前向传播
    pytorch_output = pytorch_wavenet(x_torch, x_mask_torch, g_torch)
    mlx_output = mlx_wavenet(x_mlx, x_mask_mlx, g_mlx)
    
    # 转换MLX输出到PyTorch格式
    mlx_output_torch = mlx_to_torch(mlx_output).transpose(1, 2)
    
    # 计算差异
    diff = torch.abs(pytorch_output - mlx_output_torch)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    print(f"前向传播差异:")
    print(f"  最大差异: {max_diff:.6f}")
    print(f"  平均差异: {mean_diff:.6f}")
    print(f"  差异>1e-3的元素比例: {torch.sum(diff > 1e-3).item() / diff.numel():.2%}")

if __name__ == "__main__":
    fix_mlx_wavenet_weights()
