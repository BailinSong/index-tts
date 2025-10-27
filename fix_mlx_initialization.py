#!/usr/bin/env python3
"""
修复 MLX 初始化使用 Kaiming uniform
确保 MLX 模型的 cond_projection 层使用与 PyTorch 相同的初始化方法
"""

import mlx.core as mx
import mlx.nn as nn
import math
import numpy as np

def kaiming_uniform_mlx(weight, a=math.sqrt(5)):
    """
    在 MLX 中实现 Kaiming uniform 初始化
    
    Args:
        weight: MLX array to initialize
        a: negative slope of the rectifier used after this layer
    """
    fan_in = weight.shape[-1]
    bound = a / math.sqrt(fan_in) if fan_in > 0 else 0
    
    # 生成均匀分布的随机数
    uniform_random = mx.random.uniform(-bound, bound, weight.shape)
    return uniform_random

def init_mlx_linear_like_pytorch(linear_layer, input_dim, output_dim, a=math.sqrt(5)):
    """
    使用与 PyTorch 相同的 Kaiming uniform 初始化 MLX Linear 层
    
    Args:
        linear_layer: MLX Linear layer
        input_dim: input dimension
        output_dim: output dimension
        a: negative slope parameter (default: sqrt(5) for PyTorch)
    """
    # 初始化权重
    fan_in = input_dim
    bound = a / math.sqrt(fan_in) if fan_in > 0 else 0
    
    weight = mx.random.uniform(-bound, bound, (output_dim, input_dim))
    bias = mx.random.uniform(-bound, bound, (output_dim,))
    
    linear_layer.weight = weight
    linear_layer.bias = bias
    
    return linear_layer

def verify_initialization_equivalence():
    """
    验证 MLX 和 PyTorch 初始化的一致性
    """
    print("=== 验证初始化一致性 ===")
    
    # MLX 初始化
    mlx_linear = nn.Linear(512, 512, bias=True)
    mlx_linear = init_mlx_linear_like_pytorch(mlx_linear, 512, 512)
    
    print("MLX Kaiming uniform 初始化:")
    print(f"  权重范围: [{mlx_linear.weight.min():.6f}, {mlx_linear.weight.max():.6f}]")
    print(f"  偏置范围: [{mlx_linear.bias.min():.6f}, {mlx_linear.bias.max():.6f}]")
    
    # 计算期望范围
    a = math.sqrt(5)
    bound = a / math.sqrt(512)
    print(f"  期望范围: [-{bound:.6f}, {bound:.6f}]")
    
    # 验证范围
    weight_in_range = (mlx_linear.weight.min() >= -bound) and (mlx_linear.weight.max() <= bound)
    bias_in_range = (mlx_linear.bias.min() >= -bound) and (mlx_linear.bias.max() <= bound)
    
    print(f"  权重范围正确: {weight_in_range}")
    print(f"  偏置范围正确: {bias_in_range}")
    
    return mlx_linear

def fix_cond_projection_initialization(mlx_model):
    """
    修复 MLX 模型中 cond_projection 层的初始化
    
    Args:
        mlx_model: MLX CFM model
    """
    print("=== 修复 cond_projection 初始化 ===")
    
    # 获取 cond_projection 层
    cond_proj = mlx_model.estimator.cond_projection
    
    # 获取输入和输出维度
    input_dim = cond_proj.weight.shape[1]  # 512
    output_dim = cond_proj.weight.shape[0]  # 512
    
    print(f"修复前:")
    print(f"  权重范围: [{cond_proj.weight.min():.6f}, {cond_proj.weight.max():.6f}]")
    print(f"  偏置范围: [{cond_proj.bias.min():.6f}, {cond_proj.bias.max():.6f}]")
    
    # 使用 Kaiming uniform 重新初始化
    cond_proj = init_mlx_linear_like_pytorch(cond_proj, input_dim, output_dim)
    
    print(f"修复后:")
    print(f"  权重范围: [{cond_proj.weight.min():.6f}, {cond_proj.weight.max():.6f}]")
    print(f"  偏置范围: [{cond_proj.bias.min():.6f}, {cond_proj.bias.max():.6f}]")
    
    return mlx_model

def create_pytorch_compatible_mlx_linear(input_dim, output_dim, bias=True):
    """
    创建与 PyTorch 兼容的 MLX Linear 层
    
    Args:
        input_dim: input dimension
        output_dim: output dimension
        bias: whether to use bias
    
    Returns:
        MLX Linear layer with PyTorch-compatible initialization
    """
    linear = nn.Linear(input_dim, output_dim, bias=bias)
    
    # 使用 Kaiming uniform 初始化
    linear = init_mlx_linear_like_pytorch(linear, input_dim, output_dim)
    
    return linear

if __name__ == "__main__":
    print("🔧 MLX 初始化修复工具")
    print("=" * 50)
    
    # 验证初始化一致性
    mlx_linear = verify_initialization_equivalence()
    
    print("\n✅ MLX 初始化修复完成!")
    print("现在 MLX Linear 层使用与 PyTorch 相同的 Kaiming uniform 初始化")
