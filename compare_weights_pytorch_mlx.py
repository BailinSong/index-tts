#!/usr/bin/env python3
"""
比较PyTorch和MLX缓存中的权重差异
"""

import sys
import os
import torch
import numpy as np
import pickle
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from indextts.infer_v2 import IndexTTS2
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def compare_weights_pytorch_mlx():
    """比较PyTorch和MLX缓存中的权重差异"""
    print("=== 比较PyTorch和MLX缓存中的权重差异 ===\n")
    
    # 初始化TTS系统
    print("1. 初始化TTS系统...")
    tts = IndexTTS2()
    
    # 加载PyTorch权重
    print("2. 加载PyTorch权重...")
    pytorch_state_dict = torch.load("checkpoints/s2mel.pth", map_location='cpu')
    
    # 加载MLX缓存权重
    print("3. 加载MLX缓存权重...")
    mlx_cache = np.load("checkpoints/mlx/s2mel.npz")
    
    # 比较t_embedder权重
    print("4. 比较t_embedder权重...")
    t_embedder_keys = [
        "models.cfm.estimator.t_embedder.mlp.0.weight",
        "models.cfm.estimator.t_embedder.mlp.0.bias", 
        "models.cfm.estimator.t_embedder.mlp.2.weight",
        "models.cfm.estimator.t_embedder.mlp.2.bias"
    ]
    
    for key in t_embedder_keys:
        if key in pytorch_state_dict and key in mlx_cache:
            pytorch_weight = pytorch_state_dict[key].cpu().numpy()
            mlx_weight = mlx_cache[key]
            
            print(f"   {key}:")
            print(f"     PyTorch: shape={pytorch_weight.shape}, range=[{pytorch_weight.min():.6f}, {pytorch_weight.max():.6f}]")
            print(f"     MLX:     shape={mlx_weight.shape}, range=[{mlx_weight.min():.6f}, {mlx_weight.max():.6f}]")
            
            # 计算差异
            diff = np.abs(pytorch_weight - mlx_weight)
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            print(f"     差异: 最大={max_diff:.10f}, 平均={mean_diff:.10f}")
            print(f"     一致性: {'✅ 一致' if max_diff < 1e-5 else '❌ 不一致'}")
            print()
        else:
            print(f"   ❌ 缺少权重: {key}")
    
    # 比较cond_embedder权重
    print("5. 比较cond_embedder权重...")
    cond_embedder_key = "models.cfm.estimator.cond_embedder.weight"
    
    if cond_embedder_key in pytorch_state_dict and cond_embedder_key in mlx_cache:
        pytorch_weight = pytorch_state_dict[cond_embedder_key].cpu().numpy()
        mlx_weight = mlx_cache[cond_embedder_key]
        
        print(f"   {cond_embedder_key}:")
        print(f"     PyTorch: shape={pytorch_weight.shape}, range=[{pytorch_weight.min():.6f}, {pytorch_weight.max():.6f}]")
        print(f"     MLX:     shape={mlx_weight.shape}, range=[{mlx_weight.min():.6f}, {mlx_weight.max():.6f}]")
        
        # 计算差异
        diff = np.abs(pytorch_weight - mlx_weight)
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        print(f"     差异: 最大={max_diff:.10f}, 平均={mean_diff:.10f}")
        print(f"     一致性: {'✅ 一致' if max_diff < 1e-5 else '❌ 不一致'}")
    else:
        print(f"   ❌ 缺少权重: {cond_embedder_key}")
    
    # 检查其他关键权重
    print("\n6. 检查其他关键权重...")
    other_keys = [
        "models.cfm.estimator.final_layer.adaLN_modulation.1.weight",
        "models.cfm.estimator.final_layer.adaLN_modulation.1.bias",
        "models.cfm.estimator.cond_projection.weight",
        "models.cfm.estimator.cond_projection.bias"
    ]
    
    for key in other_keys:
        if key in pytorch_state_dict and key in mlx_cache:
            pytorch_weight = pytorch_state_dict[key].cpu().numpy()
            mlx_weight = mlx_cache[key]
            
            diff = np.abs(pytorch_weight - mlx_weight)
            max_diff = diff.max()
            
            print(f"   {key}: 差异={max_diff:.10f} {'✅' if max_diff < 1e-5 else '❌'}")
        else:
            print(f"   ❌ 缺少权重: {key}")
    
    return True

if __name__ == "__main__":
    try:
        success = compare_weights_pytorch_mlx()
        if success:
            print("\n✅ 权重比较完成")
        else:
            print("\n❌ 权重比较失败")
    except Exception as e:
        print(f"\n❌ 比较失败: {e}")
        import traceback
        traceback.print_exc()
