#!/usr/bin/env python3
"""
MLX权重加载问题诊断和修复工具
专门检查t_embedder和cond_embedder的权重转换问题
"""

import torch
import mlx.core as mx
import numpy as np
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from indextts.infer_v2 import IndexTTS2
from indextts.utils.mlx_utils import mlx_to_torch, torch_to_mlx
from indextts.utils.mlx_cache import MLXModelCache

def debug_mlx_weight_loading():
    """诊断MLX权重加载问题"""
    
    print("=== MLX权重加载问题诊断 ===")
    
    # 初始化TTS系统
    print("\n1. 初始化TTS系统...")
    tts = IndexTTS2()
    
    # 获取PyTorch CFM estimator
    pytorch_cfm = tts.s2mel.models.cfm
    pytorch_estimator = pytorch_cfm.estimator
    
    print("\n2. 分析PyTorch t_embedder权重...")
    
    # 分析PyTorch t_embedder权重
    pytorch_t_embedder = pytorch_estimator.t_embedder
    print(f"PyTorch t_embedder类型: {type(pytorch_t_embedder)}")
    
    # 检查t_embedder的结构
    print(f"PyTorch t_embedder属性: {dir(pytorch_t_embedder)}")
    
    # 尝试访问权重
    if hasattr(pytorch_t_embedder, 'weight'):
        pytorch_t_weight = pytorch_t_embedder.weight
    elif hasattr(pytorch_t_embedder, 'linear'):
        pytorch_t_weight = pytorch_t_embedder.linear.weight
    elif hasattr(pytorch_t_embedder, 'mlp'):
        # mlp是一个Sequential对象，需要访问其子模块的权重
        mlp = pytorch_t_embedder.mlp
        if hasattr(mlp, '0') and hasattr(mlp[0], 'weight'):
            pytorch_t_weight = mlp[0].weight  # 第一个线性层
        elif hasattr(mlp, '1') and hasattr(mlp[1], 'weight'):
            pytorch_t_weight = mlp[1].weight  # 第二个线性层
        else:
            print("❌ 无法找到t_embedder.mlp中的权重")
            pytorch_t_weight = None
    else:
        print("❌ 无法找到t_embedder权重")
        pytorch_t_weight = None
    
    if pytorch_t_weight is not None:
        print(f"PyTorch t_embedder权重形状: {pytorch_t_weight.shape}")
        print(f"PyTorch t_embedder权重统计:")
        print(f"  数值范围: min={pytorch_t_weight.min():.6f}, max={pytorch_t_weight.max():.6f}")
        print(f"  统计: mean={pytorch_t_weight.mean():.6f}, std={pytorch_t_weight.std():.6f}")
    else:
        print("❌ 无法获取PyTorch t_embedder权重")
    
    # 分析PyTorch cond_embedder权重
    print("\n3. 分析PyTorch cond_embedder权重...")
    pytorch_cond_embedder = pytorch_estimator.cond_embedder
    print(f"PyTorch cond_embedder类型: {type(pytorch_cond_embedder)}")
    
    # 检查cond_embedder的结构
    print(f"PyTorch cond_embedder属性: {dir(pytorch_cond_embedder)}")
    
    # 尝试访问权重
    if hasattr(pytorch_cond_embedder, 'weight'):
        pytorch_cond_weight = pytorch_cond_embedder.weight
    elif hasattr(pytorch_cond_embedder, 'linear'):
        pytorch_cond_weight = pytorch_cond_embedder.linear.weight
    elif hasattr(pytorch_cond_embedder, 'mlp'):
        # mlp是一个Sequential对象，需要访问其子模块的权重
        mlp = pytorch_cond_embedder.mlp
        if hasattr(mlp, '0') and hasattr(mlp[0], 'weight'):
            pytorch_cond_weight = mlp[0].weight  # 第一个线性层
        elif hasattr(mlp, '1') and hasattr(mlp[1], 'weight'):
            pytorch_cond_weight = mlp[1].weight  # 第二个线性层
        else:
            print("❌ 无法找到cond_embedder.mlp中的权重")
            pytorch_cond_weight = None
    else:
        print("❌ 无法找到cond_embedder权重")
        pytorch_cond_weight = None
    
    if pytorch_cond_weight is not None:
        print(f"PyTorch cond_embedder权重形状: {pytorch_cond_weight.shape}")
        print(f"PyTorch cond_embedder权重统计:")
        print(f"  数值范围: min={pytorch_cond_weight.min():.6f}, max={pytorch_cond_weight.max():.6f}")
        print(f"  统计: mean={pytorch_cond_weight.mean():.6f}, std={pytorch_cond_weight.std():.6f}")
    else:
        print("❌ 无法获取PyTorch cond_embedder权重")
    
    # 手动创建MLX CFM
    print("\n4. 分析MLX权重...")
    from indextts.s2mel.modules.mlx_cfm import MLXCFM
    mlx_cfm = MLXCFM(tts.cfg.s2mel)
    mlx_estimator = mlx_cfm.estimator
    
    # 分析MLX t_embedder权重
    print("\n5. 分析MLX t_embedder权重...")
    mlx_t_embedder = mlx_estimator.t_embedder
    print(f"MLX t_embedder类型: {type(mlx_t_embedder)}")
    
    # 检查MLX t_embedder的结构
    print(f"MLX t_embedder属性: {dir(mlx_t_embedder)}")
    
    # 尝试访问权重
    if hasattr(mlx_t_embedder, 'weight'):
        mlx_t_weight = mlx_t_embedder.weight
    elif hasattr(mlx_t_embedder, 'linear'):
        mlx_t_weight = mlx_t_embedder.linear.weight
    elif hasattr(mlx_t_embedder, 'mlp'):
        # mlp是一个Sequential对象，需要访问其子模块的权重
        mlp = mlx_t_embedder.mlp
        if hasattr(mlp, '0') and hasattr(mlp[0], 'weight'):
            mlx_t_weight = mlp[0].weight  # 第一个线性层
        elif hasattr(mlp, '1') and hasattr(mlp[1], 'weight'):
            mlx_t_weight = mlp[1].weight  # 第二个线性层
        else:
            print("❌ 无法找到MLX t_embedder.mlp中的权重")
            mlx_t_weight = None
    else:
        print("❌ 无法找到MLX t_embedder权重")
        mlx_t_weight = None
    
    if mlx_t_weight is not None:
        print(f"MLX t_embedder权重形状: {mlx_t_weight.shape}")
        print(f"MLX t_embedder权重统计:")
        print(f"  数值范围: min={mlx_t_weight.min():.6f}, max={mlx_t_weight.max():.6f}")
        print(f"  统计: mean={mlx_t_weight.mean():.6f}, std={mlx_t_weight.std():.6f}")
    else:
        print("❌ 无法获取MLX t_embedder权重")
    
    # 分析MLX cond_embedder权重
    print("\n6. 分析MLX cond_embedder权重...")
    mlx_cond_embedder = mlx_estimator.cond_embedder
    print(f"MLX cond_embedder类型: {type(mlx_cond_embedder)}")
    
    # 检查MLX cond_embedder的结构
    print(f"MLX cond_embedder属性: {dir(mlx_cond_embedder)}")
    
    # 尝试访问权重
    if hasattr(mlx_cond_embedder, 'weight'):
        mlx_cond_weight = mlx_cond_embedder.weight
    elif hasattr(mlx_cond_embedder, 'linear'):
        mlx_cond_weight = mlx_cond_embedder.linear.weight
    elif hasattr(mlx_cond_embedder, 'mlp'):
        # mlp是一个Sequential对象，需要访问其子模块的权重
        mlp = mlx_cond_embedder.mlp
        if hasattr(mlp, '0') and hasattr(mlp[0], 'weight'):
            mlx_cond_weight = mlp[0].weight  # 第一个线性层
        elif hasattr(mlp, '1') and hasattr(mlp[1], 'weight'):
            mlx_cond_weight = mlp[1].weight  # 第二个线性层
        else:
            print("❌ 无法找到MLX cond_embedder.mlp中的权重")
            mlx_cond_weight = None
    else:
        print("❌ 无法找到MLX cond_embedder权重")
        mlx_cond_weight = None
    
    if mlx_cond_weight is not None:
        print(f"MLX cond_embedder权重形状: {mlx_cond_weight.shape}")
        print(f"MLX cond_embedder权重统计:")
        print(f"  数值范围: min={mlx_cond_weight.min():.6f}, max={mlx_cond_weight.max():.6f}")
        print(f"  统计: mean={mlx_cond_weight.mean():.6f}, std={mlx_cond_weight.std():.6f}")
    else:
        print("❌ 无法获取MLX cond_embedder权重")
    
    # 对比权重差异
    print("\n7. 权重差异分析...")
    
    # 对比t_embedder权重
    if pytorch_t_weight is not None and mlx_t_weight is not None:
        pytorch_t_weight_np = pytorch_t_weight.detach().cpu().numpy()
        mlx_t_weight_np = mlx_t_weight
        
        t_weight_diff = np.abs(pytorch_t_weight_np - mlx_t_weight_np)
        print(f"t_embedder权重差异:")
        print(f"  最大差异: {t_weight_diff.max():.6f}")
        print(f"  平均差异: {t_weight_diff.mean():.6f}")
        print(f"  标准差差异: {t_weight_diff.std():.6f}")
        print(f"  差异比例: {t_weight_diff.max() / pytorch_t_weight_np.std():.6f}")
    else:
        print("❌ 无法对比t_embedder权重")
    
    # 对比cond_embedder权重
    if pytorch_cond_weight is not None and mlx_cond_weight is not None:
        pytorch_cond_weight_np = pytorch_cond_weight.detach().cpu().numpy()
        mlx_cond_weight_np = mlx_cond_weight
        
        cond_weight_diff = np.abs(pytorch_cond_weight_np - mlx_cond_weight_np)
        print(f"cond_embedder权重差异:")
        print(f"  最大差异: {cond_weight_diff.max():.6f}")
        print(f"  平均差异: {cond_weight_diff.mean():.6f}")
        print(f"  标准差差异: {cond_weight_diff.std():.6f}")
        print(f"  差异比例: {cond_weight_diff.max() / pytorch_cond_weight_np.std():.6f}")
    else:
        print("❌ 无法对比cond_embedder权重")
    
    # 检查权重是否完全一致
    print("\n8. 权重一致性检查...")
    
    if pytorch_t_weight is not None and mlx_t_weight is not None:
        t_weight_identical = np.allclose(pytorch_t_weight.detach().cpu().numpy(), mlx_t_weight, atol=1e-6)
        print(f"t_embedder权重是否一致: {t_weight_identical}")
    else:
        t_weight_identical = False
        print("❌ 无法检查t_embedder权重一致性")
    
    if pytorch_cond_weight is not None and mlx_cond_weight is not None:
        cond_weight_identical = np.allclose(pytorch_cond_weight.detach().cpu().numpy(), mlx_cond_weight, atol=1e-6)
        print(f"cond_embedder权重是否一致: {cond_weight_identical}")
    else:
        cond_weight_identical = False
        print("❌ 无法检查cond_embedder权重一致性")
    
    if not t_weight_identical or not cond_weight_identical:
        print("\n❌ 发现问题: MLX权重与PyTorch权重不一致!")
        print("需要检查权重转换和加载过程")
        
        # 检查权重转换过程
        print("\n9. 检查权重转换过程...")
        
        # 检查MLX缓存
        cache_path = Path("s2mel.npz")
        if cache_path.exists():
            print(f"✅ 找到MLX缓存: {cache_path}")
            
            # 加载缓存并检查权重
            cache_data = np.load(cache_path)
            print(f"缓存中的键: {list(cache_data.keys())}")
            
            # 查找t_embedder和cond_embedder权重
            t_embedder_keys = [k for k in cache_data.keys() if 't_embedder' in k]
            cond_embedder_keys = [k for k in cache_data.keys() if 'cond_embedder' in k]
            
            print(f"缓存中的t_embedder键: {t_embedder_keys}")
            print(f"缓存中的cond_embedder键: {cond_embedder_keys}")
            
            if t_embedder_keys:
                cached_t_weight = cache_data[t_embedder_keys[0]]
                print(f"缓存的t_embedder权重形状: {cached_t_weight.shape}")
                print(f"缓存的t_embedder权重统计: min={cached_t_weight.min():.6f}, max={cached_t_weight.max():.6f}")
                
                # 对比缓存的权重和PyTorch权重
                cached_t_diff = np.abs(pytorch_t_weight - cached_t_weight)
                print(f"缓存t_embedder权重差异: 最大={cached_t_diff.max():.6f}, 平均={cached_t_diff.mean():.6f}")
            
            if cond_embedder_keys:
                cached_cond_weight = cache_data[cond_embedder_keys[0]]
                print(f"缓存的cond_embedder权重形状: {cached_cond_weight.shape}")
                print(f"缓存的cond_embedder权重统计: min={cached_cond_weight.min():.6f}, max={cached_cond_weight.max():.6f}")
                
                # 对比缓存的权重和PyTorch权重
                cached_cond_diff = np.abs(pytorch_cond_weight - cached_cond_weight)
                print(f"缓存cond_embedder权重差异: 最大={cached_cond_diff.max():.6f}, 平均={cached_cond_diff.mean():.6f}")
        else:
            print(f"❌ 未找到MLX缓存: {cache_path}")
            print("需要重新生成MLX缓存")
    
    print("\n=== 诊断总结 ===")
    if t_weight_identical and cond_weight_identical:
        print("✅ 权重加载正常，问题可能在其他地方")
    else:
        print("❌ 权重加载有问题，需要修复权重转换过程")
        print("建议:")
        print("1. 检查MLX权重转换逻辑")
        print("2. 重新生成MLX缓存")
        print("3. 验证权重加载过程")

if __name__ == "__main__":
    debug_mlx_weight_loading()
