#!/usr/bin/env python3
"""
验证MLX修复是否应用到生产代码
检查生产代码中的MLX CFM是否使用了修复后的权重
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

def verify_mlx_fix_production():
    """验证MLX修复是否应用到生产代码"""
    
    print("=== 验证MLX修复是否应用到生产代码 ===")
    
    # 初始化TTS系统
    print("\n1. 初始化TTS系统...")
    tts = IndexTTS2()
    
    # 获取PyTorch CFM estimator
    pytorch_cfm = tts.s2mel.models.cfm
    pytorch_estimator = pytorch_cfm.estimator
    pytorch_t_embedder = pytorch_estimator.t_embedder
    
    print("\n2. 分析PyTorch t_embedder权重...")
    pytorch_mlp_0_weight = pytorch_t_embedder.mlp[0].weight
    pytorch_mlp_0_bias = pytorch_t_embedder.mlp[0].bias
    pytorch_mlp_2_weight = pytorch_t_embedder.mlp[2].weight
    pytorch_mlp_2_bias = pytorch_t_embedder.mlp[2].bias
    
    print(f"PyTorch mlp_0权重范围: [{pytorch_mlp_0_weight.min():.6f}, {pytorch_mlp_0_weight.max():.6f}]")
    print(f"PyTorch mlp_2权重范围: [{pytorch_mlp_2_weight.min():.6f}, {pytorch_mlp_2_weight.max():.6f}]")
    
    print("\n3. 检查生产代码中的MLX CFM...")
    
    # 检查tts.mlx_s2mel_cfm是否存在
    if hasattr(tts, 'mlx_s2mel_cfm') and tts.mlx_s2mel_cfm is not None:
        print("✅ 找到生产代码中的MLX CFM")
        mlx_cfm = tts.mlx_s2mel_cfm
        mlx_estimator = mlx_cfm.estimator
        mlx_t_embedder = mlx_estimator.t_embedder
        
        print(f"MLX t_embedder类型: {type(mlx_t_embedder)}")
        
        # 检查MLX t_embedder权重
        print(f"MLX mlp_0权重范围: [{mlx_t_embedder.mlp_0.weight.min():.6f}, {mlx_t_embedder.mlp_0.weight.max():.6f}]")
        print(f"MLX mlp_2权重范围: [{mlx_t_embedder.mlp_2.weight.min():.6f}, {mlx_t_embedder.mlp_2.weight.max():.6f}]")
        
        # 对比权重
        mlx_mlp_0_weight_np = mlx_t_embedder.mlp_0.weight
        mlx_mlp_2_weight_np = mlx_t_embedder.mlp_2.weight
        
        pytorch_mlp_0_weight_np = pytorch_mlp_0_weight.detach().cpu().numpy()
        pytorch_mlp_2_weight_np = pytorch_mlp_2_weight.detach().cpu().numpy()
        
        mlp_0_diff = np.abs(pytorch_mlp_0_weight_np - mlx_mlp_0_weight_np)
        mlp_2_diff = np.abs(pytorch_mlp_2_weight_np - mlx_mlp_2_weight_np)
        
        print(f"mlp_0权重差异: 最大={mlp_0_diff.max():.6f}, 平均={mlp_0_diff.mean():.6f}")
        print(f"mlp_2权重差异: 最大={mlp_2_diff.max():.6f}, 平均={mlp_2_diff.mean():.6f}")
        
        if mlp_0_diff.max() < 1e-6 and mlp_2_diff.max() < 1e-6:
            print("✅ 生产代码中的MLX t_embedder权重已修复")
        else:
            print("❌ 生产代码中的MLX t_embedder权重未修复")
            
    else:
        print("❌ 生产代码中没有MLX CFM")
        print("需要手动创建MLX CFM")
        
        # 手动创建MLX CFM
        from indextts.s2mel.modules.mlx_cfm import MLXCFM
        mlx_cfm = MLXCFM(tts.cfg.s2mel)
        mlx_estimator = mlx_cfm.estimator
        mlx_t_embedder = mlx_estimator.t_embedder
        
        print(f"手动创建的MLX t_embedder类型: {type(mlx_t_embedder)}")
        
        # 检查手动创建的MLX t_embedder权重
        print(f"手动创建MLX mlp_0权重范围: [{mlx_t_embedder.mlp_0.weight.min():.6f}, {mlx_t_embedder.mlp_0.weight.max():.6f}]")
        print(f"手动创建MLX mlp_2权重范围: [{mlx_t_embedder.mlp_2.weight.min():.6f}, {mlx_t_embedder.mlp_2.weight.max():.6f}]")
        
        # 对比权重
        mlx_mlp_0_weight_np = mlx_t_embedder.mlp_0.weight
        mlx_mlp_2_weight_np = mlx_t_embedder.mlp_2.weight
        
        pytorch_mlp_0_weight_np = pytorch_mlp_0_weight.detach().cpu().numpy()
        pytorch_mlp_2_weight_np = pytorch_mlp_2_weight.detach().cpu().numpy()
        
        mlp_0_diff = np.abs(pytorch_mlp_0_weight_np - mlx_mlp_0_weight_np)
        mlp_2_diff = np.abs(pytorch_mlp_2_weight_np - mlx_mlp_2_weight_np)
        
        print(f"手动创建mlp_0权重差异: 最大={mlp_0_diff.max():.6f}, 平均={mlp_0_diff.mean():.6f}")
        print(f"手动创建mlp_2权重差异: 最大={mlp_2_diff.max():.6f}, 平均={mlp_2_diff.mean():.6f}")
        
        if mlp_0_diff.max() < 1e-6 and mlp_2_diff.max() < 1e-6:
            print("✅ 手动创建的MLX t_embedder权重已修复")
        else:
            print("❌ 手动创建的MLX t_embedder权重未修复")
            print("需要应用权重修复")
            
            # 应用权重修复
            print("\n4. 应用权重修复...")
            mlx_t_embedder.mlp_0.weight = mx.array(pytorch_mlp_0_weight_np)
            mlx_t_embedder.mlp_0.bias = mx.array(pytorch_t_embedder.mlp[0].bias.detach().cpu().numpy())
            mlx_t_embedder.mlp_2.weight = mx.array(pytorch_mlp_2_weight_np)
            mlx_t_embedder.mlp_2.bias = mx.array(pytorch_t_embedder.mlp[2].bias.detach().cpu().numpy())
            
            print("✅ 权重修复已应用")
            
            # 验证修复效果
            print("\n5. 验证修复效果...")
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            t = torch.zeros(1, device=device)
            mlx_t = torch_to_mlx(t)
            
            # 测试PyTorch t_embedder
            with torch.no_grad():
                pytorch_t_emb = pytorch_t_embedder(t)
                print(f"PyTorch t_emb范围: [{pytorch_t_emb.min():.6f}, {pytorch_t_emb.max():.6f}]")
            
            # 测试MLX t_embedder
            mlx_t_emb = mlx_t_embedder(mlx_t)
            print(f"MLX t_emb范围: [{mlx_t_emb.min():.6f}, {mlx_t_emb.max():.6f}]")
            
            # 对比输出
            mlx_t_emb_torch = mlx_to_torch(mlx_t_emb)
            if pytorch_t_emb.device != mlx_t_emb_torch.device:
                mlx_t_emb_torch = mlx_t_emb_torch.to(pytorch_t_emb.device)
            
            diff = torch.abs(pytorch_t_emb - mlx_t_emb_torch)
            print(f"t_embedder输出差异: 最大={diff.max():.6f}, 平均={diff.mean():.6f}")
            
            if diff.max() < 1e-5:
                print("✅ MLX t_embedder修复成功！")
            else:
                print("❌ MLX t_embedder修复失败")
    
    print("\n=== 验证总结 ===")
    print("1. 检查了生产代码中的MLX CFM状态")
    print("2. 验证了MLX t_embedder权重是否正确")
    print("3. 如果权重不正确，应用了修复")
    print("4. 验证了修复效果")

if __name__ == "__main__":
    verify_mlx_fix_production()
