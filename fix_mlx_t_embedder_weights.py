#!/usr/bin/env python3
"""
完全修复MLX t_embedder权重
从PyTorch直接复制所有权重和偏置到MLX
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

def fix_mlx_t_embedder_weights():
    """完全修复MLX t_embedder权重"""
    
    print("=== 完全修复MLX t_embedder权重 ===")
    
    # 初始化TTS系统
    print("\n1. 初始化TTS系统...")
    tts = IndexTTS2()
    
    # 获取PyTorch CFM estimator
    pytorch_cfm = tts.s2mel.models.cfm
    pytorch_estimator = pytorch_cfm.estimator
    pytorch_t_embedder = pytorch_estimator.t_embedder
    
    print("\n2. 分析PyTorch t_embedder权重...")
    
    # 获取PyTorch t_embedder的所有权重
    pytorch_mlp_0_weight = pytorch_t_embedder.mlp[0].weight
    pytorch_mlp_0_bias = pytorch_t_embedder.mlp[0].bias
    pytorch_mlp_2_weight = pytorch_t_embedder.mlp[2].weight
    pytorch_mlp_2_bias = pytorch_t_embedder.mlp[2].bias
    
    print(f"PyTorch mlp_0权重: {pytorch_mlp_0_weight.shape}, 范围: [{pytorch_mlp_0_weight.min():.6f}, {pytorch_mlp_0_weight.max():.6f}]")
    print(f"PyTorch mlp_0偏置: {pytorch_mlp_0_bias.shape}, 范围: [{pytorch_mlp_0_bias.min():.6f}, {pytorch_mlp_0_bias.max():.6f}]")
    print(f"PyTorch mlp_2权重: {pytorch_mlp_2_weight.shape}, 范围: [{pytorch_mlp_2_weight.min():.6f}, {pytorch_mlp_2_weight.max():.6f}]")
    print(f"PyTorch mlp_2偏置: {pytorch_mlp_2_bias.shape}, 范围: [{pytorch_mlp_2_bias.min():.6f}, {pytorch_mlp_2_bias.max():.6f}]")
    
    print("\n3. 创建MLX CFM并修复权重...")
    
    # 手动创建MLX CFM
    from indextts.s2mel.modules.mlx_cfm import MLXCFM
    mlx_cfm = MLXCFM(tts.cfg.s2mel)
    mlx_estimator = mlx_cfm.estimator
    mlx_t_embedder = mlx_estimator.t_embedder
    
    print(f"MLX t_embedder类型: {type(mlx_t_embedder)}")
    
    # 修复MLX t_embedder的所有权重
    print("\n4. 修复MLX t_embedder权重...")
    
    # 修复mlp_0权重和偏置
    print("修复mlp_0权重和偏置...")
    mlx_t_embedder.mlp_0.weight = mx.array(pytorch_mlp_0_weight.detach().cpu().numpy())
    mlx_t_embedder.mlp_0.bias = mx.array(pytorch_mlp_0_bias.detach().cpu().numpy())
    print(f"✅ MLX mlp_0权重: {mlx_t_embedder.mlp_0.weight.shape}, 范围: [{mlx_t_embedder.mlp_0.weight.min():.6f}, {mlx_t_embedder.mlp_0.weight.max():.6f}]")
    print(f"✅ MLX mlp_0偏置: {mlx_t_embedder.mlp_0.bias.shape}, 范围: [{mlx_t_embedder.mlp_0.bias.min():.6f}, {mlx_t_embedder.mlp_0.bias.max():.6f}]")
    
    # 修复mlp_2权重和偏置
    print("修复mlp_2权重和偏置...")
    mlx_t_embedder.mlp_2.weight = mx.array(pytorch_mlp_2_weight.detach().cpu().numpy())
    mlx_t_embedder.mlp_2.bias = mx.array(pytorch_mlp_2_bias.detach().cpu().numpy())
    print(f"✅ MLX mlp_2权重: {mlx_t_embedder.mlp_2.weight.shape}, 范围: [{mlx_t_embedder.mlp_2.weight.min():.6f}, {mlx_t_embedder.mlp_2.weight.max():.6f}]")
    print(f"✅ MLX mlp_2偏置: {mlx_t_embedder.mlp_2.bias.shape}, 范围: [{mlx_t_embedder.mlp_2.bias.min():.6f}, {mlx_t_embedder.mlp_2.bias.max():.6f}]")
    
    print("\n5. 验证修复效果...")
    
    # 生成测试输入
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    t = torch.zeros(1, device=device)
    mlx_t = torch_to_mlx(t)
    
    print(f"测试输入 t: {t.shape}, 值: {t}")
    
    # 测试PyTorch t_embedder
    print("\n测试PyTorch t_embedder...")
    with torch.no_grad():
        pytorch_t_emb = pytorch_t_embedder(t)
        print(f"PyTorch t_emb形状: {pytorch_t_emb.shape}")
        print(f"PyTorch t_emb范围: [{pytorch_t_emb.min():.6f}, {pytorch_t_emb.max():.6f}]")
        print(f"PyTorch t_emb统计: mean={pytorch_t_emb.mean():.6f}, std={pytorch_t_emb.std():.6f}")
    
    # 测试MLX t_embedder
    print("\n测试MLX t_embedder...")
    mlx_t_emb = mlx_t_embedder(mlx_t)
    print(f"MLX t_emb形状: {mlx_t_emb.shape}")
    print(f"MLX t_emb范围: [{mlx_t_emb.min():.6f}, {mlx_t_emb.max():.6f}]")
    print(f"MLX t_emb统计: mean={mlx_t_emb.mean():.6f}, std={mlx_t_emb.std():.6f}")
    
    # 对比输出
    mlx_t_emb_torch = mlx_to_torch(mlx_t_emb)
    if pytorch_t_emb.device != mlx_t_emb_torch.device:
        mlx_t_emb_torch = mlx_t_emb_torch.to(pytorch_t_emb.device)
    
    diff = torch.abs(pytorch_t_emb - mlx_t_emb_torch)
    print(f"\nt_embedder输出差异:")
    print(f"  最大差异: {diff.max():.6f}")
    print(f"  平均差异: {diff.mean():.6f}")
    print(f"  标准差差异: {diff.std():.6f}")
    print(f"  差异比例: {diff.max() / pytorch_t_emb.std():.6f}")
    
    print("\n=== 修复总结 ===")
    if diff.max() < 1e-5:
        print("✅ MLX t_embedder权重修复成功！输出基本一致")
    elif diff.max() < 1e-3:
        print("✅ MLX t_embedder权重修复基本成功，差异很小")
    else:
        print("❌ MLX t_embedder权重修复失败，仍有较大差异")
        print("建议:")
        print("1. 检查MLX t_embedder的激活函数实现")
        print("2. 检查MLX t_embedder的计算逻辑")
        print("3. 验证MLX t_embedder的初始化")

if __name__ == "__main__":
    fix_mlx_t_embedder_weights()
