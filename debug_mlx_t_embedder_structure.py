#!/usr/bin/env python3
"""
深入分析MLX t_embedder结构问题
对比PyTorch和MLX的t_embedder实现差异
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

def debug_mlx_t_embedder_structure():
    """深入分析MLX t_embedder结构问题"""
    
    print("=== 深入分析MLX t_embedder结构问题 ===")
    
    # 初始化TTS系统
    print("\n1. 初始化TTS系统...")
    tts = IndexTTS2()
    
    # 获取PyTorch CFM estimator
    pytorch_cfm = tts.s2mel.models.cfm
    pytorch_estimator = pytorch_cfm.estimator
    pytorch_t_embedder = pytorch_estimator.t_embedder
    
    print("\n2. 分析PyTorch t_embedder结构...")
    print(f"PyTorch t_embedder类型: {type(pytorch_t_embedder)}")
    print(f"PyTorch t_embedder属性: {[attr for attr in dir(pytorch_t_embedder) if not attr.startswith('_')]}")
    
    # 分析PyTorch t_embedder的mlp结构
    print(f"\nPyTorch t_embedder.mlp类型: {type(pytorch_t_embedder.mlp)}")
    print(f"PyTorch t_embedder.mlp长度: {len(pytorch_t_embedder.mlp)}")
    
    for i, layer in enumerate(pytorch_t_embedder.mlp):
        print(f"  layer_{i}: {type(layer)}")
        if hasattr(layer, 'weight'):
            print(f"    权重形状: {layer.weight.shape}")
            print(f"    权重范围: [{layer.weight.min():.6f}, {layer.weight.max():.6f}]")
        if hasattr(layer, 'bias'):
            print(f"    偏置形状: {layer.bias.shape}")
            print(f"    偏置范围: [{layer.bias.min():.6f}, {layer.bias.max():.6f}]")
    
    # 分析PyTorch t_embedder的其他属性
    print(f"\nPyTorch t_embedder其他属性:")
    if hasattr(pytorch_t_embedder, 'hidden_size'):
        print(f"  hidden_size: {pytorch_t_embedder.hidden_size}")
    if hasattr(pytorch_t_embedder, 'frequency_embedding_size'):
        print(f"  frequency_embedding_size: {pytorch_t_embedder.frequency_embedding_size}")
    if hasattr(pytorch_t_embedder, 'max_period'):
        print(f"  max_period: {pytorch_t_embedder.max_period}")
    if hasattr(pytorch_t_embedder, 'scale'):
        print(f"  scale: {pytorch_t_embedder.scale}")
    
    # 分析PyTorch t_embedder的freqs
    if hasattr(pytorch_t_embedder, 'freqs'):
        print(f"  freqs形状: {pytorch_t_embedder.freqs.shape}")
        print(f"  freqs范围: [{pytorch_t_embedder.freqs.min():.6f}, {pytorch_t_embedder.freqs.max():.6f}]")
    
    print("\n3. 分析MLX t_embedder结构...")
    
    # 手动创建MLX CFM
    from indextts.s2mel.modules.mlx_cfm import MLXCFM
    mlx_cfm = MLXCFM(tts.cfg.s2mel)
    mlx_estimator = mlx_cfm.estimator
    mlx_t_embedder = mlx_estimator.t_embedder
    
    print(f"MLX t_embedder类型: {type(mlx_t_embedder)}")
    print(f"MLX t_embedder属性: {[attr for attr in dir(mlx_t_embedder) if not attr.startswith('_')]}")
    
    # 分析MLX t_embedder的属性
    print(f"\nMLX t_embedder属性:")
    print(f"  hidden_size: {mlx_t_embedder.hidden_size}")
    print(f"  frequency_embedding_size: {mlx_t_embedder.frequency_embedding_size}")
    print(f"  max_period: {mlx_t_embedder.max_period}")
    print(f"  scale: {mlx_t_embedder.scale}")
    
    # 分析MLX t_embedder的mlp结构
    if hasattr(mlx_t_embedder, 'mlp_0'):
        print(f"\nMLX t_embedder.mlp_0:")
        print(f"  类型: {type(mlx_t_embedder.mlp_0)}")
        print(f"  权重形状: {mlx_t_embedder.mlp_0.weight.shape}")
        print(f"  权重范围: [{mlx_t_embedder.mlp_0.weight.min():.6f}, {mlx_t_embedder.mlp_0.weight.max():.6f}]")
        if hasattr(mlx_t_embedder.mlp_0, 'bias'):
            print(f"  偏置形状: {mlx_t_embedder.mlp_0.bias.shape}")
            print(f"  偏置范围: [{mlx_t_embedder.mlp_0.bias.min():.6f}, {mlx_t_embedder.mlp_0.bias.max():.6f}]")
    
    if hasattr(mlx_t_embedder, 'mlp_2'):
        print(f"\nMLX t_embedder.mlp_2:")
        print(f"  类型: {type(mlx_t_embedder.mlp_2)}")
        print(f"  权重形状: {mlx_t_embedder.mlp_2.weight.shape}")
        print(f"  权重范围: [{mlx_t_embedder.mlp_2.weight.min():.6f}, {mlx_t_embedder.mlp_2.weight.max():.6f}]")
        if hasattr(mlx_t_embedder.mlp_2, 'bias'):
            print(f"  偏置形状: {mlx_t_embedder.mlp_2.bias.shape}")
            print(f"  偏置范围: [{mlx_t_embedder.mlp_2.bias.min():.6f}, {mlx_t_embedder.mlp_2.bias.max():.6f}]")
    
    print("\n4. 对比t_embedder实现...")
    
    # 生成测试输入
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    t = torch.zeros(1, device=device)
    mlx_t = torch_to_mlx(t)
    
    print(f"测试输入 t: {t.shape}, 值: {t}")
    print(f"MLX测试输入 t: {mlx_t.shape}, 值: {mlx_t}")
    
    # 测试PyTorch t_embedder的timestep_embedding
    print("\n测试PyTorch t_embedder.timestep_embedding...")
    pytorch_t_freq = pytorch_t_embedder.timestep_embedding(t)
    print(f"PyTorch t_freq形状: {pytorch_t_freq.shape}")
    print(f"PyTorch t_freq范围: [{pytorch_t_freq.min():.6f}, {pytorch_t_freq.max():.6f}]")
    print(f"PyTorch t_freq统计: mean={pytorch_t_freq.mean():.6f}, std={pytorch_t_freq.std():.6f}")
    
    # 测试MLX t_embedder的timestep_embedding
    print("\n测试MLX t_embedder.timestep_embedding...")
    mlx_t_freq = mlx_t_embedder.timestep_embedding(mlx_t)
    print(f"MLX t_freq形状: {mlx_t_freq.shape}")
    print(f"MLX t_freq范围: [{mlx_t_freq.min():.6f}, {mlx_t_freq.max():.6f}]")
    print(f"MLX t_freq统计: mean={mlx_t_freq.mean():.6f}, std={mlx_t_freq.std():.6f}")
    
    # 对比timestep_embedding输出
    mlx_t_freq_torch = mlx_to_torch(mlx_t_freq)
    if pytorch_t_freq.device != mlx_t_freq_torch.device:
        mlx_t_freq_torch = mlx_t_freq_torch.to(pytorch_t_freq.device)
    
    diff = torch.abs(pytorch_t_freq - mlx_t_freq_torch)
    print(f"timestep_embedding差异: 最大={diff.max():.6f}, 平均={diff.mean():.6f}")
    
    # 测试PyTorch t_embedder的完整forward
    print("\n测试PyTorch t_embedder完整forward...")
    with torch.no_grad():
        pytorch_t_emb = pytorch_t_embedder(t)
        print(f"PyTorch t_emb形状: {pytorch_t_emb.shape}")
        print(f"PyTorch t_emb范围: [{pytorch_t_emb.min():.6f}, {pytorch_t_emb.max():.6f}]")
        print(f"PyTorch t_emb统计: mean={pytorch_t_emb.mean():.6f}, std={pytorch_t_emb.std():.6f}")
    
    # 测试MLX t_embedder的完整forward
    print("\n测试MLX t_embedder完整forward...")
    mlx_t_emb = mlx_t_embedder(mlx_t)
    print(f"MLX t_emb形状: {mlx_t_emb.shape}")
    print(f"MLX t_emb范围: [{mlx_t_emb.min():.6f}, {mlx_t_emb.max():.6f}]")
    print(f"MLX t_emb统计: mean={mlx_t_emb.mean():.6f}, std={mlx_t_emb.std():.6f}")
    
    # 对比完整输出
    mlx_t_emb_torch = mlx_to_torch(mlx_t_emb)
    if pytorch_t_emb.device != mlx_t_emb_torch.device:
        mlx_t_emb_torch = mlx_t_emb_torch.to(pytorch_t_emb.device)
    
    diff = torch.abs(pytorch_t_emb - mlx_t_emb_torch)
    print(f"完整t_embedder差异: 最大={diff.max():.6f}, 平均={diff.mean():.6f}")
    
    print("\n=== 分析总结 ===")
    print("1. ✅ cond_embedder权重修复成功")
    print("2. ❌ t_embedder仍有问题，需要深入分析结构差异")
    print("3. 可能的问题:")
    print("   - MLX t_embedder的mlp结构与PyTorch不一致")
    print("   - MLX t_embedder的timestep_embedding实现不同")
    print("   - MLX t_embedder的激活函数或计算逻辑不同")

if __name__ == "__main__":
    debug_mlx_t_embedder_structure()
