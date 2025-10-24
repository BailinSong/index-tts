#!/usr/bin/env python3
"""
直接修复MLX权重加载问题
问题根源：MLX的t_embedder和cond_embedder权重没有正确从PyTorch转换
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

def fix_mlx_weight_loading_direct():
    """直接修复MLX权重加载问题"""
    
    print("=== 直接修复MLX权重加载问题 ===")
    
    # 初始化TTS系统
    print("\n1. 初始化TTS系统...")
    tts = IndexTTS2()
    
    # 获取PyTorch CFM estimator
    pytorch_cfm = tts.s2mel.models.cfm
    pytorch_estimator = pytorch_cfm.estimator
    
    print("\n2. 分析PyTorch权重结构...")
    
    # 分析PyTorch t_embedder权重
    pytorch_t_embedder = pytorch_estimator.t_embedder
    pytorch_t_weight = pytorch_t_embedder.mlp[0].weight  # 第一个线性层
    print(f"PyTorch t_embedder权重: {pytorch_t_weight.shape}, 范围: [{pytorch_t_weight.min():.6f}, {pytorch_t_weight.max():.6f}]")
    
    # 分析PyTorch cond_embedder权重
    pytorch_cond_embedder = pytorch_estimator.cond_embedder
    pytorch_cond_weight = pytorch_cond_embedder.weight
    print(f"PyTorch cond_embedder权重: {pytorch_cond_weight.shape}, 范围: [{pytorch_cond_weight.min():.6f}, {pytorch_cond_weight.max():.6f}]")
    
    print("\n3. 手动创建MLX CFM并直接加载权重...")
    
    # 手动创建MLX CFM
    from indextts.s2mel.modules.mlx_cfm import MLXCFM
    mlx_cfm = MLXCFM(tts.cfg.s2mel)
    mlx_estimator = mlx_cfm.estimator
    
    # 直接获取MLX t_embedder和cond_embedder
    mlx_t_embedder = mlx_estimator.t_embedder
    mlx_cond_embedder = mlx_estimator.cond_embedder
    
    print(f"MLX t_embedder类型: {type(mlx_t_embedder)}")
    print(f"MLX cond_embedder类型: {type(mlx_cond_embedder)}")
    
    print("\n4. 直接修复MLX权重...")
    
    # 修复t_embedder权重
    print("修复t_embedder权重...")
    pytorch_t_weight_np = pytorch_t_weight.detach().cpu().numpy()
    
    # 直接设置MLX t_embedder权重
    if hasattr(mlx_t_embedder, 'mlp_0'):
        mlx_t_embedder.mlp_0.weight = mx.array(pytorch_t_weight_np)
        print(f"✅ 直接设置MLX t_embedder.mlp_0.weight")
        print(f"   形状: {mlx_t_embedder.mlp_0.weight.shape}")
        print(f"   范围: [{mlx_t_embedder.mlp_0.weight.min():.6f}, {mlx_t_embedder.mlp_0.weight.max():.6f}]")
    else:
        print("❌ MLX t_embedder没有mlp_0属性")
    
    # 修复cond_embedder权重
    print("修复cond_embedder权重...")
    pytorch_cond_weight_np = pytorch_cond_weight.detach().cpu().numpy()
    
    # 直接设置MLX cond_embedder权重
    mlx_cond_embedder.weight = mx.array(pytorch_cond_weight_np)
    print(f"✅ 直接设置MLX cond_embedder.weight")
    print(f"   形状: {mlx_cond_embedder.weight.shape}")
    print(f"   范围: [{mlx_cond_embedder.weight.min():.6f}, {mlx_cond_embedder.weight.max():.6f}]")
    
    print("\n5. 验证修复效果...")
    
    # 生成测试输入
    batch_size = 1
    seq_len = 100
    in_channels = 80
    hidden_dim = 512
    style_dim = 192
    
    # 获取设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    x = torch.randn(batch_size, in_channels, seq_len, device=device)
    prompt_x = torch.randn(batch_size, in_channels, seq_len, device=device)
    x_lens = torch.tensor([seq_len], device=device)
    t = torch.zeros(batch_size, device=device)
    style = torch.randn(batch_size, style_dim, device=device)
    cond = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # 准备MLX输入
    mlx_x = torch_to_mlx(x)
    mlx_prompt_x = torch_to_mlx(prompt_x)
    mlx_x_lens = torch_to_mlx(x_lens)
    mlx_t = torch_to_mlx(t)
    mlx_style = torch_to_mlx(style)
    mlx_cond = torch_to_mlx(cond)
    
    # 测试t_embedder
    print("\n测试t_embedder...")
    with torch.no_grad():
        pytorch_t_emb = pytorch_estimator.t_embedder(t)
        print(f"PyTorch t_embedder输出: {pytorch_t_emb.shape}, 范围: [{pytorch_t_emb.min():.6f}, {pytorch_t_emb.max():.6f}]")
    
    mlx_t_emb = mlx_estimator.t_embedder(mlx_t)
    print(f"MLX t_embedder输出: {mlx_t_emb.shape}, 范围: [{mlx_t_emb.min():.6f}, {mlx_t_emb.max():.6f}]")
    
    # 对比t_embedder输出
    mlx_t_emb_torch = mlx_to_torch(mlx_t_emb)
    if pytorch_t_emb.device != mlx_t_emb_torch.device:
        mlx_t_emb_torch = mlx_t_emb_torch.to(pytorch_t_emb.device)
    
    diff = torch.abs(pytorch_t_emb - mlx_t_emb_torch)
    print(f"t_embedder输出差异: 最大={diff.max():.6f}, 平均={diff.mean():.6f}")
    
    # 测试cond_embedder
    print("\n测试cond_embedder...")
    with torch.no_grad():
        pytorch_cond_emb = pytorch_estimator.cond_embedder(cond.long())
        print(f"PyTorch cond_embedder输出: {pytorch_cond_emb.shape}, 范围: [{pytorch_cond_emb.min():.6f}, {pytorch_cond_emb.max():.6f}]")
    
    mlx_cond_emb = mlx_estimator.cond_embedder(mlx_cond.astype(mx.int32))
    print(f"MLX cond_embedder输出: {mlx_cond_emb.shape}, 范围: [{mlx_cond_emb.min():.6f}, {mlx_cond_emb.max():.6f}]")
    
    # 对比cond_embedder输出
    mlx_cond_emb_torch = mlx_to_torch(mlx_cond_emb)
    if pytorch_cond_emb.device != mlx_cond_emb_torch.device:
        mlx_cond_emb_torch = mlx_cond_emb_torch.to(pytorch_cond_emb.device)
    
    diff = torch.abs(pytorch_cond_emb - mlx_cond_emb_torch)
    print(f"cond_embedder输出差异: 最大={diff.max():.6f}, 平均={diff.mean():.6f}")
    
    print("\n=== 修复总结 ===")
    if diff.max() < 1e-5:
        print("✅ 权重修复成功！MLX和PyTorch输出基本一致")
    else:
        print("❌ 权重修复失败，仍有差异")
        print("建议:")
        print("1. 检查MLX t_embedder结构是否与PyTorch一致")
        print("2. 检查权重转换过程")
        print("3. 验证MLX模型初始化")

if __name__ == "__main__":
    fix_mlx_weight_loading_direct()
