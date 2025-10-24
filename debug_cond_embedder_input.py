#!/usr/bin/env python3
"""
调试cond_embedder输入数据差异
分析为什么cond_embedder的输出差异仍然很大
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

def debug_cond_embedder_input():
    """调试cond_embedder输入数据差异"""
    
    print("=== 调试cond_embedder输入数据差异 ===")
    
    # 初始化TTS系统
    print("\n1. 初始化TTS系统...")
    tts = IndexTTS2()
    
    # 获取PyTorch CFM estimator
    pytorch_cfm = tts.s2mel.models.cfm
    pytorch_estimator = pytorch_cfm.estimator
    pytorch_cond_embedder = pytorch_estimator.cond_embedder
    
    print("\n2. 创建MLX CFM并修复权重...")
    
    # 手动创建MLX CFM
    from indextts.s2mel.modules.mlx_cfm import MLXCFM
    mlx_cfm = MLXCFM(tts.cfg.s2mel)
    mlx_estimator = mlx_cfm.estimator
    mlx_cond_embedder = mlx_estimator.cond_embedder
    
    # 直接修复权重
    pytorch_weight = pytorch_cond_embedder.weight.detach().cpu().numpy()
    mlx_cond_embedder.weight = mx.array(pytorch_weight)
    print("✅ 权重已修复")
    
    print("\n3. 分析cond_embedder的输入数据...")
    
    # 生成测试输入
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 使用与debug_problem_modules.py相同的输入
    torch.manual_seed(42)
    mx.random.seed(42)
    
    # 生成与debug_problem_modules.py相同的输入
    cond = torch.randn(1, 100, 512, device=device)
    print(f"输入cond形状: {cond.shape}")
    print(f"输入cond范围: [{cond.min():.6f}, {cond.max():.6f}]")
    print(f"输入cond统计: mean={cond.mean():.6f}, std={cond.std():.6f}")
    
    # 转换为MLX
    mlx_cond = torch_to_mlx(cond)
    print(f"MLX cond形状: {mlx_cond.shape}")
    print(f"MLX cond范围: [{mlx_cond.min():.6f}, {mlx_cond.max():.6f}]")
    print(f"MLX cond统计: mean={mlx_cond.mean():.6f}, std={mlx_cond.std():.6f}")
    
    # 检查输入差异
    mlx_cond_torch = mlx_to_torch(mlx_cond)
    if cond.device != mlx_cond_torch.device:
        mlx_cond_torch = mlx_cond_torch.to(cond.device)
    cond_diff = torch.abs(cond - mlx_cond_torch)
    print(f"输入cond差异: 最大={cond_diff.max():.6f}, 平均={cond_diff.mean():.6f}")
    
    print("\n4. 分析cond_embedder的调用方式...")
    
    # 检查PyTorch cond_embedder的调用方式
    print("PyTorch cond_embedder调用方式:")
    print(f"  输入类型: {type(cond)}")
    print(f"  输入形状: {cond.shape}")
    print(f"  输入数据类型: {cond.dtype}")
    
    # 检查MLX cond_embedder的调用方式
    print("MLX cond_embedder调用方式:")
    print(f"  输入类型: {type(mlx_cond)}")
    print(f"  输入形状: {mlx_cond.shape}")
    print(f"  输入数据类型: {mlx_cond.dtype}")
    
    print("\n5. 测试cond_embedder调用...")
    
    # 测试PyTorch cond_embedder
    print("测试PyTorch cond_embedder...")
    with torch.no_grad():
        # 注意：PyTorch cond_embedder期望的是整数索引，不是浮点数
        # 我们需要将cond转换为整数索引
        cond_indices = torch.randint(0, 1024, (1, 100, 512), device=device)
        print(f"PyTorch cond_indices形状: {cond_indices.shape}")
        print(f"PyTorch cond_indices范围: [{cond_indices.min()}, {cond_indices.max()}]")
        
        pytorch_cond_emb = pytorch_cond_embedder(cond_indices.long())
        print(f"PyTorch cond_emb形状: {pytorch_cond_emb.shape}")
        print(f"PyTorch cond_emb范围: [{pytorch_cond_emb.min():.6f}, {pytorch_cond_emb.max():.6f}]")
        print(f"PyTorch cond_emb统计: mean={pytorch_cond_emb.mean():.6f}, std={pytorch_cond_emb.std():.6f}")
    
    # 测试MLX cond_embedder
    print("测试MLX cond_embedder...")
    mlx_cond_indices = torch_to_mlx(cond_indices)
    print(f"MLX cond_indices形状: {mlx_cond_indices.shape}")
    print(f"MLX cond_indices范围: [{mlx_cond_indices.min()}, {mlx_cond_indices.max()}]")
    
    mlx_cond_emb = mlx_cond_embedder(mlx_cond_indices.astype(mx.int32))
    print(f"MLX cond_emb形状: {mlx_cond_emb.shape}")
    print(f"MLX cond_emb范围: [{mlx_cond_emb.min():.6f}, {mlx_cond_emb.max():.6f}]")
    print(f"MLX cond_emb统计: mean={mlx_cond_emb.mean():.6f}, std={mlx_cond_emb.std():.6f}")
    
    # 对比输出
    mlx_cond_emb_torch = mlx_to_torch(mlx_cond_emb)
    if pytorch_cond_emb.device != mlx_cond_emb_torch.device:
        mlx_cond_emb_torch = mlx_cond_emb_torch.to(pytorch_cond_emb.device)
    
    cond_diff = torch.abs(pytorch_cond_emb - mlx_cond_emb_torch)
    print(f"cond_embedder输出差异:")
    print(f"  最大差异: {cond_diff.max():.6f}")
    print(f"  平均差异: {cond_diff.mean():.6f}")
    print(f"  标准差差异: {cond_diff.std():.6f}")
    
    print("\n6. 分析差异分布...")
    diff_np = cond_diff.detach().cpu().numpy()
    print(f"差异分布:")
    print(f"  < 1e-6: {np.sum(diff_np < 1e-6)} / {diff_np.size} ({100*np.sum(diff_np < 1e-6)/diff_np.size:.2f}%)")
    print(f"  < 1e-5: {np.sum(diff_np < 1e-5)} / {diff_np.size} ({100*np.sum(diff_np < 1e-5)/diff_np.size:.2f}%)")
    print(f"  < 1e-4: {np.sum(diff_np < 1e-4)} / {diff_np.size} ({100*np.sum(diff_np < 1e-4)/diff_np.size:.2f}%)")
    print(f"  < 1e-3: {np.sum(diff_np < 1e-3)} / {diff_np.size} ({100*np.sum(diff_np < 1e-3)/diff_np.size:.2f}%)")
    
    print("\n=== 调试总结 ===")
    if cond_diff.max() < 1e-5:
        print("✅ cond_embedder差异已达到e-5精度要求")
    elif cond_diff.max() < 1e-4:
        print("✅ cond_embedder差异已达到e-4精度要求")
    elif cond_diff.max() < 1e-3:
        print("✅ cond_embedder差异已达到e-3精度要求")
    else:
        print("❌ cond_embedder差异仍然过大，需要进一步调试")
    
    return cond_diff.max(), cond_diff.mean()

if __name__ == "__main__":
    debug_cond_embedder_input()
