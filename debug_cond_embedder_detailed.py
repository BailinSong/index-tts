#!/usr/bin/env python3
"""
深入调试cond_embedder差异问题
分析为什么cond_embedder的差异仍然很大
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

def debug_cond_embedder_detailed():
    """深入调试cond_embedder差异问题"""
    
    print("=== 深入调试cond_embedder差异问题 ===")
    
    # 初始化TTS系统
    print("\n1. 初始化TTS系统...")
    tts = IndexTTS2()
    
    # 获取PyTorch CFM estimator
    pytorch_cfm = tts.s2mel.models.cfm
    pytorch_estimator = pytorch_cfm.estimator
    pytorch_cond_embedder = pytorch_estimator.cond_embedder
    
    print("\n2. 分析PyTorch cond_embedder...")
    pytorch_weight = pytorch_cond_embedder.weight
    print(f"PyTorch cond_embedder权重:")
    print(f"  形状: {pytorch_weight.shape}")
    print(f"  数据类型: {pytorch_weight.dtype}")
    print(f"  数值范围: [{pytorch_weight.min():.6f}, {pytorch_weight.max():.6f}]")
    print(f"  统计: mean={pytorch_weight.mean():.6f}, std={pytorch_weight.std():.6f}")
    
    print("\n3. 创建MLX CFM并分析权重...")
    
    # 手动创建MLX CFM
    from indextts.s2mel.modules.mlx_cfm import MLXCFM
    mlx_cfm = MLXCFM(tts.cfg.s2mel)
    mlx_estimator = mlx_cfm.estimator
    mlx_cond_embedder = mlx_estimator.cond_embedder
    
    print(f"MLX cond_embedder权重:")
    print(f"  形状: {mlx_cond_embedder.weight.shape}")
    print(f"  数据类型: {mlx_cond_embedder.weight.dtype}")
    print(f"  数值范围: [{mlx_cond_embedder.weight.min():.6f}, {mlx_cond_embedder.weight.max():.6f}]")
    print(f"  统计: mean={mlx_cond_embedder.weight.mean():.6f}, std={mlx_cond_embedder.weight.std():.6f}")
    
    print("\n4. 对比权重差异...")
    pytorch_weight_np = pytorch_weight.detach().cpu().numpy()
    mlx_weight_np = mlx_cond_embedder.weight
    
    weight_diff = np.abs(pytorch_weight_np - mlx_weight_np)
    print(f"权重差异:")
    print(f"  最大差异: {weight_diff.max():.6f}")
    print(f"  平均差异: {weight_diff.mean():.6f}")
    print(f"  标准差差异: {weight_diff.std():.6f}")
    
    if weight_diff.max() > 1e-5:
        print(f"  ❌ 权重差异过大，需要修复")
        
        # 直接修复权重
        print("\n5. 直接修复MLX权重...")
        mlx_cond_embedder.weight = mx.array(pytorch_weight_np)
        print("✅ 权重已修复")
        
        # 验证修复效果
        mlx_weight_fixed = mlx_cond_embedder.weight
        weight_diff_fixed = np.abs(pytorch_weight_np - mlx_weight_fixed)
        print(f"修复后权重差异:")
        print(f"  最大差异: {weight_diff_fixed.max():.6f}")
        print(f"  平均差异: {weight_diff_fixed.mean():.6f}")
        
    else:
        print(f"  ✅ 权重差异在可接受范围内")
    
    print("\n6. 测试cond_embedder输出...")
    
    # 生成测试输入
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cond = torch.randint(0, 1024, (1, 100, 512), device=device)
    
    mlx_cond = torch_to_mlx(cond)
    
    # 测试PyTorch cond_embedder
    print("测试PyTorch cond_embedder...")
    with torch.no_grad():
        pytorch_cond_emb = pytorch_cond_embedder(cond.long())
        print(f"PyTorch cond_emb形状: {pytorch_cond_emb.shape}")
        print(f"PyTorch cond_emb范围: [{pytorch_cond_emb.min():.6f}, {pytorch_cond_emb.max():.6f}]")
        print(f"PyTorch cond_emb统计: mean={pytorch_cond_emb.mean():.6f}, std={pytorch_cond_emb.std():.6f}")
    
    # 测试MLX cond_embedder
    print("测试MLX cond_embedder...")
    mlx_cond_emb = mlx_cond_embedder(mlx_cond.astype(mx.int32))
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
    
    print("\n7. 分析差异分布...")
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
    debug_cond_embedder_detailed()
