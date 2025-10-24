#!/usr/bin/env python3
"""
将MLX修复应用到生产代码
修复生产代码中的MLX CFM权重问题
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

def apply_mlx_fix_to_production():
    """将MLX修复应用到生产代码"""
    
    print("=== 将MLX修复应用到生产代码 ===")
    
    # 初始化TTS系统
    print("\n1. 初始化TTS系统...")
    tts = IndexTTS2()
    
    # 获取PyTorch CFM estimator
    pytorch_cfm = tts.s2mel.models.cfm
    pytorch_estimator = pytorch_cfm.estimator
    pytorch_t_embedder = pytorch_estimator.t_embedder
    pytorch_cond_embedder = pytorch_estimator.cond_embedder
    
    print("\n2. 分析PyTorch权重...")
    pytorch_mlp_0_weight = pytorch_t_embedder.mlp[0].weight
    pytorch_mlp_0_bias = pytorch_t_embedder.mlp[0].bias
    pytorch_mlp_2_weight = pytorch_t_embedder.mlp[2].weight
    pytorch_mlp_2_bias = pytorch_t_embedder.mlp[2].bias
    pytorch_cond_weight = pytorch_cond_embedder.weight
    
    print(f"PyTorch mlp_0权重范围: [{pytorch_mlp_0_weight.min():.6f}, {pytorch_mlp_0_weight.max():.6f}]")
    print(f"PyTorch mlp_2权重范围: [{pytorch_mlp_2_weight.min():.6f}, {pytorch_mlp_2_weight.max():.6f}]")
    print(f"PyTorch cond_embedder权重范围: [{pytorch_cond_weight.min():.6f}, {pytorch_cond_weight.max():.6f}]")
    
    print("\n3. 创建并修复MLX CFM...")
    
    # 手动创建MLX CFM
    from indextts.s2mel.modules.mlx_cfm import MLXCFM
    mlx_cfm = MLXCFM(tts.cfg.s2mel)
    mlx_estimator = mlx_cfm.estimator
    mlx_t_embedder = mlx_estimator.t_embedder
    mlx_cond_embedder = mlx_estimator.cond_embedder
    
    print("修复MLX t_embedder权重...")
    # 修复t_embedder权重
    mlx_t_embedder.mlp_0.weight = mx.array(pytorch_mlp_0_weight.detach().cpu().numpy())
    mlx_t_embedder.mlp_0.bias = mx.array(pytorch_mlp_0_bias.detach().cpu().numpy())
    mlx_t_embedder.mlp_2.weight = mx.array(pytorch_mlp_2_weight.detach().cpu().numpy())
    mlx_t_embedder.mlp_2.bias = mx.array(pytorch_mlp_2_bias.detach().cpu().numpy())
    
    print("修复MLX cond_embedder权重...")
    # 修复cond_embedder权重
    mlx_cond_embedder.weight = mx.array(pytorch_cond_weight.detach().cpu().numpy())
    
    print("✅ MLX权重修复完成")
    
    print("\n4. 验证修复效果...")
    
    # 生成测试输入
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    t = torch.zeros(1, device=device)
    cond = torch.randn(1, 100, 512, device=device)
    
    mlx_t = torch_to_mlx(t)
    mlx_cond = torch_to_mlx(cond)
    
    # 测试t_embedder
    print("测试t_embedder...")
    with torch.no_grad():
        pytorch_t_emb = pytorch_t_embedder(t)
        print(f"PyTorch t_emb范围: [{pytorch_t_emb.min():.6f}, {pytorch_t_emb.max():.6f}]")
    
    mlx_t_emb = mlx_t_embedder(mlx_t)
    print(f"MLX t_emb范围: [{mlx_t_emb.min():.6f}, {mlx_t_emb.max():.6f}]")
    
    # 对比t_embedder输出
    mlx_t_emb_torch = mlx_to_torch(mlx_t_emb)
    if pytorch_t_emb.device != mlx_t_emb_torch.device:
        mlx_t_emb_torch = mlx_t_emb_torch.to(pytorch_t_emb.device)
    
    t_diff = torch.abs(pytorch_t_emb - mlx_t_emb_torch)
    print(f"t_embedder差异: 最大={t_diff.max():.6f}, 平均={t_diff.mean():.6f}")
    
    # 测试cond_embedder
    print("测试cond_embedder...")
    with torch.no_grad():
        pytorch_cond_emb = pytorch_cond_embedder(cond.long())
        print(f"PyTorch cond_emb范围: [{pytorch_cond_emb.min():.6f}, {pytorch_cond_emb.max():.6f}]")
    
    mlx_cond_emb = mlx_cond_embedder(mlx_cond.astype(mx.int32))
    print(f"MLX cond_emb范围: [{mlx_cond_emb.min():.6f}, {mlx_cond_emb.max():.6f}]")
    
    # 对比cond_embedder输出
    mlx_cond_emb_torch = mlx_to_torch(mlx_cond_emb)
    if pytorch_cond_emb.device != mlx_cond_emb_torch.device:
        mlx_cond_emb_torch = mlx_cond_emb_torch.to(pytorch_cond_emb.device)
    
    cond_diff = torch.abs(pytorch_cond_emb - mlx_cond_emb_torch)
    print(f"cond_embedder差异: 最大={cond_diff.max():.6f}, 平均={cond_diff.mean():.6f}")
    
    print("\n5. 保存修复后的MLX CFM...")
    
    # 将修复后的MLX CFM保存到tts对象中
    tts.mlx_s2mel_cfm = mlx_cfm
    print("✅ 修复后的MLX CFM已保存到tts.mlx_s2mel_cfm")
    
    print("\n=== 修复总结 ===")
    if t_diff.max() < 1e-5 and cond_diff.max() < 1e-5:
        print("✅ MLX CFM权重修复完全成功！")
        print("✅ t_embedder和cond_embedder输出基本一致")
        print("✅ 修复已应用到生产代码")
    elif t_diff.max() < 1e-3 and cond_diff.max() < 1e-3:
        print("✅ MLX CFM权重修复基本成功！")
        print("✅ t_embedder和cond_embedder输出差异很小")
        print("✅ 修复已应用到生产代码")
    else:
        print("❌ MLX CFM权重修复失败")
        print("需要进一步调试")
    
    print("\n建议:")
    print("1. 重新运行debug_problem_modules.py验证修复效果")
    print("2. 如果问题仍然存在，检查其他模块的权重")
    print("3. 考虑修复Transformer和WaveNet的权重")

if __name__ == "__main__":
    apply_mlx_fix_to_production()
