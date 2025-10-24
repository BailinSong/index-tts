#!/usr/bin/env python3
"""
测试MLX权重加载修复是否生效
验证生产代码中的权重加载逻辑是否正确
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

def test_mlx_weight_loading_fix():
    """测试MLX权重加载修复是否生效"""
    
    print("=== 测试MLX权重加载修复是否生效 ===")
    
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
    
    print("\n3. 测试MLX权重加载修复...")
    
    # 手动创建MLX CFM并测试权重加载
    from indextts.s2mel.modules.mlx_cfm import MLXCFM
    mlx_cfm = MLXCFM(tts.cfg.s2mel)
    mlx_estimator = mlx_cfm.estimator
    mlx_t_embedder = mlx_estimator.t_embedder
    mlx_cond_embedder = mlx_estimator.cond_embedder
    
    print("测试MLX权重加载修复...")
    
    # 创建测试用的state_dict
    test_state_dict = {
        "estimator.t_embedder.mlp.0.weight": pytorch_mlp_0_weight.detach().cpu().numpy(),
        "estimator.t_embedder.mlp.0.bias": pytorch_mlp_0_bias.detach().cpu().numpy(),
        "estimator.t_embedder.mlp.2.weight": pytorch_mlp_2_weight.detach().cpu().numpy(),
        "estimator.t_embedder.mlp.2.bias": pytorch_mlp_2_bias.detach().cpu().numpy(),
        "estimator.cond_embedder.weight": pytorch_cond_weight.detach().cpu().numpy(),
    }
    
    print("测试权重加载...")
    loaded = mlx_cfm.load_weights_from_pytorch(test_state_dict, prefix="")
    
    print(f"加载了 {loaded} 个权重")
    
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
    
    print("\n=== 修复测试总结 ===")
    if t_diff.max() < 1e-5 and cond_diff.max() < 1e-5:
        print("✅ MLX权重加载修复完全成功！")
        print("✅ t_embedder和cond_embedder输出基本一致")
    elif t_diff.max() < 1e-3 and cond_diff.max() < 1e-3:
        print("✅ MLX权重加载修复基本成功！")
        print("✅ t_embedder和cond_embedder输出差异很小")
    else:
        print("❌ MLX权重加载修复失败")
        print("需要进一步调试")
    
    return t_diff.max(), cond_diff.max()

if __name__ == "__main__":
    test_mlx_weight_loading_fix()
