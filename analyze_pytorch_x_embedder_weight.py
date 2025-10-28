#!/usr/bin/env python3
"""
分析 PyTorch x_embedder.weight 的重构过程
"""

import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
import numpy as np
from indextts.infer_v2 import IndexTTS2

def analyze_pytorch_x_embedder_weight():
    print("🔍 分析 PyTorch x_embedder.weight 重构过程")
    print("=" * 60)
    
    # 加载 PyTorch 模型
    print("📦 加载 PyTorch 模型...")
    tts = IndexTTS2(use_mlx=False)
    pytorch_cfm = tts.s2mel.models['cfm']
    pytorch_dit = pytorch_cfm.estimator
    
    print(f"✅ PyTorch DiT 类型: {type(pytorch_dit)}")
    print(f"✅ x_embedder 类型: {type(pytorch_dit.x_embedder)}")
    
    # 检查 x_embedder 的属性
    print(f"\n📊 x_embedder 属性分析:")
    print(f"   weight: {pytorch_dit.x_embedder.weight.shape}")
    print(f"   bias: {pytorch_dit.x_embedder.bias.shape}")
    
    # 检查是否有 weight_g 和 weight_v
    has_weight_g = hasattr(pytorch_dit.x_embedder, 'weight_g')
    has_weight_v = hasattr(pytorch_dit.x_embedder, 'weight_v')
    
    print(f"   has weight_g: {has_weight_g}")
    print(f"   has weight_v: {has_weight_v}")
    
    if has_weight_g and has_weight_v:
        print(f"   weight_g: {pytorch_dit.x_embedder.weight_g.shape}")
        print(f"   weight_v: {pytorch_dit.x_embedder.weight_v.shape}")
        
        # 手动重构权重
        print(f"\n🔧 手动重构权重:")
        g = pytorch_dit.x_embedder.weight_g
        v = pytorch_dit.x_embedder.weight_v
        
        # 计算范数（Linear: 沿着输入维度 dim=1）
        norm_v = torch.sqrt(torch.sum(v**2, dim=1, keepdims=True))
        print(f"   norm_v shape: {norm_v.shape}")
        print(f"   norm_v range: [{norm_v.min().item():.6f}, {norm_v.max().item():.6f}]")
        
        # 重构权重
        reconstructed_weight = g * v / (norm_v + 1e-8)
        print(f"   reconstructed_weight shape: {reconstructed_weight.shape}")
        print(f"   reconstructed_weight range: [{reconstructed_weight.min().item():.6f}, {reconstructed_weight.max().item():.6f}]")
        
        # 比较原始权重和重构权重（前向前）
        original_weight = pytorch_dit.x_embedder.weight
        if original_weight.device != reconstructed_weight.device:
            reconstructed_weight = reconstructed_weight.to(original_weight.device)
        diff = torch.abs(original_weight - reconstructed_weight)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"\n📊 权重比较(前向前):")
        print(f"   原始权重 range: [{original_weight.min().item():.6f}, {original_weight.max().item():.6f}]")
        print(f"   重构权重 range: [{reconstructed_weight.min().item():.6f}, {reconstructed_weight.max().item():.6f}]")
        print(f"   差异: max={max_diff:.6f}, mean={mean_diff:.6f}")
        
        # 触发一次前向传播，weight_norm 的 pre-hook 会将 .weight 用 g、v 重构
        print("\n🧪 触发一次前向以应用 weight_norm 预钩子...")
        device = original_weight.device
        test_input = torch.randn(2, 80, device=device)
        _ = pytorch_dit.x_embedder(test_input)
        original_weight_after = pytorch_dit.x_embedder.weight
        if original_weight_after.device != reconstructed_weight.device:
            reconstructed_weight2 = reconstructed_weight.to(original_weight_after.device)
        else:
            reconstructed_weight2 = reconstructed_weight
        diff_after = torch.abs(original_weight_after - reconstructed_weight2)
        max_diff_after = torch.max(diff_after).item()
        mean_diff_after = torch.mean(diff_after).item()
        print(f"   前向后权重范围: [{original_weight_after.min().item():.6f}, {original_weight_after.max().item():.6f}]")
        print(f"   与重构权重差异(前向后): max={max_diff_after:.6f}, mean={mean_diff_after:.6f}")
        if max_diff_after < 1e-6:
            print("   ✅ 结论: PyTorch 在前向时通过 weight_norm 预钩子使用 g、v 重构 .weight")
        else:
            print("   ❌ 结论: 前向后 .weight 仍与重构不一致(需进一步排查)")

    # 列出 state_dict 中的键（确认只有 g、v 被保存）
    print(f"\n📋 state_dict 中的相关键:")
    state_dict = pytorch_dit.state_dict()
    x_embedder_keys = [k for k in state_dict.keys() if 'x_embedder' in k]
    for key in x_embedder_keys:
        value = state_dict[key]
        print(f"   {key}: {value.shape} (range: [{value.min().item():.6f}, {value.max().item():.6f}])")

if __name__ == "__main__":
    analyze_pytorch_x_embedder_weight()
