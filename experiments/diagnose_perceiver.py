#!/usr/bin/env python3
"""
诊断 Perceiver Resampler
"""
import sys
sys.path.insert(0, '/Users/bailin/index-tts')

import torch
import mlx.core as mx
import numpy as np
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def compute_correlation(a, b):
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    a_flat = a.flatten()
    b_flat = b.flatten()
    return np.corrcoef(a_flat, b_flat)[0, 1]

def main():
    print("=" * 80)
    print("🔍 诊断 Perceiver Resampler")
    print("=" * 80)
    
    # Load models
    from indextts.infer_v2 import IndexTTS2
    
    print("\n加载模型...")
    pt_model = IndexTTS2(device="mps", use_mlx=False)
    mlx_model = IndexTTS2(device="mps", use_mlx=True)
    
    # Create test input (Conformer output simulation)
    batch, seq_len, dim = 1, 60, 512  # After Conv2d subsampling
    test_conformer_out = torch.randn(batch, seq_len, dim).to("mps")
    
    print(f"\n测试输入 (Conformer output): {test_conformer_out.shape}")
    
    # ========================================================================
    # Test 1: proj_context (512 → 1280)
    # ========================================================================
    print("\n" + "=" * 80)
    print("Test 1: proj_context (512 → 1280)")
    print("=" * 80)
    
    pt_perceiver = pt_model.gpt.perceiver_encoder
    mlx_perceiver = mlx_model.mlx_transformer.conditioning_module.perceiver
    
    # PyTorch
    with torch.no_grad():
        pt_proj_out = pt_perceiver.proj_context(test_conformer_out)
    
    print(f"\n[PyTorch] proj_context:")
    print(f"  Output: {pt_proj_out.shape}")
    print(f"  Mean: {pt_proj_out.mean():.6f}, Std: {pt_proj_out.std():.6f}")
    
    # MLX
    mlx_conformer_out = torch_to_mlx(test_conformer_out.cpu())
    mlx_proj_out = mlx_perceiver.proj_context(mlx_conformer_out)
    mlx_proj_out_torch = mlx_to_torch(mlx_proj_out, device="cpu")
    
    print(f"\n[MLX] proj_context:")
    print(f"  Output: {mlx_proj_out_torch.shape}")
    print(f"  Mean: {mlx_proj_out_torch.mean():.6f}, Std: {mlx_proj_out_torch.std():.6f}")
    
    corr_proj = compute_correlation(pt_proj_out.cpu(), mlx_proj_out_torch)
    print(f"\n  Correlation: {corr_proj:.6f}")
    
    if corr_proj < 0.999:
        print("  ⚠️  proj_context 有差异")
    else:
        print("  ✅ proj_context 正确")
    
    # ========================================================================
    # Test 2: Latents
    # ========================================================================
    print("\n" + "=" * 80)
    print("Test 2: Latents")
    print("=" * 80)
    
    pt_latents = pt_perceiver.latents
    mlx_latents = mlx_perceiver.latents
    
    print(f"\n[PyTorch] latents:")
    print(f"  Shape: {pt_latents.shape}")
    print(f"  Mean: {pt_latents.mean():.6f}, Std: {pt_latents.std():.6f}")
    
    mlx_latents_np = np.array(mlx_latents)
    print(f"\n[MLX] latents:")
    print(f"  Shape: {mlx_latents_np.shape}")
    print(f"  Mean: {mlx_latents_np.mean():.6f}, Std: {mlx_latents_np.std():.6f}")
    
    corr_latents = np.corrcoef(
        pt_latents.detach().cpu().numpy().flatten(),
        mlx_latents_np.flatten()
    )[0, 1]
    print(f"\n  Correlation: {corr_latents:.6f}")
    
    if corr_latents < 0.999:
        print("  ⚠️  latents 有差异")
    else:
        print("  ✅ latents 正确")
    
    # ========================================================================
    # Test 3: First Perceiver Layer Attention
    # ========================================================================
    print("\n" + "=" * 80)
    print("Test 3: First Perceiver Layer Attention")
    print("=" * 80)
    
    # Get first layer
    pt_layer = pt_perceiver.layers[0][0]  # (attention, ff)
    mlx_layer = mlx_perceiver.layers[0]
    mlx_attn, mlx_ff = mlx_layer  # Unpack tuple
    
    # Prepare inputs
    # PyTorch: latents as query, proj_context as context
    pt_latents_expanded = pt_latents.unsqueeze(0).expand(batch, -1, -1)
    
    with torch.no_grad():
        pt_attn_out = pt_layer(pt_latents_expanded, pt_proj_out)
    
    print(f"\n[PyTorch] Attention:")
    print(f"  Output: {pt_attn_out.shape}")
    print(f"  Mean: {pt_attn_out.mean():.6f}, Std: {pt_attn_out.std():.6f}")
    
    # MLX
    mlx_latents_expanded = mx.broadcast_to(
        mlx_latents.reshape(1, mlx_latents.shape[0], mlx_latents.shape[1]),
        (batch, mlx_latents.shape[0], mlx_latents.shape[1])
    )
    
    mlx_attn_out = mlx_attn(mlx_latents_expanded, mlx_proj_out)
    mlx_attn_out_torch = mlx_to_torch(mlx_attn_out, device="cpu")
    
    print(f"\n[MLX] Attention:")
    print(f"  Output: {mlx_attn_out_torch.shape}")
    print(f"  Mean: {mlx_attn_out_torch.mean():.6f}, Std: {mlx_attn_out_torch.std():.6f}")
    
    corr_attn = compute_correlation(pt_attn_out.cpu(), mlx_attn_out_torch)
    print(f"\n  Correlation: {corr_attn:.6f}")
    
    if corr_attn < 0.999:
        print("  ⚠️  Perceiver Attention 有差异")
        print("  → 需要详细检查 to_q, to_kv, to_out")
    else:
        print("  ✅ Perceiver Attention 正确")
    
    # ========================================================================
    # Test 4: Full Perceiver Forward
    # ========================================================================
    print("\n" + "=" * 80)
    print("Test 4: Full Perceiver Forward")
    print("=" * 80)
    
    with torch.no_grad():
        pt_full_out = pt_perceiver(test_conformer_out)
    
    print(f"\n[PyTorch] Full Perceiver:")
    print(f"  Output: {pt_full_out.shape}")
    print(f"  Mean: {pt_full_out.mean():.6f}, Std: {pt_full_out.std():.6f}")
    
    # MLX - need to call the full module
    mlx_full_out = mlx_model.mlx_transformer.conditioning_module.perceiver(mlx_conformer_out)
    mlx_full_out_torch = mlx_to_torch(mlx_full_out, device="cpu")
    
    print(f"\n[MLX] Full Perceiver:")
    print(f"  Output: {mlx_full_out_torch.shape}")
    print(f"  Mean: {mlx_full_out_torch.mean():.6f}, Std: {mlx_full_out_torch.std():.6f}")
    
    corr_full = compute_correlation(pt_full_out.cpu(), mlx_full_out_torch)
    print(f"\n  Correlation: {corr_full:.6f}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 总结")
    print("=" * 80)
    
    results = {
        'proj_context': corr_proj,
        'latents': corr_latents,
        'first_attn': corr_attn,
        'full_perceiver': corr_full
    }
    
    for name, corr in results.items():
        status = "✅" if corr > 0.999 else "⚠️" if corr > 0.99 else "❌"
        print(f"{status} {name:20s}: {corr:.6f}")
    
    min_corr = min(results.values())
    print("\n" + "=" * 80)
    if min_corr > 0.999:
        print("✅ Perceiver 几乎完美！")
    elif min_corr > 0.99:
        print("⚠️  Perceiver 有小差异")
        bottleneck = min(results.items(), key=lambda x: x[1])
        print(f"   瓶颈: {bottleneck[0]} (correlation {bottleneck[1]:.6f})")
    else:
        print("❌ Perceiver 需要优化")
        bottleneck = min(results.items(), key=lambda x: x[1])
        print(f"   主要问题: {bottleneck[0]} (correlation {bottleneck[1]:.6f})")
    print("=" * 80)

if __name__ == "__main__":
    main()


