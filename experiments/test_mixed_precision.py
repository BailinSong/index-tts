#!/usr/bin/env python3
"""
测试混合精度策略对 Conformer correlation 的影响
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
    if isinstance(a, mx.array):
        a = np.array(a)
    if isinstance(b, mx.array):
        b = np.array(b)
    a_flat = a.flatten()
    b_flat = b.flatten()
    return np.corrcoef(a_flat, b_flat)[0, 1]

def test_attention_precision(use_float32=False):
    """测试 Attention 的精度策略"""
    print(f"\n{'='*80}")
    print(f"Testing Attention with {'float32' if use_float32 else 'float16'}")
    print(f"{'='*80}")
    
    # Create test data
    batch, seq_len, dim = 1, 60, 512
    num_heads = 8
    head_dim = dim // num_heads
    
    np.random.seed(42)
    q = np.random.randn(batch, num_heads, seq_len, head_dim).astype(np.float32)
    k = np.random.randn(batch, num_heads, seq_len, head_dim).astype(np.float32)
    v = np.random.randn(batch, num_heads, seq_len, head_dim).astype(np.float32)
    
    # PyTorch (baseline)
    pt_q = torch.from_numpy(q)
    pt_k = torch.from_numpy(k)
    pt_v = torch.from_numpy(v)
    
    # Attention scores
    pt_scores = torch.matmul(pt_q, pt_k.transpose(-2, -1)) / np.sqrt(head_dim)
    pt_attn = torch.softmax(pt_scores, dim=-1)
    pt_output = torch.matmul(pt_attn, pt_v)
    
    # MLX (standard float16/bfloat16)
    mlx_q = mx.array(q)
    mlx_k = mx.array(k)
    mlx_v = mx.array(v)
    
    if use_float32:
        # Mixed precision: key operations in float32
        mlx_q_hp = mlx_q.astype(mx.float32)
        mlx_k_hp = mlx_k.astype(mx.float32)
        mlx_v_hp = mlx_v.astype(mx.float32)
        
        mlx_scores = (mlx_q_hp @ mlx_k_hp.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
        mlx_attn = mx.softmax(mlx_scores.astype(mx.float32), axis=-1)
        mlx_output = mlx_attn @ mlx_v_hp
        mlx_output = mlx_output.astype(mx.float32)  # Keep in float32
    else:
        # Standard precision
        mlx_scores = (mlx_q @ mlx_k.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
        mlx_attn = mx.softmax(mlx_scores, axis=-1)
        mlx_output = mlx_attn @ mlx_v
    
    # Compare
    corr_scores = compute_correlation(pt_scores.numpy(), mlx_scores)
    corr_attn = compute_correlation(pt_attn.numpy(), mlx_attn)
    corr_output = compute_correlation(pt_output.numpy(), mlx_output)
    
    print(f"\nCorrelations:")
    print(f"  Attention scores: {corr_scores:.8f}")
    print(f"  Attention weights: {corr_attn:.8f}")
    print(f"  Attention output: {corr_output:.8f}")
    
    # Numerical differences
    max_diff_scores = np.abs(np.array(pt_scores.numpy()) - np.array(mlx_scores)).max()
    max_diff_attn = np.abs(np.array(pt_attn.numpy()) - np.array(mlx_attn)).max()
    max_diff_output = np.abs(np.array(pt_output.numpy()) - np.array(mlx_output)).max()
    
    print(f"\nMax differences:")
    print(f"  Attention scores: {max_diff_scores:.8e}")
    print(f"  Attention weights: {max_diff_attn:.8e}")
    print(f"  Attention output: {max_diff_output:.8e}")
    
    return corr_output

def test_residual_scaling(scale_factor=1.0):
    """测试残差缩放策略"""
    print(f"\n{'='*80}")
    print(f"Testing Residual Scaling (factor={scale_factor})")
    print(f"{'='*80}")
    
    # Create test data
    batch, seq_len, dim = 1, 60, 512
    
    np.random.seed(42)
    residual = np.random.randn(batch, seq_len, dim).astype(np.float32) * 0.7  # RMS ~ 0.7
    module_output = np.random.randn(batch, seq_len, dim).astype(np.float32) * 0.15  # RMS ~ 0.15
    
    # PyTorch
    pt_residual = torch.from_numpy(residual)
    pt_module = torch.from_numpy(module_output)
    pt_result = pt_residual + pt_module
    
    # MLX with scaling
    mlx_residual = mx.array(residual)
    mlx_module = mx.array(module_output)
    
    # Apply scaling to module output before adding
    mlx_result = mlx_residual + mlx_module * scale_factor
    
    # If scale_factor != 1, we need to adjust PyTorch baseline
    if scale_factor != 1.0:
        pt_result = pt_residual + pt_module * scale_factor
    
    corr = compute_correlation(pt_result.numpy(), mlx_result)
    
    print(f"\nCorrelation: {corr:.8f}")
    print(f"Residual RMS: {np.sqrt((residual**2).mean()):.4f}")
    print(f"Module RMS: {np.sqrt((module_output**2).mean()):.4f}")
    print(f"Module RMS (scaled): {np.sqrt((module_output**2).mean()) * scale_factor:.4f}")
    
    return corr

def main():
    print("="*80)
    print("🔬 混合精度和残差缩放策略测试")
    print("="*80)
    
    # Test 1: Standard precision
    print("\n" + "="*80)
    print("Test 1: Baseline (Standard Precision)")
    print("="*80)
    corr_baseline = test_attention_precision(use_float32=False)
    
    # Test 2: Mixed precision (float32 for key ops)
    print("\n" + "="*80)
    print("Test 2: Mixed Precision (float32 for Attention)")
    print("="*80)
    corr_fp32 = test_attention_precision(use_float32=True)
    
    # Test 3: Residual scaling
    print("\n" + "="*80)
    print("Test 3: Residual Scaling")
    print("="*80)
    
    scales = [1.0, 2.0, 3.0, 5.0]
    for scale in scales:
        test_residual_scaling(scale_factor=scale)
    
    # Summary
    print("\n" + "="*80)
    print("📊 总结")
    print("="*80)
    
    print(f"\n混合精度效果:")
    print(f"  Baseline (standard): {corr_baseline:.8f}")
    print(f"  Float32 (mixed):     {corr_fp32:.8f}")
    
    improvement = corr_fp32 - corr_baseline
    if improvement > 1e-6:
        print(f"  ✅ 提升: {improvement:.8f} ({improvement*100:.6f}%)")
    else:
        print(f"  ⚠️  提升不明显: {improvement:.8f}")
    
    print(f"\n残差缩放:")
    print(f"  注意: 缩放因子 != 1.0 会改变模型行为")
    print(f"  需要确保 PyTorch 和 MLX 使用相同的缩放")
    
    print("\n推荐:")
    if improvement > 1e-4:
        print("  ✅ 建议使用混合精度策略（float32 for Attention）")
        print("     - 不改变模型行为")
        print("     - 显著提升数值精度")
    else:
        print("  ⚠️  混合精度提升有限")
        print("     - 可能需要其他优化策略")
        print("     - 或接受当前精度（correlation 0.9867）")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()


