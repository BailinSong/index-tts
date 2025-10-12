#!/usr/bin/env python3
"""
在实际 Conformer 中测试混合精度策略
"""
import sys
sys.path.insert(0, '/Users/bailin/index-tts')

import torch
import mlx.core as mx
import mlx.nn as nn
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

def test_full_conformer(use_mixed_precision=False):
    """测试完整 Conformer 的混合精度"""
    print(f"\n{'='*80}")
    print(f"测试 Conformer - {'混合精度' if use_mixed_precision else '标准精度'}")
    print(f"{'='*80}")
    
    from indextts.infer_v2 import IndexTTS2
    
    # Load models
    pt_model = IndexTTS2(device="mps", use_mlx=False)
    mlx_model = IndexTTS2(device="mps", use_mlx=True)
    
    # Create test input
    torch.manual_seed(42)
    np.random.seed(42)
    
    batch, time_steps, feat_dim = 1, 121, 1024
    pt_input = torch.randn(batch, time_steps, feat_dim, device=pt_model.device)
    mlx_input = torch_to_mlx(pt_input.cpu())
    cond_lengths = torch.tensor([time_steps], device=pt_model.device)
    mlx_lengths = mx.array([time_steps])
    
    # PyTorch forward
    pt_conformer = pt_model.gpt.conditioning_encoder
    with torch.no_grad():
        pt_out, _ = pt_conformer(pt_input, cond_lengths)
    
    # MLX forward (with optional mixed precision)
    mlx_conformer = mlx_model.mlx_transformer.conditioning_module.conformer
    
    if use_mixed_precision:
        # Apply mixed precision: convert to float32 for key operations
        print("\n应用混合精度策略...")
        
        # Modify attention to use float32
        for block in mlx_conformer.blocks:
            # Wrap attention forward to use float32
            original_attn_call = block.attn.__call__
            
            def make_fp32_attn(original_fn):
                def fp32_attn_call(x, pos_emb, mask):
                    # Convert to float32
                    x_fp32 = x.astype(mx.float32)
                    pos_emb_fp32 = pos_emb.astype(mx.float32)
                    
                    # Run attention in float32
                    out_fp32 = original_fn(x_fp32, pos_emb_fp32, mask)
                    
                    # Convert back
                    return out_fp32.astype(x.dtype)
                return fp32_attn_call
            
            block.attn.__call__ = make_fp32_attn(original_attn_call)
    
    mlx_out, _ = mlx_conformer(mlx_input, mlx_lengths)
    
    # Compare
    corr = compute_correlation(pt_out.cpu(), mlx_to_torch(mlx_out, "cpu"))
    
    print(f"\n📊 Full Conformer Correlation: {corr:.6f}")
    
    # Layer-by-layer analysis
    print("\nLayer-by-layer:")
    
    # Get PyTorch layer outputs
    pt_outputs = {}
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                pt_outputs[name] = output[0].detach().cpu()
            else:
                pt_outputs[name] = output.detach().cpu()
        return hook
    
    for i in range(6):
        pt_conformer.encoders[i].register_forward_hook(make_hook(f"layer{i}"))
    
    with torch.no_grad():
        _ = pt_conformer(pt_input, cond_lengths)
    
    # Get MLX layer outputs
    mlx_x = mlx_conformer.subsampling(mlx_input)
    seq_len = mlx_x.shape[1]
    mlx_pos_emb = mlx_conformer.pos_encoding[:seq_len]
    mlx_pos_emb = mx.broadcast_to(
        mlx_pos_emb.reshape(1, seq_len, mlx_conformer.output_dim),
        (mlx_x.shape[0], seq_len, mlx_conformer.output_dim)
    )
    mlx_x = mlx_x * mlx_conformer.xscale + mlx_pos_emb
    
    correlations = []
    for i in range(6):
        mlx_x = mlx_conformer.blocks[i](mlx_x, mlx_pos_emb, None, None)
        corr_layer = compute_correlation(pt_outputs[f"layer{i}"], mlx_to_torch(mlx_x, "cpu"))
        correlations.append(corr_layer)
        
        status = "✅" if corr_layer > 0.999 else "⚠️" if corr_layer > 0.99 else "❌"
        print(f"  {status} Layer {i}: {corr_layer:.6f}")
    
    return corr, correlations

def main():
    print("="*80)
    print("🔬 Conformer 混合精度策略实战测试")
    print("="*80)
    
    # Test 1: Baseline
    print("\n" + "="*80)
    print("Test 1: 标准精度 (Baseline)")
    print("="*80)
    corr_baseline, corrs_baseline = test_full_conformer(use_mixed_precision=False)
    
    # Test 2: Mixed precision
    print("\n" + "="*80)
    print("Test 2: 混合精度 (Float32 for Attention)")
    print("="*80)
    corr_mixed, corrs_mixed = test_full_conformer(use_mixed_precision=True)
    
    # Summary
    print("\n" + "="*80)
    print("📊 对比总结")
    print("="*80)
    
    print(f"\nFull Conformer:")
    print(f"  标准精度:   {corr_baseline:.6f}")
    print(f"  混合精度:   {corr_mixed:.6f}")
    
    improvement = corr_mixed - corr_baseline
    print(f"  提升:       {improvement:.6f} ({improvement*100:.4f}%)")
    
    if improvement > 0.001:
        print("  ✅ 显著提升！")
    elif improvement > 0.0001:
        print("  ⚠️  有小幅提升")
    else:
        print("  ❌ 提升不明显")
    
    print(f"\nLayer-by-layer 对比:")
    print(f"  {'Layer':<10} {'标准精度':<12} {'混合精度':<12} {'提升':<10}")
    print(f"  {'-'*50}")
    
    for i in range(6):
        diff = corrs_mixed[i] - corrs_baseline[i]
        status = "✅" if diff > 0.001 else "⚠️" if diff > 0.0001 else ""
        print(f"  {f'Layer {i}':<10} {corrs_baseline[i]:.6f}     {corrs_mixed[i]:.6f}     {diff:+.6f}  {status}")
    
    print("\n结论:")
    if improvement > 0.005:
        print("  ✅ 混合精度策略有效！建议实施。")
        print("     - 显著提升 correlation")
        print("     - 可能解决丢字问题")
    elif improvement > 0.001:
        print("  ⚠️  混合精度有一定效果，可以尝试。")
        print("     - correlation 有提升")
        print("     - 需要权衡性能开销")
    else:
        print("  ❌ 混合精度提升有限。")
        print("     - 问题可能不是精度")
        print("     - 建议接受当前 correlation 0.9867")
        print("     - 或考虑其他优化方向")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()


