#!/usr/bin/env python3
"""
使用 Hook 追踪 Conformer 中间输出
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
    print("🔬 使用 Hook 追踪 Conformer 中间输出")
    print("=" * 80)
    
    # Load models
    from indextts.infer_v2 import IndexTTS2
    
    print("\n加载模型...")
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
    
    print(f"\n测试输入: {pt_input.shape}")
    
    # ========================================================================
    # Set up hooks for PyTorch
    # ========================================================================
    pt_outputs = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                pt_outputs[name] = output[0].detach().cpu()
            else:
                pt_outputs[name] = output.detach().cpu()
        return hook
    
    pt_conformer = pt_model.gpt.conditioning_encoder
    
    # Register hooks for each layer
    for i in range(6):
        pt_conformer.encoders[i].register_forward_hook(make_hook(f"layer{i}"))
    pt_conformer.after_norm.register_forward_hook(make_hook("after_norm"))
    
    # ========================================================================
    # Run PyTorch forward
    # ========================================================================
    print("\n运行 PyTorch forward...")
    with torch.no_grad():
        pt_out, _ = pt_conformer(pt_input, cond_lengths)
    
    print(f"PyTorch 捕获了 {len(pt_outputs)} 个中间输出")
    
    # ========================================================================
    # Run MLX forward and capture outputs manually
    # ========================================================================
    print("\n运行 MLX forward...")
    mlx_conformer = mlx_model.mlx_transformer.conditioning_module.conformer
    mlx_outputs = {}
    
    # Manually run through MLX conformer
    mlx_x, mlx_mask = mlx_conformer(mlx_input, mlx_lengths)
    
    # Now re-run step-by-step to capture intermediate outputs
    mlx_x = mlx_conformer.subsampling(mlx_input)
    seq_len = mlx_x.shape[1]
    mlx_pos_emb = mlx_conformer.pos_encoding[:seq_len]
    mlx_pos_emb = mx.broadcast_to(
        mlx_pos_emb.reshape(1, seq_len, mlx_conformer.output_dim),
        (mlx_x.shape[0], seq_len, mlx_conformer.output_dim)
    )
    mlx_x = mlx_x * mlx_conformer.xscale + mlx_pos_emb
    
    # Run through each block
    for i in range(6):
        mlx_x = mlx_conformer.blocks[i](mlx_x, mlx_pos_emb, None, None)
        mlx_outputs[f"layer{i}"] = mlx_to_torch(mlx_x, "cpu")
    
    mlx_x = mlx_conformer.after_norm(mlx_x)
    mlx_outputs["after_norm"] = mlx_to_torch(mlx_x, "cpu")
    mlx_outputs["final"] = mlx_to_torch(mlx_x, "cpu")
    
    print(f"MLX 捕获了 {len(mlx_outputs)} 个中间输出")
    
    # ========================================================================
    # Compare layer-by-layer
    # ========================================================================
    print("\n" + "=" * 80)
    print("Layer-by-Layer Correlation")
    print("=" * 80)
    
    correlations = []
    
    for i in range(6):
        key = f"layer{i}"
        corr = compute_correlation(pt_outputs[key], mlx_outputs[key])
        correlations.append(corr)
        
        status = "✅" if corr > 0.999 else "⚠️" if corr > 0.99 else "❌"
        print(f"{status} Layer {i}: {corr:.6f}", end="")
        
        if i > 0:
            drop = correlations[i-1] - corr
            if abs(drop) > 0.0001:
                print(f"  (drop: {drop:.6f})", end="")
        print()
    
    # After norm
    corr_after_norm = compute_correlation(pt_outputs["after_norm"], mlx_outputs["after_norm"])
    print(f"After Final Norm: {corr_after_norm:.6f}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 诊断总结")
    print("=" * 80)
    
    for i, corr in enumerate(correlations):
        print(f"Layer {i}: {corr:.6f}")
    print(f"After Final Norm: {corr_after_norm:.6f}")
    
    # Find problem point
    print("\n问题定位:")
    
    if correlations[0] < 0.999:
        print(f"  ❌ Layer 0 already has error (corr={correlations[0]:.6f})")
        print("     可能是 Subsampling/Pos Encoding 的问题")
    else:
        # Find first layer with drop
        for i in range(1, 6):
            if correlations[i] < 0.999 and correlations[i-1] >= 0.999:
                print(f"  ❌ Layer {i} 是第一个引入显著误差的层 (corr={correlations[i]:.6f})")
                print(f"     从 Layer {i-1} ({correlations[i-1]:.6f}) 下降到 Layer {i} ({correlations[i]:.6f})")
                break
        else:
            print("  ✅ 所有层的 correlation 都很高！")
            print("     误差可能是 6 层的微小浮点精度差异累积")
    
    # Calculate total degradation
    total_drop = 1.0 - corr_after_norm
    print(f"\n总 correlation 损失: {total_drop:.6f} ({total_drop*100:.4f}%)")
    
    if total_drop < 0.001:
        print("  ✅ 损失极小，quality 应该很好")
    elif total_drop < 0.01:
        print("  ⚠️  有一定损失，可能影响 quality")
    else:
        print("  ❌ 损失较大，需要继续优化")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

