#!/usr/bin/env python3
"""
详细检查 Layer 2 的各个组件
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
    print("🔬 详细检查 Layer 2 的各个组件")
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
    # Get input to Layer 2
    # ========================================================================
    print("\n" + "=" * 80)
    print("获取 Layer 2 的输入")
    print("=" * 80)
    
    pt_conformer = pt_model.gpt.conditioning_encoder
    mlx_conformer = mlx_model.mlx_transformer.conditioning_module.conformer
    
    # Capture Layer 1 output and pos_emb as input to Layer 2
    layer1_outputs = {}
    captured_pos_emb = {}
    
    def capture_layer1(module, input, output):
        if isinstance(output, tuple):
            layer1_outputs["pt"] = output[0].detach()
        else:
            layer1_outputs["pt"] = output.detach()
    
    def capture_embed(module, input, output):
        if isinstance(output, tuple):
            captured_pos_emb["pt"] = output[1].detach()  # pos_emb is second output
    
    pt_conformer.encoders[1].register_forward_hook(capture_layer1)
    pt_conformer.embed.register_forward_hook(capture_embed)
    
    with torch.no_grad():
        _ = pt_conformer(pt_input, cond_lengths)
    
    # Get MLX Layer 1 output
    mlx_x = mlx_conformer.subsampling(mlx_input)
    seq_len = mlx_x.shape[1]
    mlx_pos_emb = mlx_conformer.pos_encoding[:seq_len]
    mlx_pos_emb = mx.broadcast_to(
        mlx_pos_emb.reshape(1, seq_len, mlx_conformer.output_dim),
        (mlx_x.shape[0], seq_len, mlx_conformer.output_dim)
    )
    mlx_x = mlx_x * mlx_conformer.xscale + mlx_pos_emb
    
    for i in range(2):  # Run through Layer 0 and 1
        mlx_x = mlx_conformer.blocks[i](mlx_x, mlx_pos_emb, None, None)
    
    layer1_outputs["mlx"] = mlx_x
    
    print(f"\nLayer 1 输出:")
    print(f"  PyTorch: {layer1_outputs['pt'].shape}")
    print(f"  MLX: {layer1_outputs['mlx'].shape}")
    
    corr_layer1 = compute_correlation(
        layer1_outputs["pt"].cpu(),
        mlx_to_torch(layer1_outputs["mlx"], "cpu")
    )
    print(f"  Correlation: {corr_layer1:.6f}")
    
    # ========================================================================
    # Test Layer 2 components
    # ========================================================================
    print("\n" + "=" * 80)
    print("Layer 2 组件测试")
    print("=" * 80)
    
    pt_block = pt_conformer.encoders[2]
    mlx_block = mlx_conformer.blocks[2]
    
    pt_x = layer1_outputs["pt"]
    mlx_x = layer1_outputs["mlx"]
    
    # Use captured pos_emb for PyTorch
    pt_pos_emb = captured_pos_emb["pt"]
    
    # ========================================================================
    # Test 1: Attention
    # ========================================================================
    print("\n1️⃣ Attention Module")
    
    with torch.no_grad():
        # PyTorch attention
        pt_attn_in = pt_block.norm_mha(pt_x)
        pt_attn_out, _ = pt_block.self_attn(
            pt_attn_in, pt_attn_in, pt_attn_in, pos_emb=pt_pos_emb
        )
    
    # MLX attention
    mlx_attn_in = mlx_block.norm_attn(mlx_x)
    mlx_attn_out = mlx_block.attn(mlx_attn_in, mlx_pos_emb, None)
    
    corr_attn = compute_correlation(pt_attn_out.cpu(), mlx_to_torch(mlx_attn_out, "cpu"))
    print(f"  Attention output: {corr_attn:.6f}")
    
    # After residual
    pt_after_attn = pt_x + pt_attn_out
    mlx_after_attn = mlx_x + mlx_attn_out
    corr_after_attn = compute_correlation(
        pt_after_attn.cpu(), 
        mlx_to_torch(mlx_after_attn, "cpu")
    )
    print(f"  After Attention + residual: {corr_after_attn:.6f}")
    
    # ========================================================================
    # Test 2: Convolution
    # ========================================================================
    print("\n2️⃣ Convolution Module")
    
    with torch.no_grad():
        pt_conv_in = pt_block.norm_conv(pt_after_attn)
        pt_conv_out = pt_block.conv_module(pt_conv_in)
        if isinstance(pt_conv_out, tuple):
            pt_conv_out = pt_conv_out[0]
    
    mlx_conv_in = mlx_block.norm_conv(mlx_after_attn)
    mlx_conv_out = mlx_block.conv(mlx_conv_in, None)
    
    corr_conv = compute_correlation(pt_conv_out.cpu(), mlx_to_torch(mlx_conv_out, "cpu"))
    print(f"  Convolution output: {corr_conv:.6f}")
    
    # After residual
    pt_after_conv = pt_after_attn + pt_conv_out
    mlx_after_conv = mlx_after_attn + mlx_conv_out
    corr_after_conv = compute_correlation(
        pt_after_conv.cpu(),
        mlx_to_torch(mlx_after_conv, "cpu")
    )
    print(f"  After Convolution + residual: {corr_after_conv:.6f}")
    
    # ========================================================================
    # Test 3: Feed-Forward
    # ========================================================================
    print("\n3️⃣ Feed-Forward Module")
    
    with torch.no_grad():
        pt_ff_in = pt_block.norm_ff(pt_after_conv)
        pt_ff_out = pt_block.feed_forward(pt_ff_in)
    
    mlx_ff_in = mlx_block.norm_ff(mlx_after_conv)
    mlx_ff_out = mlx_block.ff(mlx_ff_in)
    
    corr_ff = compute_correlation(pt_ff_out.cpu(), mlx_to_torch(mlx_ff_out, "cpu"))
    print(f"  Feed-Forward output: {corr_ff:.6f}")
    
    # After residual
    pt_after_ff = pt_after_conv + pt_ff_out
    mlx_after_ff = mlx_after_conv + mlx_ff_out
    corr_after_ff = compute_correlation(
        pt_after_ff.cpu(),
        mlx_to_torch(mlx_after_ff, "cpu")
    )
    print(f"  After Feed-Forward + residual: {corr_after_ff:.6f}")
    
    # ========================================================================
    # Test 4: Final Norm
    # ========================================================================
    print("\n4️⃣ Final Norm")
    
    with torch.no_grad():
        pt_final = pt_block.norm_final(pt_after_ff)
    
    mlx_final = mlx_block.norm_final(mlx_after_ff)
    
    corr_final = compute_correlation(pt_final.cpu(), mlx_to_torch(mlx_final, "cpu"))
    print(f"  Final Norm output: {corr_final:.6f}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 Layer 2 组件 Correlation 总结")
    print("=" * 80)
    
    print(f"\n输入 (Layer 1 output): {corr_layer1:.6f}")
    print(f"1. Attention output: {corr_attn:.6f}")
    print(f"   After residual: {corr_after_attn:.6f}")
    print(f"2. Convolution output: {corr_conv:.6f}")
    print(f"   After residual: {corr_after_conv:.6f}")
    print(f"3. Feed-Forward output: {corr_ff:.6f}")
    print(f"   After residual: {corr_after_ff:.6f}")
    print(f"4. Final Norm: {corr_final:.6f}")
    
    # Find problem component
    print("\n问题诊断:")
    
    if corr_attn < 0.999:
        print(f"  ❌ Attention 有问题 (corr={corr_attn:.6f})")
    elif corr_after_attn < 0.999:
        print(f"  ❌ Attention residual 有问题 (corr={corr_after_attn:.6f})")
    elif corr_conv < 0.999:
        print(f"  ❌ Convolution 有问题 (corr={corr_conv:.6f})")
    elif corr_after_conv < 0.999:
        print(f"  ❌ Convolution residual 有问题 (corr={corr_after_conv:.6f})")
    elif corr_ff < 0.999:
        print(f"  ❌ Feed-Forward 有问题 (corr={corr_ff:.6f})")
    elif corr_after_ff < 0.999:
        print(f"  ❌ Feed-Forward residual 有问题 (corr={corr_after_ff:.6f})")
    elif corr_final < 0.999:
        print(f"  ❌ Final Norm 有问题 (corr={corr_final:.6f})")
    else:
        print("  ✅ 所有组件的 correlation 都很高！")
        print("     可能是浮点精度的微小累积")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

