#!/usr/bin/env python3
"""
检查各组件的数值范围
"""
import sys
sys.path.insert(0, '/Users/bailin/index-tts')

import torch
import mlx.core as mx
import numpy as np
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def compute_stats(tensor, name):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    elif not isinstance(tensor, np.ndarray):
        tensor = np.array(tensor)
    
    print(f"  {name}:")
    print(f"    Mean: {tensor.mean():.6f}")
    print(f"    Std: {tensor.std():.6f}")
    print(f"    Min: {tensor.min():.6f}")
    print(f"    Max: {tensor.max():.6f}")
    print(f"    RMS: {np.sqrt((tensor**2).mean()):.6f}")

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
    print("🔬 检查各组件的数值范围")
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
    
    # ========================================================================
    # Get Layer 1 output
    # ========================================================================
    pt_conformer = pt_model.gpt.conditioning_encoder
    mlx_conformer = mlx_model.mlx_transformer.conditioning_module.conformer
    
    layer1_outputs = {}
    captured_pos_emb = {}
    
    def capture_layer1(module, input, output):
        if isinstance(output, tuple):
            layer1_outputs["pt"] = output[0].detach()
        else:
            layer1_outputs["pt"] = output.detach()
    
    def capture_embed(module, input, output):
        if isinstance(output, tuple):
            captured_pos_emb["pt"] = output[1].detach()
    
    pt_conformer.encoders[1].register_forward_hook(capture_layer1)
    pt_conformer.embed.register_forward_hook(capture_embed)
    
    with torch.no_grad():
        _ = pt_conformer(pt_input, cond_lengths)
    
    # MLX Layer 1 output
    mlx_x = mlx_conformer.subsampling(mlx_input)
    seq_len = mlx_x.shape[1]
    mlx_pos_emb = mlx_conformer.pos_encoding[:seq_len]
    mlx_pos_emb = mx.broadcast_to(
        mlx_pos_emb.reshape(1, seq_len, mlx_conformer.output_dim),
        (mlx_x.shape[0], seq_len, mlx_conformer.output_dim)
    )
    mlx_x = mlx_x * mlx_conformer.xscale + mlx_pos_emb
    
    for i in range(2):
        mlx_x = mlx_conformer.blocks[i](mlx_x, mlx_pos_emb, None, None)
    
    layer1_outputs["mlx"] = mlx_x
    
    # ========================================================================
    # Test Layer 2 components with numerical analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("Layer 2 - Numerical Range Analysis")
    print("=" * 80)
    
    pt_block = pt_conformer.encoders[2]
    mlx_block = mlx_conformer.blocks[2]
    
    pt_x = layer1_outputs["pt"]
    mlx_x = layer1_outputs["mlx"]
    pt_pos_emb = captured_pos_emb["pt"]
    
    print("\n📊 输入 (Layer 1 output):")
    compute_stats(pt_x, "PyTorch")
    compute_stats(mlx_to_torch(mlx_x, "cpu"), "MLX")
    
    # ========================================================================
    # Attention
    # ========================================================================
    print("\n" + "=" * 80)
    print("1️⃣ Attention")
    print("=" * 80)
    
    with torch.no_grad():
        pt_attn_in = pt_block.norm_mha(pt_x)
        pt_attn_out, _ = pt_block.self_attn(
            pt_attn_in, pt_attn_in, pt_attn_in, pos_emb=pt_pos_emb
        )
    
    mlx_attn_in = mlx_block.norm_attn(mlx_x)
    mlx_attn_out = mlx_block.attn(mlx_attn_in, mlx_pos_emb, None)
    
    print("\nAttention Input (after norm):")
    compute_stats(pt_attn_in, "PyTorch")
    compute_stats(mlx_to_torch(mlx_attn_in, "cpu"), "MLX")
    
    print("\nAttention Output:")
    compute_stats(pt_attn_out, "PyTorch")
    compute_stats(mlx_to_torch(mlx_attn_out, "cpu"), "MLX")
    
    corr_attn = compute_correlation(pt_attn_out, mlx_to_torch(mlx_attn_out, "cpu"))
    print(f"\n  Correlation: {corr_attn:.6f}")
    
    # After residual
    pt_after_attn = pt_x + pt_attn_out
    mlx_after_attn = mlx_x + mlx_attn_out
    
    print("\nAfter Residual:")
    compute_stats(pt_after_attn, "PyTorch")
    compute_stats(mlx_to_torch(mlx_after_attn, "cpu"), "MLX")
    
    corr_after_attn = compute_correlation(pt_after_attn, mlx_to_torch(mlx_after_attn, "cpu"))
    print(f"  Correlation: {corr_after_attn:.6f}")
    
    # ========================================================================
    # Convolution
    # ========================================================================
    print("\n" + "=" * 80)
    print("2️⃣ Convolution")
    print("=" * 80)
    
    with torch.no_grad():
        pt_conv_in = pt_block.norm_conv(pt_after_attn)
        pt_conv_out = pt_block.conv_module(pt_conv_in)
        if isinstance(pt_conv_out, tuple):
            pt_conv_out = pt_conv_out[0]
    
    mlx_conv_in = mlx_block.norm_conv(mlx_after_attn)
    mlx_conv_out = mlx_block.conv(mlx_conv_in, None)
    
    print("\nConvolution Input (after norm):")
    compute_stats(pt_conv_in, "PyTorch")
    compute_stats(mlx_to_torch(mlx_conv_in, "cpu"), "MLX")
    
    print("\nConvolution Output:")
    compute_stats(pt_conv_out, "PyTorch")
    compute_stats(mlx_to_torch(mlx_conv_out, "cpu"), "MLX")
    
    corr_conv = compute_correlation(pt_conv_out, mlx_to_torch(mlx_conv_out, "cpu"))
    print(f"\n  Correlation: {corr_conv:.6f}")
    
    # After residual
    pt_after_conv = pt_after_attn + pt_conv_out
    mlx_after_conv = mlx_after_attn + mlx_conv_out
    
    print("\nAfter Residual:")
    compute_stats(pt_after_conv, "PyTorch")
    compute_stats(mlx_to_torch(mlx_after_conv, "cpu"), "MLX")
    
    corr_after_conv = compute_correlation(pt_after_conv, mlx_to_torch(mlx_after_conv, "cpu"))
    print(f"  Correlation: {corr_after_conv:.6f}")
    
    # ========================================================================
    # Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 数值范围分析")
    print("=" * 80)
    
    pt_x_rms = np.sqrt((pt_x.cpu().numpy()**2).mean())
    pt_attn_rms = np.sqrt((pt_attn_out.cpu().numpy()**2).mean())
    pt_conv_rms = np.sqrt((pt_conv_out.cpu().numpy()**2).mean())
    
    print(f"\nRMS 值:")
    print(f"  Residual (Layer 1 output): {pt_x_rms:.6f}")
    print(f"  Attention output: {pt_attn_rms:.6f}")
    print(f"  Convolution output: {pt_conv_rms:.6f}")
    
    print(f"\n相对比例:")
    print(f"  Attention / Residual: {pt_attn_rms / pt_x_rms:.4f}")
    print(f"  Convolution / Residual: {pt_conv_rms / pt_x_rms:.4f}")
    
    print("\n结论:")
    if pt_attn_rms < pt_x_rms * 0.1:
        print("  ⚠️  Attention 输出远小于 residual (< 10%)")
        print("     微小的绝对误差会导致 correlation 下降")
    if pt_conv_rms < pt_x_rms * 0.1:
        print("  ⚠️  Convolution 输出远小于 residual (< 10%)")
        print("     微小的绝对误差会导致 correlation 下降")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

