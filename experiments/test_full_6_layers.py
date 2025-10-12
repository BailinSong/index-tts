#!/usr/bin/env python3
"""
测试完整 6 层 Conformer，看累积效应
"""
import sys
sys.path.insert(0, '/Users/bailin/index-tts')

import torch
import mlx.core as mx
import numpy as np
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch
import torchaudio

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
    print("🔍 测试完整 6 层 Conformer + Perceiver")
    print("=" * 80)
    
    # Load models
    from indextts.infer_v2 import IndexTTS2
    
    print("\n加载模型...")
    pt_model = IndexTTS2(device="mps", use_mlx=False)
    mlx_model = IndexTTS2(device="mps", use_mlx=True)
    
    # Load real audio
    print("\n加载真实音频...")
    ref_audio = "examples/voice_01.wav"
    speaker_emb, sr = torchaudio.load(ref_audio)
    speaker_emb = torchaudio.transforms.Resample(sr, 16000)(speaker_emb)
    
    # Pad to minimum length
    min_length = 16000 * 3
    if speaker_emb.shape[1] < min_length:
        pad_length = min_length - speaker_emb.shape[1]
        speaker_emb = torch.nn.functional.pad(speaker_emb, (0, pad_length))
    
    print(f"音频 shape: {speaker_emb.shape}")
    
    # ========================================================================
    # Test PyTorch full conditioning
    # ========================================================================
    print("\n" + "=" * 80)
    print("PyTorch 完整 Conditioning (Conformer + Perceiver)")
    print("=" * 80)
    
    pt_speaker_emb = speaker_emb.to(pt_model.device)
    pt_cond_lengths = torch.tensor([pt_speaker_emb.shape[1]], device=pt_model.device)
    
    with torch.no_grad():
        pt_conditioning = pt_model.gpt.get_conditioning(
            pt_speaker_emb.unsqueeze(0),
            pt_cond_lengths
        )
    
    print(f"\n✓ PyTorch Conditioning: {pt_conditioning.shape}")
    print(f"  Mean: {pt_conditioning.mean():.6f}")
    print(f"  Std: {pt_conditioning.std():.6f}")
    print(f"  Min: {pt_conditioning.min():.6f}")
    print(f"  Max: {pt_conditioning.max():.6f}")
    
    # ========================================================================
    # Test MLX full conditioning
    # ========================================================================
    print("\n" + "=" * 80)
    print("MLX 完整 Conditioning (Conformer + Perceiver)")
    print("=" * 80)
    
    mlx_speaker_emb = torch_to_mlx(speaker_emb)
    mlx_cond_lengths = mx.array([mlx_speaker_emb.shape[1]], dtype=mx.int32)
    
    mlx_conditioning_output = mlx_model.mlx_transformer.conditioning_module(
        mlx_speaker_emb.transpose(0, 2, 1),  # (batch, time, freq)
        mlx_cond_lengths
    )
    
    mlx_conditioning_torch = mlx_to_torch(mlx_conditioning_output, device="cpu")
    pt_conditioning_cpu = pt_conditioning.cpu()
    
    print(f"\n✓ MLX Conditioning: {mlx_conditioning_torch.shape}")
    print(f"  Mean: {mlx_conditioning_torch.mean():.6f}")
    print(f"  Std: {mlx_conditioning_torch.std():.6f}")
    print(f"  Min: {mlx_conditioning_torch.min():.6f}")
    print(f"  Max: {mlx_conditioning_torch.max():.6f}")
    
    # ========================================================================
    # Correlation
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 Correlation 分析")
    print("=" * 80)
    
    # Adjust shapes if needed
    if mlx_conditioning_torch.shape != pt_conditioning_cpu.shape:
        print(f"\n⚠️  Shape mismatch:")
        print(f"  PyTorch: {pt_conditioning_cpu.shape}")
        print(f"  MLX: {mlx_conditioning_torch.shape}")
        min_len = min(pt_conditioning_cpu.shape[1], mlx_conditioning_torch.shape[1])
        pt_conditioning_cpu = pt_conditioning_cpu[:, :min_len, :]
        mlx_conditioning_torch = mlx_conditioning_torch[:, :min_len, :]
        print(f"  Truncated to: {pt_conditioning_cpu.shape}")
    
    correlation = compute_correlation(pt_conditioning_cpu, mlx_conditioning_torch)
    max_diff = (pt_conditioning_cpu - mlx_conditioning_torch).abs().max().item()
    mean_diff = (pt_conditioning_cpu - mlx_conditioning_torch).abs().mean().item()
    
    print(f"\n📊 结果:")
    print(f"  Correlation: {correlation:.6f}")
    print(f"  Max Diff: {max_diff:.6f}")
    print(f"  Mean Diff: {mean_diff:.6f}")
    
    # ========================================================================
    # Layer-by-layer correlation (逐层追踪)
    # ========================================================================
    print("\n" + "=" * 80)
    print("🔬 逐层 Correlation 追踪")
    print("=" * 80)
    
    # Get Conformer encoders
    pt_conformer = pt_model.gpt.conditioning_encoder
    mlx_conformer = mlx_model.mlx_transformer.conditioning_module.conformer
    
    # Prepare input
    pt_x = pt_speaker_emb.unsqueeze(0).transpose(1, 2).to("mps")
    
    # PyTorch: through embedding
    with torch.no_grad():
        pt_emb, pt_pos_emb, pt_mask = pt_conformer.embed(pt_x, None)
    
    # MLX: through embedding
    mlx_x = mlx_speaker_emb.transpose(0, 2, 1)  # (batch, time, freq)
    mlx_emb = mlx_conformer.subsampling(mlx_x)
    mlx_seq_len = mlx_emb.shape[1]
    mlx_pos_emb = mlx_conformer.pos_encoding[:mlx_seq_len]
    mlx_pos_emb_broadcast = mx.broadcast_to(
        mlx_pos_emb.reshape(1, mlx_seq_len, mlx_conformer.output_dim),
        (mlx_emb.shape[0], mlx_seq_len, mlx_conformer.output_dim)
    )
    mlx_emb_with_pos = mlx_emb * mlx_conformer.xscale + mlx_pos_emb_broadcast
    
    # Match shapes
    min_len = min(pt_emb.shape[1], mlx_emb_with_pos.shape[1])
    pt_emb = pt_emb[:, :min_len, :]
    pt_pos_emb = pt_pos_emb[:, :min_len, :]
    mlx_emb_with_pos = mlx_emb_with_pos[:, :min_len, :]
    mlx_pos_emb_broadcast = mlx_pos_emb_broadcast[:, :min_len, :]
    
    corr_emb = compute_correlation(pt_emb.cpu(), mlx_to_torch(mlx_emb_with_pos, "cpu"))
    print(f"\nAfter Embedding: {corr_emb:.6f}")
    
    # Through each layer
    pt_layer_out = pt_emb
    mlx_layer_out = mlx_emb_with_pos
    
    for layer_idx in range(6):
        with torch.no_grad():
            pt_layer_out = pt_conformer.encoders[layer_idx](
                pt_layer_out, None, pt_pos_emb, None
            )[0][:, :min_len, :]
        
        mlx_layer_out = mlx_conformer.blocks[layer_idx](
            mlx_layer_out, mlx_pos_emb_broadcast, None, None
        )[:, :min_len, :]
        
        corr_layer = compute_correlation(
            pt_layer_out.cpu(),
            mlx_to_torch(mlx_layer_out, "cpu")
        )
        print(f"After Layer {layer_idx}: {corr_layer:.6f}")
    
    # After final norm
    with torch.no_grad():
        pt_final = pt_conformer.after_norm(pt_layer_out)
    mlx_final = mlx_conformer.after_norm(mlx_layer_out)
    
    corr_final = compute_correlation(
        pt_final.cpu(),
        mlx_to_torch(mlx_final, "cpu")
    )
    print(f"After Final Norm: {corr_final:.6f}")
    
    # After Perceiver
    with torch.no_grad():
        pt_perceiver_out = pt_model.gpt.perceiver_encoder(pt_final)
    mlx_perceiver_out = mlx_model.mlx_transformer.conditioning_module.perceiver(mlx_final)
    
    corr_perceiver = compute_correlation(
        pt_perceiver_out.cpu(),
        mlx_to_torch(mlx_perceiver_out, "cpu")
    )
    print(f"After Perceiver: {corr_perceiver:.6f}")
    
    print("\n" + "=" * 80)
    print("💡 分析")
    print("=" * 80)
    
    if correlation > 0.99:
        print("✅ Correlation > 0.99，非常接近了！")
        print("   轻微丢字可能是:")
        print("   1. 数值精度累积误差 (0.9996^6 ≈ 0.9976)")
        print("   2. MLX 和 PyTorch 的浮点运算差异")
        print("   3. 难以完全消除")
    else:
        print(f"⚠️  Correlation = {correlation:.4f}，还有优化空间")
        print("   需要检查累积误差的来源")

if __name__ == "__main__":
    main()


