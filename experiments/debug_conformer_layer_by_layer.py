#!/usr/bin/env python3
"""
逐层诊断 MLX Conformer
找出第一个产生大差异的层
"""

import sys
sys.path.insert(0, '/Users/bailin/index-tts')

import warnings
warnings.filterwarnings('ignore')

import torch
import mlx.core as mx
import numpy as np

print("\n" + "="*70)
print("🔍 Conformer 逐层诊断")
print("="*70)

# ============================================================================
# Load models
# ============================================================================

print("\n加载模型...")
from indextts.infer_v2 import IndexTTS2

pt_model = IndexTTS2(model_dir="/Users/bailin/index-tts/checkpoints", use_mlx=False)

# For MLX, we need to load with Pure MLX Conditioning enabled temporarily
print("\n加载 MLX 模型 (临时启用 Pure MLX Conditioning 用于调试)...")
from indextts.gpt.mlx_model import UnifiedVoiceMLX
from indextts.utils.mlx_cache import MLXModelCache
from omegaconf import OmegaConf

cfg = OmegaConf.load("/Users/bailin/index-tts/checkpoints/config.yaml")
mlx_cache = MLXModelCache(cache_dir="/Users/bailin/index-tts/checkpoints/mlx")
mlx_gpt_weights = mlx_cache.get_or_convert("gpt", "/Users/bailin/index-tts/checkpoints/gpt.pth")

# Create MLX model with Pure MLX conditioning for debugging
mlx_transformer = UnifiedVoiceMLX(
    use_mlx_conditioning=True,  # Enable for debugging
    **cfg.gpt
)
mlx_transformer.load_weights_from_dict(mlx_gpt_weights)
print("✓ MLX model with Pure MLX Conditioning loaded")

# ============================================================================
# Get speaker embedding
# ============================================================================

print("\n提取 speaker embedding...")
import torchaudio

ref_audio = "/Users/bailin/index-tts/examples/voice_01.wav"
audio, sr = torchaudio.load(ref_audio)
if sr != 16000:
    audio = torchaudio.functional.resample(audio, sr, 16000)
if audio.shape[0] > 1:
    audio = audio.mean(0, keepdim=True)

with torch.no_grad():
    inputs = pt_model.extract_features(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(pt_model.device)
    attention_mask = inputs["attention_mask"].to(pt_model.device)
    speaker_emb = pt_model.get_emb(input_features, attention_mask)

print(f"✓ Speaker embedding: {speaker_emb.shape}")

# ============================================================================
# Prepare inputs for Conformer
# ============================================================================

print("\n准备 Conformer 输入...")

# Helper function for padding mask
def make_pad_mask(lengths, maxlen=None):
    """Make mask for padding."""
    batch_size = lengths.size(0)
    if maxlen is None:
        maxlen = lengths.max()
    seq_range = torch.arange(0, maxlen, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, maxlen)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask

# PyTorch conformer input
pt_cond_input = speaker_emb.transpose(1, 2)  # (1, 1024, 121)
pt_cond_lengths = torch.tensor([pt_cond_input.shape[-1]], device=pt_model.device)

# MLX conformer input
mlx_emb = mx.array(speaker_emb.cpu().numpy())  # (1, 121, 1024)
mlx_cond_lengths = mx.array([mlx_emb.shape[1]])  # Note: shape[1] not shape[-1]

print(f"PyTorch input: {pt_cond_input.shape}")
print(f"MLX input: {mlx_emb.shape}")

# ============================================================================
# Get PyTorch Conformer outputs (layer by layer)
# ============================================================================

print("\n" + "="*70)
print("PyTorch Conformer 逐层输出")
print("="*70)

pt_conformer = pt_model.gpt.conditioning_encoder

with torch.no_grad():
    # Transpose for conformer
    xs = pt_cond_input.transpose(1, 2)  # (1, 121, 1024)
    
    # Create mask
    masks = (~make_pad_mask(pt_cond_lengths, xs.size(1))).to(xs.device).unsqueeze(1)
    
    print(f"\nInput to conformer: {xs.shape}")
    print(f"  mean={xs.mean().item():.6f}, std={xs.std().item():.6f}")
    
    # Embedding (input_proj + pos_encoding)
    xs, pos_emb, masks = pt_conformer.embed(xs, masks)
    print(f"\n✓ After embed (input_proj + pos_enc):")
    print(f"  Shape: {xs.shape}")
    print(f"  mean={xs.mean().item():.6f}, std={xs.std().item():.6f}")
    print(f"  range=[{xs.min().item():.6f}, {xs.max().item():.6f}]")
    pt_after_embed = xs.clone()
    
    # Each conformer block
    for i, block in enumerate(pt_conformer.encoders):
        xs, masks, _, _ = block(xs, masks, pos_emb)
        print(f"\n✓ After Conformer block {i}:")
        print(f"  Shape: {xs.shape}")
        print(f"  mean={xs.mean().item():.6f}, std={xs.std().item():.6f}")
        print(f"  range=[{xs.min().item():.6f}, {xs.max().item():.6f}]")
    
    pt_conformer_out = xs.clone()
    
    # After norm
    if pt_conformer.after_norm is not None:
        xs = pt_conformer.after_norm(xs)
        print(f"\n✓ After final norm:")
        print(f"  Shape: {xs.shape}")
        print(f"  mean={xs.mean().item():.6f}, std={xs.std().item():.6f}")
    
    pt_final = xs.clone()

# ============================================================================
# Get MLX Conformer outputs (layer by layer)
# ============================================================================

print("\n" + "="*70)
print("MLX Conformer 逐层输出")
print("="*70)

mlx_conformer = mlx_transformer.conditioning_module.conformer

# ✅ Use the complete MLX Conformer forward pass
x, mask = mlx_conformer(mlx_emb, None)
print(f"\n✓ MLX Conformer complete output (after all layers):")
print(f"  Shape: {x.shape}")
print(f"  mean={float(x.mean()):.6f}, std={float(x.std()):.6f}")
print(f"  range=[{float(x.min()):.6f}, {float(x.max()):.6f}]")

mlx_final = x

# For detailed comparison, also run step-by-step manually
print(f"\n--- Detailed step-by-step (for debugging) ---")
x_debug = mlx_emb
print(f"Input: mean={float(x_debug.mean()):.6f}, std={float(x_debug.std()):.6f}")

# Subsampling
x_debug = mlx_conformer.subsampling(x_debug)
print(f"After subsampling: mean={float(x_debug.mean()):.6f}, std={float(x_debug.std()):.6f}")

# xscale + pos_enc
seq_len = x_debug.shape[1]
pos_emb = mlx_conformer.pos_encoding[:seq_len]
pos_emb = mx.broadcast_to(
    pos_emb.reshape(1, seq_len, mlx_conformer.output_dim),
    (x_debug.shape[0], seq_len, mlx_conformer.output_dim)
)
x_debug = x_debug * mlx_conformer.xscale + pos_emb
print(f"After xscale + pos_enc: mean={float(x_debug.mean()):.6f}, std={float(x_debug.std()):.6f}")
print(f"  xscale={mlx_conformer.xscale:.6f}")

mlx_after_embed = x_debug

# ============================================================================
# Compare layer by layer
# ============================================================================

print("\n" + "="*70)
print("📊 逐层对比分析")
print("="*70)

def compare_outputs(pt_out, mlx_out, name):
    pt_np = pt_out.cpu().numpy()
    mlx_np = np.array(mlx_out)
    
    diff = np.abs(pt_np - mlx_np)
    rel_diff = diff / (np.abs(pt_np) + 1e-8)
    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    
    print(f"\n{name}:")
    print(f"  PyTorch: mean={pt_np.mean():.6f}, std={pt_np.std():.6f}")
    print(f"  MLX:     mean={mlx_np.mean():.6f}, std={mlx_np.std():.6f}")
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")
    print(f"  Correlation: {corr:.6f}")
    
    if diff.max() > 10.0:
        status = "❌ 巨大差异"
    elif diff.max() > 5.0:
        status = "❌ 很大差异"
    elif diff.max() > 2.0:
        status = "⚠️  较大差异"
    elif diff.max() > 0.5:
        status = "⚠️  中等差异"
    else:
        status = "✓ 差异很小"
    
    print(f"  状态: {status}")
    
    return diff.max(), corr

# Compare after embed
diff_embed, corr_embed = compare_outputs(pt_after_embed, mlx_after_embed, "After Embed (input_proj + pos_enc)")

# Compare final outputs
diff_final, corr_final = compare_outputs(pt_final, mlx_final, "Final Conformer Output")

# ============================================================================
# Conclusion
# ============================================================================

print("\n" + "="*70)
print("🎯 诊断结论")
print("="*70)

print(f"\n关键发现:")
print(f"  1. After Embed: max_diff={diff_embed:.4f}, corr={corr_embed:.4f}")
print(f"  2. Final Output: max_diff={diff_final:.4f}, corr={corr_final:.4f}")

if diff_embed > 5.0:
    print(f"\n❌ 问题在 Embed 层 (input_proj or pos_encoding)!")
    print(f"  建议: 检查 input_proj 权重加载和 pos_encoding")
elif diff_final > 5.0:
    print(f"\n❌ 问题在 Conformer blocks!")
    print(f"  建议: 逐个检查每个 block 的输出")
else:
    print(f"\n✓ Conformer 基本正常，问题可能在 Perceiver")

print("\n" + "="*70 + "\n")

