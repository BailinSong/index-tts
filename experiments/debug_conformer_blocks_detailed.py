#!/usr/bin/env python3
"""
详细对比每个 Conformer Block 的内部组件
找出导致 correlation 下降的具体层
"""

import sys
sys.path.insert(0, '/Users/bailin/index-tts')

import warnings
warnings.filterwarnings('ignore')

import torch
import mlx.core as mx
import numpy as np
import torchaudio

print("\n" + "="*70)
print("🔍 Conformer Blocks 详细诊断")
print("="*70)

# Load models
print("\n加载模型...")
from indextts.infer_v2 import IndexTTS2
from indextts.gpt.mlx_model import UnifiedVoiceMLX
from indextts.utils.mlx_cache import MLXModelCache
from omegaconf import OmegaConf

pt_model = IndexTTS2(model_dir="/Users/bailin/index-tts/checkpoints", use_mlx=False)

cfg = OmegaConf.load("/Users/bailin/index-tts/checkpoints/config.yaml")
mlx_cache = MLXModelCache(cache_dir="/Users/bailin/index-tts/checkpoints/mlx")
mlx_gpt_weights = mlx_cache.get_or_convert("gpt", "/Users/bailin/index-tts/checkpoints/gpt.pth")

mlx_transformer = UnifiedVoiceMLX(
    use_mlx_conditioning=True,
    **cfg.gpt
)
mlx_transformer.load_weights_from_dict(mlx_gpt_weights)
print("✓ Models loaded")

# Helper function
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

def compare_tensors(pt_tensor, mlx_array, name):
    """Compare PyTorch and MLX tensors"""
    pt_np = pt_tensor.cpu().numpy()
    mlx_np = np.array(mlx_array)
    
    diff = np.abs(pt_np - mlx_np)
    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    
    print(f"\n{name}:")
    print(f"  Shape: PT={pt_tensor.shape}, MLX={mlx_array.shape}")
    print(f"  PT:  mean={pt_np.mean():.6f}, std={pt_np.std():.6f}")
    print(f"  MLX: mean={mlx_np.mean():.6f}, std={mlx_np.std():.6f}")
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Correlation: {corr:.6f}")
    
    if corr > 0.95:
        status = "✅"
    elif corr > 0.80:
        status = "✓"
    elif corr > 0.60:
        status = "⚠️"
    else:
        status = "❌"
    
    print(f"  Status: {status}")
    return corr

# Get speaker embedding
print("\n提取 speaker embedding...")
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
# PyTorch Conformer - detailed forward
# ============================================================================

print("\n" + "="*70)
print("PyTorch Conformer 详细前向传播")
print("="*70)

pt_conformer = pt_model.gpt.conditioning_encoder
xs = speaker_emb  # (1, 121, 1024)
cond_lengths = torch.tensor([xs.shape[1]], device=pt_model.device)

with torch.no_grad():
    # Create masks
    masks = (~make_pad_mask(cond_lengths, xs.size(1))).to(xs.device).unsqueeze(1)
    
    # Embed (Conv2d + pos_enc)
    xs, pos_emb, masks = pt_conformer.embed(xs, masks)
    print(f"\nAfter embed: {xs.shape}")
    pt_after_embed = xs.clone()
    
    # Save intermediate outputs for each block
    pt_block_outputs = []
    mask_pad = masks
    
    for i, block in enumerate(pt_conformer.encoders):
        # PyTorch Conformer block structure:
        # 1. FF (macaron style) - first half
        # 2. Self-attention
        # 3. Convolution module
        # 4. FF (second half)
        
        xs, masks, _, _ = block(xs, masks, pos_emb, mask_pad)
        pt_block_outputs.append(xs.clone())
        print(f"Block {i}: mean={xs.mean():.6f}, std={xs.std():.6f}, corr_from_prev={1.0 if i==0 else np.corrcoef(xs.cpu().numpy().flatten(), pt_block_outputs[i-1].cpu().numpy().flatten())[0,1]:.6f}")
    
    # After norm
    if pt_conformer.after_norm is not None:
        xs = pt_conformer.after_norm(xs)
    
    pt_final = xs.clone()

# ============================================================================
# MLX Conformer - detailed forward
# ============================================================================

print("\n" + "="*70)
print("MLX Conformer 详细前向传播")
print("="*70)

mlx_conformer = mlx_transformer.conditioning_module.conformer
mlx_emb = mx.array(speaker_emb.cpu().numpy())

# After embed (subsampling + xscale + pos_enc)
x = mlx_emb
x = mlx_conformer.subsampling(x)
seq_len = x.shape[1]
pos_emb = mlx_conformer.pos_encoding[:seq_len]
pos_emb = mx.broadcast_to(
    pos_emb.reshape(1, seq_len, mlx_conformer.output_dim),
    (x.shape[0], seq_len, mlx_conformer.output_dim)
)
x = x * mlx_conformer.xscale + pos_emb
print(f"\nAfter embed: {x.shape}")
mlx_after_embed = x

# Save intermediate outputs for each block
mlx_block_outputs = []

for i, block in enumerate(mlx_conformer.blocks):
    x = block(x, pos_emb, mask=None, mask_pad=None)
    mlx_block_outputs.append(x)
    print(f"Block {i}: mean={float(x.mean()):.6f}, std={float(x.std()):.6f}")

# After norm
x = mlx_conformer.after_norm(x)
mlx_final = x

# ============================================================================
# Detailed comparison
# ============================================================================

print("\n" + "="*70)
print("📊 逐层详细对比")
print("="*70)

# Compare after embed
corr_embed = compare_tensors(pt_after_embed, mlx_after_embed, "After Embed")

# Compare each block
block_corrs = []
for i in range(6):
    corr = compare_tensors(pt_block_outputs[i], mlx_block_outputs[i], f"Block {i} Output")
    block_corrs.append(corr)

# Compare final
corr_final = compare_tensors(pt_final, mlx_final, "Final Output")

# ============================================================================
# Analysis
# ============================================================================

print("\n" + "="*70)
print("🎯 分析结论")
print("="*70)

print(f"\n相关系数变化:")
print(f"  After Embed: {corr_embed:.6f}")
for i, corr in enumerate(block_corrs):
    delta = corr - (corr_embed if i == 0 else block_corrs[i-1])
    print(f"  Block {i}:     {corr:.6f} (Δ {delta:+.6f})")
print(f"  Final:       {corr_final:.6f}")

# Find the most problematic block
if len(block_corrs) > 0:
    worst_block = min(range(len(block_corrs)), key=lambda i: block_corrs[i])
    worst_corr = block_corrs[worst_block]
    
    if worst_corr < 0.80:
        print(f"\n❌ 最大问题在 Block {worst_block} (correlation: {worst_corr:.6f})")
        print(f"   建议: 详细检查 Block {worst_block} 的内部组件")
    elif corr_final < 0.80:
        print(f"\n⚠️  Blocks 基本正常，但累积误差导致 final correlation 下降")
        print(f"   建议: 检查每个 block 的细节实现")
    else:
        print(f"\n✓ Conformer blocks 都正常 (all > 0.80)")
        print(f"   问题可能在 Perceiver")

print("\n" + "="*70 + "\n")

