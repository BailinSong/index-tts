#!/usr/bin/env python3
"""
测试完整的 conditioning pipeline: Conformer + Perceiver
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
print("测试完整 Conditioning Pipeline")
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

# PyTorch conditioning
print("\n" + "="*70)
print("PyTorch Conditioning (Conformer + Perceiver)")
print("="*70)

# Call conformer and perceiver directly (like in get_conditioning)
cond_lengths = torch.tensor([speaker_emb.shape[1]], device=pt_model.device)  # [121]

with torch.no_grad():
    # Conformer: (batch, time, freq) → (batch, time', dim)
    conformer_out, mask = pt_model.gpt.conditioning_encoder(speaker_emb, cond_lengths)
    print(f"  Conformer output: {conformer_out.shape}")
    
    # Perceiver: (batch, time', dim) → (batch, num_latents, dim)
    conds_mask = pt_model.gpt.cond_mask_pad(mask.squeeze(1))
    pt_cond = pt_model.gpt.perceiver_encoder(conformer_out, conds_mask)
    print(f"  Perceiver output: {pt_cond.shape}")

print(f"\n✓ PyTorch conditioning output:")
print(f"  Shape: {pt_cond.shape}")
print(f"  Mean: {pt_cond.mean():.6f}")
print(f"  Std: {pt_cond.std():.6f}")
print(f"  Range: [{pt_cond.min():.6f}, {pt_cond.max():.6f}]")

# MLX conditioning
print("\n" + "="*70)
print("MLX Conditioning (Conformer + Perceiver)")
print("="*70)

mlx_emb = mx.array(speaker_emb.cpu().numpy())
mlx_cond = mlx_transformer.conditioning_module(mlx_emb, None)

print(f"\n✓ MLX conditioning output:")
print(f"  Shape: {mlx_cond.shape}")
print(f"  Mean: {float(mlx_cond.mean()):.6f}")
print(f"  Std: {float(mlx_cond.std()):.6f}")
print(f"  Range: [{float(mlx_cond.min()):.6f}, {float(mlx_cond.max()):.6f}]")

# Compare
print("\n" + "="*70)
print("对比分析")
print("="*70)

pt_np = pt_cond.cpu().numpy()
mlx_np = np.array(mlx_cond)

diff = np.abs(pt_np - mlx_np)
corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]

print(f"\n统计对比:")
print(f"  Max diff: {diff.max():.6f}")
print(f"  Mean diff: {diff.mean():.6f}")
print(f"  Median diff: {np.median(diff):.6f}")
print(f"  Correlation: {corr:.6f}")

if corr > 0.95:
    status = "✅ 非常接近"
elif corr > 0.85:
    status = "✓ 基本匹配"
elif corr > 0.70:
    status = "⚠️  中等差异"
else:
    status = "❌ 较大差异"

print(f"  状态: {status}")

print("\n" + "="*70 + "\n")

