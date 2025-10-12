#!/usr/bin/env python3
"""
深度诊断: 对比 PyTorch 和 MLX 的 conditioning latents 和 mel tokens
找出音质问题的根源
"""

import sys
sys.path.insert(0, '/Users/bailin/index-tts')

import warnings
warnings.filterwarnings('ignore')

import torch
import mlx.core as mx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("\n" + "="*70)
print("Conditioning & Tokens 深度对比")
print("="*70)

test_text = "今天"
ref_audio = "/Users/bailin/index-tts/examples/voice_01.wav"

from indextts.infer_v2 import IndexTTS2

# ============================================================================
# Part 1: 提取并对比 Conditioning Latents
# ============================================================================

print("\n" + "="*70)
print("Part 1: Conditioning Latents 对比")
print("="*70)

print("\n1.1: 加载模型...")
pt_model = IndexTTS2(model_dir="/Users/bailin/index-tts/checkpoints", use_mlx=False)
mlx_model = IndexTTS2(model_dir="/Users/bailin/index-tts/checkpoints", use_mlx=True)

# Get speaker embeddings
print("\n1.2: 提取 speaker embedding...")
import torchaudio

audio, sr = torchaudio.load(ref_audio)
if sr != 16000:
    audio = torchaudio.functional.resample(audio, sr, 16000)
if audio.shape[0] > 1:
    audio = audio.mean(0, keepdim=True)

with torch.no_grad():
    processor = pt_model.w2v_processor
    inputs = processor(audio.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt")
    
    input_features = inputs.input_features.to(pt_model.device)
    attention_mask = inputs.attention_mask.to(pt_model.device)
    
    speaker_emb = pt_model.get_emb(input_features, attention_mask)

print(f"✓ Speaker embedding: {speaker_emb.shape}")
print(f"  Stats: mean={speaker_emb.mean():.6f}, std={speaker_emb.std():.6f}")

# PyTorch conditioning
print("\n1.3: PyTorch conditioning...")
with torch.no_grad():
    cond_input = speaker_emb.transpose(1, 2)
    cond_lengths = torch.tensor([cond_input.shape[-1]], device=pt_model.device)
    pt_cond = pt_model.gpt.get_conditioning(cond_input, cond_lengths)

print(f"✓ PyTorch: {pt_cond.shape}")
print(f"  mean={pt_cond.mean():.6f}, std={pt_cond.std():.6f}")
print(f"  range=[{pt_cond.min():.6f}, {pt_cond.max():.6f}]")

# MLX conditioning
print("\n1.4: MLX conditioning...")
mlx_emb = mx.array(speaker_emb.cpu().numpy())
mlx_cond_lengths = mx.array([mlx_emb.shape[-1]])
mlx_cond = mlx_model.mlx_transformer.get_conditioning_mlx(mlx_emb, mlx_cond_lengths)

print(f"✓ MLX: {mlx_cond.shape}")
print(f"  mean={float(mlx_cond.mean()):.6f}, std={float(mlx_cond.std()):.6f}")
print(f"  range=[{float(mlx_cond.min()):.6f}, {float(mlx_cond.max()):.6f}]")

# Compare
print("\n1.5: Conditioning 差异分析...")
pt_cond_np = pt_cond.cpu().numpy()
mlx_cond_np = np.array(mlx_cond)

diff = np.abs(pt_cond_np - mlx_cond_np)
relative_diff = diff / (np.abs(pt_cond_np) + 1e-8)

print(f"  Absolute diff: max={diff.max():.6f}, mean={diff.mean():.6f}")
print(f"  Relative diff: max={relative_diff.max():.6f}, mean={relative_diff.mean():.6f}")

if diff.max() > 5.0:
    print(f"  ❌ 差异过大! Conditioning 可能有问题")
    cond_problem = True
elif diff.max() > 2.0:
    print(f"  ⚠️  差异较大，可能影响质量")
    cond_problem = True
else:
    print(f"  ✓ 差异可接受")
    cond_problem = False

# Visualize conditioning comparison
print(f"\n1.6: 可视化 conditioning...")
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# PyTorch
im0 = axes[0].imshow(pt_cond_np[0].T, aspect='auto', cmap='viridis', interpolation='nearest')
axes[0].set_title(f'PyTorch Conditioning (mean={pt_cond.mean():.4f}, std={pt_cond.std():.4f})')
axes[0].set_ylabel('Dimension')
plt.colorbar(im0, ax=axes[0])

# MLX
im1 = axes[1].imshow(mlx_cond_np[0].T, aspect='auto', cmap='viridis', interpolation='nearest')
axes[1].set_title(f'MLX Conditioning (mean={float(mlx_cond.mean()):.4f}, std={float(mlx_cond.std()):.4f})')
axes[1].set_ylabel('Dimension')
plt.colorbar(im1, ax=axes[1])

# Difference
im2 = axes[2].imshow(diff[0].T, aspect='auto', cmap='hot', interpolation='nearest')
axes[2].set_title(f'Absolute Difference (max={diff.max():.4f}, mean={diff.mean():.4f})')
axes[2].set_ylabel('Dimension')
axes[2].set_xlabel('Latent Index')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig('experiments/diagnose_conditioning_comparison.png', dpi=150, bbox_inches='tight')
print(f"✓ Conditioning 对比图: experiments/diagnose_conditioning_comparison.png")

# ============================================================================
# Part 2: 对比生成的 Mel Tokens
# ============================================================================

print("\n" + "="*70)
print("Part 2: Mel Tokens 对比")
print("="*70)

# Need to hook into generation to capture tokens
# For now, let's check the generated audio characteristics

print("\n2.1: 对比生成结果...")
print("  (已在之前的诊断中生成)")
print("  - experiments/diagnose_pytorch.wav")
print("  - experiments/diagnose_mlx.wav")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("问题诊断结论")
print("="*70)

print(f"\n🔍 Conditioning 质量:")
if cond_problem:
    print(f"  ⚠️  Conditioning 存在明显差异 (max diff: {diff.max():.4f})")
    print(f"  这可能导致生成的音频音色不同")
else:
    print(f"  ✓ Conditioning 差异在可接受范围")

print(f"\n🎵 音频特征 (来自之前诊断):")
print(f"  - MLX 能量正常 (89.3% of baseline)")
print(f"  - MLX 比 PyTorch 能量高 (293.4%)")
print(f"  - MLX 音频较长 (1.93s vs PyTorch 1.29s)")

print(f"\n💡 可能的问题:")
if cond_problem:
    print(f"  1. ⚠️  MLX Conditioning (Conformer/Perceiver) 输出与 PyTorch 差异较大")
    print(f"     → 导致生成的音色/音质不同")
else:
    print(f"  1. ✓ Conditioning 基本正常")

print(f"  2. MLX 生成了更多 mel tokens (推测，基于音频更长)")
print(f"  3. 可能是音色/音质问题，而非音量问题")

print(f"\n🔧 建议修复步骤:")
if cond_problem:
    print(f"  A. 检查 MLX Conformer 权重加载是否完全正确")
    print(f"  B. 检查 MLX Perceiver 权重加载是否完全正确")
    print(f"  C. 对比 PyTorch 和 MLX 的 Conformer 中间层输出")
    print(f"  D. 对比 PyTorch 和 MLX 的 Perceiver 中间层输出")
else:
    print(f"  - Conditioning 看起来正常，可能需要微调其他参数")
    print(f"  - 或者音质差异在可接受范围，需要人工评估")

print(f"\n📁 生成的分析文件:")
print(f"  - experiments/diagnose_conditioning_comparison.png")

print("\n" + "="*70 + "\n")

