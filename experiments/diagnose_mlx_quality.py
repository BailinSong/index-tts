#!/usr/bin/env python3
"""
诊断 MLX 音频质量问题
对比 PyTorch 和 MLX 在每个阶段的输出
"""

import sys
sys.path.insert(0, '/Users/bailin/index-tts')

import warnings
warnings.filterwarnings('ignore')

import torch
import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("\n" + "="*70)
print("MLX 音频质量诊断")
print("="*70)

test_text = "今天"

# ============================================================================
# Phase 1: 加载模型并获取 conditioning
# ============================================================================

print("\n" + "="*70)
print("Phase 1: Conditioning 对比")
print("="*70)

from indextts.infer_v2 import IndexTTS2

# Load PyTorch model
print("\n1.1: 加载 PyTorch 模型...")
pt_model = IndexTTS2(model_dir="/Users/bailin/index-tts/checkpoints", use_mlx=False)
print("✓ PyTorch 模型加载完成")

# Load MLX model
print("\n1.2: 加载 MLX 模型...")
mlx_model = IndexTTS2(model_dir="/Users/bailin/index-tts/checkpoints", use_mlx=True)
print("✓ MLX 模型加载完成")

# ============================================================================
# Phase 2: 提取 speaker embedding 并对比 conditioning
# ============================================================================

print("\n" + "="*70)
print("Phase 2: Speaker Embedding & Conditioning 对比")
print("="*70)

ref_audio = "/Users/bailin/index-tts/examples/voice_01.wav"

# Extract speaker embedding (same for both)
print("\n2.1: 提取 speaker embedding...")
import torchaudio
import torch.nn.functional as F

audio, sr = torchaudio.load(ref_audio)
if sr != 16000:
    audio = torchaudio.functional.resample(audio, sr, 16000)
if audio.shape[0] > 1:
    audio = audio.mean(0, keepdim=True)

# Extract features
with torch.no_grad():
    audio_mps = audio.to(pt_model.device)
    features = pt_model.w2v_model.extract_features(audio_mps)[0]
    
    input_features = features.transpose(1, 2)
    attention_mask = torch.ones(input_features.shape[0], input_features.shape[2], 
                                device=pt_model.device, dtype=torch.long)
    
    speaker_emb = pt_model.get_emb(input_features, attention_mask)

print(f"✓ Speaker embedding: {speaker_emb.shape}")
print(f"  Stats: mean={speaker_emb.mean():.6f}, std={speaker_emb.std():.6f}")

# Get PyTorch conditioning
print("\n2.2: PyTorch conditioning...")
with torch.no_grad():
    cond_input = speaker_emb.transpose(1, 2)  # (1, 1024, seq)
    cond_lengths = torch.tensor([cond_input.shape[-1]], device=pt_model.device)
    
    pt_cond = pt_model.gpt.get_conditioning(cond_input, cond_lengths)

print(f"✓ PyTorch conditioning: {pt_cond.shape}")
print(f"  Stats: mean={pt_cond.mean():.6f}, std={pt_cond.std():.6f}")
print(f"  Range: [{pt_cond.min():.6f}, {pt_cond.max():.6f}]")

# Get MLX conditioning
print("\n2.3: MLX conditioning...")
mlx_emb = mx.array(speaker_emb.cpu().numpy())
mlx_cond_lengths = mx.array([mlx_emb.shape[-1]])

mlx_cond = mlx_model.mlx_transformer.get_conditioning_mlx(mlx_emb, mlx_cond_lengths)

print(f"✓ MLX conditioning: {mlx_cond.shape}")
print(f"  Stats: mean={float(mlx_cond.mean()):.6f}, std={float(mlx_cond.std()):.6f}")
print(f"  Range: [{float(mlx_cond.min()):.6f}, {float(mlx_cond.max()):.6f}]")

# Compare conditioning
print("\n2.4: Conditioning 差异分析...")
pt_cond_np = pt_cond.cpu().numpy()
mlx_cond_np = np.array(mlx_cond)

diff = np.abs(pt_cond_np - mlx_cond_np)
relative_diff = diff / (np.abs(pt_cond_np) + 1e-8)

print(f"  Max absolute diff: {diff.max():.6f}")
print(f"  Mean absolute diff: {diff.mean():.6f}")
print(f"  Max relative diff: {relative_diff.max():.6f}")
print(f"  Mean relative diff: {relative_diff.mean():.6f}")

if diff.max() > 1.0:
    print(f"  ⚠️  警告: Conditioning 差异较大!")
    print(f"\n  Sample PyTorch values:")
    print(f"    {pt_cond_np[0, :3, :5]}")
    print(f"  Sample MLX values:")
    print(f"    {mlx_cond_np[0, :3, :5]}")
else:
    print(f"  ✓ Conditioning 差异在可接受范围内")

# ============================================================================
# Phase 3: 完整推理并对比音频
# ============================================================================

print("\n" + "="*70)
print("Phase 3: 完整推理对比")
print("="*70)

print("\n3.1: PyTorch 推理...")
pt_model.infer(
    spk_audio_prompt=ref_audio,
    text=test_text,
    output_path="experiments/diagnose_pytorch.wav",
    emo_audio_prompt=ref_audio,
    use_emo_text=True,
    emo_text=test_text,
)
print("✓ PyTorch 音频生成: experiments/diagnose_pytorch.wav")

print("\n3.2: MLX 推理...")
mlx_model.infer(
    spk_audio_prompt=ref_audio,
    text=test_text,
    output_path="experiments/diagnose_mlx.wav",
    emo_audio_prompt=ref_audio,
    use_emo_text=True,
    emo_text=test_text,
)
print("✓ MLX 音频生成: experiments/diagnose_mlx.wav")

# ============================================================================
# Phase 4: 波形对比分析
# ============================================================================

print("\n" + "="*70)
print("Phase 4: 波形对比分析")
print("="*70)

import scipy.io.wavfile as wavfile

# Load audios
_, pt_audio = wavfile.read("experiments/diagnose_pytorch.wav")
_, mlx_audio = wavfile.read("experiments/diagnose_mlx.wav")
_, base_audio = wavfile.read("gen_base.wav")

pt_audio_f = pt_audio.astype(np.float32) / 32767.0
mlx_audio_f = mlx_audio.astype(np.float32) / 32767.0
base_audio_f = base_audio.astype(np.float32) / 32767.0

print(f"\n4.1: 音频统计对比:")
print(f"  Baseline:  len={len(base_audio_f)}, mean={base_audio_f.mean():.6f}, std={base_audio_f.std():.6f}")
print(f"  PyTorch:   len={len(pt_audio_f)}, mean={pt_audio_f.mean():.6f}, std={pt_audio_f.std():.6f}")
print(f"  MLX:       len={len(mlx_audio_f)}, mean={mlx_audio_f.mean():.6f}, std={mlx_audio_f.std():.6f}")

# Energy comparison
print(f"\n4.2: 能量对比:")
base_energy = np.sqrt(np.mean(base_audio_f**2))
pt_energy = np.sqrt(np.mean(pt_audio_f**2))
mlx_energy = np.sqrt(np.mean(mlx_audio_f**2))

print(f"  Baseline RMS: {base_energy:.6f}")
print(f"  PyTorch RMS:  {pt_energy:.6f} (ratio: {pt_energy/base_energy:.3f})")
print(f"  MLX RMS:      {mlx_energy:.6f} (ratio: {mlx_energy/base_energy:.3f})")

if mlx_energy < base_energy * 0.1:
    print(f"  ⚠️  警告: MLX 音频能量过低! (仅为 baseline 的 {mlx_energy/base_energy*100:.1f}%)")
elif mlx_energy < base_energy * 0.5:
    print(f"  ⚠️  注意: MLX 音频能量偏低 (为 baseline 的 {mlx_energy/base_energy*100:.1f}%)")

# Plot waveforms
print(f"\n4.3: 绘制波形对比图...")
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Show first 0.5s
n_samples = min(11000, len(base_audio_f), len(pt_audio_f), len(mlx_audio_f))

axes[0].plot(base_audio_f[:n_samples], linewidth=0.5, alpha=0.8)
axes[0].set_title(f'Baseline (gen_base.wav) - RMS: {base_energy:.4f}', fontsize=12)
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-1, 1)

axes[1].plot(pt_audio_f[:n_samples], linewidth=0.5, alpha=0.8, color='orange')
axes[1].set_title(f'PyTorch - RMS: {pt_energy:.4f}', fontsize=12)
axes[1].set_ylabel('Amplitude')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-1, 1)

axes[2].plot(mlx_audio_f[:n_samples], linewidth=0.5, alpha=0.8, color='red')
axes[2].set_title(f'MLX - RMS: {mlx_energy:.4f}', fontsize=12)
axes[2].set_xlabel('Sample')
axes[2].set_ylabel('Amplitude')
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(-1, 1)

plt.tight_layout()
plt.savefig('experiments/diagnose_waveform_comparison.png', dpi=150, bbox_inches='tight')
print(f"✓ 波形对比图保存: experiments/diagnose_waveform_comparison.png")

# Spectrogram comparison
print(f"\n4.4: 绘制频谱图对比...")
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

for idx, (audio, title) in enumerate([
    (base_audio_f[:44100], 'Baseline (gen_base.wav)'),
    (pt_audio_f[:44100], 'PyTorch'),
    (mlx_audio_f[:44100], 'MLX')
]):
    axes[idx].specgram(audio, Fs=22050, NFFT=1024, noverlap=512, cmap='viridis')
    axes[idx].set_title(title, fontsize=12)
    axes[idx].set_ylabel('Frequency (Hz)')
    axes[idx].set_ylim(0, 8000)
    
axes[2].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig('experiments/diagnose_spectrogram_comparison.png', dpi=150, bbox_inches='tight')
print(f"✓ 频谱图对比保存: experiments/diagnose_spectrogram_comparison.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("诊断总结")
print("="*70)

print(f"\n📊 Conditioning 质量:")
if diff.max() < 0.5:
    print(f"  ✅ 优秀 (max diff: {diff.max():.6f})")
elif diff.max() < 2.0:
    print(f"  ✓ 良好 (max diff: {diff.max():.6f})")
else:
    print(f"  ⚠️  有问题 (max diff: {diff.max():.6f})")

print(f"\n🔊 音频质量:")
mlx_vs_base_ratio = mlx_energy / base_energy
if mlx_vs_base_ratio < 0.1:
    print(f"  ❌ 严重问题: MLX 能量过低 ({mlx_vs_base_ratio*100:.1f}%)")
elif mlx_vs_base_ratio < 0.5:
    print(f"  ⚠️  能量偏低: MLX ({mlx_vs_base_ratio*100:.1f}%)")
elif 0.5 <= mlx_vs_base_ratio <= 2.0:
    print(f"  ✓ 能量正常: MLX ({mlx_vs_base_ratio*100:.1f}%)")
else:
    print(f"  ⚠️  能量过高: MLX ({mlx_vs_base_ratio*100:.1f}%)")

print(f"\n📁 生成的文件:")
print(f"  - experiments/diagnose_pytorch.wav")
print(f"  - experiments/diagnose_mlx.wav")
print(f"  - experiments/diagnose_waveform_comparison.png")
print(f"  - experiments/diagnose_spectrogram_comparison.png")

print(f"\n🔍 建议:")
if diff.max() > 2.0:
    print(f"  1. Conditioning 差异过大，检查权重加载")
    print(f"  2. 验证 MLX Conformer 和 Perceiver 的前向传播")
if mlx_energy < base_energy * 0.1:
    print(f"  3. MLX 音频能量过低，可能是:")
    print(f"     - S2MEL 或 BigVGAN 的问题")
    print(f"     - GPT 生成的 mel tokens 有问题")
    print(f"     - 需要检查中间输出")

print("\n" + "="*70 + "\n")

