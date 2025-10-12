#!/usr/bin/env python3
"""
简化的 MLX 质量诊断
直接对比音频输出和波形
"""

import sys
sys.path.insert(0, '/Users/bailin/index-tts')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
from pathlib import Path

print("\n" + "="*70)
print("MLX 音频质量诊断（简化版）")
print("="*70)

test_text = "今天"
ref_audio = "/Users/bailin/index-tts/examples/voice_01.wav"

# ============================================================================
# Phase 1: 生成音频
# ============================================================================

print("\n" + "="*70)
print("Phase 1: 生成音频")
print("="*70)

from indextts.infer_v2 import IndexTTS2
import gc

# PyTorch
print("\n1.1: PyTorch 推理...")
pt_model = IndexTTS2(model_dir="/Users/bailin/index-tts/checkpoints", use_mlx=False)
pt_model.infer(
    spk_audio_prompt=ref_audio,
    text=test_text,
    output_path="experiments/diagnose_pytorch.wav",
    emo_audio_prompt=ref_audio,
    use_emo_text=True,
    emo_text=test_text,
)
print("✓ PyTorch 音频: experiments/diagnose_pytorch.wav")
del pt_model
gc.collect()

# MLX
print("\n1.2: MLX 推理...")
mlx_model = IndexTTS2(model_dir="/Users/bailin/index-tts/checkpoints", use_mlx=True)
mlx_model.infer(
    spk_audio_prompt=ref_audio,
    text=test_text,
    output_path="experiments/diagnose_mlx.wav",
    emo_audio_prompt=ref_audio,
    use_emo_text=True,
    emo_text=test_text,
)
print("✓ MLX 音频: experiments/diagnose_mlx.wav")

# ============================================================================
# Phase 2: 波形对比分析
# ============================================================================

print("\n" + "="*70)
print("Phase 2: 波形对比分析")
print("="*70)

import scipy.io.wavfile as wavfile

# Load audios
_, pt_audio = wavfile.read("experiments/diagnose_pytorch.wav")
_, mlx_audio = wavfile.read("experiments/diagnose_mlx.wav")
_, base_audio = wavfile.read("gen_base.wav")

pt_audio_f = pt_audio.astype(np.float32) / 32767.0
mlx_audio_f = mlx_audio.astype(np.float32) / 32767.0
base_audio_f = base_audio.astype(np.float32) / 32767.0

print(f"\n2.1: 基本统计:")
print(f"  Baseline:  len={len(base_audio_f)/22050:.2f}s, mean={base_audio_f.mean():.6f}, std={base_audio_f.std():.6f}")
print(f"  PyTorch:   len={len(pt_audio_f)/22050:.2f}s, mean={pt_audio_f.mean():.6f}, std={pt_audio_f.std():.6f}")
print(f"  MLX:       len={len(mlx_audio_f)/22050:.2f}s, mean={mlx_audio_f.mean():.6f}, std={mlx_audio_f.std():.6f}")

# Energy comparison (RMS)
print(f"\n2.2: 能量对比 (RMS):")
base_energy = np.sqrt(np.mean(base_audio_f**2))
pt_energy = np.sqrt(np.mean(pt_audio_f**2))
mlx_energy = np.sqrt(np.mean(mlx_audio_f**2))

print(f"  Baseline: {base_energy:.6f}")
print(f"  PyTorch:  {pt_energy:.6f} (vs baseline: {pt_energy/base_energy:.1%})")
print(f"  MLX:      {mlx_energy:.6f} (vs baseline: {mlx_energy/base_energy:.1%})")

# Peak analysis
print(f"\n2.3: 峰值分析:")
print(f"  Baseline: max={np.abs(base_audio_f).max():.4f}")
print(f"  PyTorch:  max={np.abs(pt_audio_f).max():.4f}")
print(f"  MLX:      max={np.abs(mlx_audio_f).max():.4f}")

# Check for silent audio
silence_threshold = 0.001
is_silent = mlx_energy < silence_threshold

if is_silent:
    print(f"\n  ❌ 警告: MLX 音频几乎无声! (RMS < {silence_threshold})")
elif mlx_energy < base_energy * 0.1:
    print(f"\n  ⚠️  警告: MLX 音频能量过低! (仅为 baseline 的 {mlx_energy/base_energy*100:.1f}%)")
elif mlx_energy < base_energy * 0.5:
    print(f"\n  ⚠️  注意: MLX 音频能量偏低 (为 baseline 的 {mlx_energy/base_energy*100:.1f}%)")
else:
    print(f"\n  ✓ MLX 音频能量正常")

# ============================================================================
# Phase 3: 可视化对比
# ============================================================================

print(f"\n" + "="*70)
print("Phase 3: 可视化对比")
print("="*70)

# Plot waveforms (first 0.5s)
print(f"\n3.1: 绘制波形对比图...")
fig, axes = plt.subplots(3, 1, figsize=(16, 10))

n_samples = 11000  # 0.5s @ 22050Hz

axes[0].plot(base_audio_f[:n_samples], linewidth=0.5, alpha=0.8, color='blue')
axes[0].set_title(f'Baseline (gen_base.wav) - RMS: {base_energy:.4f}, Max: {np.abs(base_audio_f).max():.4f}', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Amplitude', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-1, 1)
axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

axes[1].plot(pt_audio_f[:n_samples], linewidth=0.5, alpha=0.8, color='orange')
axes[1].set_title(f'PyTorch - RMS: {pt_energy:.4f}, Max: {np.abs(pt_audio_f).max():.4f}', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Amplitude', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-1, 1)
axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

axes[2].plot(mlx_audio_f[:n_samples], linewidth=0.5, alpha=0.8, color='red')
axes[2].set_title(f'MLX - RMS: {mlx_energy:.4f}, Max: {np.abs(mlx_audio_f).max():.4f}', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Sample (first 0.5s)', fontsize=11)
axes[2].set_ylabel('Amplitude', fontsize=11)
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(-1, 1)
axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

plt.tight_layout()
plt.savefig('experiments/diagnose_waveform.png', dpi=150, bbox_inches='tight')
print(f"✓ 波形图: experiments/diagnose_waveform.png")

# Plot spectrograms
print(f"\n3.2: 绘制频谱图...")
fig, axes = plt.subplots(3, 1, figsize=(16, 10))

# Show first 2s
n_spec_samples = min(44100, len(base_audio_f), len(pt_audio_f), len(mlx_audio_f))

for idx, (audio, title, color) in enumerate([
    (base_audio_f[:n_spec_samples], 'Baseline (gen_base.wav)', 'viridis'),
    (pt_audio_f[:n_spec_samples], 'PyTorch', 'plasma'),
    (mlx_audio_f[:n_spec_samples], 'MLX', 'inferno')
]):
    spec, freqs, t, im = axes[idx].specgram(audio, Fs=22050, NFFT=1024, noverlap=512, cmap=color)
    axes[idx].set_title(title, fontsize=13, fontweight='bold')
    axes[idx].set_ylabel('Frequency (Hz)', fontsize=11)
    axes[idx].set_ylim(0, 8000)
    plt.colorbar(im, ax=axes[idx], label='Power/Frequency (dB/Hz)')
    
axes[2].set_xlabel('Time (s)', fontsize=11)
plt.tight_layout()
plt.savefig('experiments/diagnose_spectrogram.png', dpi=150, bbox_inches='tight')
print(f"✓ 频谱图: experiments/diagnose_spectrogram.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("诊断总结")
print("="*70)

print(f"\n📊 音频长度:")
print(f"  Baseline: {len(base_audio_f)/22050:.2f}s")
print(f"  PyTorch:  {len(pt_audio_f)/22050:.2f}s (diff: {abs(len(pt_audio_f)-len(base_audio_f))/22050:.2f}s)")
print(f"  MLX:      {len(mlx_audio_f)/22050:.2f}s (diff: {abs(len(mlx_audio_f)-len(base_audio_f))/22050:.2f}s)")

print(f"\n🔊 音频能量 (RMS):")
mlx_vs_base = mlx_energy / base_energy
mlx_vs_pt = mlx_energy / pt_energy

if mlx_energy < 0.001:
    print(f"  ❌ MLX 几乎无声 (RMS: {mlx_energy:.6f})")
    status = "CRITICAL"
elif mlx_vs_base < 0.1:
    print(f"  ❌ MLX 能量过低 (仅为 baseline 的 {mlx_vs_base*100:.1f}%)")
    status = "VERY_LOW"
elif mlx_vs_base < 0.5:
    print(f"  ⚠️  MLX 能量偏低 (为 baseline 的 {mlx_vs_base*100:.1f}%)")
    status = "LOW"
elif 0.5 <= mlx_vs_base <= 2.0:
    print(f"  ✓ MLX 能量正常 (为 baseline 的 {mlx_vs_base*100:.1f}%)")
    status = "NORMAL"
else:
    print(f"  ⚠️  MLX 能量过高 (为 baseline 的 {mlx_vs_base*100:.1f}%)")
    status = "HIGH"

print(f"\n  PyTorch vs Baseline: {pt_energy/base_energy*100:.1f}%")
print(f"  MLX vs Baseline: {mlx_vs_base*100:.1f}%")
print(f"  MLX vs PyTorch: {mlx_vs_pt*100:.1f}%")

print(f"\n📁 生成的文件:")
print(f"  🔊 experiments/diagnose_pytorch.wav")
print(f"  🔊 experiments/diagnose_mlx.wav")
print(f"  📊 experiments/diagnose_waveform.png")
print(f"  📊 experiments/diagnose_spectrogram.png")

print(f"\n🔍 问题分析:")
if status in ["CRITICAL", "VERY_LOW", "LOW"]:
    print(f"  ⚠️  MLX 音频质量异常!")
    print(f"\n  可能原因:")
    print(f"  1. MLX Conditioning (Conformer/Perceiver) 输出异常")
    print(f"  2. MLX Transformer 生成的 mel tokens 有问题")
    print(f"  3. S2MEL 或 BigVGAN 处理 MLX tokens 时出错")
    print(f"\n  建议调试步骤:")
    print(f"  A. 对比 MLX 和 PyTorch 的 conditioning latents")
    print(f"  B. 对比生成的 mel tokens 序列")
    print(f"  C. 检查 S2MEL 的输入/输出")
    print(f"  D. 验证 BigVGAN 的输入/输出")
else:
    print(f"  ✓ MLX 音频能量正常")
    print(f"  建议人工听音对比质量")

print("\n" + "="*70 + "\n")

