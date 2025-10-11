#!/usr/bin/env python3
"""
Step 6.4 & 6.5: Final MLX vs PyTorch Comparison
Lightweight version - load models separately to avoid memory issues

Focus on:
1. Audio quality comparison (most important)
2. Performance benchmark
"""

import sys
sys.path.insert(0, '/Users/bailin/index-tts')

import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import gc

print("\n" + "="*70)
print("MLX vs PyTorch Final Comparison")
print("="*70)

test_text = "今天天气很好"
ref_audio = "/Users/bailin/index-tts/examples/voice_01.wav"
ref_text = "今天天气很好"

#==================================================================
# Part 1: Generate PyTorch Audio
#==================================================================

print("\n" + "="*70)
print("Part 1: PyTorch Audio Generation")
print("="*70)

print("\n1. Loading PyTorch model...")
from indextts.infer_v2 import IndexTTS2

pytorch_model = IndexTTS2(
    model_dir="/Users/bailin/index-tts/checkpoints",
    use_mlx=False
)
print("✓ PyTorch model loaded")

print("\n2. Generating PyTorch audio...")
pt_start = time.time()
pytorch_model.infer(
    spk_audio_prompt=ref_audio,
    text=test_text,
    output_path="experiments/final_pytorch.wav",
    emo_audio_prompt=ref_audio,
    use_emo_text=True,
    emo_text=ref_text,
)
pt_time = time.time() - pt_start

print(f"✓ PyTorch generation completed in {pt_time:.2f}s")
print(f"  Output: experiments/final_pytorch.wav")

# Load and analyze
import scipy.io.wavfile as wavfile
pt_sr, pt_audio = wavfile.read("experiments/final_pytorch.wav")
pt_audio_float = pt_audio.astype(np.float32) / 32767.0
pt_duration = len(pt_audio) / pt_sr

print(f"  Duration: {pt_duration:.2f}s")
print(f"  RTF: {pt_time/pt_duration:.3f}x")
print(f"  Stats: mean={pt_audio_float.mean():.4f}, std={pt_audio_float.std():.4f}")

# Cleanup
del pytorch_model
gc.collect()

#==================================================================
# Part 2: Generate MLX Audio
#==================================================================

print("\n" + "="*70)
print("Part 2: MLX Audio Generation")
print("="*70)

print("\n1. Loading MLX model...")
mlx_model = IndexTTS2(
    model_dir="/Users/bailin/index-tts/checkpoints",
    use_mlx=True
)
print("✓ MLX model loaded")

print("\n2. Generating MLX audio...")
mlx_start = time.time()
mlx_model.infer(
    spk_audio_prompt=ref_audio,
    text=test_text,
    output_path="experiments/final_mlx.wav",
    emo_audio_prompt=ref_audio,
    use_emo_text=True,
    emo_text=ref_text,
)
mlx_time = time.time() - mlx_start

print(f"✓ MLX generation completed in {mlx_time:.2f}s")
print(f"  Output: experiments/final_mlx.wav")

# Load and analyze
mlx_sr, mlx_audio = wavfile.read("experiments/final_mlx.wav")
mlx_audio_float = mlx_audio.astype(np.float32) / 32767.0
mlx_duration = len(mlx_audio) / mlx_sr

print(f"  Duration: {mlx_duration:.2f}s")
print(f"  RTF: {mlx_time/mlx_duration:.3f}x")
print(f"  Stats: mean={mlx_audio_float.mean():.4f}, std={mlx_audio_float.std():.4f}")

#==================================================================
# Part 3: Comparison Summary
#==================================================================

print("\n" + "="*70)
print("FINAL COMPARISON SUMMARY")
print("="*70)

print(f"\n📊 Audio Generation:")
print(f"  PyTorch: {pt_duration:.2f}s in {pt_time:.2f}s (RTF={pt_time/pt_duration:.3f}x)")
print(f"  MLX:     {mlx_duration:.2f}s in {mlx_time:.2f}s (RTF={mlx_time/mlx_duration:.3f}x)")

duration_diff = abs(pt_duration - mlx_duration)
print(f"\n  Duration match: {duration_diff:.3f}s difference", end="")
if duration_diff < 0.1:
    print(" ✅ Excellent")
elif duration_diff < 0.5:
    print(" ✓ Good")
else:
    print(" ⚠️  Significant")

print(f"\n⚡ Performance:")
if mlx_time < pt_time:
    speedup = pt_time / mlx_time
    improvement = (pt_time - mlx_time) / pt_time * 100
    print(f"  MLX is {speedup:.2f}x faster ({improvement:.1f}% improvement)")
else:
    slowdown = mlx_time / pt_time
    degradation = (mlx_time - pt_time) / pt_time * 100
    print(f"  MLX is {slowdown:.2f}x slower ({degradation:.1f}% slower)")

print(f"\n📁 Generated Files:")
print(f"  PyTorch: experiments/final_pytorch.wav")
print(f"  MLX:     experiments/final_mlx.wav")

print(f"\n✅ Please listen to both files and compare quality!")

print("\n" + "="*70)
print("Comparison Complete!")
print("="*70 + "\n")
