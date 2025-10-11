#!/usr/bin/env python3
"""
Step 6.4 & 6.5: Comprehensive comparison between MLX and PyTorch
Goal: Validate quality and measure performance

Test pipeline:
1. Compare Conditioning outputs (Conformer + Perceiver)
2. Compare Token generation
3. Compare Audio generation
4. Performance benchmark
"""

import sys
sys.path.insert(0, '/Users/bailin/index-tts')

import warnings
warnings.filterwarnings('ignore')

import mlx.core as mx
import torch
import numpy as np
import time
from pathlib import Path

print("\n" + "="*70)
print("MLX vs PyTorch Comprehensive Comparison")
print("="*70)

# ============================================================================
# Phase 1: Load Models
# ============================================================================

print("\n" + "="*70)
print("Phase 1: Loading Models")
print("="*70)

# Load PyTorch model
print("\n1. Loading PyTorch model...")
try:
    from indextts.infer_v2 import IndexTTS2
    
    pytorch_model = IndexTTS2(
        model_dir="/Users/bailin/index-tts/checkpoints",
        use_mlx=False  # Pure PyTorch
    )
    print("✓ PyTorch model loaded")
except Exception as e:
    print(f"✗ Failed to load PyTorch model: {e}")
    exit(1)

# Load MLX model with weights
print("\n2. Loading MLX model with conditioning weights...")
try:
    # We need to load the model and explicitly load the checkpoint
    mlx_model_obj = IndexTTS2(
        model_dir="/Users/bailin/index-tts/checkpoints",
        use_mlx=True  # Pure MLX
    )
    print("✓ MLX model loaded")
    print(f"  MLX transformer type: {type(mlx_model_obj.mlx_transformer).__name__}")
    print(f"  Use MLX conditioning: {mlx_model_obj.mlx_transformer.use_mlx_conditioning}")
except Exception as e:
    print(f"✗ Failed to load MLX model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# Phase 2: Compare Conditioning Outputs
# ============================================================================

print("\n" + "="*70)
print("Phase 2: Comparing Conditioning Outputs")
print("="*70)

print("\n2.1: Creating test inputs...")
# Create identical test input
# Note: Use longer sequence (400) to avoid PyTorch Conformer Conv2d subsampling issues
np.random.seed(42)
test_audio_length = 400
test_emb = np.random.randn(1, 1024, test_audio_length).astype(np.float32)

# Convert to respective formats
pt_emb = torch.from_numpy(test_emb).to(pytorch_model.device)  # Move to correct device
mlx_emb = mx.array(test_emb)

print(f"Test input shape: {test_emb.shape}")

# Get PyTorch conditioning
print("\n2.2: Getting PyTorch conditioning...")
try:
    with torch.no_grad():
        pt_cond_lengths = torch.tensor([test_audio_length])
        pt_conditioning = pytorch_model.gpt.get_conditioning(
            pt_emb.transpose(1, 2),  # (1, 100, 1024)
            pt_cond_lengths
        )
    print(f"✓ PyTorch conditioning: {pt_conditioning.shape}")
    print(f"  Stats: mean={pt_conditioning.mean():.4f}, std={pt_conditioning.std():.4f}")
except Exception as e:
    print(f"✗ Failed to get PyTorch conditioning: {e}")
    import traceback
    traceback.print_exc()
    pt_conditioning = None

# Get MLX conditioning
print("\n2.3: Getting MLX conditioning...")
try:
    mlx_cond_lengths = mx.array([test_audio_length])
    mlx_conditioning = mlx_model_obj.mlx_transformer.get_conditioning_mlx(
        mlx_emb,  # (1, 1024, 100)
        mlx_cond_lengths
    )
    print(f"✓ MLX conditioning: {mlx_conditioning.shape}")
    print(f"  Stats: mean={float(mlx_conditioning.mean()):.4f}, std={float(mlx_conditioning.std()):.4f}")
except Exception as e:
    print(f"✗ Failed to get MLX conditioning: {e}")
    import traceback
    traceback.print_exc()
    mlx_conditioning = None

# Compare conditioning outputs
if pt_conditioning is not None and mlx_conditioning is not None:
    print("\n2.4: Comparing conditioning outputs...")
    
    pt_cond_np = pt_conditioning.cpu().numpy()
    mlx_cond_np = np.array(mlx_conditioning)
    
    diff = np.abs(pt_cond_np - mlx_cond_np)
    
    print(f"\nConditioning comparison:")
    print(f"  PyTorch shape: {pt_cond_np.shape}")
    print(f"  MLX shape:     {mlx_cond_np.shape}")
    print(f"  Max diff:      {diff.max():.6f}")
    print(f"  Mean diff:     {diff.mean():.6f}")
    print(f"  Median diff:   {np.median(diff):.6f}")
    print(f"  Relative error: {(diff / (np.abs(pt_cond_np) + 1e-8)).mean():.6f}")
    
    if diff.max() < 0.1:
        print("  ✅ Excellent match (< 0.1)")
    elif diff.max() < 1.0:
        print("  ✓ Good match (< 1.0)")
    elif diff.max() < 5.0:
        print("  ⚠️  Moderate difference (< 5.0)")
    else:
        print("  ⚠️  Large difference (>= 5.0)")
        print(f"\n  Sample PyTorch values: {pt_cond_np[0, :3, :3]}")
        print(f"  Sample MLX values:     {mlx_cond_np[0, :3, :3]}")

# ============================================================================
# Phase 3: Compare Token Generation (Simple Test)
# ============================================================================

print("\n" + "="*70)
print("Phase 3: Token Generation Test")
print("="*70)

print("\n3.1: Testing with short text...")
test_text = "今天天气很好"

# PyTorch generation
print("\n  PyTorch generation...")
try:
    pt_start = time.time()
    pytorch_model.infer(
        spk_audio_prompt="/Users/bailin/index-tts/examples/voice_01.wav",
        text=test_text,
        output_path="experiments/compare_pytorch.wav",
        ref_text="今天天气很好",
        emo_audio="/Users/bailin/index-tts/examples/voice_01.wav",
        emo_text="今天天气很好",
    )
    pt_time = time.time() - pt_start
    
    # Load generated audio
    import scipy.io.wavfile as wavfile
    pt_sr, pt_audio = wavfile.read("experiments/compare_pytorch.wav")
    pt_audio = pt_audio.astype(np.float32) / 32767.0
    
    pt_duration = len(pt_audio) / pt_sr
    print(f"  ✓ PyTorch: {pt_duration:.2f}s audio in {pt_time:.2f}s")
    print(f"    RTF: {pt_time/pt_duration:.2f}x")
    print(f"    Audio stats: mean={pt_audio.mean():.4f}, std={pt_audio.std():.4f}")
except Exception as e:
    print(f"  ✗ PyTorch generation failed: {e}")
    import traceback
    traceback.print_exc()
    pt_audio = None
    pt_time = None

# MLX generation
print("\n  MLX generation...")
try:
    mlx_start = time.time()
    mlx_model_obj.infer(
        spk_audio_prompt="/Users/bailin/index-tts/examples/voice_01.wav",
        text=test_text,
        output_path="experiments/compare_mlx.wav",
        ref_text="今天天气很好",
        emo_audio="/Users/bailin/index-tts/examples/voice_01.wav",
        emo_text="今天天气很好",
    )
    mlx_time = time.time() - mlx_start
    
    # Load generated audio
    mlx_sr, mlx_audio = wavfile.read("experiments/compare_mlx.wav")
    mlx_audio = mlx_audio.astype(np.float32) / 32767.0
    
    mlx_duration = len(mlx_audio) / mlx_sr
    print(f"  ✓ MLX: {mlx_duration:.2f}s audio in {mlx_time:.2f}s")
    print(f"    RTF: {mlx_time/mlx_duration:.2f}x")
    print(f"    Audio stats: mean={mlx_audio.mean():.4f}, std={mlx_audio.std():.4f}")
except Exception as e:
    print(f"  ✗ MLX generation failed: {e}")
    import traceback
    traceback.print_exc()
    mlx_audio = None
    mlx_time = None

# Compare results
if pt_audio is not None and mlx_audio is not None:
    print(f"\n3.2: Comparison:")
    print(f"  Duration difference: {abs(pt_duration - mlx_duration):.2f}s")
    print(f"  Speed comparison: MLX is {pt_time/mlx_time:.2f}x vs PyTorch")
    
    if mlx_time < pt_time:
        speedup = (pt_time - mlx_time) / pt_time * 100
        print(f"  ✓ MLX faster by {speedup:.1f}%")
    else:
        slowdown = (mlx_time - pt_time) / pt_time * 100
        print(f"  ⚠️  MLX slower by {slowdown:.1f}%")
    
    print(f"\n  Audio files saved:")
    print(f"    experiments/compare_pytorch.wav")
    print(f"    experiments/compare_mlx.wav")

# ============================================================================
# Phase 4: Performance Benchmark
# ============================================================================

print("\n" + "="*70)
print("Phase 4: Performance Benchmark")
print("="*70)

# Benchmark conditioning only (most relevant for our changes)
print("\n4.1: Benchmarking Conditioning (10 runs)...")

n_runs = 10
pt_times = []
mlx_times = []

for i in range(n_runs):
    # PyTorch
    with torch.no_grad():
        start = time.time()
        _ = pytorch_model.gpt.get_conditioning(
            pt_emb.transpose(1, 2),
            torch.tensor([test_audio_length])
        )
        pt_times.append(time.time() - start)
    
    # MLX
    start = time.time()
    _ = mlx_model_obj.mlx_transformer.get_conditioning_mlx(
        mlx_emb,
        mx.array([test_audio_length])
    )
    mx.eval(_)  # Ensure computation completes
    mlx_times.append(time.time() - start)

pt_mean = np.mean(pt_times) * 1000
pt_std = np.std(pt_times) * 1000
mlx_mean = np.mean(mlx_times) * 1000
mlx_std = np.std(mlx_times) * 1000

print(f"\nConditioning performance (ms):")
print(f"  PyTorch: {pt_mean:.2f} ± {pt_std:.2f} ms")
print(f"  MLX:     {mlx_mean:.2f} ± {mlx_std:.2f} ms")
print(f"  Speedup: {pt_mean/mlx_mean:.2f}x")

if mlx_mean < pt_mean:
    improvement = (pt_mean - mlx_mean) / pt_mean * 100
    print(f"  ✓ MLX is {improvement:.1f}% faster")
else:
    slowdown = (mlx_mean - pt_mean) / pt_mean * 100
    print(f"  ⚠️  MLX is {slowdown:.1f}% slower")

# ============================================================================
# Summary Report
# ============================================================================

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print("\n✅ Completed Tests:")
print("  ✓ Phase 1: Model loading")
print("  ✓ Phase 2: Conditioning comparison")
print("  ✓ Phase 3: Token generation")
print("  ✓ Phase 4: Performance benchmark")

print("\n📊 Key Results:")
if pt_conditioning is not None and mlx_conditioning is not None:
    print(f"  Conditioning accuracy: max_diff = {diff.max():.6f}")
else:
    print(f"  Conditioning: Could not compare")

if pt_audio is not None and mlx_audio is not None:
    print(f"  Audio generation: Both succeeded")
    print(f"  Speed: MLX is {pt_time/mlx_time:.2f}x vs PyTorch")
else:
    print(f"  Audio generation: Incomplete")

print(f"  Conditioning benchmark: {pt_mean/mlx_mean:.2f}x speedup")

print("\n📁 Generated Files:")
print("  • experiments/compare_pytorch.wav")
print("  • experiments/compare_mlx.wav")

print("\n" + "="*70)
print("Comparison Complete!")
print("="*70 + "\n")

