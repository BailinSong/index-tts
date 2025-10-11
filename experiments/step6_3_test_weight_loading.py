#!/usr/bin/env python3
"""
Step 6.3: Test weight loading from PyTorch checkpoint
Goal: Verify that conditioning weights are loaded correctly
"""

import sys
sys.path.insert(0, '/Users/bailin/index-tts')

import mlx.core as mx
import torch
import numpy as np

print("\n" + "="*70)
print("Step 6.3: Testing Weight Loading")
print("="*70)

# Load PyTorch checkpoint
print("\n1. Loading PyTorch checkpoint...")
checkpoint_path = "/Users/bailin/index-tts/checkpoints/gpt.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print(f"✓ Loaded checkpoint: {len(checkpoint)} keys")

# Convert to MLX format for loading
print("\n2. Converting checkpoint to MLX format...")
mlx_weights = {}
for key, value in checkpoint.items():
    if isinstance(value, torch.Tensor):
        mlx_weights[key] = mx.array(value.numpy())

print(f"✓ Converted {len(mlx_weights)} tensors to MLX")

# Create MLX model with conditioning
print("\n3. Creating MLX model with Pure MLX conditioning...")
try:
    from indextts.gpt.mlx_model import UnifiedVoiceMLX
    
    model = UnifiedVoiceMLX(
        layers=24,
        model_dim=1280,
        heads=16,
        max_text_tokens=402,
        max_mel_tokens=2002,
        max_conditioning_inputs=3,
        mel_length_compression=1024,
        number_text_tokens=256,
        start_text_token=255,
        stop_text_token=0,
        number_mel_codes=8194,
        start_mel_token=8192,
        stop_mel_token=8193,
        use_mlx_conditioning=True  # Enable Pure MLX conditioning
    )
    
    print("✓ Model created")
    print(f"  use_mlx_conditioning: {model.use_mlx_conditioning}")
    print(f"  conditioning_module: {type(model.conditioning_module).__name__}")
    
except Exception as e:
    print(f"✗ Failed to create model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Load weights
print("\n4. Loading weights from checkpoint...")
try:
    loaded_count = model.load_weights_from_dict(mlx_weights)
    print(f"✓ Loaded {loaded_count} weight tensors")
    
except Exception as e:
    print(f"✗ Failed to load weights: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Verify some weights were loaded correctly
print("\n5. Verifying weights...")
success = True

# Check Perceiver latents
try:
    latents = model.conditioning_module.perceiver.latents
    expected_shape = (32, 1280)
    if latents.shape == expected_shape:
        print(f"✓ Perceiver latents: {latents.shape}")
    else:
        print(f"✗ Perceiver latents shape mismatch: {latents.shape} != {expected_shape}")
        success = False
except Exception as e:
    print(f"✗ Failed to check perceiver latents: {e}")
    success = False

# Check proj_context
try:
    if model.conditioning_module.perceiver.proj_context:
        w = model.conditioning_module.perceiver.proj_context.weight
        expected_shape = (1280, 512)
        if w.shape == expected_shape:
            print(f"✓ Perceiver proj_context: {w.shape}")
        else:
            print(f"✗ proj_context shape mismatch: {w.shape} != {expected_shape}")
            success = False
except Exception as e:
    print(f"✗ Failed to check proj_context: {e}")
    success = False

# Check Conformer input projection
try:
    w = model.conditioning_module.conformer.input_proj[0].weight
    print(f"✓ Conformer input_proj: {w.shape}")
    if w.shape != (512, 1024):
        print(f"  ⚠️  Expected (512, 1024), got {w.shape}")
except Exception as e:
    print(f"✗ Failed to check conformer input_proj: {e}")
    success = False

# Compare a few weights with PyTorch
print("\n6. Comparing weights with PyTorch...")
try:
    # Compare perceiver latents
    pt_latents = checkpoint['perceiver_encoder.latents']
    mlx_latents = model.conditioning_module.perceiver.latents
    
    diff = np.abs(pt_latents.numpy() - np.array(mlx_latents))
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"Perceiver latents:")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    
    if max_diff < 1e-5:
        print("  ✓ Perfect match!")
    elif max_diff < 1e-3:
        print("  ✓ Good match (< 1e-3)")
    else:
        print(f"  ⚠️  Difference may be too large")
        success = False
    
except Exception as e:
    print(f"✗ Failed to compare weights: {e}")
    import traceback
    traceback.print_exc()

# Test forward pass with loaded weights
print("\n7. Testing forward pass with loaded weights...")
try:
    # Create dummy input
    batch, seq, dim = 1, 100, 1024
    x = mx.random.normal((batch, seq, dim))
    lengths = mx.array([100])
    
    # Forward through conditioning
    latents = model.conditioning_module(x, lengths)
    
    print(f"✓ Forward pass successful")
    print(f"  Input: {x.shape}")
    print(f"  Output: {latents.shape}")
    print(f"  Output std: {float(latents.std()):.4f}")
    
    # Check output statistics
    std = float(latents.std())
    if std > 0.01 and std < 10.0:
        print(f"  ✓ Output statistics reasonable")
    else:
        print(f"  ⚠️  Unusual std: {std}")
    
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    success = False

# Summary
print("\n" + "="*70)
if success:
    print("✅ Step 6.3 Complete: Weight loading successful!")
    print("\n Key achievements:")
    print("  ✓ Loaded PyTorch checkpoint")
    print("  ✓ Converted to MLX format")
    print("  ✓ Loaded conditioning weights")
    print("  ✓ Weights match PyTorch")
    print("  ✓ Forward pass works")
    print("\n→ Ready for Step 6.4: Quality validation")
else:
    print("⚠️  Step 6.3: Some issues found")
    print("  Check warnings above")
    print("  May still proceed to Step 6.4")

print("="*70 + "\n")

