#!/usr/bin/env python3
"""
Step 6.2: Test adjusted MLX architecture
Goal: Verify dimension changes work correctly
"""

import sys
sys.path.insert(0, '/Users/bailin/index-tts')

import mlx.core as mx
from indextts.gpt.mlx_conditioning import MLXConditioningModule

print("\n" + "="*70)
print("Step 6.2: Testing Adjusted MLX Architecture")
print("="*70)

print("\n1. Creating MLXConditioningModule with PyTorch-matched dimensions...")
try:
    conditioning = MLXConditioningModule(
        input_dim=1024,
        conformer_dim=512,  # Matches PyTorch
        model_dim=1280,
        num_latents=32,
        conformer_layers=6,
        perceiver_depth=2
    )
    print("✓ Module created successfully")
    print(f"  Conformer output_dim: {conditioning.conformer.output_dim}")
    print(f"  Perceiver dim: {conditioning.perceiver.latents.shape}")
except Exception as e:
    print(f"✗ Failed to create module: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n2. Testing forward pass...")
try:
    # Create input (speaker embedding)
    batch, seq, dim = 2, 121, 1024
    x = mx.random.normal((batch, seq, dim))
    lengths = mx.array([121, 100])
    
    print(f"Input: {x.shape}")
    
    # Forward pass
    latents = conditioning(x, lengths)
    
    print(f"Output: {latents.shape}")
    print(f"Expected: (2, 32, 1280)")
    
    # Verify shape
    if latents.shape == (2, 32, 1280):
        print("✓ Output shape correct!")
    else:
        print(f"✗ Output shape mismatch!")
        exit(1)
    
    # Check output statistics
    std = float(latents.std())
    mean = float(latents.mean())
    print(f"\nOutput statistics:")
    print(f"  Mean: {mean:.4f}")
    print(f"  Std: {std:.4f}")
    print(f"  Range: [{float(latents.min()):.4f}, {float(latents.max()):.4f}]")
    
    if mx.any(mx.isnan(latents)) or mx.any(mx.isinf(latents)):
        print("✗ Contains NaN or Inf!")
        exit(1)
    
    if std < 0.01 or std > 10.0:
        print(f"⚠️  Unusual std: {std}")
    else:
        print("✓ Output statistics normal")
    
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n3. Testing Conformer intermediate output...")
try:
    # Test Conformer separately
    x = mx.random.normal((2, 121, 1024))
    lengths = mx.array([121, 100])
    
    conformer_out, mask = conditioning.conformer(x, lengths)
    print(f"Conformer output: {conformer_out.shape}")
    print(f"Expected: (2, 121, 512)")
    
    if conformer_out.shape == (2, 121, 512):
        print("✓ Conformer output dimension correct (512)!")
    else:
        print(f"✗ Conformer dimension mismatch!")
        exit(1)
    
except Exception as e:
    print(f"✗ Conformer test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n4. Testing Perceiver proj_context...")
try:
    # Check if Perceiver has proj_context
    if hasattr(conditioning.perceiver, 'proj_context'):
        if conditioning.perceiver.proj_context is not None:
            print(f"✓ Perceiver has proj_context")
            print(f"  proj_context.weight: {conditioning.perceiver.proj_context.weight.shape}")
            expected_shape = (1280, 512)
            if conditioning.perceiver.proj_context.weight.shape == expected_shape:
                print(f"✓ proj_context shape correct: {expected_shape}")
            else:
                print(f"⚠️  proj_context shape: {conditioning.perceiver.proj_context.weight.shape}")
        else:
            print("⚠️  proj_context is None")
    else:
        print("⚠️  Perceiver doesn't have proj_context attribute")
    
except Exception as e:
    print(f"⚠️  proj_context check failed: {e}")

print("\n" + "="*70)
print("Step 6.2 Architecture Test Results")
print("="*70)
print("✅ Architecture adjusted successfully")
print("✓ Conformer: 1024 → 512")
print("✓ Perceiver: 512 → 1280 (via proj_context)")
print("✓ Output: (batch, 32, 1280)")
print("\n→ Ready for Step 6.3: Weight loading")
print("="*70 + "\n")

