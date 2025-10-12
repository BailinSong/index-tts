#!/usr/bin/env python3
"""
Debug MLX Conv1d input format
"""

import mlx.core as mx
import mlx.nn as nn

print("=" * 60)
print("MLX Conv1d Input Format Test")
print("=" * 60)

# Test parameters
batch = 2
in_channels = 64
out_channels = 128
length = 100
kernel_size = 3

print(f"\nTest config:")
print(f"   batch={batch}, in_channels={in_channels}, out_channels={out_channels}")
print(f"   length={length}, kernel_size={kernel_size}")

# Create MLX Conv1d
mlx_conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
print(f"\nMLX Conv1d weight shape: {mlx_conv.weight.shape}")

# Test different input formats
print("\n>> Testing different input formats:")

# Format 1: (batch, channels, length) - PyTorch style
print("\n   Format 1: (batch, channels, length) = (2, 64, 100)")
try:
    input1 = mx.random.normal((batch, in_channels, length))
    output1 = mlx_conv(input1)
    print(f"   ✅ SUCCESS! Output shape: {output1.shape}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Format 2: (batch, length, channels) - Alternative
print("\n   Format 2: (batch, length, channels) = (2, 100, 64)")
try:
    input2 = mx.random.normal((batch, length, in_channels))
    output2 = mlx_conv(input2)
    print(f"   ✅ SUCCESS! Output shape: {output2.shape}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

print("\n" + "=" * 60)

