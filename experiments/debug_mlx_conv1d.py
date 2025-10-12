#!/usr/bin/env python3
"""
Debug MLX Conv1d weight format
"""

import mlx.core as mx
import mlx.nn as nn
import torch
import numpy as np

print("=" * 60)
print("MLX Conv1d Weight Format Test")
print("=" * 60)

# Test parameters
batch = 1
in_channels = 1024
out_channels = 1024
length = 172
kernel_size = 3

# Create PyTorch Conv1d
print("\n>> PyTorch Conv1d:")
pt_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
print(f"   Weight shape: {pt_conv.weight.shape}")  # (out, in, kernel)
print(f"   Expected: (out_channels={out_channels}, in_channels={in_channels}, kernel={kernel_size})")

# Create input
pt_input = torch.randn(batch, in_channels, length)
print(f"   Input shape: {pt_input.shape}")

# Forward
pt_output = pt_conv(pt_input)
print(f"   Output shape: {pt_output.shape}")

# Create MLX Conv1d
print("\n>> MLX Conv1d:")
mlx_conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
print(f"   Weight shape: {mlx_conv.weight.shape}")
print(f"   Expected format: ???")

# Try to convert PyTorch weights
print("\n>> Weight conversion attempts:")

# Attempt 1: Direct copy (out, in, kernel)
print("   Attempt 1: Direct copy (out, in, kernel)")
try:
    mlx_conv.weight = mx.array(pt_conv.weight.detach().cpu().numpy())
    mlx_input = mx.array(pt_input.numpy())
    mlx_output = mlx_conv(mlx_input)
    print(f"   ✅ SUCCESS! Output shape: {mlx_output.shape}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Attempt 2: Transpose to (out, kernel, in)
print("\n   Attempt 2: Transpose to (out, kernel, in)")
try:
    mlx_conv2 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
    pt_weight = pt_conv.weight.detach().cpu().numpy()  # (out, in, kernel)
    mlx_weight = np.transpose(pt_weight, (0, 2, 1))  # (out, kernel, in)
    mlx_conv2.weight = mx.array(mlx_weight)
    mlx_input = mx.array(pt_input.numpy())
    mlx_output = mlx_conv2(mlx_input)
    print(f"   ✅ SUCCESS! Output shape: {mlx_output.shape}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

print("\n" + "=" * 60)

