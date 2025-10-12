#!/usr/bin/env python3
"""
Debug interpolate indices
"""

import torch
import torch.nn.functional as F
import mlx.core as mx
import numpy as np

print("=" * 60)
print("Interpolate Indices Debug")
print("=" * 60)

# Simple test
src_len, tgt_len = 10, 17  # ratio = 1.7

# Create a simple input with indices as values
pt_input = torch.arange(src_len, dtype=torch.float32).view(1, 1, src_len)
print(f"\nPyTorch input (as indices): {pt_input.squeeze().numpy()}")

# PyTorch interpolate
pt_output = F.interpolate(pt_input, size=tgt_len, mode='nearest')
print(f"PyTorch output: {pt_output.squeeze().numpy()}")

# Manually compute indices using PyTorch's formula
# https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
# For nearest mode:
# out[i] = in[floor((i + 0.5) * (in_len / out_len))]

scale = src_len / tgt_len
print(f"\nScale: {scale}")

out_positions = np.arange(tgt_len)
in_positions_formula1 = np.floor((out_positions + 0.5) * scale).astype(int)
print(f"\nFormula 1: floor((i + 0.5) * scale)")
print(f"Indices: {in_positions_formula1}")

# Alternative formula
in_positions_formula2 = np.round(out_positions * scale).astype(int)
print(f"\nFormula 2: round(i * scale)")
print(f"Indices: {in_positions_formula2}")

# Another formula
in_positions_formula3 = np.floor(out_positions * scale + 0.5).astype(int)
print(f"\nFormula 3: floor(i * scale + 0.5)")
print(f"Indices: {in_positions_formula3}")

print("\nCompare with PyTorch output:")
pt_indices = pt_output.squeeze().numpy().astype(int)
print(f"PyTorch indices: {pt_indices}")

if np.array_equal(pt_indices, in_positions_formula1):
    print("✅ Formula 1 matches!")
elif np.array_equal(pt_indices, in_positions_formula2):
    print("✅ Formula 2 matches!")
elif np.array_equal(pt_indices, in_positions_formula3):
    print("✅ Formula 3 matches!")
else:
    print("❌ None of the formulas match!")

print("\n" + "=" * 60)

