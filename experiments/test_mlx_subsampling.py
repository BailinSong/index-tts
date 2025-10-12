#!/usr/bin/env python3
"""
测试 MLX Conv2d Subsampling 的形状和基本功能
"""

import sys
sys.path.insert(0, '/Users/bailin/index-tts')

import mlx.core as mx
import numpy as np

print("\n" + "="*70)
print("测试 MLX Conv2d Subsampling")
print("="*70)

# Test Conv2d
print("\n1. 测试 MLX Conv2d...")
from indextts.gpt.mlx_subsampling import MLXConv2d

conv = MLXConv2d(in_channels=1, out_channels=512, kernel_size=3, stride=2)

# Input: (batch=1, height=121, width=1024, channels=1)
x = mx.random.normal((1, 121, 1024, 1))
print(f"  Input shape: {x.shape}")

out = conv(x)
print(f"  Output shape: {out.shape}")
print(f"  Expected: (1, 60, 512, 512) approximately")

# Test Conv2dSubsampling2
print("\n2. 测试 MLXConv2dSubsampling2Fixed...")
from indextts.gpt.mlx_subsampling import MLXConv2dSubsampling2Fixed

subsampling = MLXConv2dSubsampling2Fixed(idim=1024, odim=512)

# Input: (batch=1, time=121, idim=1024)
x = mx.random.normal((1, 121, 1024))
print(f"  Input shape: {x.shape}")

out = subsampling(x)
print(f"  Output shape: {out.shape}")
print(f"  Expected: (1, 60, 512) ✅" if out.shape == (1, 60, 512) else f"  Expected: (1, 60, 512) ❌")

# Test with MLX Conformer Encoder
print("\n3. 测试完整的 MLX Conformer Encoder...")
from indextts.gpt.mlx_conditioning import MLXConformerEncoder

encoder = MLXConformerEncoder(
    input_dim=1024,
    output_dim=512,
    num_layers=6
)

x = mx.random.normal((1, 121, 1024))
lengths = mx.array([121])

print(f"  Input shape: {x.shape}")
out, mask = encoder(x, lengths)
print(f"  Output shape: {out.shape}")
print(f"  Expected: (1, 60, 512) ✅" if out.shape == (1, 60, 512) else f"  Expected: (1, 60, 512) ❌")

print("\n" + "="*70)
print("✅ 形状测试完成！")
print("="*70 + "\n")

