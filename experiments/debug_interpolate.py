#!/usr/bin/env python3
"""
Debug interpolate function
"""

import torch
import torch.nn.functional as F
import mlx.core as mx
import numpy as np
import sys
sys.path.insert(0, '/Users/bailin/index-tts')

from indextts.s2mel.mlx_modules.length_regulator import mlx_interpolate_nearest_1d


def test_interpolate():
    print("=" * 60)
    print("Interpolate Function Test")
    print("=" * 60)
    
    # Test parameters
    batch, channels, src_len, tgt_len = 1, 64, 100, 172
    
    # PyTorch interpolate
    pt_input = torch.randn(batch, channels, src_len)
    pt_output = F.interpolate(pt_input, size=tgt_len, mode='nearest')
    
    print(f"\nPyTorch F.interpolate:")
    print(f"   Input shape: {pt_input.shape}")
    print(f"   Output shape: {pt_output.shape}")
    
    # MLX interpolate
    mlx_input = mx.array(pt_input.numpy())
    mlx_output = mlx_interpolate_nearest_1d(mlx_input, tgt_len)
    
    print(f"\nMLX interpolate:")
    print(f"   Input shape: {mlx_input.shape}")
    print(f"   Output shape: {mlx_output.shape}")
    
    # Compare
    pt_np = pt_output.detach().cpu().numpy()
    mlx_np = np.array(mlx_output)
    
    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    max_diff = np.max(np.abs(pt_np - mlx_np))
    
    print(f"\nComparison:")
    print(f"   Correlation: {corr:.6f}")
    print(f"   Max difference: {max_diff:.6e}")
    
    if corr > 0.99:
        print(f"   ✅ SUCCESS!")
    else:
        print(f"   ❌ FAILED")
        # Show first few values
        print(f"\n   First 10 values comparison:")
        print(f"   PyTorch: {pt_np[0, 0, :10]}")
        print(f"   MLX:     {mlx_np[0, 0, :10]}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    test_interpolate()

