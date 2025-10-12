#!/usr/bin/env python3
"""
Debug MLX Length Regulator layer by layer
"""

import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sys
sys.path.insert(0, '/Users/bailin/index-tts')

from indextts.s2mel.modules.length_regulator import InterpolateRegulator


def test_conv1d_conversion():
    """Test Conv1d weight conversion in isolation"""
    print("=" * 60)
    print("Conv1d Weight Conversion Test")
    print("=" * 60)
    
    # Simple test
    batch, in_ch, out_ch, length, kernel = 1, 64, 128, 50, 3
    
    # PyTorch Conv1d
    pt_conv = torch.nn.Conv1d(in_ch, out_ch, kernel, padding=1)
    pt_input = torch.randn(batch, in_ch, length)
    pt_output = pt_conv(pt_input)
    
    print(f"\nPyTorch:")
    print(f"   Weight shape: {pt_conv.weight.shape}")  # (out, in, kernel)
    print(f"   Input shape: {pt_input.shape}")  # (batch, in, length)
    print(f"   Output shape: {pt_output.shape}")  # (batch, out, length)
    
    # MLX Conv1d - Direct weight copy test
    print(f"\nMLX Attempt 1: Direct weight copy (no transpose)")
    mlx_conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=1)
    # MLX weight format: (out, kernel, in)
    # PyTorch weight format: (out, in, kernel)
    # So we need: (out, in, kernel) -> (out, kernel, in)
    pt_weight_np = pt_conv.weight.detach().cpu().numpy()
    mlx_weight = np.transpose(pt_weight_np, (0, 2, 1))  # (out, kernel, in)
    mlx_conv1.weight = mx.array(mlx_weight)
    mlx_conv1.bias = mx.array(pt_conv.bias.detach().cpu().numpy())
    
    # MLX expects (batch, length, channels)
    mlx_input = mx.array(pt_input.permute(0, 2, 1).numpy())  # (batch, length, in)
    mlx_output1 = mlx_conv1(mlx_input)  # (batch, length, out)
    mlx_output1_np = np.array(mlx_output1)
    
    # Convert back to PyTorch format for comparison
    mlx_output1_pt_format = np.transpose(mlx_output1_np, (0, 2, 1))  # (batch, out, length)
    pt_output_np = pt_output.detach().cpu().numpy()
    
    corr1 = np.corrcoef(pt_output_np.flatten(), mlx_output1_pt_format.flatten())[0, 1]
    print(f"   MLX output shape: {mlx_output1.shape}")
    print(f"   Correlation: {corr1:.6f}")
    
    if corr1 > 0.99:
        print(f"   ✅ SUCCESS!")
    else:
        print(f"   ❌ FAILED")
        print(f"   Max diff: {np.max(np.abs(pt_output_np - mlx_output1_pt_format)):.6e}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    test_conv1d_conversion()

