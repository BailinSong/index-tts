#!/usr/bin/env python3
"""
Step 1.1: Test basic MLX Conv1d operations
Goal: Understand input/output format and weight layout
"""

import mlx.core as mx
import mlx.nn as nn
import torch
import numpy as np


def test_1_basic_conv1d():
    """Test 1: Basic Conv1d - understand input/output format"""
    print("\n" + "="*60)
    print("Test 1: Basic Conv1d")
    print("="*60)
    
    batch, channels_in, seq_len = 1, 64, 100
    channels_out = 128
    kernel_size = 3
    
    print(f"\nConfig: in_channels={channels_in}, out_channels={channels_out}, kernel={kernel_size}")
    print(f"Input: batch={batch}, channels={channels_in}, seq={seq_len}")
    
    # PyTorch expects: (batch, channels, seq)
    x_pt = torch.randn(batch, channels_in, seq_len)
    print(f"\nPyTorch input shape: {x_pt.shape}")
    
    conv_pt = torch.nn.Conv1d(channels_in, channels_out, kernel_size, padding=kernel_size//2)
    out_pt = conv_pt(x_pt)
    print(f"PyTorch output shape: {out_pt.shape}")
    print(f"PyTorch weight shape: {conv_pt.weight.shape}")  # (out, in, kernel)
    
    # MLX test - try different input formats
    print("\n" + "-"*60)
    print("MLX Test:")
    
    # Try format 1: (batch, seq, channels) - NLC format
    try:
        x_mlx_nlc = mx.array(x_pt.permute(0, 2, 1).numpy())  # (1, 100, 64)
        print(f"\nTrying NLC format: {x_mlx_nlc.shape}")
        
        conv_mlx = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, kernel_size=kernel_size)
        out_mlx = conv_mlx(x_mlx_nlc)
        print(f"✓ MLX output shape (NLC input): {out_mlx.shape}")
        print(f"  MLX weight shape: {conv_mlx.weight.shape}")
        
    except Exception as e:
        print(f"✗ NLC format failed: {e}")
    
    # Try format 2: (batch, channels, seq) - NCL format (like PyTorch)
    try:
        x_mlx_ncl = mx.array(x_pt.numpy())  # (1, 64, 100)
        print(f"\nTrying NCL format: {x_mlx_ncl.shape}")
        
        conv_mlx2 = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, kernel_size=kernel_size)
        out_mlx2 = conv_mlx2(x_mlx_ncl)
        print(f"✓ MLX output shape (NCL input): {out_mlx2.shape}")
        
    except Exception as e:
        print(f"✗ NCL format failed: {e}")


def test_2_depthwise_conv():
    """Test 2: Depthwise convolution (groups=channels)"""
    print("\n" + "="*60)
    print("Test 2: Depthwise Convolution")
    print("="*60)
    
    batch, channels, seq_len = 1, 64, 100
    kernel_size = 31
    
    print(f"\nDepthwise: channels={channels}, groups={channels}, kernel={kernel_size}")
    
    # PyTorch depthwise
    x_pt = torch.randn(batch, channels, seq_len)
    conv_pt = torch.nn.Conv1d(channels, channels, kernel_size, groups=channels, padding=kernel_size//2)
    out_pt = conv_pt(x_pt)
    print(f"PyTorch output: {out_pt.shape}")
    print(f"PyTorch weight: {conv_pt.weight.shape}")  # Should be (channels, 1, kernel)
    
    # MLX depthwise
    try:
        x_mlx = mx.array(x_pt.permute(0, 2, 1).numpy())  # Try NLC
        print(f"\nMLX input (NLC): {x_mlx.shape}")
        
        # MLX Conv1d with groups parameter
        conv_mlx = nn.Conv1d(channels, channels, kernel_size)
        out_mlx = conv_mlx(x_mlx)
        print(f"✓ MLX output: {out_mlx.shape}")
        print(f"  MLX weight: {conv_mlx.weight.shape}")
        
    except Exception as e:
        print(f"✗ MLX depthwise failed: {e}")


def test_3_pointwise_conv():
    """Test 3: Pointwise convolution (kernel_size=1)"""
    print("\n" + "="*60)
    print("Test 3: Pointwise Convolution (1x1)")
    print("="*60)
    
    batch, channels_in, seq_len = 1, 64, 100
    channels_out = 128
    
    print(f"\nPointwise: {channels_in} -> {channels_out}, kernel=1")
    
    # PyTorch
    x_pt = torch.randn(batch, channels_in, seq_len)
    conv_pt = torch.nn.Conv1d(channels_in, channels_out, kernel_size=1)
    out_pt = conv_pt(x_pt)
    print(f"PyTorch output: {out_pt.shape}")
    
    # MLX
    try:
        x_mlx = mx.array(x_pt.permute(0, 2, 1).numpy())
        print(f"\nMLX input: {x_mlx.shape}")
        
        conv_mlx = nn.Conv1d(channels_in, channels_out, kernel_size=1)
        out_mlx = conv_mlx(x_mlx)
        print(f"✓ MLX output: {out_mlx.shape}")
        
        # This is equivalent to Linear layer
        print("\nCompare with Linear layer:")
        linear_mlx = nn.Linear(channels_in, channels_out)
        out_linear = linear_mlx(x_mlx)
        print(f"  Linear output: {out_linear.shape}")
        print(f"  → Pointwise Conv1d ≈ Linear for kernel_size=1")
        
    except Exception as e:
        print(f"✗ MLX pointwise failed: {e}")


def test_4_weight_copying():
    """Test 4: Copy weights from PyTorch to MLX"""
    print("\n" + "="*60)
    print("Test 4: Weight Transfer PyTorch → MLX")
    print("="*60)
    
    batch, channels_in, seq_len = 2, 32, 50
    channels_out = 64
    kernel_size = 3
    
    print(f"\nConfig: {channels_in} -> {channels_out}, kernel={kernel_size}")
    
    # Create same input
    x_np = np.random.randn(batch, channels_in, seq_len).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_mlx = mx.array(x_np.transpose(0, 2, 1))  # (batch, seq, channels)
    
    print(f"PyTorch input: {x_pt.shape}")
    print(f"MLX input: {x_mlx.shape}")
    
    # PyTorch conv
    conv_pt = torch.nn.Conv1d(channels_in, channels_out, kernel_size, padding=kernel_size//2, bias=True)
    out_pt = conv_pt(x_pt)
    
    print(f"\nPyTorch weight shape: {conv_pt.weight.shape}")  # (out, in, kernel)
    print(f"PyTorch bias shape: {conv_pt.bias.shape}")
    print(f"PyTorch output: {out_pt.shape}")
    
    # MLX conv
    conv_mlx = nn.Conv1d(channels_in, channels_out, kernel_size)
    print(f"\nMLX weight shape: {conv_mlx.weight.shape}")
    
    # Try to copy weights - need to understand the format
    try:
        # MLX Conv1d weight format might be different
        # Let's check what works
        print("\n" + "-"*40)
        print("Attempting weight transfer...")
        
        # Get PyTorch weights
        pt_weight = conv_pt.weight.detach().numpy()  # (out, in, kernel)
        pt_bias = conv_pt.bias.detach().numpy()  # (out,)
        
        # Try different permutations
        print(f"PT weight: {pt_weight.shape}")
        
        # The question is: what format does MLX expect?
        # We'll test by checking the output
        
        out_mlx = conv_mlx(x_mlx)
        print(f"MLX output: {out_mlx.shape}")
        
        print("\n→ Need to determine correct weight format for MLX Conv1d")
        
    except Exception as e:
        print(f"✗ Weight transfer failed: {e}")
        import traceback
        traceback.print_exc()


def test_5_numerical_comparison():
    """Test 5: Numerical comparison with weight sharing"""
    print("\n" + "="*60)
    print("Test 5: Numerical Comparison")
    print("="*60)
    
    # Small test case
    batch, channels_in, seq_len = 1, 4, 10
    channels_out = 8
    kernel_size = 3
    
    # Fixed random seed
    np.random.seed(42)
    x_np = np.random.randn(batch, channels_in, seq_len).astype(np.float32)
    
    # PyTorch
    torch.manual_seed(42)
    x_pt = torch.from_numpy(x_np)
    conv_pt = torch.nn.Conv1d(channels_in, channels_out, kernel_size, padding=1, bias=False)
    out_pt = conv_pt(x_pt)
    
    # MLX - try to match PyTorch output
    try:
        x_mlx = mx.array(x_np.transpose(0, 2, 1))  # (1, 10, 4)
        conv_mlx = nn.Conv1d(channels_in, channels_out, kernel_size)
        
        # Copy weights (if we figure out the right format)
        # For now, just run and see the shapes
        out_mlx = conv_mlx(x_mlx)
        
        print(f"PyTorch output: {out_pt.shape}, mean={out_pt.mean():.4f}, std={out_pt.std():.4f}")
        print(f"MLX output: {out_mlx.shape}, mean={float(out_mlx.mean()):.4f}, std={float(out_mlx.std()):.4f}")
        
    except Exception as e:
        print(f"✗ Comparison failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MLX Conv1d API Research - Step 1")
    print("="*60)
    print("\nGoal: Understand MLX Conv1d input/output format and weight layout")
    print("Reference: PyTorch Conv1d uses (batch, channels, seq)")
    
    try:
        test_1_basic_conv1d()
    except Exception as e:
        print(f"\nTest 1 crashed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_2_depthwise_conv()
    except Exception as e:
        print(f"\nTest 2 crashed: {e}")
    
    try:
        test_3_pointwise_conv()
    except Exception as e:
        print(f"\nTest 3 crashed: {e}")
    
    try:
        test_4_weight_copying()
    except Exception as e:
        print(f"\nTest 4 crashed: {e}")
    
    try:
        test_5_numerical_comparison()
    except Exception as e:
        print(f"\nTest 5 crashed: {e}")
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print("✓ Tests completed - check results above")
    print("→ Next: Analyze results and document findings")
    print("="*60 + "\n")

