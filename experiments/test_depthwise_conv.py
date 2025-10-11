#!/usr/bin/env python3
"""
Step 2: Implement and test MLX Depthwise Separable Convolution
Goal: Create a working depthwise conv that matches PyTorch behavior
"""

import mlx.core as mx
import mlx.nn as nn
import torch
import numpy as np


class MLXDepthwiseConv1d(nn.Module):
    """
    Depthwise 1D Convolution in MLX.
    
    Each input channel is convolved with its own set of filters.
    Input: (batch, seq, channels)
    Output: (batch, seq_out, channels)
    """
    
    def __init__(self, channels: int, kernel_size: int, padding: int = 0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Weight: one kernel per channel
        # Shape: (channels, kernel_size)
        self.weight = mx.random.normal((channels, kernel_size)) * 0.02
        self.bias = mx.zeros((channels,))
    
    def __call__(self, x):
        """
        Args:
            x: (batch, seq, channels)
        
        Returns:
            out: (batch, seq_out, channels)
        """
        batch, seq, channels = x.shape
        
        # Apply padding if needed
        if self.padding > 0:
            # Pad along sequence dimension
            pad_config = [(0, 0), (self.padding, self.padding), (0, 0)]
            x = mx.pad(x, pad_config)
            seq = seq + 2 * self.padding
        
        # Output length
        seq_out = seq - self.kernel_size + 1
        
        # Method: Use sliding window and element-wise multiply
        # This is the most straightforward approach
        
        outputs = []
        for i in range(seq_out):
            # Extract window: (batch, kernel_size, channels)
            window = x[:, i:i+self.kernel_size, :]
            
            # Element-wise multiply with weights and sum over kernel dimension
            # window: (batch, kernel_size, channels)
            # weight: (channels, kernel_size) → broadcast to (1, kernel_size, channels)
            weight_broadcast = self.weight.T.reshape(1, self.kernel_size, channels)
            
            # Multiply and sum
            out_i = mx.sum(window * weight_broadcast, axis=1)  # (batch, channels)
            outputs.append(out_i)
        
        # Stack outputs
        out = mx.stack(outputs, axis=1)  # (batch, seq_out, channels)
        
        # Add bias
        out = out + self.bias
        
        return out


class MLXPointwiseConv1d(nn.Module):
    """Pointwise Conv1d is just a Linear layer."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
    
    def __call__(self, x):
        return self.linear(x)


class MLXDepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise Separable Convolution = Depthwise Conv + Pointwise Conv
    
    This is the standard mobile/efficient convolution pattern.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0
    ):
        super().__init__()
        
        # Step 1: Depthwise conv (each channel independently)
        self.depthwise = MLXDepthwiseConv1d(in_channels, kernel_size, padding)
        
        # Step 2: Pointwise conv (mix channels)
        self.pointwise = MLXPointwiseConv1d(in_channels, out_channels)
    
    def __call__(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


def test_depthwise_conv():
    """Test depthwise convolution matches PyTorch"""
    print("\n" + "="*60)
    print("Test 1: Depthwise Convolution")
    print("="*60)
    
    batch, channels, seq = 2, 8, 20
    kernel_size = 3
    padding = kernel_size // 2
    
    # Create same input
    np.random.seed(42)
    x_np = np.random.randn(batch, channels, seq).astype(np.float32)
    
    # PyTorch
    x_pt = torch.from_numpy(x_np)
    conv_pt = torch.nn.Conv1d(channels, channels, kernel_size, groups=channels, padding=padding, bias=True)
    
    # Get weights
    with torch.no_grad():
        # PyTorch depthwise weight: (out_channels, in_channels/groups, kernel_size)
        # For groups=channels: (channels, 1, kernel_size)
        pt_weight = conv_pt.weight.numpy()  # (8, 1, 3)
        pt_bias = conv_pt.bias.numpy()  # (8,)
    
    out_pt = conv_pt(x_pt)
    
    print(f"PyTorch input: {x_pt.shape}")
    print(f"PyTorch weight: {pt_weight.shape}")
    print(f"PyTorch output: {out_pt.shape}")
    
    # MLX
    x_mlx = mx.array(x_np.transpose(0, 2, 1))  # (batch, seq, channels)
    conv_mlx = MLXDepthwiseConv1d(channels, kernel_size, padding)
    
    # Copy weights from PyTorch
    # PyTorch: (channels, 1, kernel_size) → MLX: (channels, kernel_size)
    conv_mlx.weight = mx.array(pt_weight.squeeze(1))  # (8, 3)
    conv_mlx.bias = mx.array(pt_bias)
    
    out_mlx = conv_mlx(x_mlx)
    
    print(f"\nMLX input: {x_mlx.shape}")
    print(f"MLX weight: {conv_mlx.weight.shape}")
    print(f"MLX output: {out_mlx.shape}")
    
    # Compare outputs
    out_pt_np = out_pt.detach().numpy().transpose(0, 2, 1)  # (batch, seq, channels)
    out_mlx_np = np.array(out_mlx)
    
    diff = np.abs(out_pt_np - out_mlx_np)
    print(f"\n{'='*60}")
    print("Numerical Comparison:")
    print(f"{'='*60}")
    print(f"Max diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")
    print(f"Median diff: {np.median(diff):.6f}")
    
    if diff.max() < 1e-4:
        print("✅ PASS: Outputs match within tolerance")
        return True
    else:
        print("❌ FAIL: Outputs differ significantly")
        print(f"\nPyTorch output sample:\n{out_pt_np[0, :3, :3]}")
        print(f"\nMLX output sample:\n{out_mlx_np[0, :3, :3]}")
        return False


def test_depthwise_separable():
    """Test full depthwise separable convolution"""
    print("\n" + "="*60)
    print("Test 2: Depthwise Separable Convolution")
    print("="*60)
    
    batch, in_channels, out_channels, seq = 2, 16, 32, 20
    kernel_size = 3
    padding = kernel_size // 2
    
    # Create input
    np.random.seed(123)
    x_np = np.random.randn(batch, in_channels, seq).astype(np.float32)
    
    # PyTorch: Depthwise separable = Depthwise + Pointwise
    x_pt = torch.from_numpy(x_np)
    
    # Depthwise
    dw_pt = torch.nn.Conv1d(in_channels, in_channels, kernel_size, groups=in_channels, padding=padding)
    # Pointwise
    pw_pt = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    with torch.no_grad():
        out_pt = pw_pt(dw_pt(x_pt))
    
    print(f"PyTorch output: {out_pt.shape}")
    
    # MLX
    x_mlx = mx.array(x_np.transpose(0, 2, 1))
    conv_mlx = MLXDepthwiseSeparableConv1d(in_channels, out_channels, kernel_size, padding)
    
    # Copy weights
    with torch.no_grad():
        conv_mlx.depthwise.weight = mx.array(dw_pt.weight.numpy().squeeze(1))
        conv_mlx.depthwise.bias = mx.array(dw_pt.bias.numpy())
        conv_mlx.pointwise.linear.weight = mx.array(pw_pt.weight.squeeze(-1).numpy())
        conv_mlx.pointwise.linear.bias = mx.array(pw_pt.bias.numpy())
    
    out_mlx = conv_mlx(x_mlx)
    
    print(f"MLX output: {out_mlx.shape}")
    
    # Compare
    out_pt_np = out_pt.detach().numpy().transpose(0, 2, 1)
    out_mlx_np = np.array(out_mlx)
    
    diff = np.abs(out_pt_np - out_mlx_np)
    print(f"\n{'='*60}")
    print("Numerical Comparison:")
    print(f"{'='*60}")
    print(f"Max diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")
    
    if diff.max() < 1e-4:
        print("✅ PASS: Depthwise separable matches PyTorch")
        return True
    else:
        print("❌ FAIL: Outputs differ")
        return False


def test_conformer_convolution_module():
    """Test the full Conformer convolution module with GLU"""
    print("\n" + "="*60)
    print("Test 3: Conformer Convolution Module")
    print("="*60)
    
    class MLXConformerConvModule(nn.Module):
        """Conformer Convolution Module with proper Conv1d"""
        
        def __init__(self, channels: int, kernel_size: int = 31):
            super().__init__()
            
            # Layer norm
            self.norm = nn.LayerNorm(channels)
            
            # Pointwise expansion (for GLU)
            self.pointwise1 = nn.Linear(channels, 2 * channels)
            
            # Depthwise convolution
            padding = kernel_size // 2
            self.depthwise = MLXDepthwiseConv1d(channels, kernel_size, padding)
            
            # Batch norm (use LayerNorm in MLX)
            self.bn = nn.LayerNorm(channels)
            
            # Pointwise projection
            self.pointwise2 = nn.Linear(channels, channels)
        
        def __call__(self, x):
            # x: (batch, seq, channels)
            
            # Norm
            x = self.norm(x)
            
            # Pointwise expansion
            x = self.pointwise1(x)  # (batch, seq, 2*channels)
            
            # GLU: split and gate
            x1, x2 = mx.split(x, 2, axis=-1)
            x = x1 * nn.sigmoid(x2)  # (batch, seq, channels)
            
            # Depthwise conv
            x = self.depthwise(x)
            
            # Batch norm
            x = self.bn(x)
            
            # Swish activation
            x = x * nn.sigmoid(x)
            
            # Pointwise projection
            x = self.pointwise2(x)
            
            return x
    
    # Test
    batch, seq, channels = 2, 100, 256
    kernel_size = 31
    
    x_mlx = mx.random.normal((batch, seq, channels))
    
    conv_module = MLXConformerConvModule(channels, kernel_size)
    out = conv_module(x_mlx)
    
    print(f"Input: {x_mlx.shape}")
    print(f"Output: {out.shape}")
    print(f"Output stats: mean={float(out.mean()):.4f}, std={float(out.std()):.4f}")
    
    # Check shape
    if out.shape == x_mlx.shape:
        print("✅ PASS: Output shape matches input (residual-friendly)")
        return True
    else:
        print("❌ FAIL: Shape mismatch")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Step 2: MLX Depthwise Separable Convolution Implementation")
    print("="*70)
    
    results = []
    
    try:
        results.append(("Depthwise Conv", test_depthwise_conv()))
    except Exception as e:
        print(f"\n❌ Test 1 crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Depthwise Conv", False))
    
    try:
        results.append(("Depthwise Separable", test_depthwise_separable()))
    except Exception as e:
        print(f"\n❌ Test 2 crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Depthwise Separable", False))
    
    try:
        results.append(("Conformer Conv Module", test_conformer_convolution_module()))
    except Exception as e:
        print(f"\n❌ Test 3 crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Conformer Conv Module", False))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + "="*70)
    if all_passed:
        print("🎉 All tests passed! Ready for Step 3.")
    else:
        print("⚠️  Some tests failed. Review and debug.")
    print("="*70 + "\n")

