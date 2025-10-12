#!/usr/bin/env python3
"""
测试 Depthwise Conv 数值精度
"""
import sys
sys.path.insert(0, '/Users/bailin/index-tts')

import torch
import mlx.core as mx
import numpy as np

def main():
    print("=" * 80)
    print("🔍 测试 Depthwise Conv 数值精度")
    print("=" * 80)
    
    # Test parameters
    batch = 1
    seq_len = 20
    channels = 512
    kernel_size = 15
    padding = kernel_size // 2
    
    # Create random input and weights
    np.random.seed(42)
    torch.manual_seed(42)
    
    input_data = np.random.randn(batch, seq_len, channels).astype(np.float32)
    weights = np.random.randn(channels, kernel_size).astype(np.float32) * 0.01
    bias = np.random.randn(channels).astype(np.float32) * 0.01
    
    print(f"\n测试配置:")
    print(f"  Input: ({batch}, {seq_len}, {channels})")
    print(f"  Kernel size: {kernel_size}")
    print(f"  Padding: {padding}")
    
    # ========================================================================
    # PyTorch Implementation
    # ========================================================================
    print("\n" + "=" * 80)
    print("PyTorch Depthwise Conv1d")
    print("=" * 80)
    
    # PyTorch expects (batch, channels, seq_len)
    pt_input = torch.from_numpy(input_data).permute(0, 2, 1)  # (batch, channels, seq_len)
    pt_weight = torch.from_numpy(weights).unsqueeze(1)  # (channels, 1, kernel_size)
    pt_bias = torch.from_numpy(bias)
    
    pt_conv = torch.nn.Conv1d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        padding=padding,
        groups=channels,  # Depthwise
        bias=True
    )
    
    # Set weights
    pt_conv.weight.data = pt_weight
    pt_conv.bias.data = pt_bias
    
    with torch.no_grad():
        pt_output = pt_conv(pt_input)  # (batch, channels, seq_len)
        pt_output = pt_output.permute(0, 2, 1)  # (batch, seq_len, channels)
    
    print(f"\nPyTorch Output:")
    print(f"  Shape: {pt_output.shape}")
    print(f"  Mean: {pt_output.mean():.6f}, Std: {pt_output.std():.6f}")
    print(f"  Sample [0, 0, :5]: {pt_output[0, 0, :5].numpy()}")
    
    # ========================================================================
    # MLX Implementation (current)
    # ========================================================================
    print("\n" + "=" * 80)
    print("MLX Depthwise Conv (当前实现)")
    print("=" * 80)
    
    from indextts.gpt.mlx_conditioning import MLXDepthwiseConv1d
    
    mlx_conv = MLXDepthwiseConv1d(channels, kernel_size, padding)
    mlx_conv.weight = mx.array(weights)
    mlx_conv.bias = mx.array(bias)
    
    mlx_input = mx.array(input_data)
    mlx_output = mlx_conv(mlx_input)
    mlx_output_np = np.array(mlx_output)
    
    print(f"\nMLX Output:")
    print(f"  Shape: {mlx_output_np.shape}")
    print(f"  Mean: {mlx_output_np.mean():.6f}, Std: {mlx_output_np.std():.6f}")
    print(f"  Sample [0, 0, :5]: {mlx_output_np[0, 0, :5]}")
    
    # ========================================================================
    # Comparison
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 对比")
    print("=" * 80)
    
    pt_np = pt_output.numpy()
    diff = np.abs(pt_np - mlx_output_np)
    
    correlation = np.corrcoef(pt_np.flatten(), mlx_output_np.flatten())[0, 1]
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"\nCorrelation: {correlation:.6f}")
    print(f"Max diff: {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")
    
    # Find worst position
    worst_idx = np.unravel_index(diff.argmax(), diff.shape)
    print(f"\n最大差异位置: {worst_idx}")
    print(f"  PyTorch: {pt_np[worst_idx]:.6f}")
    print(f"  MLX: {mlx_output_np[worst_idx]:.6f}")
    print(f"  Diff: {diff[worst_idx]:.6f}")
    
    if correlation < 0.9999:
        print("\n❌ 实现有差异！")
        print("   需要检查 MLXDepthwiseConv1d 的实现")
    elif correlation < 0.99999:
        print("\n⚠️  有小的数值差异")
        print("   可能是浮点精度累积")
    else:
        print("\n✅ 实现完全一致！")
    
    # ========================================================================
    # Manual verification for one position
    # ========================================================================
    print("\n" + "=" * 80)
    print("🔍 手动验证（第一个位置，第一个通道）")
    print("=" * 80)
    
    # Channel 0, position 0
    channel_idx = 0
    pos_idx = 0
    
    # Get input for this channel (with padding)
    input_channel = input_data[0, :, channel_idx]  # (seq_len,)
    input_padded = np.pad(input_channel, (padding, padding), mode='constant')
    
    # Get kernel for this channel
    kernel_channel = weights[channel_idx, :]  # (kernel_size,)
    bias_channel = bias[channel_idx]
    
    # Manual convolution
    manual_result = 0.0
    for k in range(kernel_size):
        manual_result += input_padded[pos_idx + k] * kernel_channel[k]
    manual_result += bias_channel
    
    pt_result = pt_np[0, pos_idx, channel_idx]
    mlx_result = mlx_output_np[0, pos_idx, channel_idx]
    
    print(f"\nManual: {manual_result:.6f}")
    print(f"PyTorch: {pt_result:.6f}")
    print(f"MLX: {mlx_result:.6f}")
    print(f"PyTorch vs Manual: {abs(pt_result - manual_result):.6f}")
    print(f"MLX vs Manual: {abs(mlx_result - manual_result):.6f}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

