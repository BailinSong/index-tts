#!/usr/bin/env python3
"""
测试 MLX Conv1d 实现的具体问题
"""

import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np

def test_mlx_conv1d_implementation():
    """测试 MLX Conv1d 实现的具体问题"""
    
    # 使用更简单的参数进行测试
    in_channels = 2
    out_channels = 4
    kernel_size = 3
    dilation = 1
    
    print(f"测试参数:")
    print(f"  in_channels: {in_channels}")
    print(f"  out_channels: {out_channels}")
    print(f"  kernel_size: {kernel_size}")
    print(f"  dilation: {dilation}")
    
    # 创建简单的测试数据
    batch_size, seq_len = 1, 5
    
    # PyTorch 格式: (batch, channels, seq_len)
    x_pytorch = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0],
                              [0.5, 1.5, 2.5, 3.5, 4.5]]], dtype=torch.float32)
    
    # MLX 格式: (batch, seq_len, channels)
    x_mlx = mx.array(x_pytorch.numpy().transpose(0, 2, 1))
    
    print(f"\n输入数据:")
    print(f"  PyTorch: {x_pytorch}")
    print(f"  MLX: {x_mlx}")
    
    # 创建简单的权重
    torch_weight = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # 输出通道 0
                                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # 输出通道 1
                                [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],  # 输出通道 2
                                [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]], dtype=torch.float32)  # 输出通道 3
    
    torch_bias = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    
    print(f"\n权重形状:")
    print(f"  PyTorch weight: {torch_weight.shape}")
    print(f"  PyTorch bias: {torch_bias.shape}")
    
    # 转换为 MLX 格式
    mlx_weight = np.transpose(torch_weight.numpy(), (0, 2, 1))  # (O, I, K) -> (O, K, I)
    mlx_bias = torch_bias.numpy()
    
    print(f"  MLX weight: {mlx_weight.shape}")
    print(f"  MLX bias: {mlx_bias.shape}")
    
    # PyTorch 卷积
    torch_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, 
                                dilation=dilation, bias=True, padding=1)
    torch_conv.weight.data = torch_weight
    torch_conv.bias.data = torch_bias
    
    torch_output = torch_conv(x_pytorch)
    
    print(f"\nPyTorch 输出:")
    print(f"  形状: {torch_output.shape}")
    print(f"  输出: {torch_output[0]}")
    
    # MLX 卷积
    mlx_conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                        dilation=dilation, bias=True, padding=1)
    mlx_conv.weight = mx.array(mlx_weight)
    mlx_conv.bias = mx.array(mlx_bias)
    
    mlx_output = mlx_conv(x_mlx)
    
    print(f"\nMLX 输出:")
    print(f"  形状: {mlx_output.shape}")
    print(f"  输出: {mlx_output[0]}")
    
    # 比较结果
    mlx_output_torch_format = torch.from_numpy(np.array(mlx_output).transpose(0, 2, 1))
    diff = torch.abs(torch_output - mlx_output_torch_format)
    
    print(f"\n差异分析:")
    print(f"  最大差异: {diff.max().item():.6f}")
    print(f"  平均差异: {diff.mean().item():.6f}")
    print(f"  形状匹配: {torch_output.shape == mlx_output_torch_format.shape}")
    
    if diff.max().item() > 1e-6:
        print(f"  ❌ 存在差异!")
        print(f"  详细差异:")
        print(f"    PyTorch: {torch_output[0]}")
        print(f"    MLX:     {mlx_output_torch_format[0]}")
        print(f"    差异:    {diff[0]}")
    else:
        print(f"  ✅ 结果一致!")
    
    # 测试手动实现的卷积
    print(f"\n手动实现卷积测试:")
    
    # PyTorch 手动实现
    def manual_conv1d_pytorch(x, weight, bias, padding=1):
        """手动实现 PyTorch Conv1d"""
        B, C, T = x.shape
        O, I, K = weight.shape
        
        # 应用 padding
        if padding > 0:
            x_padded = torch.nn.functional.pad(x, (padding, padding), mode='constant', value=0)
        else:
            x_padded = x
        
        # 计算输出长度
        out_len = x_padded.shape[2] - K + 1
        output = torch.zeros(B, O, out_len)
        
        for b in range(B):
            for o in range(O):
                for t in range(out_len):
                    for i in range(I):
                        for k in range(K):
                            output[b, o, t] += x_padded[b, i, t + k] * weight[o, i, k]
                output[b, o, :] += bias[o]
        
        return output
    
    torch_manual = manual_conv1d_pytorch(x_pytorch, torch_weight, torch_bias, padding=1)
    print(f"  PyTorch 手动实现: {torch_manual[0]}")
    
    # MLX 手动实现
    def manual_conv1d_mlx(x, weight, bias, padding=1):
        """手动实现 MLX Conv1d"""
        B, T, C = x.shape  # MLX 格式
        O, K, I = weight.shape  # MLX 格式
        
        # 应用 padding
        if padding > 0:
            x_padded = mx.pad(x, ((0, 0), (padding, padding), (0, 0)), mode='constant', constant_values=0)
        else:
            x_padded = x
        
        # 计算输出长度
        out_len = x_padded.shape[1] - K + 1
        output = mx.zeros((B, out_len, O))
        
        for b in range(B):
            for o in range(O):
                for t in range(out_len):
                    for i in range(I):
                        for k in range(K):
                            output = mx.concatenate([
                                output[:, :t, :],
                                mx.concatenate([
                                    output[:, t:t+1, :o],
                                    mx.array([[output[b, t, o] + x_padded[b, t + k, i] * weight[o, k, i]]]),
                                    output[:, t:t+1, o+1:]
                                ], axis=2),
                                output[:, t+1:, :]
                            ], axis=1)
                output = mx.concatenate([
                    output[:, :, :o],
                    output[:, :, o:o+1] + bias[o],
                    output[:, :, o+1:]
                ], axis=2)
        
        return output
    
    # 简化版本
    def simple_manual_conv1d_mlx(x, weight, bias):
        """简化的 MLX Conv1d 手动实现"""
        B, T, C = x.shape
        O, K, I = weight.shape
        
        # 不使用 padding 进行测试
        out_len = T - K + 1
        output = mx.zeros((B, out_len, O))
        
        for b in range(B):
            for o in range(O):
                for t in range(out_len):
                    val = 0.0
                    for i in range(I):
                        for k in range(K):
                            val += x[b, t + k, i] * weight[o, k, i]
                    output = mx.concatenate([
                        output[:, :t, :],
                        mx.concatenate([
                            output[:, t:t+1, :o],
                            mx.array([[val + bias[o]]]),
                            output[:, t:t+1, o+1:]
                        ], axis=2),
                        output[:, t+1:, :]
                    ], axis=1)
        
        return output
    
    # 测试无 padding 的情况
    x_mlx_no_pad = x_mlx[:, 1:-1, :]  # 去掉首尾，模拟无 padding
    mlx_manual = simple_manual_conv1d_mlx(x_mlx_no_pad, mx.array(mlx_weight), mx.array(mlx_bias))
    print(f"  MLX 手动实现: {mlx_manual[0]}")
    
    # 比较手动实现
    mlx_manual_torch_format = torch.from_numpy(np.array(mlx_manual).transpose(0, 2, 1))
    diff_manual = torch.abs(torch_manual - mlx_manual_torch_format)
    print(f"  手动实现差异: {diff_manual.max().item():.6f}")

if __name__ == "__main__":
    test_mlx_conv1d_implementation()