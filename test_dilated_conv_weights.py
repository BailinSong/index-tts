#!/usr/bin/env python3
"""
测试扩张卷积的权重转换
"""

import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np

def test_dilated_conv_weights():
    """测试扩张卷积权重的转换"""
    
    # 模拟 WaveNet 第0层的参数
    in_channels = 512
    out_channels = 1024  # 2 * 512
    kernel_size = 5
    dilation = 1
    
    print(f"测试参数:")
    print(f"  in_channels: {in_channels}")
    print(f"  out_channels: {out_channels}")
    print(f"  kernel_size: {kernel_size}")
    print(f"  dilation: {dilation}")
    
    # 创建 PyTorch 权重 (模拟从检查点加载)
    torch_weight = torch.randn(out_channels, in_channels, kernel_size)
    torch_bias = torch.randn(out_channels)
    
    print(f"\nPyTorch 权重形状:")
    print(f"  weight: {torch_weight.shape}")
    print(f"  bias: {torch_bias.shape}")
    
    # 转换为 MLX 格式
    def convert_conv1d_weight(w):
        """PyTorch Conv1d (O, I, K) -> MLX (O, K, I)"""
        print(f"  转换前形状: {w.shape}")
        if len(w.shape) == 3:
            # 使用numpy的transpose，因为w是numpy数组
            result = np.transpose(w, (0, 2, 1))  # (O, I, K) -> (O, K, I)
            print(f"  转换后形状: {result.shape}")
            return result
        print(f"  不是3D，直接返回")
        return w
    
    # 转换权重
    mlx_weight = convert_conv1d_weight(torch_weight.numpy())
    mlx_bias = torch_bias.numpy()
    
    print(f"\nMLX 权重形状:")
    print(f"  weight: {mlx_weight.shape}")
    print(f"  bias: {mlx_bias.shape}")
    
    # 创建测试输入
    batch_size, seq_len = 2, 10
    
    # PyTorch 格式: (batch, channels, seq_len)
    x_pytorch = torch.randn(batch_size, in_channels, seq_len)
    
    # MLX 格式: (batch, seq_len, channels)
    x_mlx = mx.array(x_pytorch.numpy().transpose(0, 2, 1))
    
    print(f"\n输入形状:")
    print(f"  PyTorch: {x_pytorch.shape}")
    print(f"  MLX: {x_mlx.shape}")
    
    # PyTorch 卷积
    torch_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, 
                                dilation=dilation, bias=True)
    torch_conv.weight.data = torch_weight
    torch_conv.bias.data = torch_bias
    
    # 应用 padding
    padding_total = kernel_size - 1
    padding_right = padding_total // 2
    padding_left = padding_total - padding_right
    x_pytorch_padded = torch.nn.functional.pad(x_pytorch, (padding_left, padding_right), mode='reflect')
    
    torch_output = torch_conv(x_pytorch_padded)
    
    print(f"\nPyTorch 输出:")
    print(f"  形状: {torch_output.shape}")
    print(f"  前3个值: {torch_output[0, 0, :3]}")
    print(f"  后3个值: {torch_output[0, 0, -3:]}")
    
    # MLX 卷积
    mlx_conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                        dilation=dilation, bias=True)
    mlx_conv.weight = mx.array(mlx_weight)
    mlx_conv.bias = mx.array(mlx_bias)
    
    # 应用 padding (使用当前的实现)
    def pad1d_mlx(x, paddings, mode='zero', value=0.):
        """MLX版本的pad1d，完全复刻PyTorch F.pad的行为"""
        import torch
        
        length = x.shape[1]  # MLX格式: (batch, seq_len, channels)
        padding_left, padding_right = paddings
        assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
        if mode == 'reflect':
            max_pad = max(padding_left, padding_right)
            extra_pad = 0
            if length <= max_pad:
                extra_pad = max_pad - length + 1
                x = mx.pad(x, ((0, 0), (0, extra_pad), (0, 0)), mode='constant', constant_values=0)
            
            # 使用PyTorch的reflect padding逻辑，完全复刻
            # 转换为PyTorch格式进行padding，然后转换回MLX格式
            x_torch = torch.from_numpy(np.array(x).transpose(0, 2, 1))  # MLX -> PyTorch格式
            x_padded_torch = torch.nn.functional.pad(x_torch, (padding_left, padding_right), mode='reflect')
            x_padded_mlx = mx.array(x_padded_torch.numpy().transpose(0, 2, 1))  # PyTorch -> MLX格式
            
            # 关键修复：截断到原始长度，完全复刻PyTorch的pad1d行为
            end = x_padded_mlx.shape[1] - extra_pad
            return x_padded_mlx[:, :end, :]
        else:
            return mx.pad(x, ((0, 0), paddings, (0, 0)), mode='constant', constant_values=value)
    
    x_mlx_padded = pad1d_mlx(x_mlx, (padding_left, padding_right), mode='reflect')
    mlx_output = mlx_conv(x_mlx_padded)
    
    print(f"\nMLX 输出:")
    print(f"  形状: {mlx_output.shape}")
    print(f"  前3个值: {mlx_output[0, :3, 0]}")
    print(f"  后3个值: {mlx_output[0, -3:, 0]}")
    
    # 比较结果
    mlx_output_torch_format = torch.from_numpy(np.array(mlx_output).transpose(0, 2, 1))
    diff = torch.abs(torch_output - mlx_output_torch_format)
    
    print(f"\n差异分析:")
    print(f"  最大差异: {diff.max().item():.6f}")
    print(f"  平均差异: {diff.mean().item():.6f}")
    print(f"  形状匹配: {torch_output.shape == mlx_output_torch_format.shape}")
    
    if diff.max().item() > 1e-5:
        print(f"  ❌ 存在显著差异!")
        print(f"  差异分布:")
        print(f"    min: {diff.min().item():.6f}")
        print(f"    max: {diff.max().item():.6f}")
        print(f"    mean: {diff.mean().item():.6f}")
        print(f"    std: {diff.std().item():.6f}")
    else:
        print(f"  ✅ 结果一致!")

if __name__ == "__main__":
    test_dilated_conv_weights()
