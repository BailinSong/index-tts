#!/usr/bin/env python3
"""
测试 SConv1d 的 padding 行为差异
"""

import torch
import mlx.core as mx
import numpy as np

def test_padding_behavior():
    """测试 PyTorch 和 MLX 的 padding 行为"""
    
    # 创建测试数据
    batch_size, seq_len, channels = 2, 10, 512
    kernel_size, stride, dilation = 5, 1, 1
    
    # PyTorch 格式: (batch, channels, seq_len)
    x_pytorch = torch.randn(batch_size, channels, seq_len)
    
    # MLX 格式: (batch, seq_len, channels)  
    x_mlx = mx.array(x_pytorch.numpy().transpose(0, 2, 1))
    
    print(f"输入形状:")
    print(f"  PyTorch: {x_pytorch.shape}")
    print(f"  MLX:     {x_mlx.shape}")
    
    # 计算 padding 参数
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    padding_total = effective_kernel_size - stride
    padding_right = padding_total // 2
    padding_left = padding_total - padding_right
    
    print(f"\nPadding 参数:")
    print(f"  kernel_size: {kernel_size}, dilation: {dilation}")
    print(f"  effective_kernel_size: {effective_kernel_size}")
    print(f"  padding_total: {padding_total}")
    print(f"  padding_left: {padding_left}, padding_right: {padding_right}")
    
    # PyTorch padding
    x_pytorch_padded = torch.nn.functional.pad(x_pytorch, (padding_left, padding_right), mode='reflect')
    print(f"\nPyTorch padding 结果:")
    print(f"  形状: {x_pytorch_padded.shape}")
    print(f"  前3个值: {x_pytorch_padded[0, 0, :3]}")
    print(f"  后3个值: {x_pytorch_padded[0, 0, -3:]}")
    
    # MLX padding (当前实现)
    def pad1d_mlx_current(x, paddings, mode='zero', value=0.):
        """当前的 MLX pad1d 实现"""
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
    
    x_mlx_padded = pad1d_mlx_current(x_mlx, (padding_left, padding_right), mode='reflect')
    print(f"\nMLX padding 结果 (当前实现):")
    print(f"  形状: {x_mlx_padded.shape}")
    print(f"  前3个值: {x_mlx_padded[0, :3, 0]}")
    print(f"  后3个值: {x_mlx_padded[0, -3:, 0]}")
    
    # 比较结果
    x_mlx_padded_torch_format = torch.from_numpy(np.array(x_mlx_padded).transpose(0, 2, 1))
    diff = torch.abs(x_pytorch_padded - x_mlx_padded_torch_format)
    print(f"\n差异分析:")
    print(f"  最大差异: {diff.max().item():.6f}")
    print(f"  平均差异: {diff.mean().item():.6f}")
    print(f"  形状匹配: {x_pytorch_padded.shape == x_mlx_padded_torch_format.shape}")
    
    # 测试正确的 MLX padding 实现
    def pad1d_mlx_correct(x, paddings, mode='zero', value=0.):
        """正确的 MLX pad1d 实现"""
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
            
            # 直接使用 MLX 的 padding，但需要正确处理维度
            # MLX 的 padding 格式: ((batch_pad), (seq_pad), (channel_pad))
            x_padded = mx.pad(x, ((0, 0), (padding_left, padding_right), (0, 0)), mode='constant', constant_values=0)
            
            # 应用 reflect padding 逻辑
            if padding_left > 0:
                # 左 padding: 反射前 padding_left 个元素
                for i in range(padding_left):
                    x_padded = mx.concatenate([
                        x_padded[:, padding_left-i:padding_left-i+1, :],  # 反射
                        x_padded
                    ], axis=1)
            
            if padding_right > 0:
                # 右 padding: 反射后 padding_right 个元素
                for i in range(padding_right):
                    x_padded = mx.concatenate([
                        x_padded,
                        x_padded[:, -2-i:-1-i, :]  # 反射
                    ], axis=1)
            
            # 截断到正确长度
            end = x_padded.shape[1] - extra_pad
            return x_padded[:, :end, :]
        else:
            return mx.pad(x, ((0, 0), paddings, (0, 0)), mode='constant', constant_values=value)
    
    x_mlx_padded_correct = pad1d_mlx_correct(x_mlx, (padding_left, padding_right), mode='reflect')
    print(f"\nMLX padding 结果 (正确实现):")
    print(f"  形状: {x_mlx_padded_correct.shape}")
    print(f"  前3个值: {x_mlx_padded_correct[0, :3, 0]}")
    print(f"  后3个值: {x_mlx_padded_correct[0, -3:, 0]}")
    
    # 比较正确实现的结果
    x_mlx_padded_correct_torch_format = torch.from_numpy(np.array(x_mlx_padded_correct).transpose(0, 2, 1))
    diff_correct = torch.abs(x_pytorch_padded - x_mlx_padded_correct_torch_format)
    print(f"\n正确实现差异分析:")
    print(f"  最大差异: {diff_correct.max().item():.6f}")
    print(f"  平均差异: {diff_correct.mean().item():.6f}")
    print(f"  形状匹配: {x_pytorch_padded.shape == x_mlx_padded_correct_torch_format.shape}")

if __name__ == "__main__":
    test_padding_behavior()
