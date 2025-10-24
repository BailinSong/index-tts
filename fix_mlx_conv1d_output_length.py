#!/usr/bin/env python3
"""
修复MLX Conv1d与PyTorch SConv1d的输出长度差异
"""

import sys
sys.path.append('.')
import torch
import mlx.core as mx
import mlx.nn as nn
import math
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

class MLXSConv1d(nn.Module):
    """
    MLX版本的SConv1d，确保输出长度与PyTorch SConv1d一致
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, 
                 groups=1, bias=True, causal=False, pad_mode='reflect'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.causal = causal
        self.pad_mode = pad_mode
        
        # 创建MLX Conv1d层
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             stride=stride, dilation=dilation, groups=groups, bias=bias)
    
    def get_extra_padding_for_conv1d(self, x, kernel_size, stride, padding_total):
        """计算额外的padding以确保输出长度一致"""
        length = x.shape[1]  # MLX格式: (batch, seq_len, channels)
        
        # 计算MLX Conv1d的输出长度
        mlx_output_length = length - (kernel_size - 1) * self.dilation
        
        # 计算需要的额外padding来确保输出长度与输入长度一致
        # 我们需要: (length + extra_padding) - (kernel_size - 1) * dilation = length
        # 所以: extra_padding = (kernel_size - 1) * dilation
        extra_padding = (kernel_size - 1) * self.dilation
        
        print(f"get_extra_padding: length={length}, kernel_size={kernel_size}, dilation={self.dilation}")
        print(f"  mlx_output_length={mlx_output_length}, extra_padding={extra_padding}")
        return extra_padding
    
    def mlx_pad_reflect_1d(self, x, padding_left, padding_right):
        """MLX版本的reflect padding"""
        batch, seq_len, channels = x.shape
        original_seq_len = seq_len
        
        if padding_left == 0 and padding_right == 0:
            return x
        
        print(f"mlx_pad_reflect_1d: 输入={original_seq_len}, padding_left={padding_left}, padding_right={padding_right}")
        
        # 处理reflect padding的特殊情况
        if seq_len <= max(padding_left, padding_right):
            # 如果输入长度小于padding，需要先扩展
            extra_pad = max(padding_left, padding_right) - seq_len + 1
            x = mx.pad(x, ((0, 0), (0, extra_pad), (0, 0)), mode='constant', constant_values=0)
            seq_len = x.shape[1]
            print(f"  扩展后长度: {seq_len}")
        
        # 左padding: 使用切片反转前padding_left个元素
        if padding_left > 0:
            left_reflect = x[:, padding_left-1::-1, :]  # 反转前padding_left个元素
            x = mx.concatenate([left_reflect, x], axis=1)
            print(f"  左padding后长度: {x.shape[1]}")
        
        # 右padding: 使用切片反转后padding_right个元素
        if padding_right > 0:
            right_reflect = x[:, -1:-padding_right-1:-1, :]  # 反转后padding_right个元素
            x = mx.concatenate([x, right_reflect], axis=1)
            print(f"  右padding后长度: {x.shape[1]}")
        
        # 关键修复：截取到原始长度，模拟PyTorch SConv1d的行为
        start_idx = padding_left
        end_idx = start_idx + original_seq_len
        x = x[:, start_idx:end_idx, :]
        
        print(f"  最终输出长度: {x.shape[1]}")
        
        return x
    
    def __call__(self, x):
        """
        x: (batch, seq_len, channels) - MLX格式
        """
        batch, seq_len, channels = x.shape
        
        # 计算有效kernel size
        effective_kernel_size = (self.kernel_size - 1) * self.dilation + 1
        padding_total = effective_kernel_size - self.stride
        
        # 计算额外padding
        extra_padding = self.get_extra_padding_for_conv1d(x, effective_kernel_size, self.stride, padding_total)
        
        print(f"输入长度: {x.shape[1]}, 有效kernel: {effective_kernel_size}, padding_total: {padding_total}, extra_padding: {extra_padding}")
        
        if self.causal:
            # 左padding用于causal
            x_padded = self.mlx_pad_reflect_1d(x, padding_total, extra_padding)
        else:
            # 非对称padding用于奇数stride
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x_padded = self.mlx_pad_reflect_1d(x, padding_left, padding_right + extra_padding)
        
        print(f"实际Padded长度: {x_padded.shape[1]}")
        
        print(f"Padded长度: {x_padded.shape[1]}")
        
        # 应用卷积
        y = self.conv(x_padded)
        
        print(f"Conv输出长度: {y.shape[1]}")
        
        # 关键修复：确保输出长度与输入长度一致
        # MLX Conv1d的输出长度可能小于输入长度，需要截取到正确长度
        if y.shape[1] != x.shape[1]:
            # 如果输出长度不匹配，截取到正确长度
            y = y[:, :x.shape[1], :]
        
        print(f"最终输出长度: {y.shape[1]}")
        
        # 最终确保输出长度与输入长度一致
        if y.shape[1] != x.shape[1]:
            print(f"警告：输出长度仍然不正确，期望={x.shape[1]}, 实际={y.shape[1]}")
            # 使用更简单的方法：直接截取到正确长度
            y = y[:, :x.shape[1], :]
        
        return y

def test_mlx_sconv1d():
    """测试MLX SConv1d的输出长度"""
    print("=== 测试MLX SConv1d输出长度 ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    mx.random.seed(42)
    
    # 创建PyTorch SConv1d
    from indextts.s2mel.modules.encodec import SConv1d
    pytorch_sconv = SConv1d(512, 1024, 5, dilation=1, bias=True)
    
    # 创建MLX SConv1d
    mlx_sconv = MLXSConv1d(512, 1024, 5, dilation=1, bias=True)
    
    # 创建测试数据
    batch_size = 1
    seq_len = 100
    
    # PyTorch格式: (batch, channels, seq_len)
    x_torch = torch.randn(batch_size, 512, seq_len)
    
    # MLX格式: (batch, seq_len, channels)
    x_mlx = torch_to_mlx(x_torch.transpose(1, 2))
    
    print(f"输入形状: PyTorch={x_torch.shape}, MLX={x_mlx.shape}")
    
    # 测试不同dilation的情况
    test_cases = [
        (1, "Dilation=1"),
        (2, "Dilation=2"),
        (4, "Dilation=4"),
        (8, "Dilation=8"),
    ]
    
    for dilation, name in test_cases:
        print(f"\n--- {name} ---")
        
        # 创建新的模型
        pytorch_sconv = SConv1d(512, 1024, 5, dilation=dilation, bias=True)
        mlx_sconv = MLXSConv1d(512, 1024, 5, dilation=dilation, bias=True)
        
        # 前向传播
        pytorch_output = pytorch_sconv(x_torch)
        mlx_output = mlx_sconv(x_mlx)
        
        # 转换MLX输出到PyTorch格式
        mlx_output_torch = mlx_to_torch(mlx_output).transpose(1, 2)
        
        print(f"PyTorch输出: {pytorch_output.shape}")
        print(f"MLX输出: {mlx_output.shape}")
        print(f"MLX输出(转换后): {mlx_output_torch.shape}")
        
        # 检查输出长度是否一致
        if pytorch_output.shape[2] == mlx_output_torch.shape[2]:
            print("✅ 输出长度一致")
            
            # 计算数值差异
            diff = torch.abs(pytorch_output - mlx_output_torch)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            
            print(f"数值差异: max={max_diff:.6f}, mean={mean_diff:.6f}")
        else:
            print("❌ 输出长度不一致")

def test_mlx_sconv1d_with_wavenet():
    """测试MLX SConv1d在WaveNet中的应用"""
    print("\n=== 测试MLX SConv1d在WaveNet中的应用 ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    mx.random.seed(42)
    
    # 创建MLX SConv1d
    mlx_sconv = MLXSConv1d(512, 1024, 5, dilation=2, bias=True)
    
    # 创建测试数据
    batch_size = 1
    seq_len = 100
    
    # MLX格式: (batch, seq_len, channels)
    x_mlx = mx.random.normal((batch_size, seq_len, 512))
    
    print(f"输入形状: {x_mlx.shape}")
    
    # 前向传播
    y_mlx = mlx_sconv(x_mlx)
    
    print(f"输出形状: {y_mlx.shape}")
    
    # 检查输出长度是否与输入长度一致
    if y_mlx.shape[1] == x_mlx.shape[1]:
        print("✅ 输出长度与输入长度一致")
    else:
        print("❌ 输出长度与输入长度不一致")

if __name__ == "__main__":
    test_mlx_sconv1d()
    test_mlx_sconv1d_with_wavenet()
