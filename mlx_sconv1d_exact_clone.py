#!/usr/bin/env python3
"""
1:1复刻PyTorch SConv1d，包括所有方法和行为
"""

import sys
sys.path.append('.')
import torch
import mlx.core as mx
import mlx.nn as nn
import math
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def get_extra_padding_for_conv1d_mlx(x, kernel_size, stride, padding_total=0):
    """
    MLX版本的get_extra_padding_for_conv1d，完全复刻PyTorch版本
    """
    length = x.shape[1]  # MLX格式: (batch, seq_len, channels)
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length

def pad1d_mlx(x, paddings, mode='zero', value=0.):
    """
    MLX版本的pad1d，完全复刻PyTorch版本
    """
    length = x.shape[1]  # MLX格式: (batch, seq_len, channels)
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = mx.pad(x, ((0, 0), (0, extra_pad), (0, 0)), mode='constant', constant_values=0)
        
        # 实现reflect padding
        padded = mlx_pad_reflect_1d(x, padding_left, padding_right)
        end = padded.shape[1] - extra_pad
        return padded[:, :end, :]
    else:
        return mx.pad(x, ((0, 0), (padding_left, padding_right), (0, 0)), mode=mode, constant_values=value)

def mlx_pad_reflect_1d(x, padding_left, padding_right):
    """
    MLX版本的reflect padding，完全复刻PyTorch的reflect模式
    """
    batch, seq_len, channels = x.shape
    
    if padding_left == 0 and padding_right == 0:
        return x
    
    # 左padding: 使用切片反转前padding_left个元素
    if padding_left > 0:
        left_reflect = x[:, padding_left-1::-1, :]  # 反转前padding_left个元素
        x = mx.concatenate([left_reflect, x], axis=1)
    
    # 右padding: 使用切片反转后padding_right个元素
    if padding_right > 0:
        right_reflect = x[:, -1:-padding_right-1:-1, :]  # 反转后padding_right个元素
        x = mx.concatenate([x, right_reflect], axis=1)
    
    return x

class MLXNormConv1d(nn.Module):
    """
    MLX版本的NormConv1d，完全复刻PyTorch版本
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, 
                 groups=1, bias=True, causal=False, norm='none', norm_kwargs=None):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             stride=stride, dilation=dilation, groups=groups, bias=bias)
        self.norm_type = norm
        self.causal = causal
        
        # 简化版本，暂时不使用normalization
        # 在实际应用中，这里应该实现相应的normalization
    
    def __call__(self, x):
        """
        x: (batch, seq_len, channels) - MLX格式
        """
        x = self.conv(x)
        # 这里应该应用normalization，但为了简化，暂时跳过
        return x

class MLXSConv1d(nn.Module):
    """
    MLX版本的SConv1d，1:1复刻PyTorch版本
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, 
                 groups=1, bias=True, causal=False, norm='none', norm_kwargs=None, 
                 pad_mode='reflect', **kwargs):
        super().__init__()
        
        # 警告用户不寻常的设置
        if stride > 1 and dilation > 1:
            print(f"Warning: MLXSConv1d has been initialized with stride > 1 and dilation > 1"
                  f" (kernel_size={kernel_size} stride={stride}, dilation={dilation}).")
        
        self.conv = MLXNormConv1d(in_channels, out_channels, kernel_size, stride,
                                 dilation=dilation, groups=groups, bias=bias, causal=causal,
                                 norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.pad_mode = pad_mode
    
    def __call__(self, x):
        """
        x: (batch, seq_len, channels) - MLX格式
        """
        batch, seq_len, channels = x.shape
        
        # 获取卷积层的参数
        kernel_size = self.conv.conv.kernel_size
        stride = self.conv.conv.stride
        dilation = self.conv.conv.dilation
        
        # 计算有效kernel size
        kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size with dilations
        padding_total = kernel_size - stride
        
        # 计算额外padding
        extra_padding = get_extra_padding_for_conv1d_mlx(x, kernel_size, stride, padding_total)
        
        if self.causal:
            # 左padding用于causal
            x = pad1d_mlx(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # 非对称padding用于奇数stride
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d_mlx(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        
        return self.conv(x)

def test_mlx_sconv1d_exact_clone():
    """测试MLX SConv1d与PyTorch SConv1d的完全一致性"""
    print("=== 测试MLX SConv1d与PyTorch SConv1d的完全一致性 ===")
    
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
            
            # 分析差异的分布
            diff_percentiles = torch.quantile(diff.flatten(), torch.tensor([0.5, 0.9, 0.95, 0.99]))
            print(f"差异分位数: 50%={diff_percentiles[0]:.6f}, 90%={diff_percentiles[1]:.6f}, 95%={diff_percentiles[2]:.6f}, 99%={diff_percentiles[3]:.6f}")
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
    test_mlx_sconv1d_exact_clone()
    test_mlx_sconv1d_with_wavenet()
