#!/usr/bin/env python3
"""
逐行翻译PyTorch SConv1d到MLX版本
"""

import sys
sys.path.append('.')
import torch
import mlx.core as mx
import mlx.nn as nn
import math
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

# 逐行翻译 get_extra_padding_for_conv1d
def get_extra_padding_for_conv1d_mlx(x, kernel_size, stride, padding_total=0):
    """
    逐行翻译PyTorch的get_extra_padding_for_conv1d函数
    """
    # PyTorch: length = x.shape[-1]
    # MLX: length = x.shape[1]  # MLX格式: (batch, seq_len, channels)
    length = x.shape[1]
    
    # PyTorch: n_frames = (length - kernel_size + padding_total) / stride + 1
    # MLX: 完全相同
    n_frames = (length - kernel_size + padding_total) / stride + 1
    
    # PyTorch: ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    # MLX: 完全相同
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    
    # PyTorch: return ideal_length - length
    # MLX: 完全相同
    return ideal_length - length

# 逐行翻译 pad1d
def pad1d_mlx(x, paddings, mode='zero', value=0.):
    """
    逐行翻译PyTorch的pad1d函数
    """
    # PyTorch: length = x.shape[-1]
    # MLX: length = x.shape[1]  # MLX格式: (batch, seq_len, channels)
    length = x.shape[1]
    
    # PyTorch: padding_left, padding_right = paddings
    # MLX: 完全相同
    padding_left, padding_right = paddings
    
    # PyTorch: assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    # MLX: 完全相同
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    
    # PyTorch: if mode == 'reflect':
    # MLX: 完全相同
    if mode == 'reflect':
        # PyTorch: max_pad = max(padding_left, padding_right)
        # MLX: 完全相同
        max_pad = max(padding_left, padding_right)
        
        # PyTorch: extra_pad = 0
        # MLX: 完全相同
        extra_pad = 0
        
        # PyTorch: if length <= max_pad:
        # MLX: 完全相同
        if length <= max_pad:
            # PyTorch: extra_pad = max_pad - length + 1
            # MLX: 完全相同
            extra_pad = max_pad - length + 1
            
            # PyTorch: x = F.pad(x, (0, extra_pad))
            # MLX: x = mx.pad(x, ((0, 0), (0, extra_pad), (0, 0)), mode='constant', constant_values=0)
            x = mx.pad(x, ((0, 0), (0, extra_pad), (0, 0)), mode='constant', constant_values=0)
        
        # PyTorch: padded = F.pad(x, paddings, mode, value)
        # MLX: padded = mlx_pad_reflect_1d(x, padding_left, padding_right)
        padded = mlx_pad_reflect_1d(x, padding_left, padding_right)
        
        # PyTorch: end = padded.shape[-1] - extra_pad
        # MLX: end = padded.shape[1] - extra_pad
        end = padded.shape[1] - extra_pad
        
        # PyTorch: return padded[..., :end]
        # MLX: return padded[:, :end, :]
        return padded[:, :end, :]
    else:
        # PyTorch: return F.pad(x, paddings, mode, value)
        # MLX: return mx.pad(x, ((0, 0), (padding_left, padding_right), (0, 0)), mode=mode, constant_values=value)
        return mx.pad(x, ((0, 0), (padding_left, padding_right), (0, 0)), mode=mode, constant_values=value)

def mlx_pad_reflect_1d(x, padding_left, padding_right):
    """
    MLX版本的reflect padding，逐行翻译PyTorch的reflect模式
    """
    # PyTorch: 在F.pad中实现reflect模式
    # MLX: 手动实现reflect模式
    
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

# 逐行翻译 NormConv1d
class MLXNormConv1d(nn.Module):
    """
    逐行翻译PyTorch的NormConv1d类
    """
    
    def __init__(self, *args, causal=False, norm='none', norm_kwargs=None, **kwargs):
        # PyTorch: super().__init__()
        # MLX: 完全相同
        super().__init__()
        
        # PyTorch: self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        # MLX: self.conv = nn.Conv1d(*args, **kwargs)  # 简化版本，暂时不使用parametrization_norm
        self.conv = nn.Conv1d(*args, **kwargs)
        
        # PyTorch: self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        # MLX: 简化版本，暂时不使用normalization
        # self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        
        # PyTorch: self.norm_type = norm
        # MLX: 完全相同
        self.norm_type = norm
    
    def __call__(self, x):
        # PyTorch: x = self.conv(x)
        # MLX: 完全相同
        x = self.conv(x)
        
        # PyTorch: x = self.norm(x)
        # MLX: 简化版本，暂时跳过normalization
        # x = self.norm(x)
        
        # PyTorch: return x
        # MLX: 完全相同
        return x

# 逐行翻译 SConv1d
class MLXSConv1d(nn.Module):
    """
    逐行翻译PyTorch的SConv1d类
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, 
                 groups=1, bias=True, causal=False, norm='none', norm_kwargs=None, 
                 pad_mode='reflect', **kwargs):
        # PyTorch: super().__init__()
        # MLX: 完全相同
        super().__init__()
        
        # PyTorch: if stride > 1 and dilation > 1:
        # MLX: 完全相同
        if stride > 1 and dilation > 1:
            # PyTorch: warnings.warn('SConv1d has been initialized with stride > 1 and dilation > 1'
            #                       f' (kernel_size={kernel_size} stride={stride}, dilation={dilation}).')
            # MLX: print(f"Warning: MLXSConv1d has been initialized with stride > 1 and dilation > 1"
            #            f" (kernel_size={kernel_size} stride={stride}, dilation={dilation}).")
            print(f"Warning: MLXSConv1d has been initialized with stride > 1 and dilation > 1"
                  f" (kernel_size={kernel_size} stride={stride}, dilation={dilation}).")
        
        # PyTorch: self.conv = NormConv1d(in_channels, out_channels, kernel_size, stride,
        #                                dilation=dilation, groups=groups, bias=bias, causal=causal,
        #                                norm=norm, norm_kwargs=norm_kwargs)
        # MLX: 完全相同
        self.conv = MLXNormConv1d(in_channels, out_channels, kernel_size, stride,
                                 dilation=dilation, groups=groups, bias=bias, causal=causal,
                                 norm=norm, norm_kwargs=norm_kwargs)
        
        # PyTorch: self.causal = causal
        # MLX: 完全相同
        self.causal = causal
        
        # PyTorch: self.pad_mode = pad_mode
        # MLX: 完全相同
        self.pad_mode = pad_mode
    
    def __call__(self, x):
        # PyTorch: B, C, T = x.shape
        # MLX: batch, channels, seq_len = x.shape  # MLX格式: (batch, seq_len, channels)
        batch, seq_len, channels = x.shape
        
        # PyTorch: kernel_size = self.conv.conv.kernel_size[0]
        # MLX: kernel_size = self.conv.conv.weight.shape[1]  # MLX Conv1d的kernel_size在weight的第二个维度
        kernel_size = self.conv.conv.weight.shape[1]
        
        # PyTorch: stride = self.conv.conv.stride[0]
        # MLX: stride = self.conv.conv.stride
        stride = self.conv.conv.stride
        
        # PyTorch: dilation = self.conv.conv.dilation[0]
        # MLX: dilation = self.conv.conv.dilation
        dilation = self.conv.conv.dilation
        
        # PyTorch: kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size with dilations
        # MLX: 完全相同
        kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size with dilations
        
        # PyTorch: padding_total = kernel_size - stride
        # MLX: 完全相同
        padding_total = kernel_size - stride
        
        # PyTorch: extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        # MLX: extra_padding = get_extra_padding_for_conv1d_mlx(x, kernel_size, stride, padding_total)
        extra_padding = get_extra_padding_for_conv1d_mlx(x, kernel_size, stride, padding_total)
        
        # PyTorch: if self.causal:
        # MLX: 完全相同
        if self.causal:
            # PyTorch: x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
            # MLX: x = pad1d_mlx(x, (padding_total, extra_padding), mode=self.pad_mode)
            x = pad1d_mlx(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # PyTorch: padding_right = padding_total // 2
            # MLX: 完全相同
            padding_right = padding_total // 2
            
            # PyTorch: padding_left = padding_total - padding_right
            # MLX: 完全相同
            padding_left = padding_total - padding_right
            
            # PyTorch: x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
            # MLX: x = pad1d_mlx(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
            x = pad1d_mlx(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        
        # PyTorch: return self.conv(x)
        # MLX: 完全相同
        return self.conv(x)

def test_mlx_sconv1d_line_by_line():
    """测试逐行翻译的MLX SConv1d与PyTorch SConv1d的完全一致性"""
    print("=== 测试逐行翻译的MLX SConv1d与PyTorch SConv1d的完全一致性 ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    mx.random.seed(42)
    
    # 创建PyTorch SConv1d
    from indextts.s2mel.modules.encodec import SConv1d
    pytorch_sconv = SConv1d(512, 1024, 5, dilation=1, bias=True)
    
    # 创建MLX SConv1d
    mlx_sconv = MLXSConv1d(512, 1024, 5, dilation=1, bias=True)
    
    # 同步权重：将PyTorch权重复制到MLX
    pytorch_weight = pytorch_sconv.conv.conv.weight.detach().cpu().numpy()
    pytorch_bias = pytorch_sconv.conv.conv.bias.detach().cpu().numpy()
    
    # 转换权重格式：PyTorch (out_channels, in_channels, kernel_size) -> MLX (out_channels, kernel_size, in_channels)
    mlx_weight = mx.array(pytorch_weight.transpose(0, 2, 1))
    mlx_bias = mx.array(pytorch_bias)
    
    # 设置MLX权重
    mlx_sconv.conv.conv.weight = mlx_weight
    mlx_sconv.conv.conv.bias = mlx_bias
    
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
        
        # 同步权重：将PyTorch权重复制到MLX
        pytorch_weight = pytorch_sconv.conv.conv.weight.detach().cpu().numpy()
        pytorch_bias = pytorch_sconv.conv.conv.bias.detach().cpu().numpy()
        
        # 转换权重格式：PyTorch (out_channels, in_channels, kernel_size) -> MLX (out_channels, kernel_size, in_channels)
        mlx_weight = mx.array(pytorch_weight.transpose(0, 2, 1))
        mlx_bias = mx.array(pytorch_bias)
        
        # 设置MLX权重
        mlx_sconv.conv.conv.weight = mlx_weight
        mlx_sconv.conv.conv.bias = mlx_bias
        
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

if __name__ == "__main__":
    test_mlx_sconv1d_line_by_line()
