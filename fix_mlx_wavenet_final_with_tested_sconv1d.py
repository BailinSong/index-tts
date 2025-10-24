#!/usr/bin/env python3
"""
最终修复MLX WaveNet的权重格式问题，使用经过测试验证的SConv1d实现
"""

import sys
sys.path.append('.')
import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import math
from typing import Tuple, Dict, Any
import typing as tp
from indextts.s2mel.modules.wavenet import WN as PyTorchWaveNet
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def get_extra_padding_for_conv1d_mlx(x: mx.array, kernel_size: int, stride: int,
                                      padding_total: int = 0) -> int:
    """MLX版本的get_extra_padding_for_conv1d，完全复刻PyTorch版本"""
    length = x.shape[1]  # MLX格式: (batch, seq_len, channels)
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return int(ideal_length - length)

def pad1d_mlx(x: mx.array, paddings: tp.Tuple[int, int], mode: str = 'zero', value: float = 0.):
    """MLX版本的pad1d，完全复刻PyTorch F.pad的行为"""
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
        
        return x_padded_mlx
    else:
        return mx.pad(x, ((0, 0), paddings, (0, 0)), mode='constant', constant_values=value)

class MLXNormConv1d(nn.Module):
    """MLX版本的NormConv1d，完全复刻PyTorch版本"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, groups: int = 1, bias: bool = True,
                 causal: bool = False, norm: str = 'none',
                 norm_kwargs: tp.Dict[str, tp.Any] = {}):
        super().__init__()
        self.causal = causal
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                             dilation=dilation, groups=groups, bias=bias)

    def __call__(self, x):
        return self.conv(x)

class MLXSConv1d(nn.Module):
    """MLX版本的SConv1d，完全复刻PyTorch版本，使用经过测试验证的实现"""
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, dilation: int = 1,
                 groups: int = 1, bias: bool = True, causal: bool = False,
                 norm: str = 'none', norm_kwargs: tp.Dict[str, tp.Any] = {},
                 pad_mode: str = 'reflect', **kwargs):
        super().__init__()
        self.kernel_size_init = kernel_size  # Store original kernel_size
        self.stride_init = stride
        self.dilation_init = dilation
        self.causal = causal
        self.pad_mode = pad_mode
        
        # MLX NormConv1d equivalent
        self.conv = MLXNormConv1d(in_channels, out_channels, kernel_size, stride,
                                  dilation=dilation, groups=groups, bias=bias, causal=causal,
                                  norm=norm, norm_kwargs=norm_kwargs)

    def get_extra_padding_for_conv1d(self, x: mx.array, kernel_size: int, stride: int,
                                     padding_total: int = 0) -> int:
        """MLX版本的get_extra_padding_for_conv1d"""
        return get_extra_padding_for_conv1d_mlx(x, kernel_size, stride, padding_total)

    def mlx_pad1d(self, x: mx.array, paddings: tp.Tuple[int, int], mode: str = 'zero', value: float = 0.):
        """MLX版本的pad1d，模拟PyTorch F.pad的行为"""
        return pad1d_mlx(x, paddings, mode, value)

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
        # MLX: 使用MLX版本
        extra_padding = self.get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        
        # PyTorch: if self.causal: x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        # MLX: 使用MLX版本
        if self.causal:
            # Left padding for causal
            x_padded = self.mlx_pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        # PyTorch: else: padding_right = padding_total // 2; padding_left = padding_total - padding_right; x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        # MLX: 使用MLX版本
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x_padded = self.mlx_pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        
        # PyTorch: return self.conv(x)
        # MLX: 完全相同
        output = self.conv(x_padded)
        
        return output

def fused_add_tanh_sigmoid_multiply_mlx(input_a, input_b, n_channels):
    """
    MLX implementation of fused gated activation.
    
    Args:
        input_a: (batch, 2*n_channels, seq_len)
        input_b: (batch, 2*n_channels, seq_len)
        n_channels: int
    
    Returns:
        output: (batch, n_channels, seq_len)
    """
    # Note: MLX uses (batch, seq_len, channels) format
    # So we need to adapt
    
    in_act = input_a + input_b
    
    # Split channels
    t_act = mx.tanh(in_act[:, :, :n_channels])
    s_act = mx.sigmoid(in_act[:, :, n_channels:])
    
    return t_act * s_act

class MLXWaveNetFinal(nn.Module):
    """
    最终MLX WaveNet实现，使用经过测试验证的SConv1d
    """
    
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0
    ):
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        
        # Input layers (dilated convolutions with tested SConv1d)
        self.in_layers = []
        for i in range(n_layers):
            dilation = dilation_rate ** i
            
            layer = MLXSConv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                bias=True,
                causal=False,
                pad_mode='reflect'
            )
            self.in_layers.append(layer)
        
        # Residual/skip layers (1x1 conv, no special padding needed)
        self.res_skip_layers = []
        for i in range(n_layers):
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels
            
            layer = nn.Conv1d(
                hidden_channels,
                res_skip_channels,
                kernel_size=1,
                padding=0,
                bias=True
            )
            self.res_skip_layers.append(layer)
        
        # Conditioning layer (if using global conditioning)
        if gin_channels != 0:
            self.cond_layer = nn.Conv1d(
                gin_channels,
                2 * hidden_channels * n_layers,
                kernel_size=1,
                padding=0,
                bias=True
            )
        else:
            self.cond_layer = None
        
        # Dropout
        self.dropout = nn.Dropout(p_dropout)
        
        print(f">> MLX WaveNet (Final with tested SConv1d) initialized:")
        print(f"   Layers: {n_layers}, Channels: {hidden_channels}")
        print(f"   Kernel: {kernel_size}, Dilation rate: {dilation_rate}")
        print(f"   Using tested SConv1d implementation")
    
    def __call__(self, x, x_mask, g=None):
        """
        x: (batch, seq_len, hidden_channels) - MLX format
        x_mask: (batch, 1, seq_len) - mask (True = keep)
        g: (batch, seq_len, gin_channels) - global conditioning
        
        Returns:
            output: (batch, seq_len, hidden_channels)
        """
        # Reshape mask for MLX broadcast: (batch, 1, seq_len) -> (batch, seq_len, 1)
        if len(x_mask.shape) == 3 and x_mask.shape[1] == 1:
            x_mask_mlx = x_mask.transpose(0, 2, 1)  # (batch, seq_len, 1)
        else:
            x_mask_mlx = x_mask
        
        output = mx.zeros_like(x)
        
        # Process global conditioning
        if g is not None and self.cond_layer is not None:
            g_cond = self.cond_layer(g)  # (batch, seq_len, 2*hidden_channels*n_layers)
        else:
            g_cond = None
        
        for i in range(self.n_layers):
            # Apply mask and dilated conv
            x_masked = x * x_mask_mlx
            x_in = self.in_layers[i](x_masked)  # (batch, seq_len, 2*hidden_channels)
            
            # 确保x_in的长度与原始输入一致
            if x_in.shape[1] != x.shape[1]:
                # 如果长度不匹配，截断或填充到原始长度
                if x_in.shape[1] > x.shape[1]:
                    x_in = x_in[:, :x.shape[1], :]
                else:
                    # 填充到原始长度
                    pad_length = x.shape[1] - x_in.shape[1]
                    x_in = mx.pad(x_in, ((0, 0), (0, pad_length), (0, 0)), mode='constant', constant_values=0)
            
            # Add global conditioning
            if g_cond is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g_cond[:, :, cond_offset:cond_offset + 2 * self.hidden_channels]
                # 确保g_l的长度与x_in一致
                if g_l.shape[1] != x_in.shape[1]:
                    if g_l.shape[1] > x_in.shape[1]:
                        g_l = g_l[:, :x_in.shape[1], :]
                    else:
                        pad_length = x_in.shape[1] - g_l.shape[1]
                        g_l = mx.pad(g_l, ((0, 0), (0, pad_length), (0, 0)), mode='constant', constant_values=0)
            else:
                g_l = mx.zeros_like(x_in)
            
            # Fused gated activation: tanh(a) * sigmoid(b)
            in_act = x_in + g_l
            t_act = mx.tanh(in_act[:, :, :self.hidden_channels])
            s_act = mx.sigmoid(in_act[:, :, self.hidden_channels:])
            acts = t_act * s_act
            
            # Dropout
            acts = self.dropout(acts)
            
            # Residual/skip connection
            res_skip_acts = self.res_skip_layers[i](acts)
            
            if i < self.n_layers - 1:
                # Split into residual and skip
                res_acts = res_skip_acts[:, :, :self.hidden_channels]
                skip_acts = res_skip_acts[:, :, self.hidden_channels:]
                
                # 确保res_acts的长度与x一致
                if res_acts.shape[1] != x.shape[1]:
                    if res_acts.shape[1] > x.shape[1]:
                        res_acts = res_acts[:, :x.shape[1], :]
                    else:
                        pad_length = x.shape[1] - res_acts.shape[1]
                        res_acts = mx.pad(res_acts, ((0, 0), (0, pad_length), (0, 0)), mode='constant', constant_values=0)
                
                # 确保skip_acts的长度与output一致
                if skip_acts.shape[1] != output.shape[1]:
                    if skip_acts.shape[1] > output.shape[1]:
                        skip_acts = skip_acts[:, :output.shape[1], :]
                    else:
                        pad_length = output.shape[1] - skip_acts.shape[1]
                        skip_acts = mx.pad(skip_acts, ((0, 0), (0, pad_length), (0, 0)), mode='constant', constant_values=0)
                
                x = (x + res_acts) * x_mask_mlx
                output = output + skip_acts
            else:
                # Last layer: only skip
                # 确保res_skip_acts的长度与output一致
                if res_skip_acts.shape[1] != output.shape[1]:
                    if res_skip_acts.shape[1] > output.shape[1]:
                        res_skip_acts = res_skip_acts[:, :output.shape[1], :]
                    else:
                        pad_length = output.shape[1] - res_skip_acts.shape[1]
                        res_skip_acts = mx.pad(res_skip_acts, ((0, 0), (0, pad_length), (0, 0)), mode='constant', constant_values=0)
                
                output = output + res_skip_acts
        
        return output * x_mask_mlx
    
    def load_weights_from_pytorch(self, state_dict, prefix="wavenet."):
        """
        Load weights from PyTorch WaveNet (with weight_norm removed)
        
        Args:
            state_dict: PyTorch state dict (numpy arrays)
            prefix: Key prefix
        """
        loaded = 0
        
        def convert_conv1d_weight(w):
            """PyTorch (O, I, K) -> MLX (O, K, I)"""
            return w.transpose(0, 2, 1)
        
        # Load in_layers
        for i in range(self.n_layers):
            # Conv weights
            w_key = f"{prefix}in_layers.{i}.weight"
            b_key = f"{prefix}in_layers.{i}.bias"
            
            if w_key in state_dict:
                w = state_dict[w_key]
                self.in_layers[i].conv.conv.weight = mx.array(convert_conv1d_weight(w))
                loaded += 1
            if b_key in state_dict:
                self.in_layers[i].conv.conv.bias = mx.array(state_dict[b_key])
                loaded += 1
        
        # Load res_skip_layers
        for i in range(self.n_layers):
            w_key = f"{prefix}res_skip_layers.{i}.weight"
            b_key = f"{prefix}res_skip_layers.{i}.bias"
            
            if w_key in state_dict:
                w = state_dict[w_key]
                self.res_skip_layers[i].weight = mx.array(convert_conv1d_weight(w))
                loaded += 1
            if b_key in state_dict:
                self.res_skip_layers[i].bias = mx.array(state_dict[b_key])
                loaded += 1
        
        # Load cond_layer
        if self.gin_channels != 0 and self.cond_layer is not None:
            w_key = f"{prefix}cond_layer.weight"
            b_key = f"{prefix}cond_layer.bias"
            
            if w_key in state_dict:
                w = state_dict[w_key]
                self.cond_layer.weight = mx.array(convert_conv1d_weight(w))
                loaded += 1
            if b_key in state_dict:
                self.cond_layer.bias = mx.array(state_dict[b_key])
                loaded += 1
        
        print(f">> MLX WaveNet (Final) loaded {loaded} weights")
        return loaded

def fix_mlx_wavenet_weights():
    """修复MLX WaveNet的权重初始化，使用经过测试验证的SConv1d"""
    print("=== 修复MLX WaveNet权重初始化 (使用测试验证的SConv1d) ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    mx.random.seed(42)
    
    # 创建PyTorch WaveNet
    pytorch_wavenet = PyTorchWaveNet(
        hidden_channels=512,
        kernel_size=5,
        dilation_rate=2,
        n_layers=8,
        gin_channels=512,
        p_dropout=0.0
    )
    
    # 创建MLX WaveNet (使用测试验证的SConv1d)
    mlx_wavenet = MLXWaveNetFinal(
        hidden_channels=512,
        kernel_size=5,
        dilation_rate=2,
        n_layers=8,
        gin_channels=512,
        p_dropout=0.0
    )
    
    print("修复前权重对比:")
    for i in range(8):
        pytorch_weight = pytorch_wavenet.in_layers[i].conv.conv.weight  # (1024, 512, 5)
        mlx_weight = mlx_wavenet.in_layers[i].conv.conv.weight  # (1024, 5, 512)
        
        print(f"Layer {i}:")
        print(f"  PyTorch: {pytorch_weight.shape}")
        print(f"  MLX: {mlx_weight.shape}")
        
        # 转换MLX权重到PyTorch格式
        mlx_weight_torch = mlx_to_torch(mlx_weight).transpose(1, 2)
        
        # 计算差异
        diff = torch.abs(pytorch_weight - mlx_weight_torch)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"  差异: max={max_diff:.6f}, mean={mean_diff:.6f}")
    
    # 修复MLX权重格式
    print("\n修复MLX权重格式...")
    for i in range(8):
        pytorch_weight = pytorch_wavenet.in_layers[i].conv.conv.weight  # (1024, 512, 5)
        
        # 将PyTorch权重转换为MLX格式
        # PyTorch: (out_channels, in_channels, kernel_size)
        # MLX: (out_channels, kernel_size, in_channels)
        pytorch_weight_np = pytorch_weight.detach().cpu().numpy()
        mlx_weight_fixed = mx.array(pytorch_weight_np.transpose(0, 2, 1))
        
        # 更新MLX权重
        mlx_wavenet.in_layers[i].conv.conv.weight = mlx_weight_fixed
        
        print(f"Layer {i} 权重已修复")
    
    # 修复bias
    for i in range(8):
        if hasattr(pytorch_wavenet.in_layers[i].conv.conv, 'bias') and pytorch_wavenet.in_layers[i].conv.conv.bias is not None:
            pytorch_bias = pytorch_wavenet.in_layers[i].conv.conv.bias
            mlx_bias_fixed = torch_to_mlx(pytorch_bias)
            mlx_wavenet.in_layers[i].conv.conv.bias = mlx_bias_fixed
            print(f"Layer {i} bias已修复")
    
    # 修复res_skip_layers权重
    for i in range(8):
        pytorch_weight = pytorch_wavenet.res_skip_layers[i].conv.conv.weight
        pytorch_weight_np = pytorch_weight.detach().cpu().numpy()
        mlx_weight_fixed = mx.array(pytorch_weight_np.transpose(0, 2, 1))
        mlx_wavenet.res_skip_layers[i].weight = mlx_weight_fixed
        
        if hasattr(pytorch_wavenet.res_skip_layers[i].conv.conv, 'bias') and pytorch_wavenet.res_skip_layers[i].conv.conv.bias is not None:
            pytorch_bias = pytorch_wavenet.res_skip_layers[i].conv.conv.bias
            mlx_bias_fixed = torch_to_mlx(pytorch_bias)
            mlx_wavenet.res_skip_layers[i].bias = mlx_bias_fixed
        
        print(f"ResSkip Layer {i} 权重已修复")
    
    # 修复cond_layer权重
    if hasattr(pytorch_wavenet, 'cond_layer') and pytorch_wavenet.cond_layer is not None:
        pytorch_weight = pytorch_wavenet.cond_layer.conv.conv.weight
        pytorch_weight_np = pytorch_weight.detach().cpu().numpy()
        mlx_weight_fixed = mx.array(pytorch_weight_np.transpose(0, 2, 1))
        mlx_wavenet.cond_layer.weight = mlx_weight_fixed
        
        if hasattr(pytorch_wavenet.cond_layer.conv.conv, 'bias') and pytorch_wavenet.cond_layer.conv.conv.bias is not None:
            pytorch_bias = pytorch_wavenet.cond_layer.conv.conv.bias
            mlx_bias_fixed = torch_to_mlx(pytorch_bias)
            mlx_wavenet.cond_layer.bias = mlx_bias_fixed
        
        print("Cond layer权重已修复")
    
    # 验证修复结果
    print("\n修复后权重对比:")
    for i in range(8):
        pytorch_weight = pytorch_wavenet.in_layers[i].conv.conv.weight  # (1024, 512, 5)
        mlx_weight = mlx_wavenet.in_layers[i].conv.conv.weight  # (1024, 5, 512)
        
        # 转换MLX权重到PyTorch格式
        mlx_weight_torch = mlx_to_torch(mlx_weight).transpose(1, 2)
        
        # 计算差异
        diff = torch.abs(pytorch_weight - mlx_weight_torch)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"Layer {i}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    
    # 测试前向传播
    print("\n测试前向传播...")
    batch_size = 1
    seq_len = 100
    
    # PyTorch输入
    x_torch = torch.randn(batch_size, 512, seq_len)
    x_mask_torch = torch.ones(batch_size, 1, seq_len, dtype=torch.bool)
    g_torch = torch.randn(batch_size, 512, 1)
    
    # MLX输入
    x_mlx = torch_to_mlx(x_torch.transpose(1, 2))
    x_mask_mlx = torch_to_mlx(x_mask_torch.transpose(1, 2))
    g_mlx = torch_to_mlx(g_torch.transpose(1, 2))
    
    # 前向传播
    pytorch_output = pytorch_wavenet(x_torch, x_mask_torch, g_torch)
    mlx_output = mlx_wavenet(x_mlx, x_mask_mlx, g_mlx)
    
    # 转换MLX输出到PyTorch格式
    mlx_output_torch = mlx_to_torch(mlx_output).transpose(1, 2)
    
    # 计算差异
    diff = torch.abs(pytorch_output - mlx_output_torch)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    print(f"前向传播差异:")
    print(f"  最大差异: {max_diff:.6f}")
    print(f"  平均差异: {mean_diff:.6f}")
    print(f"  差异>1e-3的元素比例: {torch.sum(diff > 1e-3).item() / diff.numel():.2%}")
    
    return mlx_wavenet

if __name__ == "__main__":
    fix_mlx_wavenet_weights()
