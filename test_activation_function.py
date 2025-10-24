#!/usr/bin/env python3
"""
测试PyTorch和MLX的fused_add_tanh_sigmoid_multiply函数差异
"""

import sys
sys.path.append('.')
import torch
import mlx.core as mx
import numpy as np
from indextts.s2mel.modules.commons import fused_add_tanh_sigmoid_multiply
from indextts.s2mel.modules.mlx_wavenet import fused_add_tanh_sigmoid_multiply_mlx
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def test_activation_function():
    """测试激活函数的差异"""
    print("=== 测试fused_add_tanh_sigmoid_multiply函数 ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    mx.random.seed(42)
    
    # 创建测试数据
    batch_size = 1
    seq_len = 100
    n_channels = 512
    
    # PyTorch格式: (batch, 2*n_channels, seq_len)
    input_a_torch = torch.randn(batch_size, 2 * n_channels, seq_len)
    input_b_torch = torch.randn(batch_size, 2 * n_channels, seq_len)
    n_channels_tensor = torch.IntTensor([n_channels])
    
    # MLX格式: (batch, seq_len, 2*n_channels)
    input_a_mlx = torch_to_mlx(input_a_torch.transpose(1, 2))
    input_b_mlx = torch_to_mlx(input_b_torch.transpose(1, 2))
    
    print(f"输入形状:")
    print(f"  PyTorch: input_a={input_a_torch.shape}, input_b={input_b_torch.shape}")
    print(f"  MLX: input_a={input_a_mlx.shape}, input_b={input_b_mlx.shape}")
    
    # 测试PyTorch版本
    pytorch_output = fused_add_tanh_sigmoid_multiply(input_a_torch, input_b_torch, n_channels_tensor)
    print(f"PyTorch输出: {pytorch_output.shape}, range=[{torch.min(pytorch_output):.6f}, {torch.max(pytorch_output):.6f}]")
    
    # 测试MLX版本
    mlx_output = fused_add_tanh_sigmoid_multiply_mlx(input_a_mlx, input_b_mlx, n_channels)
    print(f"MLX输出: {mlx_output.shape}, range=[{mx.min(mlx_output):.6f}, {mx.max(mlx_output):.6f}]")
    
    # 转换MLX输出到PyTorch格式进行对比
    mlx_output_torch = mlx_to_torch(mlx_output).transpose(1, 2)
    print(f"MLX输出(转换后): {mlx_output_torch.shape}, range=[{torch.min(mlx_output_torch):.6f}, {torch.max(mlx_output_torch):.6f}]")
    
    # 计算差异
    diff = torch.abs(pytorch_output - mlx_output_torch)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    print(f"\n差异分析:")
    print(f"  最大差异: {max_diff:.6f}")
    print(f"  平均差异: {mean_diff:.6f}")
    print(f"  差异>1e-3的元素比例: {torch.sum(diff > 1e-3).item() / diff.numel():.2%}")
    
    # 详细分析中间步骤
    print(f"\n=== 中间步骤分析 ===")
    
    # PyTorch中间步骤
    in_act_torch = input_a_torch + input_b_torch
    t_act_torch = torch.tanh(in_act_torch[:, :n_channels, :])
    s_act_torch = torch.sigmoid(in_act_torch[:, n_channels:, :])
    
    print(f"PyTorch中间步骤:")
    print(f"  in_act: {in_act_torch.shape}, range=[{torch.min(in_act_torch):.6f}, {torch.max(in_act_torch):.6f}]")
    print(f"  t_act: {t_act_torch.shape}, range=[{torch.min(t_act_torch):.6f}, {torch.max(t_act_torch):.6f}]")
    print(f"  s_act: {s_act_torch.shape}, range=[{torch.min(s_act_torch):.6f}, {torch.max(s_act_torch):.6f}]")
    
    # MLX中间步骤
    in_act_mlx = input_a_mlx + input_b_mlx
    t_act_mlx = mx.tanh(in_act_mlx[:, :, :n_channels])
    s_act_mlx = mx.sigmoid(in_act_mlx[:, :, n_channels:])
    
    print(f"MLX中间步骤:")
    print(f"  in_act: {in_act_mlx.shape}, range=[{mx.min(in_act_mlx):.6f}, {mx.max(in_act_mlx):.6f}]")
    print(f"  t_act: {t_act_mlx.shape}, range=[{mx.min(t_act_mlx):.6f}, {mx.max(t_act_mlx):.6f}]")
    print(f"  s_act: {s_act_mlx.shape}, range=[{mx.min(s_act_mlx):.6f}, {mx.max(s_act_mlx):.6f}]")
    
    # 对比中间步骤
    in_act_mlx_torch = mlx_to_torch(in_act_mlx).transpose(1, 2)
    t_act_mlx_torch = mlx_to_torch(t_act_mlx).transpose(1, 2)
    s_act_mlx_torch = mlx_to_torch(s_act_mlx).transpose(1, 2)
    
    in_act_diff = torch.abs(in_act_torch - in_act_mlx_torch)
    t_act_diff = torch.abs(t_act_torch - t_act_mlx_torch)
    s_act_diff = torch.abs(s_act_torch - s_act_mlx_torch)
    
    print(f"\n中间步骤差异:")
    print(f"  in_act差异: max={torch.max(in_act_diff):.6f}, mean={torch.mean(in_act_diff):.6f}")
    print(f"  t_act差异: max={torch.max(t_act_diff):.6f}, mean={torch.mean(t_act_diff):.6f}")
    print(f"  s_act差异: max={torch.max(s_act_diff):.6f}, mean={torch.mean(s_act_diff):.6f}")

if __name__ == "__main__":
    test_activation_function()
