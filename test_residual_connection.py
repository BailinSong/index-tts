#!/usr/bin/env python3
"""
测试PyTorch和MLX WaveNet的残差连接实现差异
"""

import sys
sys.path.append('.')
import torch
import mlx.core as mx
import numpy as np
from indextts.s2mel.modules.wavenet import WN as PyTorchWaveNet
from indextts.s2mel.modules.mlx_wavenet import MLXWaveNet
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def test_residual_connection():
    """测试残差连接的差异"""
    print("=== 测试WaveNet残差连接实现 ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    mx.random.seed(42)
    
    # 创建WaveNet实例
    hidden_channels = 512
    kernel_size = 5
    dilation_rate = 2
    n_layers = 8
    gin_channels = 512
    
    pytorch_wavenet = PyTorchWaveNet(
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        n_layers=n_layers,
        gin_channels=gin_channels,
        p_dropout=0.0
    )
    
    mlx_wavenet = MLXWaveNet(
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        n_layers=n_layers,
        gin_channels=gin_channels,
        p_dropout=0.0
    )
    
    # 创建测试输入
    batch_size = 1
    seq_len = 100
    
    # PyTorch格式: (batch, channels, seq_len)
    x_torch = torch.randn(batch_size, hidden_channels, seq_len)
    x_mask_torch = torch.ones(batch_size, 1, seq_len, dtype=torch.bool)
    g_torch = torch.randn(batch_size, gin_channels, 1)
    
    # MLX格式: (batch, seq_len, channels)
    x_mlx = torch_to_mlx(x_torch.transpose(1, 2))
    x_mask_mlx = torch_to_mlx(x_mask_torch.transpose(1, 2))
    g_mlx = torch_to_mlx(g_torch.transpose(1, 2))
    
    print(f"输入形状:")
    print(f"  PyTorch: x={x_torch.shape}, mask={x_mask_torch.shape}, g={g_torch.shape}")
    print(f"  MLX: x={x_mlx.shape}, mask={x_mask_mlx.shape}, g={g_mlx.shape}")
    
    # 测试PyTorch版本
    pytorch_output = pytorch_wavenet(x_torch, x_mask_torch, g_torch)
    print(f"PyTorch输出: {pytorch_output.shape}, range=[{torch.min(pytorch_output):.6f}, {torch.max(pytorch_output):.6f}]")
    
    # 测试MLX版本
    mlx_output = mlx_wavenet(x_mlx, x_mask_mlx, g_mlx)
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
    
    # 逐层分析残差连接
    print(f"\n=== 逐层残差连接分析 ===")
    
    # 手动执行PyTorch WaveNet的逐层计算
    pytorch_x = x_torch.clone()
    pytorch_output_manual = torch.zeros_like(pytorch_x)
    n_channels_tensor = torch.IntTensor([hidden_channels])
    
    if g_torch is not None:
        pytorch_g_cond = pytorch_wavenet.cond_layer(g_torch)
    
    for i in range(n_layers):
        print(f"\n--- Layer {i} ---")
        
        # PyTorch计算
        pytorch_x_in = pytorch_wavenet.in_layers[i](pytorch_x)
        if g_torch is not None:
            cond_offset = i * 2 * hidden_channels
            pytorch_g_l = pytorch_g_cond[:, cond_offset:cond_offset + 2 * hidden_channels, :]
        else:
            pytorch_g_l = torch.zeros_like(pytorch_x_in)
        
        from indextts.s2mel.modules.commons import fused_add_tanh_sigmoid_multiply
        pytorch_acts = fused_add_tanh_sigmoid_multiply(pytorch_x_in, pytorch_g_l, n_channels_tensor)
        pytorch_acts = pytorch_wavenet.drop(pytorch_acts)
        
        pytorch_res_skip_acts = pytorch_wavenet.res_skip_layers[i](pytorch_acts)
        
        if i < n_layers - 1:
            pytorch_res_acts = pytorch_res_skip_acts[:, :hidden_channels, :]
            pytorch_x = (pytorch_x + pytorch_res_acts) * x_mask_torch
            pytorch_output_manual = pytorch_output_manual + pytorch_res_skip_acts[:, hidden_channels:, :]
        else:
            pytorch_output_manual = pytorch_output_manual + pytorch_res_skip_acts
        
        print(f"PyTorch: x_in={pytorch_x_in.shape}, acts={pytorch_acts.shape}, res_acts={pytorch_res_acts.shape if i < n_layers-1 else 'N/A'}")
        print(f"  x range: [{torch.min(pytorch_x):.6f}, {torch.max(pytorch_x):.6f}]")
        print(f"  output range: [{torch.min(pytorch_output_manual):.6f}, {torch.max(pytorch_output_manual):.6f}]")
    
    pytorch_final = pytorch_output_manual * x_mask_torch
    print(f"\nPyTorch手动计算最终输出: {pytorch_final.shape}, range=[{torch.min(pytorch_final):.6f}, {torch.max(pytorch_final):.6f}]")
    
    # 对比手动计算和直接调用
    manual_vs_direct_diff = torch.abs(pytorch_final - pytorch_output)
    print(f"PyTorch手动vs直接调用差异: max={torch.max(manual_vs_direct_diff):.6f}, mean={torch.mean(manual_vs_direct_diff):.6f}")

if __name__ == "__main__":
    test_residual_connection()
