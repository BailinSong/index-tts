#!/usr/bin/env python3
"""
详细对比PyTorch和MLX WaveNet实现
确保MLX版本与PyTorch版本完全一致
"""

import sys
sys.path.append('.')
import torch
import mlx.core as mx
import numpy as np
from indextts.s2mel.modules.wavenet import WN as PyTorchWaveNet
from indextts.s2mel.modules.mlx_wavenet import MLXWaveNet
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def test_fused_activation():
    """测试fused_add_tanh_sigmoid_multiply函数"""
    print("=== 测试fused activation函数 ===")
    
    # 创建测试数据
    batch, seq_len, channels = 1, 100, 512
    n_channels = 256  # 一半通道用于tanh，一半用于sigmoid
    
    # PyTorch格式: (batch, channels, seq_len)
    input_a_torch = torch.randn(batch, 2 * n_channels, seq_len)
    input_b_torch = torch.randn(batch, 2 * n_channels, seq_len)
    n_channels_tensor = torch.IntTensor([n_channels])
    
    # MLX格式: (batch, seq_len, channels)
    input_a_mlx = torch_to_mlx(input_a_torch.transpose(1, 2))
    input_b_mlx = torch_to_mlx(input_b_torch.transpose(1, 2))
    
    # PyTorch实现
    from indextts.s2mel.modules.commons import fused_add_tanh_sigmoid_multiply
    pytorch_out = fused_add_tanh_sigmoid_multiply(input_a_torch, input_b_torch, n_channels_tensor)
    
    # MLX实现
    from indextts.s2mel.modules.mlx_wavenet import fused_add_tanh_sigmoid_multiply_mlx
    mlx_out = fused_add_tanh_sigmoid_multiply_mlx(input_a_mlx, input_b_mlx, n_channels)
    
    # 转换MLX输出到PyTorch格式进行对比
    mlx_out_torch = mlx_to_torch(mlx_out).transpose(1, 2)
    
    print(f"PyTorch输出形状: {pytorch_out.shape}")
    print(f"MLX输出形状: {mlx_out.shape}")
    print(f"MLX转PyTorch格式形状: {mlx_out_torch.shape}")
    
    # 计算差异
    diff = torch.abs(pytorch_out - mlx_out_torch)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    print(f"最大差异: {max_diff:.6f}")
    print(f"平均差异: {mean_diff:.6f}")
    
    return max_diff < 1e-5, pytorch_out, mlx_out_torch

def test_wavenet_forward():
    """测试完整的WaveNet forward pass"""
    print("\n=== 测试完整WaveNet forward pass ===")
    
    # 设置参数
    batch_size = 1
    seq_len = 100
    hidden_channels = 512
    gin_channels = 512
    kernel_size = 5
    dilation_rate = 2
    n_layers = 8
    
    # 创建PyTorch WaveNet
    pytorch_wavenet = PyTorchWaveNet(
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        n_layers=n_layers,
        gin_channels=gin_channels,
        p_dropout=0.0
    )
    
    # 创建MLX WaveNet
    mlx_wavenet = MLXWaveNet(
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        n_layers=n_layers,
        gin_channels=gin_channels,
        p_dropout=0.0
    )
    
    # 创建测试输入
    torch.manual_seed(42)
    x_torch = torch.randn(batch_size, hidden_channels, seq_len)
    x_mask_torch = torch.ones(batch_size, 1, seq_len, dtype=torch.bool)
    g_torch = torch.randn(batch_size, gin_channels, 1)
    
    # 转换到MLX格式
    x_mlx = torch_to_mlx(x_torch.transpose(1, 2))  # (batch, seq_len, channels)
    x_mask_mlx = torch_to_mlx(x_mask_torch.transpose(1, 2))  # (batch, seq_len, 1)
    g_mlx = torch_to_mlx(g_torch.transpose(1, 2))  # (batch, 1, channels)
    
    print(f"输入形状:")
    print(f"  PyTorch x: {x_torch.shape}, mask: {x_mask_torch.shape}, g: {g_torch.shape}")
    print(f"  MLX x: {x_mlx.shape}, mask: {x_mask_mlx.shape}, g: {g_mlx.shape}")
    
    # 运行PyTorch WaveNet
    with torch.no_grad():
        pytorch_out = pytorch_wavenet(x_torch, x_mask_torch, g_torch)
    
    # 运行MLX WaveNet
    mlx_out = mlx_wavenet(x_mlx, x_mask_mlx, g_mlx)
    
    # 转换MLX输出到PyTorch格式
    mlx_out_torch = mlx_to_torch(mlx_out).transpose(1, 2)
    
    print(f"\n输出形状:")
    print(f"  PyTorch: {pytorch_out.shape}")
    print(f"  MLX: {mlx_out.shape}")
    print(f"  MLX转PyTorch: {mlx_out_torch.shape}")
    
    # 计算差异
    diff = torch.abs(pytorch_out - mlx_out_torch)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    print(f"\n差异统计:")
    print(f"  最大差异: {max_diff:.6f}")
    print(f"  平均差异: {mean_diff:.6f}")
    print(f"  差异>1e-3的元素比例: {torch.sum(diff > 1e-3).item() / diff.numel():.2%}")
    
    return max_diff, mean_diff, pytorch_out, mlx_out_torch

def analyze_layer_differences():
    """分析逐层差异"""
    print("\n=== 分析逐层差异 ===")
    
    # 这里需要修改MLX WaveNet以支持逐层输出
    # 暂时跳过，因为需要修改MLX WaveNet的内部实现
    print("需要修改MLX WaveNet以支持逐层调试")

def main():
    """主测试函数"""
    print("开始对比PyTorch和MLX WaveNet实现...")
    
    # 设置随机种子
    torch.manual_seed(42)
    mx.random.seed(42)
    
    # 测试fused activation函数
    success, pytorch_act, mlx_act = test_fused_activation()
    if not success:
        print("❌ fused activation函数差异过大!")
        return
    
    # 测试完整WaveNet
    max_diff, mean_diff, pytorch_out, mlx_out = test_wavenet_forward()
    
    if max_diff < 1e-3:
        print("✅ WaveNet实现基本一致!")
    elif max_diff < 1e-1:
        print("⚠️ WaveNet实现有差异，但可接受")
    else:
        print("❌ WaveNet实现差异过大，需要修复!")
    
    # 分析逐层差异
    analyze_layer_differences()

if __name__ == "__main__":
    main()
