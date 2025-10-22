"""
调试dilated convolution的差异
"""

import torch
import numpy as np
import mlx.core as mx
import mlx.nn as nn

def torch_to_mlx(tensor):
    if isinstance(tensor, torch.Tensor):
        return mx.array(tensor.detach().cpu().numpy())
    return tensor

def test_dilated_conv():
    """测试简单的dilated convolution"""
    
    # 参数
    in_channels = 512
    out_channels = 1024
    kernel_size = 5
    dilation = 1  # 第一层 dilation=1
    
    # 计算padding
    # PyTorch SConv1d: padding = int((kernel_size * dilation - dilation) / 2)
    padding_pytorch = int((kernel_size * dilation - dilation) / 2)
    print(f"Dilated Conv参数:")
    print(f"  kernel_size={kernel_size}, dilation={dilation}")
    print(f"  PyTorch padding={padding_pytorch}")
    
    # 创建PyTorch Conv1d
    torch_conv = torch.nn.Conv1d(
        in_channels, out_channels,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding_pytorch,
        bias=True
    )
    torch_conv.eval()
    
    # 创建MLX Conv1d
    mlx_conv = nn.Conv1d(
        in_channels, out_channels,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding_pytorch,
        bias=True
    )
    
    # 复制权重
    # PyTorch: (O, I, K)
    # MLX: (O, K, I)
    w_torch = torch_conv.weight.data.numpy()
    b_torch = torch_conv.bias.data.numpy()
    
    w_mlx = w_torch.transpose(0, 2, 1)
    mlx_conv.weight = mx.array(w_mlx)
    mlx_conv.bias = mx.array(b_torch)
    
    # 测试数据
    batch = 1
    seq_len = 30
    x_torch = torch.randn(batch, in_channels, seq_len)
    
    # PyTorch forward
    with torch.no_grad():
        y_torch = torch_conv(x_torch)
    
    print(f"\nPyTorch Conv1d:")
    print(f"  Input: {x_torch.shape}")
    print(f"  Output: {y_torch.shape}")
    print(f"  Mean: {y_torch.mean():.6f}, Std: {y_torch.std():.6f}")
    
    # MLX forward
    x_mlx = torch_to_mlx(x_torch).transpose(0, 2, 1)  # (B, T, C)
    y_mlx = mlx_conv(x_mlx)
    mx.eval(y_mlx)
    
    print(f"\nMLX Conv1d:")
    print(f"  Input: {x_mlx.shape}")
    print(f"  Output: {y_mlx.shape}")
    print(f"  Mean: {float(mx.mean(y_mlx)):.6f}, Std: {float(mx.std(y_mlx)):.6f}")
    
    # 对比
    y_torch_np = y_torch.transpose(1, 2).numpy()
    y_mlx_np = np.array(y_mlx)
    
    max_diff = np.abs(y_torch_np - y_mlx_np).max()
    mean_diff = np.abs(y_torch_np - y_mlx_np).mean()
    corr = np.corrcoef(y_torch_np.flatten(), y_mlx_np.flatten())[0, 1]
    
    print(f"\n对比:")
    print(f"  Max diff: {max_diff:.8f}")
    print(f"  Mean diff: {mean_diff:.8f}")
    print(f"  Correlation: {corr:.6f}")
    
    if corr > 0.9999:
        print("✅ 完美一致！")
    elif corr > 0.99:
        print("✅ 很好！")
    else:
        print(f"❌ 有差异！")
        
        # 详细分析
        print("\n详细分析前几个输出:")
        print(f"PyTorch: {y_torch_np[0, :3, :5]}")
        print(f"MLX: {y_mlx_np[0, :3, :5]}")

if __name__ == "__main__":
    test_dilated_conv()

