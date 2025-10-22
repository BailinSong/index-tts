"""
对比PyTorch和MLX的reflect padding实现
"""

import torch
import torch.nn.functional as F
import mlx.core as mx
import numpy as np

# 创建测试数据
x_torch = torch.arange(1, 11, dtype=torch.float32).reshape(1, 1, 10)
print("Original PyTorch tensor: (1, 1, 10)")
print(x_torch.squeeze())

# PyTorch reflect padding
# F.pad format: (left, right) for last dimension
padding_left = 2
padding_right = 2
x_padded_torch = F.pad(x_torch, (padding_left, padding_right), mode='reflect')
print(f"\nPyTorch reflect pad ({padding_left}, {padding_right}):")
print(f"Shape: {x_padded_torch.shape}")
print(f"Values: {x_padded_torch.squeeze()}")

# MLX手动实现
x_mlx = mx.array(x_torch.squeeze().numpy()).reshape(1, 10, 1)  # (B, T, C)
print(f"\nMLX input: {x_mlx.shape}")
print(f"Values: {x_mlx.squeeze()}")

# 使用mlx_wavenet.py中的函数
from indextts.s2mel.modules.mlx_wavenet import mlx_pad_reflect_1d
x_padded_mlx = mlx_pad_reflect_1d(x_mlx, padding_left, padding_right)

print(f"\nMLX手动reflect pad:")
print(f"Shape: {x_padded_mlx.shape}")
print(f"Values: {x_padded_mlx.squeeze()}")

# 对比
torch_padded_np = x_padded_torch.squeeze().numpy()
mlx_padded_np = np.array(x_padded_mlx.squeeze())

print(f"\n对比:")
print(f"PyTorch: {torch_padded_np}")
print(f"MLX:     {mlx_padded_np}")
print(f"差异: {np.abs(torch_padded_np - mlx_padded_np).max():.6f}")

if np.allclose(torch_padded_np, mlx_padded_np):
    print("✅ Reflect padding完全一致！")
else:
    print("❌ Reflect padding有差异！")
    
    # 详细分析
    print("\n详细对比:")
    print("Index | PyTorch | MLX | 说明")
    print("------|---------|-----|-----")
    for i in range(len(torch_padded_np)):
        if i < padding_left:
            note = "左padding"
        elif i >= len(torch_padded_np) - padding_right:
            note = "右padding"
        else:
            note = "原始"
        print(f"{i:5d} | {torch_padded_np[i]:7.1f} | {mlx_padded_np[i]:7.1f} | {note}")

