"""
调试reflect padding的具体实现
"""

import mlx.core as mx
import numpy as np

# 创建测试数据 (1, 10, 1)
x = mx.array([[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]]])
print(f"Original x: shape={x.shape}")
print(f"Values (squeezed): {x.squeeze()}")

padding_left = 2
padding_right = 2
batch, seq_len, channels = x.shape

print(f"\n目标: PyTorch reflect [3, 2, 1,2,3,4,5,6,7,8,9,10, 9, 8]")

# 左侧padding
print(f"\n=== 左侧padding ===")
pad_size = min(padding_left, seq_len - 1)
print(f"pad_size = {pad_size}")

# 取切片
left_slice = x[:, 1:pad_size+1, :]  # 索引1到2（不含3）
print(f"left_slice (索引1到{pad_size}): {left_slice.squeeze()}")

# 反转
reverse_indices = mx.arange(pad_size - 1, -1, -1)
print(f"reverse_indices: {reverse_indices}")
left_pad = left_slice[:, reverse_indices, :]
print(f"left_pad (reversed): {left_pad.squeeze()}")

# 拼接
x_with_left = mx.concatenate([left_pad, x], axis=1)
print(f"After left pad: {x_with_left.squeeze()}")

# 右侧padding
print(f"\n=== 右侧padding ===")
original_end_idx = seq_len + padding_left - 1
print(f"original_end_idx = {original_end_idx} (值应该是10)")

pad_size_r = min(padding_right, seq_len - 1)
# 取切片: [original_end_idx-pad_size:original_end_idx]
# 应该是索引10,11（值9,10）...不对，10是索引9的值
# original_end_idx = 10 + 2 - 1 = 11 (这是padding后的索引11，值是10)

print(f"需要取的索引range: [{original_end_idx-pad_size_r}:{original_end_idx}]")
right_slice = x_with_left[:, original_end_idx-pad_size_r:original_end_idx, :]
print(f"right_slice: {right_slice.squeeze()}")

reverse_indices_r = mx.arange(pad_size_r - 1, -1, -1)
print(f"reverse_indices: {reverse_indices_r}")
right_pad = right_slice[:, reverse_indices_r, :]
print(f"right_pad (reversed): {right_pad.squeeze()}")

# 最终
x_final = mx.concatenate([x_with_left, right_pad], axis=1)
print(f"\nFinal: {x_final.squeeze()}")
print(f"Expected: [3, 2, 1,2,3,4,5,6,7,8,9,10, 9, 8]")

