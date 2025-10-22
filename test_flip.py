"""
测试MLX的flip和索引
"""

import mlx.core as mx
import numpy as np

# 创建简单数组
x = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])  # (1, 5)
print(f"Original: {x}")

# 测试切片
slice1 = x[:, 1:3]  # 索引1,2 -> 值2,3
print(f"Slice [1:3]: {slice1}")

# 测试手动反转（使用负索引步长）
# MLX不支持负步长，需要手动构建反转索引
reversed_indices = mx.array([1, 0])  # 反转[0,1]
flipped = slice1[:, reversed_indices]
print(f"Flipped (manual): {flipped}")

# 预期: [3, 2]
expected = np.array([[3.0, 2.0]])
actual = np.array(flipped)
print(f"\nExpected: {expected}")
print(f"Actual:   {actual}")
print(f"Match: {np.allclose(expected, actual)}")

