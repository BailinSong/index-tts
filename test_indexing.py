"""
测试MLX的索引选择
"""

import mlx.core as mx

x = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])  # (1, 5)
print(f"Original: {x}")

# 切片
slice1 = x[:, 1:3]  # 索引1,2 -> 值2,3
print(f"Slice [1:3]: {slice1}")  # 应该是[[2, 3]]

# 反转索引选择
indices = mx.array([1, 0])
result = slice1[:, indices]
print(f"slice1[:, [1,0]]: {result}")  # 应该是[[3, 2]]

# 直接从原数组选择
indices_direct = mx.array([2, 1])  # 直接选索引2,1
result_direct = x[:, indices_direct]
print(f"x[:, [2,1]]: {result_direct}")  # 应该也是[[3, 2]]

