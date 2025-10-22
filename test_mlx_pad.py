"""
测试MLX的pad功能
"""

import mlx.core as mx
import numpy as np

# 创建测试数据
x = mx.array([[[1, 2, 3, 4, 5]]])  # (1, 1, 5)
print(f"Original: {x.shape}")
print(f"Values: {x}")

# 测试不同的padding格式
print("\n测试padding格式:")

# 格式1: ((dim0), (dim1), (dim2))
try:
    pad_width = ((0, 0), (2, 2), (0, 0))
    padded = mx.pad(x, pad_width)
    print(f"✅ 格式1成功: pad_width={pad_width}, 结果shape={padded.shape}")
    print(f"   Values: {padded}")
except Exception as e:
    print(f"❌ 格式1失败: {e}")

# 格式2: [left, right] for all dims
try:
    pad_width = [0, 0, 2, 2, 0, 0]
    padded = mx.pad(x, pad_width)
    print(f"✅ 格式2成功: pad_width={pad_width}, 结果shape={padded.shape}")
    print(f"   Values: {padded}")
except Exception as e:
    print(f"❌ 格式2失败: {e}")

# 测试mode参数
print("\n测试mode:")
try:
    pad_width = ((0, 0), (2, 2), (0, 0))
    padded_reflect = mx.pad(x, pad_width, mode='reflect')
    print(f"✅ mode='reflect'成功")
    print(f"   Values: {padded_reflect}")
except Exception as e:
    print(f"❌ mode='reflect'失败: {e}")

try:
    padded_constant = mx.pad(x, pad_width, mode='constant', constant_values=0)
    print(f"✅ mode='constant'成功")
    print(f"   Values: {padded_constant}")
except Exception as e:
    print(f"❌ mode='constant'失败: {e}")

# 测试reflect的正确性
print("\n验证reflect padding:")
x_test = mx.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
pad_width = ((0, 0), (2, 2), (0, 0))
try:
    padded = mx.pad(x_test, pad_width, mode='reflect')
    print(f"Input: {x_test.squeeze()}")
    print(f"Padded: {padded.squeeze()}")
    print(f"Expected: [3, 2, 1, 2, 3, 4, 5, 4, 3] (reflect at boundaries)")
except Exception as e:
    print(f"Error: {e}")

