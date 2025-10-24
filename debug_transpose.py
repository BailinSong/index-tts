#!/usr/bin/env python3
"""
调试MLX transpose操作
"""

import mlx.core as mx
import numpy as np

# 创建测试数据
x_mask = mx.ones((1, 1, 100), dtype=mx.bool_)
print(f"原始mask形状: {x_mask.shape}")

# 测试不同的transpose方法
try:
    # 方法1: 使用tuple
    x_mask_1 = x_mask.transpose((0, 2, 1))
    print(f"方法1结果: {x_mask_1.shape}")
except Exception as e:
    print(f"方法1失败: {e}")

try:
    # 方法2: 使用list
    x_mask_2 = x_mask.transpose([0, 2, 1])
    print(f"方法2结果: {x_mask_2.shape}")
except Exception as e:
    print(f"方法2失败: {e}")

try:
    # 方法3: 使用reshape
    x_mask_3 = x_mask.reshape(1, 100, 1)
    print(f"方法3结果: {x_mask_3.shape}")
except Exception as e:
    print(f"方法3失败: {e}")

try:
    # 方法4: 使用squeeze和expand
    x_mask_4 = x_mask.squeeze(1).expand_dims(2)
    print(f"方法4结果: {x_mask_4.shape}")
except Exception as e:
    print(f"方法4失败: {e}")
