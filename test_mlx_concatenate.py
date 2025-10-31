#!/usr/bin/env python3
"""
测试 MLX concatenate 功能
"""

import mlx.core as mx
import numpy as np

def test_concatenate():
    # 创建测试数据
    a = mx.array(np.random.randn(2, 415, 80))
    b = mx.array(np.random.randn(2, 415, 80))
    c = mx.array(np.random.randn(2, 415, 512))
    
    print(f"a shape: {a.shape}")
    print(f"b shape: {b.shape}")
    print(f"c shape: {c.shape}")
    
    try:
        result = mx.concatenate([a, b, c], -1)
        print(f"Success! Result shape: {result.shape}")
    except Exception as e:
        print(f"Failed: {e}")
        print(f"Error type: {type(e)}")

if __name__ == "__main__":
    test_concatenate()







