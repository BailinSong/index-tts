#!/usr/bin/env python3
"""
测试PyTorch的reflect padding行为
"""

import torch
import torch.nn.functional as F
import numpy as np

def test_pytorch_reflect_padding():
    """测试PyTorch的reflect padding行为"""
    print("=== 测试PyTorch的reflect padding行为 ===")
    
    # 创建简单的测试数据
    x = torch.randn(1, 3, 5)  # (batch, channels, seq_len)
    print(f"输入形状: {x.shape}")
    print(f"输入数据:\n{x[0, 0, :]}")
    
    # 测试不同的padding
    paddings = [(2, 2), (1, 3), (3, 1), (0, 4), (4, 0)]
    
    for padding_left, padding_right in paddings:
        print(f"\n--- Padding: ({padding_left}, {padding_right}) ---")
        
        # 使用PyTorch的F.pad
        padded = F.pad(x, (padding_left, padding_right), mode='reflect')
        print(f"Padded形状: {padded.shape}")
        print(f"Padded数据:\n{padded[0, 0, :]}")
        
        # 手动实现reflect padding
        manual_padded = manual_reflect_pad(x, padding_left, padding_right)
        print(f"Manual padded形状: {manual_padded.shape}")
        print(f"Manual padded数据:\n{manual_padded[0, 0, :]}")
        
        # 比较差异
        diff = torch.abs(padded - manual_padded)
        print(f"差异: max={diff.max():.10f}, mean={diff.mean():.10f}")

def manual_reflect_pad(x, padding_left, padding_right):
    """手动实现reflect padding"""
    batch, channels, seq_len = x.shape
    
    if padding_left == 0 and padding_right == 0:
        return x
    
    # 左padding: 反转前padding_left个元素
    if padding_left > 0:
        left_reflect = x[:, :, padding_left-1::-1]  # 反转前padding_left个元素
        x = torch.cat([left_reflect, x], dim=2)
    
    # 右padding: 反转后padding_right个元素
    if padding_right > 0:
        right_reflect = x[:, :, -1:-padding_right-1:-1]  # 反转后padding_right个元素
        x = torch.cat([x, right_reflect], dim=2)
    
    return x

if __name__ == "__main__":
    test_pytorch_reflect_padding()