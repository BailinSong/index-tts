#!/usr/bin/env python3
"""
分析 PyTorch 和 MLX 转置函数 API 参数差异
"""

import sys
import os
sys.path.append('/Users/bailin/index-tts')

import torch
import mlx.core as mx
import numpy as np

def analyze_transpose_api_differences():
    """分析转置函数 API 差异"""
    print("🔍 分析 PyTorch 和 MLX 转置函数 API 差异")
    print("="*60)
    
    # 创建测试张量
    print("📊 创建测试张量:")
    pytorch_tensor = torch.randn(2, 80, 415)
    mlx_tensor = mx.array(pytorch_tensor.detach().cpu().numpy())
    
    print(f"   原始形状: PyTorch {pytorch_tensor.shape}, MLX {mlx_tensor.shape}")
    
    # 测试不同的转置操作
    print(f"\n🔍 转置操作对比:")
    print("="*40)
    
    # PyTorch transpose(1, 2) - 交换维度1和2
    pytorch_transpose_1_2 = pytorch_tensor.transpose(1, 2)
    print(f"PyTorch transpose(1, 2): {pytorch_transpose_1_2.shape}")
    
    # MLX transpose(0, 2, 1) - 交换维度0和2
    mlx_transpose_0_2_1 = mlx_tensor.transpose(0, 2, 1)
    print(f"MLX transpose(0, 2, 1): {mlx_transpose_0_2_1.shape}")
    
    # 检查是否等价
    pytorch_np = pytorch_transpose_1_2.detach().cpu().numpy()
    mlx_np = np.array(mlx_transpose_0_2_1)
    
    if pytorch_np.shape == mlx_np.shape:
        diff = np.abs(pytorch_np - mlx_np)
        max_diff = np.max(diff)
        print(f"✅ 形状一致，数值差异: {max_diff:.2e}")
    else:
        print(f"❌ 形状不一致")
    
    # 测试其他转置组合
    print(f"\n🔍 其他转置组合测试:")
    print("="*40)
    
    # PyTorch transpose(0, 1) - 交换维度0和1
    pytorch_transpose_0_1 = pytorch_tensor.transpose(0, 1)
    print(f"PyTorch transpose(0, 1): {pytorch_transpose_0_1.shape}")
    
    # MLX transpose(1, 0, 2) - 交换维度1和0
    mlx_transpose_1_0_2 = mlx_tensor.transpose(1, 0, 2)
    print(f"MLX transpose(1, 0, 2): {mlx_transpose_1_0_2.shape}")
    
    # 检查是否等价
    pytorch_np_01 = pytorch_transpose_0_1.detach().cpu().numpy()
    mlx_np_102 = np.array(mlx_transpose_1_0_2)
    
    if pytorch_np_01.shape == mlx_np_102.shape:
        diff_01 = np.abs(pytorch_np_01 - mlx_np_102)
        max_diff_01 = np.max(diff_01)
        print(f"✅ 形状一致，数值差异: {max_diff_01:.2e}")
    else:
        print(f"❌ 形状不一致")
    
    # 测试 MLX 的其他转置方式
    print(f"\n🔍 MLX 其他转置方式:")
    print("="*40)
    
    # MLX transpose(0, 1, 2) - 交换维度0和1
    mlx_transpose_0_1_2 = mlx_tensor.transpose(0, 1, 2)
    print(f"MLX transpose(0, 1, 2): {mlx_transpose_0_1_2.shape}")
    
    # MLX transpose(1, 2, 0) - 交换维度1和2
    mlx_transpose_1_2_0 = mlx_tensor.transpose(1, 2, 0)
    print(f"MLX transpose(1, 2, 0): {mlx_transpose_1_2_0.shape}")
    
    # 检查哪个与 PyTorch transpose(1, 2) 等价
    pytorch_np_12 = pytorch_transpose_1_2.detach().cpu().numpy()
    mlx_np_012 = np.array(mlx_transpose_0_1_2)
    mlx_np_120 = np.array(mlx_transpose_1_2_0)
    
    print(f"PyTorch transpose(1, 2) 形状: {pytorch_np_12.shape}")
    print(f"MLX transpose(0, 1, 2) 形状: {mlx_np_012.shape}")
    print(f"MLX transpose(1, 2, 0) 形状: {mlx_np_120.shape}")
    
    # 只比较形状相同的
    if pytorch_np_12.shape == mlx_np_012.shape:
        diff_012 = np.abs(pytorch_np_12 - mlx_np_012)
        print(f"MLX transpose(0, 1, 2) vs PyTorch transpose(1, 2): max_diff={np.max(diff_012):.2e}")
    else:
        print(f"MLX transpose(0, 1, 2) vs PyTorch transpose(1, 2): 形状不匹配")
    
    if pytorch_np_12.shape == mlx_np_120.shape:
        diff_120 = np.abs(pytorch_np_12 - mlx_np_120)
        print(f"MLX transpose(1, 2, 0) vs PyTorch transpose(1, 2): max_diff={np.max(diff_120):.2e}")
    else:
        print(f"MLX transpose(1, 2, 0) vs PyTorch transpose(1, 2): 形状不匹配")
    
    # 分析 DiT 中的具体转置操作
    print(f"\n🔍 DiT 中的转置操作分析:")
    print("="*40)
    
    # 模拟 DiT 中的转置操作
    print("1. x.transpose(1, 2) - PyTorch 版本:")
    print(f"   输入: [2, 80, 415]")
    print(f"   输出: {pytorch_transpose_1_2.shape}")
    
    print("\n2. x.transpose(0, 2, 1) - MLX 版本:")
    print(f"   输入: [2, 80, 415]")
    print(f"   输出: {mlx_transpose_0_2_1.shape}")
    
    # 检查 MLX 是否有等效的 transpose(1, 2)
    print(f"\n🔍 寻找 MLX 的等效 transpose(1, 2):")
    print("="*40)
    
    # 尝试不同的 MLX 转置方式
    mlx_candidates = [
        ("transpose(0, 1, 2)", mlx_tensor.transpose(0, 1, 2)),
        ("transpose(0, 2, 1)", mlx_tensor.transpose(0, 2, 1)),
        ("transpose(1, 0, 2)", mlx_tensor.transpose(1, 0, 2)),
        ("transpose(1, 2, 0)", mlx_tensor.transpose(1, 2, 0)),
        ("transpose(2, 0, 1)", mlx_tensor.transpose(2, 0, 1)),
        ("transpose(2, 1, 0)", mlx_tensor.transpose(2, 1, 0)),
    ]
    
    for name, result in mlx_candidates:
        result_np = np.array(result)
        if result_np.shape == pytorch_np_12.shape:
            diff = np.abs(pytorch_np_12 - result_np)
            max_diff = np.max(diff)
            print(f"   {name}: 形状一致 {result_np.shape}, 数值差异: {max_diff:.2e}")
            if max_diff < 1e-6:
                print(f"   ✅ {name} 与 PyTorch transpose(1, 2) 完全等价!")
        else:
            print(f"   {name}: 形状不一致 {result_np.shape}")

def main():
    analyze_transpose_api_differences()

if __name__ == "__main__":
    main()
