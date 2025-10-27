#!/usr/bin/env python3
"""
直接分析analyze_cfm_differences_fixed.py中的输入差异来源
"""

import sys
import os
import torch
import numpy as np
import pickle
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def analyze_input_differences_direct():
    """直接分析输入差异"""
    print("=== 直接分析analyze_cfm_differences_fixed.py中的输入差异 ===\n")
    
    # 从analyze_cfm_differences_fixed.py输出中提取的数据
    print("1. 从analyze_cfm_differences_fixed.py输出中提取的数据:")
    pytorch_x_range = [-4.465604, 4.479084]
    mlx_x_range = [-4.462968, 3.760155]
    
    print(f"   PyTorch x范围: [{pytorch_x_range[0]:.6f}, {pytorch_x_range[1]:.6f}]")
    print(f"   MLX x范围: [{mlx_x_range[0]:.6f}, {mlx_x_range[1]:.6f}]")
    
    # 计算差异
    min_diff = abs(pytorch_x_range[0] - mlx_x_range[0])
    max_diff = abs(pytorch_x_range[1] - mlx_x_range[1])
    
    print(f"\n2. 差异分析:")
    print(f"   最小值差异: {min_diff:.6f}")
    print(f"   最大值差异: {max_diff:.6f}")
    print(f"   范围差异: {max_diff - min_diff:.6f}")
    
    # 分析差异特征
    print(f"\n3. 差异特征分析:")
    print(f"   - 最小值差异很小 ({min_diff:.6f})，说明基础数值接近")
    print(f"   - 最大值差异较大 ({max_diff:.6f})，说明MLX版本的上界被压缩")
    print(f"   - 范围差异: {max_diff - min_diff:.6f}，MLX版本范围更小")
    
    # 分析可能的原因
    print(f"\n4. 可能的原因分析:")
    print(f"   1. **数值精度差异**: MLX和PyTorch的数值计算精度不同")
    print(f"   2. **数据类型转换**: 在PyTorch->MLX转换过程中的精度损失")
    print(f"   3. **随机数生成差异**: 不同框架的随机数生成器产生不同序列")
    print(f"   4. **推理步骤差异**: 不同的推理路径或时间点")
    print(f"   5. **缓存加载差异**: 不同的缓存加载时机或内容")
    
    # 检查缓存文件
    print(f"\n5. 检查相关缓存文件:")
    cache_files = [
        "cfm_inputs.pkl",
        "cfm_outputs_mlx.pkl",
        "cfm_outputs_pytorch.pkl",
        "s2mel.npz"
    ]
    
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            print(f"   ✅ 找到: {cache_file}")
            try:
                if cache_file.endswith('.pkl'):
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    print(f"      键: {list(data.keys())}")
                    
                    if 'x' in data:
                        x_data = data['x']
                        print(f"      x形状: {x_data.shape}")
                        print(f"      x范围: [{x_data.min():.6f}, {x_data.max():.6f}]")
                        
                        # 检查是否与analyze_cfm_differences_fixed.py一致
                        if abs(x_data.min() - mlx_x_range[0]) < 0.001 and abs(x_data.max() - mlx_x_range[1]) < 0.001:
                            print(f"      ✅ 与MLX范围一致")
                        elif abs(x_data.min() - pytorch_x_range[0]) < 0.001 and abs(x_data.max() - pytorch_x_range[1]) < 0.001:
                            print(f"      ✅ 与PyTorch范围一致")
                        else:
                            print(f"      ❌ 与两者都不一致")
                            
                elif cache_file.endswith('.npz'):
                    data = np.load(cache_file)
                    print(f"      键: {list(data.keys())}")
                    
            except Exception as e:
                print(f"      ❌ 读取失败: {e}")
        else:
            print(f"   ❌ 未找到: {cache_file}")
    
    # 分析差异的影响
    print(f"\n6. 差异影响分析:")
    print(f"   - 最小值差异 {min_diff:.6f} < 1e-5: {'✅ 可接受' if min_diff < 1e-5 else '❌ 不可接受'}")
    print(f"   - 最大值差异 {max_diff:.6f} < 1e-5: {'✅ 可接受' if max_diff < 1e-5 else '❌ 不可接受'}")
    print(f"   - 整体差异: {'✅ 在e-5范围内' if max_diff < 1e-5 else '❌ 超出e-5范围'}")
    
    # 建议的修复方向
    print(f"\n7. 建议的修复方向:")
    if max_diff > 1e-5:
        print(f"   ❌ 输入差异超出e-5要求，需要修复:")
        print(f"   1. 检查PyTorch->MLX转换过程中的精度损失")
        print(f"   2. 统一随机数生成器")
        print(f"   3. 检查推理步骤的一致性")
        print(f"   4. 验证缓存加载的正确性")
    else:
        print(f"   ✅ 输入差异在e-5范围内，无需修复")
    
    return {
        'min_diff': min_diff,
        'max_diff': max_diff,
        'within_e5': max_diff < 1e-5
    }

if __name__ == "__main__":
    try:
        results = analyze_input_differences_direct()
        print(f"\n✅ 输入差异分析完成")
        print(f"   结果: 差异在e-5范围内: {results['within_e5']}")
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
