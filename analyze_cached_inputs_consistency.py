#!/usr/bin/env python3
"""
分析cfm_inputs_mlx.pkl和cfm_inputs_torch.pkl两个缓存输入数据是否一致
"""

import sys
import os
import pickle
import numpy as np
from pathlib import Path

def analyze_cached_inputs_consistency():
    """分析两个缓存输入数据的一致性"""
    print("=== 分析缓存输入数据一致性 ===\n")
    
    # 检查缓存文件是否存在
    mlx_cache_file = "cfm_inputs_mlx.pkl"
    torch_cache_file = "cfm_inputs_torch.pkl"
    
    if not os.path.exists(mlx_cache_file):
        print(f"❌ 未找到MLX缓存文件: {mlx_cache_file}")
        return False
    
    if not os.path.exists(torch_cache_file):
        print(f"❌ 未找到PyTorch缓存文件: {torch_cache_file}")
        return False
    
    print(f"✅ 找到MLX缓存文件: {mlx_cache_file}")
    print(f"✅ 找到PyTorch缓存文件: {torch_cache_file}")
    
    # 加载缓存数据
    print("\n1. 加载缓存数据...")
    try:
        with open(mlx_cache_file, 'rb') as f:
            mlx_data = pickle.load(f)
        print(f"   MLX缓存数据键: {list(mlx_data.keys())}")
        
        with open(torch_cache_file, 'rb') as f:
            torch_data = pickle.load(f)
        print(f"   PyTorch缓存数据键: {list(torch_data.keys())}")
        
    except Exception as e:
        print(f"   ❌ 加载缓存数据失败: {e}")
        return False
    
    # 比较数据键
    print("\n2. 比较数据键...")
    mlx_keys = set(mlx_data.keys())
    torch_keys = set(torch_data.keys())
    
    common_keys = mlx_keys & torch_keys
    mlx_only_keys = mlx_keys - torch_keys
    torch_only_keys = torch_keys - mlx_keys
    
    print(f"   共同键: {len(common_keys)} 个")
    print(f"   仅MLX键: {len(mlx_only_keys)} 个")
    print(f"   仅PyTorch键: {len(torch_only_keys)} 个")
    
    if mlx_only_keys:
        print(f"   仅MLX键: {list(mlx_only_keys)}")
    if torch_only_keys:
        print(f"   仅PyTorch键: {list(torch_only_keys)}")
    
    # 比较共同键的数据
    print("\n3. 比较共同键的数据...")
    max_diff = 0.0
    mean_diff = 0.0
    total_elements = 0
    
    for key in sorted(common_keys):
        mlx_value = mlx_data[key]
        torch_value = torch_data[key]
        
        print(f"\n   --- 键: {key} ---")
        print(f"   MLX类型: {type(mlx_value)}")
        print(f"   PyTorch类型: {type(torch_value)}")
        
        # 处理numpy数组
        if isinstance(mlx_value, np.ndarray) and isinstance(torch_value, np.ndarray):
            print(f"   MLX形状: {mlx_value.shape}, 范围: [{mlx_value.min():.6f}, {mlx_value.max():.6f}]")
            print(f"   PyTorch形状: {torch_value.shape}, 范围: [{torch_value.min():.6f}, {torch_value.max():.6f}]")
            
            if mlx_value.shape == torch_value.shape:
                diff = np.abs(mlx_value - torch_value)
                key_max_diff = diff.max()
                key_mean_diff = diff.mean()
                key_elements = diff.size
                
                print(f"   差异: 最大 {key_max_diff:.10f}, 平均 {key_mean_diff:.10f}")
                
                if key_max_diff < 1e-5:
                    print(f"   ✅ 差异在e-5范围内")
                elif key_max_diff < 1e-3:
                    print(f"   ⚠️ 差异在e-3范围内")
                else:
                    print(f"   ❌ 差异较大")
                
                max_diff = max(max_diff, key_max_diff)
                mean_diff += key_mean_diff * key_elements
                total_elements += key_elements
            else:
                print(f"   ❌ 形状不匹配")
        
        # 处理标量值
        elif isinstance(mlx_value, (int, float)) and isinstance(torch_value, (int, float)):
            print(f"   MLX值: {mlx_value}")
            print(f"   PyTorch值: {torch_value}")
            diff = abs(mlx_value - torch_value)
            print(f"   差异: {diff:.10f}")
            
            if diff < 1e-5:
                print(f"   ✅ 差异在e-5范围内")
            elif diff < 1e-3:
                print(f"   ⚠️ 差异在e-3范围内")
            else:
                print(f"   ❌ 差异较大")
            
            max_diff = max(max_diff, diff)
            mean_diff += diff
            total_elements += 1
        
        # 处理列表
        elif isinstance(mlx_value, list) and isinstance(torch_value, list):
            print(f"   MLX长度: {len(mlx_value)}")
            print(f"   PyTorch长度: {len(torch_value)}")
            
            if len(mlx_value) == len(torch_value):
                print(f"   MLX内容: {mlx_value}")
                print(f"   PyTorch内容: {torch_value}")
                
                if mlx_value == torch_value:
                    print(f"   ✅ 内容完全一致")
                else:
                    print(f"   ❌ 内容不一致")
            else:
                print(f"   ❌ 长度不匹配")
        
        else:
            print(f"   MLX值: {mlx_value}")
            print(f"   PyTorch值: {torch_value}")
            print(f"   ❌ 类型不匹配或无法比较")
    
    # 计算总体差异
    if total_elements > 0:
        mean_diff = mean_diff / total_elements
        print(f"\n4. 总体差异统计:")
        print(f"   最大差异: {max_diff:.10f}")
        print(f"   平均差异: {mean_diff:.10f}")
        print(f"   总元素数: {total_elements}")
        
        if max_diff < 1e-5:
            print(f"   ✅ 缓存输入数据完全一致 (差异 < e-5)")
        elif max_diff < 1e-3:
            print(f"   ⚠️ 缓存输入数据基本一致 (差异 < e-3)")
        else:
            print(f"   ❌ 缓存输入数据存在显著差异")
    else:
        print(f"\n4. 无法计算总体差异")
    
    # 保存分析结果
    print("\n5. 保存分析结果...")
    analysis_result = {
        'mlx_keys': list(mlx_keys),
        'torch_keys': list(torch_keys),
        'common_keys': list(common_keys),
        'mlx_only_keys': list(mlx_only_keys),
        'torch_only_keys': list(torch_only_keys),
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'total_elements': total_elements,
        'consistent': max_diff < 1e-5 if total_elements > 0 else False
    }
    
    with open('cached_inputs_consistency_analysis.pkl', 'wb') as f:
        pickle.dump(analysis_result, f)
    
    print(f"   分析结果已保存到: cached_inputs_consistency_analysis.pkl")
    
    return max_diff < 1e-5 if total_elements > 0 else False

if __name__ == "__main__":
    try:
        consistent = analyze_cached_inputs_consistency()
        if consistent:
            print("\n✅ 缓存输入数据完全一致")
        else:
            print("\n❌ 缓存输入数据存在差异")
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
