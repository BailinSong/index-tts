#!/usr/bin/env python3
"""
调试MLX CFM缓存加载错误
分析'Module does not have parameter named "0"'的原因
"""

import sys
import os
sys.path.append('.')

import torch
import numpy as np
import mlx.core as mx
from indextts.infer_v2 import IndexTTS2

def debug_cache_loading_error():
    """调试缓存加载错误"""
    
    print("=== 调试MLX CFM缓存加载错误 ===")
    
    # 初始化TTS
    tts = IndexTTS2(use_mlx=True)
    
    if tts.mlx_s2mel_cfm is None:
        print("❌ MLX CFM未初始化")
        return
    
    mlx_cfm = tts.mlx_s2mel_cfm
    
    # 加载缓存权重
    cache_file = "checkpoints/mlx/s2mel.npz"
    if not os.path.exists(cache_file):
        print(f"❌ 缓存文件不存在: {cache_file}")
        return
    
    print(f"📁 加载缓存文件: {cache_file}")
    cache_data = mx.load(cache_file)
    
    # 提取CFM权重
    cfm_weights = {}
    for key, value in cache_data.items():
        if key.startswith("models.cfm.estimator."):
            cfm_weights[key] = value
    
    print(f"📊 提取的CFM权重数量: {len(cfm_weights)}")
    
    # 分析权重键名
    print(f"\n🔍 分析权重键名:")
    sample_keys = list(cfm_weights.keys())[:10]
    for key in sample_keys:
        print(f"  {key}")
    
    # 测试unflatten_parameters函数
    print(f"\n🔧 测试unflatten_parameters函数:")
    
    def unflatten_parameters(flat_dict, prefix="models.cfm.estimator"):
        """Reconstruct nested dict from flattened parameters"""
        nested = {}
        matched_keys = []
        for key, value in flat_dict.items():
            if not key.startswith(prefix + "."):
                continue
            matched_keys.append(key)
            # Remove prefix
            rel_key = key[len(prefix) + 1:]
            parts = rel_key.split('.')
            
            # Navigate/create nested structure
            current = nested
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set leaf value
            current[parts[-1]] = value
        
        print(f"  匹配的权重键数量: {len(matched_keys)}")
        return nested
    
    nested_params = unflatten_parameters(cfm_weights, "models.cfm.estimator")
    
    # 分析嵌套结构
    print(f"\n📊 分析嵌套结构:")
    
    def analyze_nested_structure(obj, prefix="", max_depth=3, current_depth=0):
        """递归分析嵌套结构"""
        if current_depth >= max_depth:
            print(f"{prefix}... (max depth reached)")
            return
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, dict):
                    print(f"{prefix}{key}: dict with {len(value)} keys")
                    analyze_nested_structure(value, prefix + "  ", max_depth, current_depth + 1)
                else:
                    if hasattr(value, 'shape'):
                        print(f"{prefix}{key}: {type(value).__name__} {value.shape}")
                    else:
                        print(f"{prefix}{key}: {type(value).__name__}")
        else:
            print(f"{prefix}{type(obj).__name__}")
    
    analyze_nested_structure(nested_params)
    
    # 检查是否有数字键
    print(f"\n🔍 检查数字键:")
    
    def find_numeric_keys(obj, prefix=""):
        """查找数字键"""
        numeric_keys = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(key, str) and key.isdigit():
                    numeric_keys.append(f"{prefix}{key}")
                if isinstance(value, dict):
                    numeric_keys.extend(find_numeric_keys(value, f"{prefix}{key}."))
        return numeric_keys
    
    numeric_keys = find_numeric_keys(nested_params)
    if numeric_keys:
        print(f"  发现数字键: {numeric_keys}")
    else:
        print(f"  未发现数字键")
    
    # 检查MLX estimator的参数结构
    print(f"\n🔍 检查MLX estimator参数结构:")
    
    estimator = mlx_cfm.estimator
    print(f"  Estimator类型: {type(estimator)}")
    
    # 尝试获取estimator的参数
    try:
        # 检查estimator是否有parameters方法
        if hasattr(estimator, 'parameters'):
            params = estimator.parameters()
            print(f"  Estimator参数数量: {len(params)}")
            
            # 显示前几个参数名
            param_names = list(params.keys())[:10]
            print(f"  前10个参数名: {param_names}")
            
            # 检查是否有数字参数名
            numeric_param_names = [name for name in param_names if name.isdigit()]
            if numeric_param_names:
                print(f"  数字参数名: {numeric_param_names}")
            else:
                print(f"  无数字参数名")
                
        else:
            print(f"  Estimator没有parameters方法")
            
    except Exception as e:
        print(f"  ❌ 获取estimator参数失败: {e}")
    
    # 尝试手动更新一个简单的参数
    print(f"\n🔧 测试手动更新参数:")
    
    try:
        # 创建一个简单的测试参数
        test_params = {
            "cond_projection": {
                "weight": cfm_weights.get("models.cfm.estimator.cond_projection.weight")
            }
        }
        
        if test_params["cond_projection"]["weight"] is not None:
            print(f"  测试更新cond_projection.weight...")
            estimator.update(test_params)
            print(f"  ✅ 测试更新成功")
        else:
            print(f"  ❌ 找不到cond_projection.weight")
            
    except Exception as e:
        print(f"  ❌ 测试更新失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    
    print("开始调试MLX CFM缓存加载错误...")
    
    debug_cache_loading_error()
    
    print(f"\n🎉 调试完成！")

if __name__ == "__main__":
    main()
