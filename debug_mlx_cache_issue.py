#!/usr/bin/env python3
"""
修复 MLX 缓存加载中的参数名问题

这个脚本专门用于诊断和修复 "Module does not have parameter named '0'" 错误
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append('/Users/bailin/index-tts')

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("⚠️ MLX not available")

from indextts.utils.mlx_cache import MLXModelCache


class MLXCacheDebugger:
    """MLX 缓存调试器"""
    
    def __init__(self):
        self.mlx_cache = MLXModelCache(cache_dir="checkpoints/mlx")
    
    def debug_cache_structure(self, model_name: str = "s2mel"):
        """调试缓存结构"""
        print(f"🔍 Debugging cache structure for: {model_name}")
        
        cache_path = self.mlx_cache.get_cache_path(model_name)
        if not os.path.exists(cache_path):
            print(f"❌ Cache file not found: {cache_path}")
            return
        
        print(f"📁 Cache file: {cache_path}")
        
        # 加载缓存
        try:
            cache_data = mx.load(cache_path)
            print(f"✅ Cache loaded successfully")
            print(f"   Keys count: {len(cache_data)}")
            
            # 分析键的结构
            self.analyze_cache_keys(cache_data)
            
            return cache_data
            
        except Exception as e:
            print(f"❌ Failed to load cache: {e}")
            return None
    
    def analyze_cache_keys(self, cache_data):
        """分析缓存键的结构"""
        print(f"\n📊 Analyzing cache keys structure...")
        
        # 按前缀分组
        prefixes = {}
        for key in cache_data.keys():
            parts = key.split('.')
            if len(parts) > 0:
                prefix = parts[0]
                if prefix not in prefixes:
                    prefixes[prefix] = []
                prefixes[prefix].append(key)
        
        print(f"   Found {len(prefixes)} prefix groups:")
        for prefix, keys in prefixes.items():
            print(f"   - {prefix}: {len(keys)} keys")
            # 显示前几个键作为示例
            sample_keys = keys[:5]
            for sample_key in sample_keys:
                print(f"     {sample_key}")
            if len(keys) > 5:
                print(f"     ... and {len(keys) - 5} more")
        
        # 检查是否有数字开头的键
        numeric_keys = [k for k in cache_data.keys() if k.split('.')[0].isdigit()]
        if numeric_keys:
            print(f"\n⚠️ Found {len(numeric_keys)} keys starting with numbers:")
            for key in numeric_keys[:10]:  # 只显示前10个
                print(f"   {key}")
            if len(numeric_keys) > 10:
                print(f"   ... and {len(numeric_keys) - 10} more")
        
        return prefixes
    
    def test_parameter_loading(self, cache_data):
        """测试参数加载过程"""
        print(f"\n🧪 Testing parameter loading process...")
        
        if not cache_data:
            print("❌ No cache data to test")
            return
        
        # 模拟 load_from_cache 中的 unflatten_parameters 过程
        def unflatten_parameters(flat_dict, prefix="models.cfm.estimator"):
            """Reconstruct nested dict from flattened parameters"""
            nested = {}
            for key, value in flat_dict.items():
                if not key.startswith(prefix + "."):
                    continue
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
            
            return nested
        
        # 测试不同的前缀
        test_prefixes = [
            "models.cfm.estimator",
            "cfm.estimator", 
            "estimator",
            "models.cfm",
            "cfm"
        ]
        
        for prefix in test_prefixes:
            print(f"\n   Testing prefix: '{prefix}'")
            try:
                nested = unflatten_parameters(cache_data, prefix)
                print(f"   ✅ Successfully unflattened {len(nested)} top-level keys")
                
                # 显示嵌套结构的示例
                if nested:
                    self.show_nested_structure(nested, max_depth=3)
                    
            except Exception as e:
                print(f"   ❌ Failed with prefix '{prefix}': {e}")
    
    def show_nested_structure(self, nested_dict, prefix="", max_depth=3, current_depth=0):
        """显示嵌套结构"""
        if current_depth >= max_depth:
            return
        
        for key, value in nested_dict.items():
            if isinstance(value, dict):
                print(f"   {prefix}{key}/")
                self.show_nested_structure(value, prefix + "  ", max_depth, current_depth + 1)
            else:
                if hasattr(value, 'shape'):
                    print(f"   {prefix}{key}: {value.shape}")
                else:
                    print(f"   {prefix}{key}: {type(value)}")
    
    def find_problematic_keys(self, cache_data):
        """查找有问题的键"""
        print(f"\n🔍 Looking for problematic keys...")
        
        problematic_patterns = [
            # 数字开头的键
            lambda k: k.split('.')[0].isdigit(),
            # 包含特殊字符的键
            lambda k: any(c in k for c in ['[', ']', '(', ')', ' ', '\t']),
            # 空键或只有点的键
            lambda k: not k or k == '.' or k.startswith('.') or k.endswith('.'),
            # 包含连续点的键
            lambda k: '..' in k,
        ]
        
        for i, pattern in enumerate(problematic_patterns):
            matching_keys = [k for k in cache_data.keys() if pattern(k)]
            if matching_keys:
                print(f"   Pattern {i+1}: Found {len(matching_keys)} matching keys")
                for key in matching_keys[:5]:  # 只显示前5个
                    print(f"     {key}")
                if len(matching_keys) > 5:
                    print(f"     ... and {len(matching_keys) - 5} more")
    
    def suggest_fixes(self, cache_data):
        """建议修复方案"""
        print(f"\n💡 Suggested fixes:")
        
        # 检查是否有数字开头的键
        numeric_keys = [k for k in cache_data.keys() if k.split('.')[0].isdigit()]
        if numeric_keys:
            print(f"   1. Found {len(numeric_keys)} keys starting with numbers")
            print(f"      These keys might be causing the 'parameter named \"0\"' error")
            print(f"      Suggested fix: Rename these keys to use proper prefixes")
            
            # 显示重命名示例
            print(f"      Example renames:")
            for key in numeric_keys[:3]:
                new_key = f"layer_{key}"
                print(f"        '{key}' -> '{new_key}'")
        
        # 检查前缀一致性
        prefixes = set()
        for key in cache_data.keys():
            parts = key.split('.')
            if len(parts) > 0:
                prefixes.add(parts[0])
        
        print(f"   2. Found {len(prefixes)} different prefixes:")
        for prefix in sorted(prefixes):
            count = len([k for k in cache_data.keys() if k.startswith(prefix)])
            print(f"      - '{prefix}': {count} keys")
        
        # 建议统一前缀
        if len(prefixes) > 1:
            print(f"   3. Consider using a unified prefix like 'models.cfm.estimator'")
            print(f"      This would make the unflatten_parameters function work correctly")
    
    def create_fixed_cache(self, original_cache_data, model_name: str = "s2mel_fixed"):
        """创建修复后的缓存"""
        print(f"\n🔧 Creating fixed cache: {model_name}")
        
        if not original_cache_data:
            print("❌ No original cache data")
            return
        
        fixed_data = {}
        
        for key, value in original_cache_data.items():
            # 修复数字开头的键
            if key.split('.')[0].isdigit():
                new_key = f"layer_{key}"
                print(f"   Fixed: '{key}' -> '{new_key}'")
                fixed_data[new_key] = value
            else:
                fixed_data[key] = value
        
        # 保存修复后的缓存
        cache_path = self.mlx_cache.get_cache_path(model_name)
        try:
            mx.savez(cache_path, **fixed_data)
            print(f"✅ Fixed cache saved to: {cache_path}")
            return cache_path
        except Exception as e:
            print(f"❌ Failed to save fixed cache: {e}")
            return None
    
    def run_full_debug(self, model_name: str = "s2mel"):
        """运行完整的调试流程"""
        print(f"🚀 Starting full MLX cache debug for: {model_name}")
        print("=" * 60)
        
        # 1. 调试缓存结构
        cache_data = self.debug_cache_structure(model_name)
        
        if cache_data is None:
            print("❌ Cannot proceed without cache data")
            return
        
        # 2. 分析键结构
        prefixes = self.analyze_cache_keys(cache_data)
        
        # 3. 测试参数加载
        self.test_parameter_loading(cache_data)
        
        # 4. 查找问题键
        self.find_problematic_keys(cache_data)
        
        # 5. 建议修复方案
        self.suggest_fixes(cache_data)
        
        # 6. 创建修复后的缓存
        fixed_cache_path = self.create_fixed_cache(cache_data, f"{model_name}_fixed")
        
        print(f"\n🎉 Debug completed!")
        if fixed_cache_path:
            print(f"   Fixed cache available at: {fixed_cache_path}")
            print(f"   You can test loading with the fixed cache")


def main():
    """主函数"""
    print("🔧 MLX Cache Parameter Name Issue Debugger")
    print("=" * 50)
    
    debugger = MLXCacheDebugger()
    
    # 检查缓存文件
    cache_files = [
        "s2mel",
        "s2mel_cfm", 
        "gpt"
    ]
    
    for cache_name in cache_files:
        cache_path = debugger.mlx_cache.get_cache_path(cache_name)
        if os.path.exists(cache_path):
            print(f"📁 Found cache: {cache_name}")
            debugger.run_full_debug(cache_name)
            print("\n" + "="*60 + "\n")
        else:
            print(f"❌ Cache not found: {cache_name}")
    
    print("🎯 Debug session completed!")


if __name__ == "__main__":
    main()
