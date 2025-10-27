#!/usr/bin/env python3
"""
简化的 MLX 缓存问题测试脚本

快速验证 "Module does not have parameter named '0'" 错误的原因
"""

import os
import sys
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


def quick_test():
    """快速测试缓存问题"""
    print("🔍 Quick MLX Cache Issue Test")
    print("=" * 40)
    
    if not MLX_AVAILABLE:
        print("❌ MLX not available")
        return
    
    # 检查缓存文件
    cache_dir = "checkpoints/mlx"
    if not os.path.exists(cache_dir):
        print(f"❌ Cache directory not found: {cache_dir}")
        return
    
    # 查找缓存文件
    npz_files = list(Path(cache_dir).glob("*.npz"))
    if not npz_files:
        print(f"❌ No .npz files found in {cache_dir}")
        return
    
    print(f"📁 Found {len(npz_files)} cache files:")
    for npz_file in npz_files:
        print(f"   - {npz_file.name}")
    
    # 测试每个文件
    for npz_file in npz_files:
        print(f"\n🔍 Testing: {npz_file.name}")
        test_cache_file(npz_file)


def test_cache_file(cache_path):
    """测试单个缓存文件"""
    try:
        # 加载缓存
        cache_data = mx.load(cache_path)
        print(f"   ✅ Loaded {len(cache_data)} keys")
        
        # 检查问题键
        problematic_keys = []
        for key in cache_data.keys():
            if is_problematic_key(key):
                problematic_keys.append(key)
        
        if problematic_keys:
            print(f"   ⚠️ Found {len(problematic_keys)} problematic keys:")
            for key in problematic_keys[:5]:
                print(f"      {key}")
            if len(problematic_keys) > 5:
                print(f"      ... and {len(problematic_keys) - 5} more")
            
            # 这就是导致 "Module does not have parameter named '0'" 错误的原因！
            print(f"   🎯 This is likely causing the parameter name error!")
        else:
            print(f"   ✅ No problematic keys found")
        
        # 测试 unflatten_parameters
        test_unflatten_parameters(cache_data)
        
    except Exception as e:
        print(f"   ❌ Failed to load: {e}")


def is_problematic_key(key):
    """检查键是否有问题"""
    parts = key.split('.')
    
    # 检查数字开头的键
    if parts[0].isdigit():
        return True
    
    # 检查特殊字符
    if any(c in key for c in ['[', ']', '(', ')', ' ', '\t']):
        return True
    
    # 检查空键或只有点
    if not key or key == '.' or key.startswith('.') or key.endswith('.'):
        return True
    
    # 检查连续点
    if '..' in key:
        return True
    
    return False


def test_unflatten_parameters(cache_data):
    """测试 unflatten_parameters 函数"""
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
    
    print(f"   🧪 Testing unflatten_parameters:")
    for prefix in test_prefixes:
        try:
            nested = unflatten_parameters(cache_data, prefix)
            print(f"      ✅ '{prefix}': {len(nested)} keys")
        except Exception as e:
            print(f"      ❌ '{prefix}': {e}")


def show_solution():
    """显示解决方案"""
    print(f"\n💡 Solution:")
    print(f"   1. The error 'Module does not have parameter named \"0\"' occurs because")
    print(f"      the cache contains keys that start with numbers (like '0.weight')")
    print(f"   2. MLX's update() method expects parameter names that are valid Python identifiers")
    print(f"   3. Numbers are not valid Python identifiers")
    print(f"   4. Fix: Rename numeric keys to use proper prefixes like 'layer_0.weight'")
    print(f"   5. Run the fix script: python fix_mlx_cache_issue.py")


if __name__ == "__main__":
    quick_test()
    show_solution()
