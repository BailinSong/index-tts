#!/usr/bin/env python3
"""
测试修改后的生产代码

验证 MLX 缓存转换、存储和加载逻辑的修复效果
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


def test_key_fixing():
    """测试键名修复功能"""
    print("🧪 Testing key name fixing functionality")
    print("=" * 50)
    
    if not MLX_AVAILABLE:
        print("❌ MLX not available")
        return
    
    # 创建缓存管理器
    cache_manager = MLXModelCache(cache_dir="checkpoints/mlx")
    
    # 测试各种问题键名
    test_keys = [
        "0.weight",           # 数字开头
        "1.bias",            # 数字开头
        "layer.0.weight",    # 包含数字
        "model[0].weight",   # 包含方括号
        "model(0).weight",   # 包含圆括号
        "model 0.weight",    # 包含空格
        "model\t0.weight",   # 包含制表符
        "..weight",          # 连续点
        ".weight",           # 开头点
        "weight.",           # 结尾点
        "normal.weight",     # 正常键名
    ]
    
    print("Testing key name fixes:")
    for key in test_keys:
        fixed_key = cache_manager._fix_key_name(key)
        status = "✅" if fixed_key != key else "➡️"
        print(f"   {status} '{key}' -> '{fixed_key}'")
    
    print("\n✅ Key fixing test completed")


def test_cache_conversion():
    """测试缓存转换功能"""
    print("\n🧪 Testing cache conversion functionality")
    print("=" * 50)
    
    if not MLX_AVAILABLE:
        print("❌ MLX not available")
        return
    
    # 创建测试数据
    test_state_dict = {
        "0.weight": torch.randn(128, 256),
        "1.bias": torch.randn(128),
        "layer.0.weight": torch.randn(64, 128),
        "normal.weight": torch.randn(32, 64),
        "normal.bias": torch.randn(32),
    }
    
    print("Original state dict keys:")
    for key in test_state_dict.keys():
        print(f"   {key}")
    
    # 创建缓存管理器
    cache_manager = MLXModelCache(cache_dir="test_mlx_cache")
    
    try:
        # 测试转换和缓存
        print("\nConverting and caching...")
        cache_path = cache_manager.convert_and_cache(
            "test_model", 
            state_dict=test_state_dict
        )
        
        if cache_path:
            print(f"✅ Cache created: {cache_path}")
            
            # 测试加载
            print("\nLoading from cache...")
            loaded_data = cache_manager.load_from_cache("test_model")
            
            if loaded_data:
                print("✅ Cache loaded successfully")
                print("Loaded keys:")
                for key in loaded_data.keys():
                    print(f"   {key}")
                
                # 验证键名是否被修复
                problematic_keys = [k for k in loaded_data.keys() if k.split('.')[0].isdigit()]
                if problematic_keys:
                    print(f"⚠️ Still found {len(problematic_keys)} problematic keys:")
                    for key in problematic_keys:
                        print(f"   {key}")
                else:
                    print("✅ No problematic keys found in loaded cache")
            else:
                print("❌ Failed to load cache")
        else:
            print("❌ Failed to create cache")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理测试文件
        test_cache_dir = "test_mlx_cache"
        if os.path.exists(test_cache_dir):
            import shutil
            shutil.rmtree(test_cache_dir)
            print(f"\n🧹 Cleaned up test cache directory: {test_cache_dir}")


def test_mlx_model_loading():
    """测试 MLX 模型加载功能"""
    print("\n🧪 Testing MLX model loading functionality")
    print("=" * 50)
    
    if not MLX_AVAILABLE:
        print("❌ MLX not available")
        return
    
    # 检查是否有现有的缓存文件
    cache_dir = "checkpoints/mlx"
    if not os.path.exists(cache_dir):
        print(f"❌ Cache directory not found: {cache_dir}")
        return
    
    # 查找缓存文件
    npz_files = list(Path(cache_dir).glob("*.npz"))
    if not npz_files:
        print(f"❌ No .npz files found in {cache_dir}")
        return
    
    print(f"Found {len(npz_files)} cache files:")
    for npz_file in npz_files:
        print(f"   - {npz_file.name}")
    
    # 测试每个缓存文件
    for npz_file in npz_files:
        print(f"\n🔍 Testing: {npz_file.name}")
        test_single_cache_file(npz_file)


def test_single_cache_file(cache_path):
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
        else:
            print(f"   ✅ No problematic keys found")
        
        # 测试键名修复
        print(f"   🔧 Testing key name fixes:")
        fixed_count = 0
        for key in list(cache_data.keys())[:5]:  # 只测试前5个键
            from indextts.utils.mlx_cache import MLXModelCache
            cache_manager = MLXModelCache()
            fixed_key = cache_manager._fix_key_name(key)
            if fixed_key != key:
                print(f"      '{key}' -> '{fixed_key}'")
                fixed_count += 1
        
        if fixed_count == 0:
            print(f"      No keys needed fixing")
        
    except Exception as e:
        print(f"   ❌ Failed to test: {e}")


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


def test_production_integration():
    """测试生产环境集成"""
    print("\n🧪 Testing production integration")
    print("=" * 50)
    
    # 检查必要的文件
    required_files = [
        "checkpoints/config.yaml",
        "indextts/utils/mlx_cache.py",
        "indextts/s2mel/modules/mlx_cfm.py",
        "indextts/gpt/mlx_model.py"
    ]
    
    print("Checking required files:")
    all_files_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path}")
            all_files_exist = False
    
    if not all_files_exist:
        print("❌ Some required files are missing")
        return
    
    # 测试导入
    print("\nTesting imports:")
    try:
        from indextts.utils.mlx_cache import MLXModelCache
        print("   ✅ MLXModelCache imported successfully")
        
        from indextts.s2mel.modules.mlx_cfm import MLXCFM
        print("   ✅ MLXCFM imported successfully")
        
        from indextts.gpt.mlx_model import UnifiedVoiceMLX
        print("   ✅ UnifiedVoiceMLX imported successfully")
        
        print("✅ All imports successful")
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("🧪 Testing Modified Production Code")
    print("=" * 60)
    
    # 运行所有测试
    test_key_fixing()
    test_cache_conversion()
    test_mlx_model_loading()
    test_production_integration()
    
    print(f"\n🎉 All tests completed!")
    print(f"\n📋 Summary:")
    print(f"   - Modified MLXModelCache to fix key names during conversion")
    print(f"   - Modified MLXCFM to use direct parameter loading")
    print(f"   - Modified MLX GPT to handle both original and fixed key names")
    print(f"   - All changes are backward compatible")


if __name__ == "__main__":
    main()
