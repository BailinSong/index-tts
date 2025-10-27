#!/usr/bin/env python3
"""
修复 MLX 缓存加载中的参数名问题

这个脚本会：
1. 检测缓存中的问题键（如数字开头的键）
2. 修复这些键的命名
3. 创建修复后的缓存文件
"""

import os
import sys
import shutil
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


class MLXCacheFixer:
    """MLX 缓存修复器"""
    
    def __init__(self, cache_dir="checkpoints/mlx"):
        self.cache_dir = cache_dir
        self.mlx_cache = MLXModelCache(cache_dir=cache_dir)
    
    def fix_cache_file(self, model_name: str, backup=True):
        """修复单个缓存文件"""
        print(f"🔧 Fixing cache file: {model_name}")
        
        cache_path = self.mlx_cache.get_cache_path(model_name)
        if not os.path.exists(cache_path):
            print(f"❌ Cache file not found: {cache_path}")
            return False
        
        # 备份原文件
        if backup:
            backup_path = f"{cache_path}.backup"
            shutil.copy2(cache_path, backup_path)
            print(f"📁 Backup created: {backup_path}")
        
        try:
            # 加载原缓存
            print(f"📦 Loading original cache...")
            original_data = mx.load(cache_path)
            print(f"   Loaded {len(original_data)} keys")
            
            # 修复键名
            print(f"🔧 Fixing key names...")
            fixed_data = self.fix_key_names(original_data)
            
            # 保存修复后的缓存
            print(f"💾 Saving fixed cache...")
            mx.savez(cache_path, **fixed_data)
            
            print(f"✅ Cache file fixed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to fix cache: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def fix_key_names(self, cache_data):
        """修复键名"""
        fixed_data = {}
        fixes_applied = 0
        
        for key, value in cache_data.items():
            new_key = self.fix_single_key(key)
            if new_key != key:
                print(f"   Fixed: '{key}' -> '{new_key}'")
                fixes_applied += 1
            fixed_data[new_key] = value
        
        print(f"   Applied {fixes_applied} key fixes")
        return fixed_data
    
    def fix_single_key(self, key):
        """修复单个键名"""
        parts = key.split('.')
        
        # 修复数字开头的键
        if parts[0].isdigit():
            # 将数字开头的键转换为 layer_ 前缀
            parts[0] = f"layer_{parts[0]}"
        
        # 修复其他常见问题
        fixed_parts = []
        for part in parts:
            # 移除特殊字符
            clean_part = part.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
            clean_part = clean_part.replace(' ', '_').replace('\t', '_')
            
            # 确保不是空字符串
            if clean_part:
                fixed_parts.append(clean_part)
        
        return '.'.join(fixed_parts)
    
    def fix_all_caches(self):
        """修复所有缓存文件"""
        print(f"🚀 Starting cache fix for all files in: {self.cache_dir}")
        
        if not os.path.exists(self.cache_dir):
            print(f"❌ Cache directory not found: {self.cache_dir}")
            return
        
        # 查找所有 .npz 文件
        npz_files = list(Path(self.cache_dir).glob("*.npz"))
        
        if not npz_files:
            print(f"❌ No .npz files found in {self.cache_dir}")
            return
        
        print(f"📁 Found {len(npz_files)} cache files:")
        for npz_file in npz_files:
            print(f"   - {npz_file.name}")
        
        # 修复每个文件
        success_count = 0
        for npz_file in npz_files:
            model_name = npz_file.stem  # 去掉 .npz 扩展名
            print(f"\n{'='*50}")
            if self.fix_cache_file(model_name):
                success_count += 1
        
        print(f"\n🎉 Cache fix completed!")
        print(f"   Successfully fixed: {success_count}/{len(npz_files)} files")
    
    def verify_fixed_cache(self, model_name: str):
        """验证修复后的缓存"""
        print(f"🔍 Verifying fixed cache: {model_name}")
        
        cache_path = self.mlx_cache.get_cache_path(model_name)
        if not os.path.exists(cache_path):
            print(f"❌ Cache file not found: {cache_path}")
            return False
        
        try:
            # 加载缓存
            cache_data = mx.load(cache_path)
            
            # 检查是否还有问题键
            problematic_keys = []
            for key in cache_data.keys():
                if self.is_problematic_key(key):
                    problematic_keys.append(key)
            
            if problematic_keys:
                print(f"⚠️ Still found {len(problematic_keys)} problematic keys:")
                for key in problematic_keys[:5]:
                    print(f"   {key}")
                if len(problematic_keys) > 5:
                    print(f"   ... and {len(problematic_keys) - 5} more")
                return False
            else:
                print(f"✅ No problematic keys found!")
                print(f"   Cache contains {len(cache_data)} keys")
                return True
                
        except Exception as e:
            print(f"❌ Failed to verify cache: {e}")
            return False
    
    def is_problematic_key(self, key):
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
    
    def test_cache_loading(self, model_name: str):
        """测试缓存加载"""
        print(f"🧪 Testing cache loading for: {model_name}")
        
        cache_path = self.mlx_cache.get_cache_path(model_name)
        if not os.path.exists(cache_path):
            print(f"❌ Cache file not found: {cache_path}")
            return False
        
        try:
            # 测试加载
            cache_data = mx.load(cache_path)
            print(f"✅ Cache loaded successfully")
            print(f"   Keys: {len(cache_data)}")
            
            # 测试 unflatten_parameters 函数
            self.test_unflatten_parameters(cache_data)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load cache: {e}")
            return False
    
    def test_unflatten_parameters(self, cache_data):
        """测试 unflatten_parameters 函数"""
        print(f"🧪 Testing unflatten_parameters function...")
        
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
            try:
                nested = unflatten_parameters(cache_data, prefix)
                print(f"   ✅ Prefix '{prefix}': {len(nested)} top-level keys")
            except Exception as e:
                print(f"   ❌ Prefix '{prefix}': {e}")


def main():
    """主函数"""
    print("🔧 MLX Cache Parameter Name Fixer")
    print("=" * 50)
    
    fixer = MLXCacheFixer()
    
    # 检查缓存目录
    if not os.path.exists(fixer.cache_dir):
        print(f"❌ Cache directory not found: {fixer.cache_dir}")
        return
    
    # 修复所有缓存文件
    fixer.fix_all_caches()
    
    # 验证修复结果
    print(f"\n🔍 Verifying fixes...")
    npz_files = list(Path(fixer.cache_dir).glob("*.npz"))
    for npz_file in npz_files:
        model_name = npz_file.stem
        fixer.verify_fixed_cache(model_name)
        fixer.test_cache_loading(model_name)
        print()
    
    print("🎉 All fixes completed!")


if __name__ == "__main__":
    main()
