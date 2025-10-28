#!/usr/bin/env python3
"""
检查MLX缓存中的键名
"""

import numpy as np

def check_mlx_cache_keys():
    """检查MLX缓存中的键名"""
    print("🔍 检查MLX缓存中的键名")
    print("="*60)
    
    try:
        # 加载MLX缓存
        mlx_cache = np.load("checkpoints/mlx/s2mel.npz")
        
        print(f"📊 MLX缓存包含 {len(mlx_cache.keys())} 个键")
        
        # 查找x_embedder相关的键
        x_embedder_keys = [k for k in mlx_cache.keys() if 'x_embedder' in k]
        print(f"\n📊 x_embedder相关键:")
        for key in x_embedder_keys:
            weight = mlx_cache[key]
            print(f"   {key}: {weight.shape} {weight.dtype}")
            print(f"      min={weight.min():.6f}, max={weight.max():.6f}, avg={weight.mean():.6f}")
        
        # 查找所有estimator相关的键
        estimator_keys = [k for k in mlx_cache.keys() if 'estimator' in k]
        print(f"\n📊 estimator相关键 (前20个):")
        for key in sorted(estimator_keys)[:20]:
            weight = mlx_cache[key]
            print(f"   {key}: {weight.shape} {weight.dtype}")
        
        if len(estimator_keys) > 20:
            print(f"   ... 还有 {len(estimator_keys) - 20} 个estimator相关键")
        
        # 检查键名模式
        print(f"\n📊 键名模式分析:")
        patterns = {}
        for key in mlx_cache.keys():
            if '.' in key:
                prefix = key.split('.')[0]
                patterns[prefix] = patterns.get(prefix, 0) + 1
        
        for prefix, count in sorted(patterns.items()):
            print(f"   {prefix}.*: {count} 个键")
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_mlx_cache_keys()
