#!/usr/bin/env python3
"""
DiT 权重差异分析工具
直接比较 PyTorch 和 MLX DiT 权重文件
"""

import os
import numpy as np
import torch
import mlx.core as mx

def load_pytorch_dit_weights():
    """从 PyTorch 检查点加载 DiT 权重"""
    try:
        # 加载 PyTorch 检查点
        checkpoint_path = "checkpoints/s2mel.pth"
        if not os.path.exists(checkpoint_path):
            print(f"❌ 未找到 PyTorch 检查点: {checkpoint_path}")
            return None
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 提取 DiT 权重
        dit_weights = {}
        for key, value in checkpoint['net']['cfm'].items():
            if key.startswith('estimator.'):
                # 移除 'estimator.' 前缀
                clean_key = key[len('estimator.'):]
                if isinstance(value, torch.Tensor):
                    dit_weights[clean_key] = value.detach().cpu().numpy()
                else:
                    dit_weights[clean_key] = value
                
        print(f"✅ 加载了 {len(dit_weights)} 个 PyTorch DiT 权重")
        return dit_weights
        
    except Exception as e:
        print(f"❌ 加载 PyTorch 权重失败: {e}")
        return None

def load_mlx_dit_weights():
    """从 MLX 缓存加载 DiT 权重"""
    try:
        # 加载 MLX 缓存
        cache_path = "checkpoints/mlx/s2mel.npz"
        if not os.path.exists(cache_path):
            print(f"❌ 未找到 MLX 缓存: {cache_path}")
            return None
            
        with np.load(cache_path, allow_pickle=True) as data:
            cache_dict = {k: data[k] for k in data.files}
        
        # 提取 DiT 权重
        dit_weights = {}
        for key, value in cache_dict.items():
            if key.startswith('models.cfm.estimator.'):
                # 移除前缀
                clean_key = key[len('models.cfm.estimator.'):]
                dit_weights[clean_key] = value
                
        print(f"✅ 加载了 {len(dit_weights)} 个 MLX DiT 权重")
        return dit_weights
        
    except Exception as e:
        print(f"❌ 加载 MLX 权重失败: {e}")
        return None

def compare_weights(pytorch_weights, mlx_weights):
    """比较 PyTorch 和 MLX 权重"""
    print(f"\n{'='*60}")
    print(f"🔍 DiT 权重比较分析")
    print(f"{'='*60}")
    
    # 找到共同的权重键
    common_keys = set(pytorch_weights.keys()) & set(mlx_weights.keys())
    print(f"共同权重键数量: {len(common_keys)}")
    
    # 只比较 PyTorch 独有的键
    pytorch_only = set(pytorch_weights.keys()) - set(mlx_weights.keys())
    if pytorch_only:
        print(f"PyTorch 独有键: {sorted(pytorch_only)}")
    
    # 只比较 MLX 独有的键
    mlx_only = set(mlx_weights.keys()) - set(pytorch_weights.keys())
    if mlx_only:
        print(f"MLX 独有键: {sorted(mlx_only)}")
    
    print(f"\n📊 详细权重比较:")
    
    # 按层分组比较
    layer_groups = {
        'x_embedder': [k for k in common_keys if 'x_embedder' in k],
        't_embedder': [k for k in common_keys if 't_embedder' in k],
        'cond_projection': [k for k in common_keys if 'cond_projection' in k],
        'cond_embedder': [k for k in common_keys if 'cond_embedder' in k],
        'cond_x_merge_linear': [k for k in common_keys if 'cond_x_merge_linear' in k],
        'transformer': [k for k in common_keys if 'transformer' in k],
        'final_layer': [k for k in common_keys if 'final_layer' in k or 'final_mlp' in k],
    }
    
    for layer_name, keys in layer_groups.items():
        if not keys:
            continue
            
        print(f"\n🔍 {layer_name} 层:")
        for key in sorted(keys):
            pt_w = pytorch_weights[key]
            mlx_w = mlx_weights[key]
            
            print(f"   {key}:")
            print(f"     PyTorch: {pt_w.shape}, min={pt_w.min():.6f}, max={pt_w.max():.6f}")
            print(f"     MLX:     {mlx_w.shape}, min={mlx_w.min():.6f}, max={mlx_w.max():.6f}")
            
            if pt_w.shape == mlx_w.shape:
                diff = np.abs(pt_w - mlx_w)
                print(f"     差异: max={diff.max():.6f}, mean={diff.mean():.6f}")
                if diff.max() < 1e-6:
                    print(f"     ✅ 一致")
                else:
                    print(f"     ❌ 不一致")
            elif pt_w.shape == mlx_w.shape[::-1]:
                diff = np.abs(pt_w - mlx_w.T)
                print(f"     差异(转置后): max={diff.max():.6f}, mean={diff.mean():.6f}")
                if diff.max() < 1e-6:
                    print(f"     ✅ 一致(需转置)")
                else:
                    print(f"     ❌ 不一致(需转置)")
            else:
                print(f"     ❌ 形状不匹配: {pt_w.shape} vs {mlx_w.shape}")

def analyze_weight_statistics(pytorch_weights, mlx_weights):
    """分析权重统计信息"""
    print(f"\n{'='*60}")
    print(f"🔍 权重统计信息分析")
    print(f"{'='*60}")
    
    def get_stats(weights_dict, name):
        stats = {}
        for key, value in weights_dict.items():
            if isinstance(value, np.ndarray) and value.dtype in [np.float32, np.float64]:
                stats[key] = {
                    'min': float(value.min()),
                    'max': float(value.max()),
                    'mean': float(value.mean()),
                    'std': float(value.std()),
                    'shape': value.shape
                }
        return stats
    
    pt_stats = get_stats(pytorch_weights, "PyTorch")
    mlx_stats = get_stats(mlx_weights, "MLX")
    
    print(f"PyTorch 权重统计:")
    for key, stats in pt_stats.items():
        print(f"   {key}: shape={stats['shape']}, min={stats['min']:.6f}, max={stats['max']:.6f}, mean={stats['mean']:.6f}")
    
    print(f"\nMLX 权重统计:")
    for key, stats in mlx_stats.items():
        print(f"   {key}: shape={stats['shape']}, min={stats['min']:.6f}, max={stats['max']:.6f}, mean={stats['mean']:.6f}")

def main():
    print("🔍 DiT 权重差异分析工具")
    print("="*60)
    
    # 加载权重
    pytorch_weights = load_pytorch_dit_weights()
    mlx_weights = load_mlx_dit_weights()
    
    if pytorch_weights is None or mlx_weights is None:
        print("❌ 无法加载权重，退出分析")
        return
    
    # 比较权重
    compare_weights(pytorch_weights, mlx_weights)
    
    # 分析统计信息
    analyze_weight_statistics(pytorch_weights, mlx_weights)

if __name__ == "__main__":
    main()
