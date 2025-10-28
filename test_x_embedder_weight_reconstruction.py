#!/usr/bin/env python3
"""
测试 x_embedder 权重重构是否正确
验证 load_weight_norm 函数的正确性
"""

import numpy as np
import torch
import mlx.core as mx

def test_weight_norm_reconstruction():
    """测试权重归一化重构"""
    print("🔍 测试 x_embedder 权重重构")
    print("="*60)
    
    try:
        # 加载 MLX 缓存文件
        mlx_weights = np.load("checkpoints/mlx/s2mel.npz")
        
        # 获取 x_embedder 的 weight_g 和 weight_v
        weight_g_key = "models.cfm.estimator.x_embedder.weight_g"
        weight_v_key = "models.cfm.estimator.x_embedder.weight_v"
        
        if weight_g_key in mlx_weights and weight_v_key in mlx_weights:
            g = mlx_weights[weight_g_key]  # (512, 1)
            v = mlx_weights[weight_v_key]  # (512, 80)
            
            print(f"📊 原始权重数据:")
            print(f"   weight_g: shape={g.shape}, min={g.min():.6f}, max={g.max():.6f}")
            print(f"   weight_v: shape={v.shape}, min={v.min():.6f}, max={v.max():.6f}")
            
            # 重构权重
            norm_v = np.sqrt(np.sum(v**2, axis=1, keepdims=True))  # (512, 1)
            w_reconstructed = g * v / (norm_v + 1e-8)
            
            print(f"\n📊 重构权重:")
            print(f"   norm_v: shape={norm_v.shape}, min={norm_v.min():.6f}, max={norm_v.max():.6f}")
            print(f"   w_reconstructed: shape={w_reconstructed.shape}, min={w_reconstructed.min():.6f}, max={w_reconstructed.max():.6f}")
            
            # 加载 PyTorch 原始权重进行对比
            pytorch_weights = torch.load("checkpoints/s2mel.pth", map_location='cpu')
            
            # 找到 x_embedder 的权重
            x_embedder_weight = None
            for key, value in pytorch_weights.items():
                if 'x_embedder' in key and 'weight' in key and 'weight_g' not in key and 'weight_v' not in key:
                    x_embedder_weight = value
                    print(f"   找到PyTorch权重: {key}")
                    break
            
            if x_embedder_weight is not None:
                pytorch_weight_np = x_embedder_weight.detach().cpu().numpy()
                
                print(f"\n📊 PyTorch 原始权重:")
                print(f"   shape={pytorch_weight_np.shape}, min={pytorch_weight_np.min():.6f}, max={pytorch_weight_np.max():.6f}")
                
                # 比较差异
                diff = np.abs(pytorch_weight_np - w_reconstructed)
                max_diff = np.max(diff)
                avg_diff = np.mean(diff)
                
                print(f"\n📊 重构权重与PyTorch权重差异:")
                print(f"   最大差异: {max_diff:.6f}")
                print(f"   平均差异: {avg_diff:.6f}")
                
                if max_diff < 1e-6:
                    print(f"   ✅ 权重重构正确")
                else:
                    print(f"   ❌ 权重重构有问题")
                    
                    # 找出差异最大的位置
                    max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
                    print(f"   最大差异位置: {max_diff_idx}")
                    print(f"   PyTorch值: {pytorch_weight_np[max_diff_idx]:.6f}")
                    print(f"   重构值:    {w_reconstructed[max_diff_idx]:.6f}")
                    
                    # 检查是否是完全不同的权重
                    if np.array_equal(pytorch_weight_np, w_reconstructed):
                        print(f"   ✅ 权重完全相同")
                    else:
                        different_count = np.sum(diff > 1e-10)
                        total_count = pytorch_weight_np.size
                        print(f"   ❌ 不同元素: {different_count}/{total_count} ({different_count/total_count*100:.2f}%)")
            else:
                print(f"   ❌ 未找到PyTorch x_embedder权重")
                
        else:
            print(f"❌ 未找到MLX缓存中的weight_g或weight_v")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_load_weight_norm_function():
    """测试 load_weight_norm 函数"""
    print(f"\n{'='*60}")
    print(f"🔍 测试 load_weight_norm 函数")
    print(f"{'='*60}")
    
    try:
        # 导入函数
        from indextts.s2mel.modules.mlx_diffusion_transformer_weights import load_weight_norm
        
        # 加载 MLX 缓存文件
        mlx_weights = np.load("checkpoints/mlx/s2mel.npz")
        
        # 测试 load_weight_norm 函数
        reconstructed_weight = load_weight_norm(mlx_weights, "models.cfm.estimator.x_embedder")
        
        if reconstructed_weight is not None:
            print(f"📊 load_weight_norm 结果:")
            print(f"   shape={reconstructed_weight.shape}, min={reconstructed_weight.min():.6f}, max={reconstructed_weight.max():.6f}")
            
            # 与PyTorch权重对比
            pytorch_weights = torch.load("checkpoints/s2mel.pth", map_location='cpu')
            pytorch_weight = None
            for key, value in pytorch_weights.items():
                if 'x_embedder' in key and 'weight' in key and 'weight_g' not in key and 'weight_v' not in key:
                    pytorch_weight = value.detach().cpu().numpy()
                    break
            
            if pytorch_weight is not None:
                diff = np.abs(pytorch_weight - reconstructed_weight)
                max_diff = np.max(diff)
                avg_diff = np.mean(diff)
                
                print(f"\n📊 与PyTorch权重差异:")
                print(f"   最大差异: {max_diff:.6f}")
                print(f"   平均差异: {avg_diff:.6f}")
                
                if max_diff < 1e-6:
                    print(f"   ✅ load_weight_norm 函数正确")
                else:
                    print(f"   ❌ load_weight_norm 函数有问题")
        else:
            print(f"❌ load_weight_norm 返回 None")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    test_weight_norm_reconstruction()
    test_load_weight_norm_function()

if __name__ == "__main__":
    main()
