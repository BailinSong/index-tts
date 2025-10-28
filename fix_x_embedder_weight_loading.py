#!/usr/bin/env python3
"""
修复 x_embedder 权重加载问题
确保 MLX 正确加载 PyTorch 的 weight_norm 权重
"""

import torch
import mlx.core as mx
import numpy as np
from indextts.s2mel.modules.mlx_diffusion_transformer_weights import load_weight_norm

def test_x_embedder_weight_loading():
    """测试 x_embedder 权重加载"""
    print("🔍 测试 x_embedder 权重加载")
    print("="*60)
    
    try:
        # 加载 PyTorch 权重
        pytorch_weights = torch.load("checkpoints/s2mel.pth", map_location='cpu')
        cfm_weights = pytorch_weights['net']['cfm']
        
        print("📊 PyTorch CFM 权重中的 x_embedder 键:")
        x_embedder_keys = [k for k in cfm_weights.keys() if 'x_embedder' in k]
        for key in x_embedder_keys:
            weight = cfm_weights[key]
            print(f"   {key}: {weight.shape} {weight.dtype}")
            print(f"      min={weight.min():.6f}, max={weight.max():.6f}, avg={weight.mean():.6f}")
        
        # 测试不同的前缀
        prefixes_to_test = [
            "",  # 无前缀
            "estimator.",  # estimator 前缀
            "models.cfm.estimator.",  # 完整前缀
        ]
        
        for prefix in prefixes_to_test:
            print(f"\n📊 测试前缀: '{prefix}'")
            
            # 测试 load_weight_norm 函数
            reconstructed_weight = load_weight_norm(cfm_weights, f"{prefix}x_embedder")
            
            if reconstructed_weight is not None:
                print(f"   ✅ 成功重构权重: {reconstructed_weight.shape}")
                print(f"      min={reconstructed_weight.min():.6f}, max={reconstructed_weight.max():.6f}, avg={reconstructed_weight.mean():.6f}")
                
                # 验证重构是否正确
                if prefix == "estimator.":
                    # 手动重构验证
                    g = cfm_weights["estimator.x_embedder.weight_g"].numpy()
                    v = cfm_weights["estimator.x_embedder.weight_v"].numpy()
                    
                    norm_v = np.sqrt(np.sum(v**2, axis=1, keepdims=True))
                    w_manual = g * v / (norm_v + 1e-8)
                    
                    diff = np.abs(reconstructed_weight - w_manual)
                    max_diff = np.max(diff)
                    
                    print(f"   🔍 与手动重构差异: {max_diff:.6f}")
                    if max_diff < 1e-6:
                        print(f"   ✅ 重构正确")
                    else:
                        print(f"   ❌ 重构有问题")
            else:
                print(f"   ❌ 无法重构权重")
        
        # 找到正确的前缀
        print(f"\n📊 寻找正确的前缀:")
        correct_prefix = None
        for prefix in prefixes_to_test:
            g_key = f"{prefix}x_embedder.weight_g"
            v_key = f"{prefix}x_embedder.weight_v"
            
            if g_key in cfm_weights and v_key in cfm_weights:
                print(f"   ✅ 找到正确前缀: '{prefix}'")
                correct_prefix = prefix
                break
        
        if correct_prefix is not None:
            print(f"\n📊 使用正确前缀 '{correct_prefix}' 重构权重:")
            reconstructed_weight = load_weight_norm(cfm_weights, f"{correct_prefix}x_embedder")
            
            if reconstructed_weight is not None:
                print(f"   ✅ 重构成功: {reconstructed_weight.shape}")
                print(f"      min={reconstructed_weight.min():.6f}, max={reconstructed_weight.max():.6f}, avg={reconstructed_weight.mean():.6f}")
                
                # 保存重构的权重用于后续测试
                np.save("x_embedder_reconstructed.npy", reconstructed_weight)
                print(f"   💾 已保存重构权重到 x_embedder_reconstructed.npy")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def fix_mlx_weight_loading():
    """修复 MLX 权重加载"""
    print(f"\n{'='*60}")
    print(f"🔧 修复 MLX 权重加载")
    print(f"{'='*60}")
    
    try:
        # 加载 MLX 模型
        from indextts.infer_v2 import IndexTTS2
        mlx_tts = IndexTTS2(use_mlx=True)
        mlx_dit = mlx_tts.mlx_s2mel_cfm.estimator
        
        # 加载 PyTorch 权重
        pytorch_weights = torch.load("checkpoints/s2mel.pth", map_location='cpu')
        cfm_weights = pytorch_weights['net']['cfm']
        
        # 使用正确的前缀重构权重
        reconstructed_weight = load_weight_norm(cfm_weights, "estimator.x_embedder")
        
        if reconstructed_weight is not None:
            print(f"📊 重构权重成功: {reconstructed_weight.shape}")
            
            # 更新 MLX 模型权重
            mlx_dit.x_embedder.weight = mx.array(reconstructed_weight)
            
            # 更新偏置
            if "estimator.x_embedder.bias" in cfm_weights:
                mlx_dit.x_embedder.bias = mx.array(cfm_weights["estimator.x_embedder.bias"].numpy())
            
            print(f"✅ MLX x_embedder 权重已更新")
            
            # 验证更新后的权重
            mlx_weight = mlx_dit.x_embedder.weight
            print(f"📊 更新后的 MLX 权重:")
            print(f"   shape={mlx_weight.shape}, min={mx.min(mlx_weight):.6f}, max={mx.max(mlx_weight):.6f}, avg={mx.mean(mlx_weight):.6f}")
            
            # 与重构权重比较
            diff = np.abs(reconstructed_weight - np.array(mlx_weight))
            max_diff = np.max(diff)
            avg_diff = np.mean(diff)
            
            print(f"📊 与重构权重差异:")
            print(f"   最大差异: {max_diff:.6f}")
            print(f"   平均差异: {avg_diff:.6f}")
            
            if max_diff < 1e-6:
                print(f"   ✅ MLX 权重更新成功")
            else:
                print(f"   ❌ MLX 权重更新有问题")
        else:
            print(f"❌ 无法重构权重")
        
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    test_x_embedder_weight_loading()
    fix_mlx_weight_loading()

if __name__ == "__main__":
    main()
