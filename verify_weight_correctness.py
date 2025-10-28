#!/usr/bin/env python3
"""
验证MLX权重是正确的，PyTorch权重是错误的
"""

import torch
import mlx.core as mx
import numpy as np
from indextts.s2mel.modules.mlx_diffusion_transformer_weights import load_weight_norm

def verify_weight_correctness():
    """验证权重正确性"""
    print("🔍 验证MLX权重是正确的，PyTorch权重是错误的")
    print("="*60)
    
    try:
        # 加载PyTorch权重文件
        pytorch_weights = torch.load("checkpoints/s2mel.pth", map_location='cpu')
        cfm_weights = pytorch_weights['net']['cfm']
        
        # 重构正确的权重
        correct_weight = load_weight_norm(cfm_weights, "estimator.x_embedder")
        
        print(f"📊 正确的权重（从weight_g和weight_v重构）:")
        print(f"   shape={correct_weight.shape}, min={correct_weight.min():.6f}, max={correct_weight.max():.6f}")
        
        # 加载PyTorch模型
        from indextts.infer_v2 import IndexTTS2
        pytorch_tts = IndexTTS2(use_mlx=False)
        pytorch_dit = pytorch_tts.s2mel.models['cfm'].estimator
        
        pytorch_weight = pytorch_dit.x_embedder.weight.detach().cpu().numpy()
        
        print(f"\n📊 PyTorch模型中的权重:")
        print(f"   shape={pytorch_weight.shape}, min={pytorch_weight.min():.6f}, max={pytorch_weight.max():.6f}")
        
        # 比较差异
        diff = np.abs(correct_weight - pytorch_weight)
        max_diff = np.max(diff)
        avg_diff = np.mean(diff)
        
        print(f"\n📊 正确权重与PyTorch权重差异:")
        print(f"   最大差异: {max_diff:.6f}")
        print(f"   平均差异: {avg_diff:.6f}")
        
        if max_diff > 0.1:
            print(f"   ❌ PyTorch权重与正确权重差异很大")
            print(f"   💡 这说明PyTorch的weight_norm加载有问题")
        else:
            print(f"   ✅ PyTorch权重与正确权重基本一致")
        
        # 加载MLX模型
        mlx_tts = IndexTTS2(use_mlx=True)
        mlx_dit = mlx_tts.mlx_s2mel_cfm.estimator
        
        mlx_weight = np.array(mlx_dit.x_embedder.weight)
        
        print(f"\n📊 MLX模型中的权重:")
        print(f"   shape={mlx_weight.shape}, min={mlx_weight.min():.6f}, max={mlx_weight.max():.6f}")
        
        # 比较MLX与正确权重
        diff_mlx = np.abs(correct_weight - mlx_weight)
        max_diff_mlx = np.max(diff_mlx)
        avg_diff_mlx = np.mean(diff_mlx)
        
        print(f"\n📊 正确权重与MLX权重差异:")
        print(f"   最大差异: {max_diff_mlx:.6f}")
        print(f"   平均差异: {avg_diff_mlx:.6f}")
        
        if max_diff_mlx < 1e-6:
            print(f"   ✅ MLX权重与正确权重完全一致")
            print(f"   💡 MLX版本是正确的！")
        else:
            print(f"   ❌ MLX权重与正确权重有差异")
        
        # 结论
        print(f"\n📊 结论:")
        if max_diff_mlx < max_diff:
            print(f"   ✅ MLX权重更接近正确权重")
            print(f"   ❌ PyTorch权重有问题")
            print(f"   💡 应该修复PyTorch的weight_norm加载，而不是MLX")
        else:
            print(f"   ❌ MLX权重也有问题")
            print(f"   💡 需要进一步检查")
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_weight_correctness()
