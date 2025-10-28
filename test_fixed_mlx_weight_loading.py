#!/usr/bin/env python3
"""
测试修复后的MLX权重加载
"""

import torch
import mlx.core as mx
import numpy as np
from indextts.infer_v2 import IndexTTS2

def test_fixed_mlx_weight_loading():
    """测试修复后的MLX权重加载"""
    print("🔍 测试修复后的MLX权重加载")
    print("="*60)
    
    try:
        # 加载 MLX 模型
        print("📊 加载 MLX 模型...")
        mlx_tts = IndexTTS2(use_mlx=True)
        mlx_dit = mlx_tts.mlx_s2mel_cfm.estimator
        
        # 获取 x_embedder 权重
        mlx_weight = mlx_dit.x_embedder.weight
        mlx_bias = mlx_dit.x_embedder.bias
        
        print(f"📊 MLX x_embedder 权重:")
        print(f"   weight: shape={mlx_weight.shape}, min={mx.min(mlx_weight):.6f}, max={mx.max(mlx_weight):.6f}, avg={mx.mean(mlx_weight):.6f}")
        
        # 检查bias的类型
        if hasattr(mlx_bias, 'shape'):
            if isinstance(mlx_bias, mx.array):  # MLX array
                print(f"   bias:   shape={mlx_bias.shape}, min={mx.min(mlx_bias):.6f}, max={mx.max(mlx_bias):.6f}, avg={mx.mean(mlx_bias):.6f}")
            else:  # numpy array
                print(f"   bias:   shape={mlx_bias.shape}, min={mlx_bias.min():.6f}, max={mlx_bias.max():.6f}, avg={mlx_bias.mean():.6f}")
        else:
            print(f"   bias:   {mlx_bias}")
        
        # 加载 PyTorch 模型进行对比
        print(f"\n📊 加载 PyTorch 模型进行对比...")
        pytorch_tts = IndexTTS2(use_mlx=False)
        pytorch_dit = pytorch_tts.s2mel.models['cfm'].estimator
        
        pytorch_weight = pytorch_dit.x_embedder.weight
        pytorch_bias = pytorch_dit.x_embedder.bias
        
        print(f"📊 PyTorch x_embedder 权重:")
        print(f"   weight: shape={pytorch_weight.shape}, min={pytorch_weight.min():.6f}, max={pytorch_weight.max():.6f}, avg={pytorch_weight.mean():.6f}")
        print(f"   bias:   shape={pytorch_bias.shape}, min={pytorch_bias.min():.6f}, max={pytorch_bias.max():.6f}, avg={pytorch_bias.mean():.6f}")
        
        # 比较权重差异
        print(f"\n📊 权重差异分析:")
        
        # 权重差异
        pytorch_weight_np = pytorch_weight.detach().cpu().numpy()
        mlx_weight_np = np.array(mlx_weight)
        weight_diff = np.abs(pytorch_weight_np - mlx_weight_np)
        weight_max_diff = np.max(weight_diff)
        weight_avg_diff = np.mean(weight_diff)
        
        print(f"   权重差异: max={weight_max_diff:.6f}, avg={weight_avg_diff:.6f}")
        
        # 偏置差异
        pytorch_bias_np = pytorch_bias.detach().cpu().numpy()
        mlx_bias_np = np.array(mlx_bias)
        bias_diff = np.abs(pytorch_bias_np - mlx_bias_np)
        bias_max_diff = np.max(bias_diff)
        bias_avg_diff = np.mean(bias_diff)
        
        print(f"   偏置差异: max={bias_max_diff:.6f}, avg={bias_avg_diff:.6f}")
        
        # 判断是否修复成功
        if weight_max_diff < 1e-6 and bias_max_diff < 1e-6:
            print(f"\n✅ x_embedder 权重加载修复成功！")
            print(f"   PyTorch 和 MLX 权重完全一致")
        else:
            print(f"\n❌ x_embedder 权重加载仍有问题")
            print(f"   需要进一步检查")
            
            # 找出差异最大的位置
            if weight_max_diff > 1e-6:
                max_diff_idx = np.unravel_index(np.argmax(weight_diff), weight_diff.shape)
                print(f"   权重最大差异位置: {max_diff_idx}")
                print(f"   PyTorch值: {pytorch_weight_np[max_diff_idx]:.6f}")
                print(f"   MLX值:     {mlx_weight_np[max_diff_idx]:.6f}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_mlx_weight_loading()
