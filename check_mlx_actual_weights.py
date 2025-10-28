#!/usr/bin/env python3
"""
检查MLX实际加载的权重
"""

import torch
import mlx.core as mx
import numpy as np
from indextts.infer_v2 import IndexTTS2

def check_mlx_actual_weights():
    """检查MLX实际加载的权重"""
    print("🔍 检查MLX实际加载的权重")
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
        print(f"   bias:   shape={mlx_bias.shape}, min={mx.min(mlx_bias):.6f}, max={mx.max(mlx_bias):.6f}, avg={mx.mean(mlx_bias):.6f}")
        
        # 加载 PyTorch 模型进行对比
        print(f"\n📊 加载 PyTorch 模型进行对比...")
        pytorch_tts = IndexTTS2(use_mlx=False)
        pytorch_dit = pytorch_tts.s2mel.models['cfm'].estimator
        
        pytorch_weight = pytorch_dit.x_embedder.weight
        pytorch_bias = pytorch_dit.x_embedder.bias
        
        print(f"📊 PyTorch x_embedder 权重:")
        print(f"   weight: shape={pytorch_weight.shape}, min={pytorch_weight.min():.6f}, max={pytorch_weight.max():.6f}, avg={pytorch_weight.mean():.6f}")
        print(f"   bias:   shape={pytorch_bias.shape}, min={pytorch_bias.min():.6f}, max={pytorch_bias.max():.6f}, avg={pytorch_bias.mean():.6f}")
        
        # 检查MLX权重是否来自PyTorch
        print(f"\n📊 检查MLX权重来源...")
        
        # 检查MLX权重是否与PyTorch权重相同
        pytorch_weight_np = pytorch_weight.detach().cpu().numpy()
        mlx_weight_np = np.array(mlx_weight)
        
        # 检查是否完全相同
        is_identical = np.array_equal(pytorch_weight_np, mlx_weight_np)
        print(f"   MLX权重与PyTorch权重完全相同: {is_identical}")
        
        if not is_identical:
            # 找出差异
            diff = np.abs(pytorch_weight_np - mlx_weight_np)
            max_diff = np.max(diff)
            avg_diff = np.mean(diff)
            
            print(f"   权重差异: max={max_diff:.6f}, avg={avg_diff:.6f}")
            
            # 找出差异最大的位置
            max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"   最大差异位置: {max_diff_idx}")
            print(f"   PyTorch值: {pytorch_weight_np[max_diff_idx]:.6f}")
            print(f"   MLX值:     {mlx_weight_np[max_diff_idx]:.6f}")
            
            # 检查是否有符号差异
            sign_diff = np.sum(np.sign(pytorch_weight_np) != np.sign(mlx_weight_np))
            print(f"   符号差异数量: {sign_diff}")
            
            # 检查是否有数值差异
            value_diff = np.sum(np.abs(pytorch_weight_np - mlx_weight_np) > 1e-6)
            print(f"   数值差异数量: {value_diff}")
            
            # 检查MLX权重是否来自重构
            print(f"\n📊 检查MLX权重是否来自重构...")
            
            # 从权重文件重构权重
            pytorch_weights = torch.load("checkpoints/s2mel.pth", map_location='cpu')
            cfm_weights = pytorch_weights['net']['cfm']
            
            from indextts.s2mel.modules.mlx_diffusion_transformer_weights import load_weight_norm
            reconstructed_weight = load_weight_norm(cfm_weights, "estimator.x_embedder")
            
            print(f"   重构权重: shape={reconstructed_weight.shape}, min={reconstructed_weight.min():.6f}, max={reconstructed_weight.max():.6f}, avg={reconstructed_weight.mean():.6f}")
            
            # 比较重构权重与MLX权重
            diff_recon = np.abs(reconstructed_weight - mlx_weight_np)
            max_diff_recon = np.max(diff_recon)
            avg_diff_recon = np.mean(diff_recon)
            
            print(f"   重构权重与MLX权重差异: max={max_diff_recon:.6f}, avg={avg_diff_recon:.6f}")
            
            if max_diff_recon < 1e-6:
                print(f"   ✅ MLX权重来自重构")
            else:
                print(f"   ❌ MLX权重不是来自重构")
                
                # 检查MLX权重是否来自PyTorch模型
                diff_pytorch = np.abs(pytorch_weight_np - mlx_weight_np)
                max_diff_pytorch = np.max(diff_pytorch)
                
                if max_diff_pytorch < 1e-6:
                    print(f"   ✅ MLX权重来自PyTorch模型")
                else:
                    print(f"   ❌ MLX权重既不是来自重构也不是来自PyTorch模型")
                    print(f"   💡 MLX权重来源未知")
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_mlx_actual_weights()
