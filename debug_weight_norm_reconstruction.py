#!/usr/bin/env python3
"""
深入调试 x_embedder 权重重构过程
"""

import torch
import numpy as np
from indextts.s2mel.modules.mlx_diffusion_transformer_weights import load_weight_norm

def debug_weight_norm_reconstruction():
    """调试权重归一化重构过程"""
    print("🔍 深入调试 x_embedder 权重重构过程")
    print("="*60)
    
    try:
        # 加载 PyTorch 权重
        pytorch_weights = torch.load("checkpoints/s2mel.pth", map_location='cpu')
        cfm_weights = pytorch_weights['net']['cfm']
        
        # 获取原始的 weight_g 和 weight_v
        g = cfm_weights["estimator.x_embedder.weight_g"].numpy()  # (512, 1)
        v = cfm_weights["estimator.x_embedder.weight_v"].numpy()  # (512, 80)
        
        print(f"📊 原始权重数据:")
        print(f"   weight_g: shape={g.shape}, min={g.min():.6f}, max={g.max():.6f}")
        print(f"   weight_v: shape={v.shape}, min={v.min():.6f}, max={v.max():.6f}")
        
        # 手动重构权重
        print(f"\n📊 手动重构权重:")
        norm_v = np.sqrt(np.sum(v**2, axis=1, keepdims=True))  # (512, 1)
        w_manual = g * v / (norm_v + 1e-8)
        
        print(f"   norm_v: shape={norm_v.shape}, min={norm_v.min():.6f}, max={norm_v.max():.6f}")
        print(f"   w_manual: shape={w_manual.shape}, min={w_manual.min():.6f}, max={w_manual.max():.6f}")
        
        # 使用 load_weight_norm 函数重构
        print(f"\n📊 使用 load_weight_norm 函数重构:")
        w_function = load_weight_norm(cfm_weights, "estimator.x_embedder")
        
        if w_function is not None:
            print(f"   w_function: shape={w_function.shape}, min={w_function.min():.6f}, max={w_function.max():.6f}")
            
            # 比较两种方法
            diff = np.abs(w_manual - w_function)
            max_diff = np.max(diff)
            avg_diff = np.mean(diff)
            
            print(f"\n📊 两种方法差异:")
            print(f"   最大差异: {max_diff:.6f}")
            print(f"   平均差异: {avg_diff:.6f}")
            
            if max_diff < 1e-6:
                print(f"   ✅ 两种方法结果一致")
            else:
                print(f"   ❌ 两种方法结果不一致")
                
                # 找出差异最大的位置
                max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
                print(f"   最大差异位置: {max_diff_idx}")
                print(f"   手动重构值: {w_manual[max_diff_idx]:.6f}")
                print(f"   函数重构值: {w_function[max_diff_idx]:.6f}")
        else:
            print(f"   ❌ load_weight_norm 返回 None")
        
        # 检查 PyTorch 模型中的实际权重
        print(f"\n📊 检查 PyTorch 模型中的实际权重:")
        from indextts.infer_v2 import IndexTTS2
        pytorch_tts = IndexTTS2(use_mlx=False)
        pytorch_dit = pytorch_tts.s2mel.models['cfm'].estimator
        
        pytorch_weight = pytorch_dit.x_embedder.weight.detach().cpu().numpy()
        print(f"   PyTorch实际权重: shape={pytorch_weight.shape}, min={pytorch_weight.min():.6f}, max={pytorch_weight.max():.6f}")
        
        # 与重构权重比较
        if w_function is not None:
            diff_pytorch = np.abs(pytorch_weight - w_function)
            max_diff_pytorch = np.max(diff_pytorch)
            avg_diff_pytorch = np.mean(diff_pytorch)
            
            print(f"\n📊 与PyTorch实际权重差异:")
            print(f"   最大差异: {max_diff_pytorch:.6f}")
            print(f"   平均差异: {avg_diff_pytorch:.6f}")
            
            if max_diff_pytorch < 1e-6:
                print(f"   ✅ 重构权重与PyTorch实际权重一致")
            else:
                print(f"   ❌ 重构权重与PyTorch实际权重不一致")
                
                # 找出差异最大的位置
                max_diff_idx = np.unravel_index(np.argmax(diff_pytorch), diff_pytorch.shape)
                print(f"   最大差异位置: {max_diff_idx}")
                print(f"   PyTorch值: {pytorch_weight[max_diff_idx]:.6f}")
                print(f"   重构值:    {w_function[max_diff_idx]:.6f}")
                
                # 检查是否是完全不同的权重
                if np.array_equal(pytorch_weight, w_function):
                    print(f"   ✅ 权重完全相同")
                else:
                    different_count = np.sum(diff_pytorch > 1e-10)
                    total_count = pytorch_weight.size
                    print(f"   ❌ 不同元素: {different_count}/{total_count} ({different_count/total_count*100:.2f}%)")
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_weight_norm_reconstruction()
