#!/usr/bin/env python3
"""
深入分析PyTorch weight_norm的加载过程
"""

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm

def analyze_pytorch_weight_norm_loading():
    """分析PyTorch weight_norm的加载过程"""
    print("🔍 分析PyTorch weight_norm的加载过程")
    print("="*60)
    
    try:
        # 加载PyTorch权重文件
        pytorch_weights = torch.load("checkpoints/s2mel.pth", map_location='cpu')
        cfm_weights = pytorch_weights['net']['cfm']
        
        print(f"📊 权重文件中的x_embedder相关键:")
        x_embedder_keys = [k for k in cfm_weights.keys() if 'x_embedder' in k]
        for key in x_embedder_keys:
            print(f"   {key}: shape={cfm_weights[key].shape}")
        
        # 获取权重文件中的weight_g和weight_v
        weight_g = cfm_weights["estimator.x_embedder.weight_g"].numpy()
        weight_v = cfm_weights["estimator.x_embedder.weight_v"].numpy()
        
        print(f"\n📊 权重文件中的权重:")
        print(f"   weight_g: shape={weight_g.shape}, min={weight_g.min():.6f}, max={weight_g.max():.6f}")
        print(f"   weight_v: shape={weight_v.shape}, min={weight_v.min():.6f}, max={weight_v.max():.6f}")
        print(f"   weight:   不存在（weight_norm只保存weight_g和weight_v）")
        
        # 手动重构权重
        norm_v = np.sqrt(np.sum(weight_v**2, axis=1, keepdims=True))
        w_manual = weight_g * weight_v / (norm_v + 1e-8)
        
        print(f"\n📊 手动重构权重:")
        print(f"   w_manual: shape={w_manual.shape}, min={w_manual.min():.6f}, max={w_manual.max():.6f}")
        
        # 加载PyTorch模型
        from indextts.infer_v2 import IndexTTS2
        pytorch_tts = IndexTTS2(use_mlx=False)
        pytorch_dit = pytorch_tts.s2mel.models['cfm'].estimator
        
        pytorch_weight = pytorch_dit.x_embedder.weight.detach().cpu().numpy()
        
        print(f"\n📊 PyTorch模型中的权重:")
        print(f"   pytorch_weight: shape={pytorch_weight.shape}, min={pytorch_weight.min():.6f}, max={pytorch_weight.max():.6f}")
        
        # 权重文件中没有weight，只有weight_g和weight_v
        print(f"\n📊 权重文件中没有weight，只有weight_g和weight_v")
        
        # 比较PyTorch模型权重与重构权重
        diff_pytorch_manual = np.abs(pytorch_weight - w_manual)
        max_diff_pytorch_manual = np.max(diff_pytorch_manual)
        avg_diff_pytorch_manual = np.mean(diff_pytorch_manual)
        
        print(f"\n📊 PyTorch模型权重与重构权重差异:")
        print(f"   最大差异: {max_diff_pytorch_manual:.6f}")
        print(f"   平均差异: {avg_diff_pytorch_manual:.6f}")
        
        if max_diff_pytorch_manual < 1e-6:
            print(f"   ✅ PyTorch模型权重与重构权重一致")
        else:
            print(f"   ❌ PyTorch模型权重与重构权重不一致")
            
            # 找出差异最大的位置
            max_diff_idx = np.unravel_index(np.argmax(diff_pytorch_manual), diff_pytorch_manual.shape)
            print(f"   最大差异位置: {max_diff_idx}")
            print(f"   PyTorch值: {pytorch_weight[max_diff_idx]:.6f}")
            print(f"   重构值:    {w_manual[max_diff_idx]:.6f}")
        
        # 分析PyTorch的weight_norm行为
        print(f"\n📊 分析PyTorch的weight_norm行为:")
        
        # 检查PyTorch模型是否使用了weight_norm
        has_weight_g = hasattr(pytorch_dit.x_embedder, 'weight_g')
        has_weight_v = hasattr(pytorch_dit.x_embedder, 'weight_v')
        
        print(f"   PyTorch模型有weight_g: {has_weight_g}")
        print(f"   PyTorch模型有weight_v: {has_weight_v}")
        
        if has_weight_g and has_weight_v:
            pytorch_g = pytorch_dit.x_embedder.weight_g.detach().cpu().numpy()
            pytorch_v = pytorch_dit.x_embedder.weight_v.detach().cpu().numpy()
            
            print(f"   PyTorch模型weight_g: shape={pytorch_g.shape}, min={pytorch_g.min():.6f}, max={pytorch_g.max():.6f}")
            print(f"   PyTorch模型weight_v: shape={pytorch_v.shape}, min={pytorch_v.min():.6f}, max={pytorch_v.max():.6f}")
            
            # 比较PyTorch模型的weight_g和weight_v与权重文件
            diff_g = np.abs(pytorch_g - weight_g)
            diff_v = np.abs(pytorch_v - weight_v)
            
            print(f"   weight_g差异: max={np.max(diff_g):.6f}, avg={np.mean(diff_g):.6f}")
            print(f"   weight_v差异: max={np.max(diff_v):.6f}, avg={np.mean(diff_v):.6f}")
            
            if np.max(diff_g) < 1e-6 and np.max(diff_v) < 1e-6:
                print(f"   ✅ PyTorch模型的weight_g和weight_v与权重文件一致")
                print(f"   💡 问题在于PyTorch的weight_norm重构过程")
            else:
                print(f"   ❌ PyTorch模型的weight_g和weight_v与权重文件不一致")
                print(f"   💡 问题在于权重加载过程")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_pytorch_weight_norm_loading()
