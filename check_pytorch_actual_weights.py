#!/usr/bin/env python3
"""
检查PyTorch模型实际使用的权重
"""

import torch
import numpy as np
from indextts.s2mel.modules.mlx_diffusion_transformer_weights import load_weight_norm

def check_pytorch_actual_weights():
    """检查PyTorch模型实际使用的权重"""
    print("🔍 检查PyTorch模型实际使用的权重")
    print("="*60)
    
    try:
        # 加载PyTorch权重文件
        pytorch_weights = torch.load("checkpoints/s2mel.pth", map_location='cpu')
        cfm_weights = pytorch_weights['net']['cfm']
        
        # 重构权重
        reconstructed_weight = load_weight_norm(cfm_weights, "estimator.x_embedder")
        
        print(f"📊 从weight_g和weight_v重构的权重:")
        print(f"   shape={reconstructed_weight.shape}, min={reconstructed_weight.min():.6f}, max={reconstructed_weight.max():.6f}")
        
        # 加载PyTorch模型
        from indextts.infer_v2 import IndexTTS2
        pytorch_tts = IndexTTS2(use_mlx=False)
        pytorch_dit = pytorch_tts.s2mel.models['cfm'].estimator
        
        pytorch_weight = pytorch_dit.x_embedder.weight.detach().cpu().numpy()
        
        print(f"\n📊 PyTorch模型中的权重:")
        print(f"   shape={pytorch_weight.shape}, min={pytorch_weight.min():.6f}, max={pytorch_weight.max():.6f}")
        
        # 比较差异
        diff = np.abs(reconstructed_weight - pytorch_weight)
        max_diff = np.max(diff)
        avg_diff = np.mean(diff)
        
        print(f"\n📊 重构权重与PyTorch权重差异:")
        print(f"   最大差异: {max_diff:.6f}")
        print(f"   平均差异: {avg_diff:.6f}")
        
        if max_diff < 1e-6:
            print(f"   ✅ 重构权重与PyTorch权重一致")
        else:
            print(f"   ❌ 重构权重与PyTorch权重不一致")
            
            # 找出差异最大的位置
            max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"   最大差异位置: {max_diff_idx}")
            print(f"   重构值: {reconstructed_weight[max_diff_idx]:.6f}")
            print(f"   PyTorch值: {pytorch_weight[max_diff_idx]:.6f}")
        
        # 检查PyTorch模型的weight_g和weight_v
        print(f"\n📊 PyTorch模型的weight_g和weight_v:")
        pytorch_g = pytorch_dit.x_embedder.weight_g.detach().cpu().numpy()
        pytorch_v = pytorch_dit.x_embedder.weight_v.detach().cpu().numpy()
        
        print(f"   weight_g: shape={pytorch_g.shape}, min={pytorch_g.min():.6f}, max={pytorch_g.max():.6f}")
        print(f"   weight_v: shape={pytorch_v.shape}, min={pytorch_v.min():.6f}, max={pytorch_v.max():.6f}")
        
        # 与权重文件中的比较
        file_g = cfm_weights["estimator.x_embedder.weight_g"].numpy()
        file_v = cfm_weights["estimator.x_embedder.weight_v"].numpy()
        
        print(f"\n📊 权重文件中的weight_g和weight_v:")
        print(f"   weight_g: shape={file_g.shape}, min={file_g.min():.6f}, max={file_g.max():.6f}")
        print(f"   weight_v: shape={file_v.shape}, min={file_v.min():.6f}, max={file_v.max():.6f}")
        
        # 比较weight_g和weight_v
        diff_g = np.abs(pytorch_g - file_g)
        diff_v = np.abs(pytorch_v - file_v)
        
        print(f"\n📊 weight_g和weight_v差异:")
        print(f"   weight_g差异: max={np.max(diff_g):.6f}, avg={np.mean(diff_g):.6f}")
        print(f"   weight_v差异: max={np.max(diff_v):.6f}, avg={np.mean(diff_v):.6f}")
        
        if np.max(diff_g) < 1e-6 and np.max(diff_v) < 1e-6:
            print(f"   ✅ weight_g和weight_v一致")
            print(f"   💡 问题在于PyTorch的weight重构过程")
        else:
            print(f"   ❌ weight_g和weight_v不一致")
            print(f"   💡 问题在于权重加载过程")
        
        # 手动重构PyTorch模型的权重
        print(f"\n📊 手动重构PyTorch模型的权重:")
        norm_v_pytorch = np.sqrt(np.sum(pytorch_v**2, axis=1, keepdims=True))
        w_manual_pytorch = pytorch_g * pytorch_v / (norm_v_pytorch + 1e-8)
        
        print(f"   w_manual_pytorch: shape={w_manual_pytorch.shape}, min={w_manual_pytorch.min():.6f}, max={w_manual_pytorch.max():.6f}")
        
        # 比较手动重构与PyTorch权重
        diff_manual = np.abs(w_manual_pytorch - pytorch_weight)
        max_diff_manual = np.max(diff_manual)
        
        print(f"   手动重构与PyTorch权重差异: {max_diff_manual:.6f}")
        
        if max_diff_manual < 1e-6:
            print(f"   ✅ 手动重构与PyTorch权重一致")
        else:
            print(f"   ❌ 手动重构与PyTorch权重不一致")
            print(f"   💡 PyTorch的weight_norm实现有问题")
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_pytorch_actual_weights()
