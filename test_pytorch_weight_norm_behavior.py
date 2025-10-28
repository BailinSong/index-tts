#!/usr/bin/env python3
"""
重新分析MLX weight_norm重构问题
"""

import torch
import numpy as np
from torch.nn.utils import weight_norm

def test_pytorch_weight_norm_behavior():
    """测试PyTorch weight_norm的实际行为"""
    print("🔍 测试PyTorch weight_norm的实际行为")
    print("="*60)
    
    try:
        # 创建一个weight_norm层
        linear = torch.nn.Linear(80, 512, bias=True)
        wn_linear = weight_norm(linear)
        
        print(f"📊 原始Linear权重:")
        print(f"   weight: shape={linear.weight.shape}, min={linear.weight.min():.6f}, max={linear.weight.max():.6f}")
        
        print(f"\n📊 Weight Norm Linear:")
        print(f"   weight_g: shape={wn_linear.weight_g.shape}, min={wn_linear.weight_g.min():.6f}, max={wn_linear.weight_g.max():.6f}")
        print(f"   weight_v: shape={wn_linear.weight_v.shape}, min={wn_linear.weight_v.min():.6f}, max={wn_linear.weight_v.max():.6f}")
        print(f"   weight: shape={wn_linear.weight.shape}, min={wn_linear.weight.min():.6f}, max={wn_linear.weight.max():.6f}")
        
        # 手动重构权重
        norm_v = torch.sqrt(torch.sum(wn_linear.weight_v**2, dim=1, keepdim=True))
        w_manual = wn_linear.weight_g * wn_linear.weight_v / (norm_v + 1e-8)
        
        print(f"\n📊 手动重构权重:")
        print(f"   w_manual: shape={w_manual.shape}, min={w_manual.min():.6f}, max={w_manual.max():.6f}")
        
        # 比较
        diff = torch.abs(wn_linear.weight - w_manual)
        max_diff = torch.max(diff)
        
        print(f"\n📊 与weight属性差异: {max_diff:.6f}")
        
        if max_diff < 1e-6:
            print(f"   ✅ 手动重构正确")
        else:
            print(f"   ❌ 手动重构有问题")
        
        # 测试load_state_dict
        print(f"\n📊 测试load_state_dict:")
        
        # 保存state_dict
        state_dict = wn_linear.state_dict()
        
        # 创建新的weight_norm层
        new_linear = torch.nn.Linear(80, 512, bias=True)
        new_wn_linear = weight_norm(new_linear)
        
        # 加载state_dict
        new_wn_linear.load_state_dict(state_dict)
        
        print(f"   原始weight: shape={wn_linear.weight.shape}, min={wn_linear.weight.min():.6f}, max={wn_linear.weight.max():.6f}")
        print(f"   加载后weight: shape={new_wn_linear.weight.shape}, min={new_wn_linear.weight.min():.6f}, max={new_wn_linear.weight.max():.6f}")
        
        # 比较加载前后的weight
        diff_load = torch.abs(wn_linear.weight - new_wn_linear.weight)
        max_diff_load = torch.max(diff_load)
        
        print(f"   加载前后差异: {max_diff_load:.6f}")
        
        if max_diff_load < 1e-6:
            print(f"   ✅ load_state_dict正确")
        else:
            print(f"   ❌ load_state_dict有问题")
            
            # 检查是否是初始化问题
            print(f"\n📊 检查初始化问题:")
            print(f"   原始weight_g: shape={wn_linear.weight_g.shape}, min={wn_linear.weight_g.min():.6f}, max={wn_linear.weight_g.max():.6f}")
            print(f"   加载后weight_g: shape={new_wn_linear.weight_g.shape}, min={new_wn_linear.weight_g.min():.6f}, max={new_wn_linear.weight_g.max():.6f}")
            
            print(f"   原始weight_v: shape={wn_linear.weight_v.shape}, min={wn_linear.weight_v.min():.6f}, max={wn_linear.weight_v.max():.6f}")
            print(f"   加载后weight_v: shape={new_wn_linear.weight_v.shape}, min={new_wn_linear.weight_v.min():.6f}, max={new_wn_linear.weight_v.max():.6f}")
            
            # 检查weight_g和weight_v是否相同
            diff_g = torch.abs(wn_linear.weight_g - new_wn_linear.weight_g)
            diff_v = torch.abs(wn_linear.weight_v - new_wn_linear.weight_v)
            
            print(f"   weight_g差异: {torch.max(diff_g):.6f}")
            print(f"   weight_v差异: {torch.max(diff_v):.6f}")
            
            if torch.max(diff_g) < 1e-6 and torch.max(diff_v) < 1e-6:
                print(f"   ✅ weight_g和weight_v相同")
                print(f"   💡 问题在于weight_norm的重构过程")
            else:
                print(f"   ❌ weight_g和weight_v不同")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pytorch_weight_norm_behavior()
