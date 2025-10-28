#!/usr/bin/env python3
"""
检查PyTorch weight_norm的自动处理
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

def test_pytorch_weight_norm():
    """测试PyTorch weight_norm的自动处理"""
    print("🔍 测试PyTorch weight_norm的自动处理")
    print("="*60)
    
    try:
        # 创建一个简单的weight_norm层
        linear = nn.Linear(80, 512, bias=True)
        wn_linear = weight_norm(linear)
        
        print(f"📊 原始Linear层:")
        print(f"   weight: shape={linear.weight.shape}, min={linear.weight.min():.6f}, max={linear.weight.max():.6f}")
        
        print(f"\n📊 Weight Norm Linear层:")
        print(f"   weight_g: shape={wn_linear.weight_g.shape}, min={wn_linear.weight_g.min():.6f}, max={wn_linear.weight_g.max():.6f}")
        print(f"   weight_v: shape={wn_linear.weight_v.shape}, min={wn_linear.weight_v.min():.6f}, max={wn_linear.weight_v.max():.6f}")
        
        # 检查weight属性
        print(f"\n📊 Weight Norm Linear的weight属性:")
        print(f"   weight: shape={wn_linear.weight.shape}, min={wn_linear.weight.min():.6f}, max={wn_linear.weight.max():.6f}")
        
        # 手动重构权重
        norm_v = torch.sqrt(torch.sum(wn_linear.weight_v**2, dim=1, keepdim=True))
        w_manual = wn_linear.weight_g * wn_linear.weight_v / (norm_v + 1e-8)
        
        print(f"\n📊 手动重构权重:")
        print(f"   w_manual: shape={w_manual.shape}, min={w_manual.min():.6f}, max={w_manual.max():.6f}")
        
        # 比较
        diff = torch.abs(wn_linear.weight - w_manual)
        max_diff = torch.max(diff)
        avg_diff = torch.mean(diff)
        
        print(f"\n📊 与weight属性差异:")
        print(f"   最大差异: {max_diff:.6f}")
        print(f"   平均差异: {avg_diff:.6f}")
        
        if max_diff < 1e-6:
            print(f"   ✅ weight属性与手动重构一致")
        else:
            print(f"   ❌ weight属性与手动重构不一致")
        
        # 测试state_dict
        print(f"\n📊 State Dict:")
        state_dict = wn_linear.state_dict()
        for key, value in state_dict.items():
            print(f"   {key}: {value.shape}")
        
        # 测试load_state_dict
        print(f"\n📊 测试load_state_dict:")
        new_wn_linear = weight_norm(nn.Linear(80, 512, bias=True))
        new_wn_linear.load_state_dict(state_dict)
        
        print(f"   加载后weight: shape={new_wn_linear.weight.shape}, min={new_wn_linear.weight.min():.6f}, max={new_wn_linear.weight.max():.6f}")
        
        # 比较加载前后的weight
        diff_load = torch.abs(wn_linear.weight - new_wn_linear.weight)
        max_diff_load = torch.max(diff_load)
        
        print(f"   加载前后差异: {max_diff_load:.6f}")
        
        if max_diff_load < 1e-6:
            print(f"   ✅ load_state_dict正确")
        else:
            print(f"   ❌ load_state_dict有问题")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pytorch_weight_norm()
