#!/usr/bin/env python3
"""
检查PyTorch weight_norm在推理时的动态重构
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

def test_weight_norm_forward():
    """测试weight_norm在forward时的行为"""
    print("🔍 测试PyTorch weight_norm在forward时的行为")
    print("="*60)
    
    try:
        # 创建一个weight_norm层
        linear = torch.nn.Linear(80, 512, bias=True)
        wn_linear = weight_norm(linear)
        
        print(f"📊 Weight Norm Linear初始状态:")
        print(f"   weight_g: shape={wn_linear.weight_g.shape}, min={wn_linear.weight_g.min():.6f}, max={wn_linear.weight_g.max():.6f}")
        print(f"   weight_v: shape={wn_linear.weight_v.shape}, min={wn_linear.weight_v.min():.6f}, max={wn_linear.weight_v.max():.6f}")
        print(f"   weight: shape={wn_linear.weight.shape}, min={wn_linear.weight.min():.6f}, max={wn_linear.weight.max():.6f}")
        
        # 手动重构权重
        norm_v = torch.sqrt(torch.sum(wn_linear.weight_v**2, dim=1, keepdim=True))
        w_manual = wn_linear.weight_g * wn_linear.weight_v / (norm_v + 1e-8)
        
        print(f"\n📊 手动重构权重:")
        print(f"   w_manual: shape={w_manual.shape}, min={w_manual.min():.6f}, max={w_manual.max():.6f}")
        
        # 比较weight属性
        diff = torch.abs(wn_linear.weight - w_manual)
        max_diff = torch.max(diff)
        
        print(f"\n📊 weight属性与手动重构差异: {max_diff:.6f}")
        
        if max_diff < 1e-6:
            print(f"   ✅ weight属性与手动重构一致")
        else:
            print(f"   ❌ weight属性与手动重构不一致")
        
        # 测试forward
        print(f"\n📊 测试forward:")
        x = torch.randn(2, 80)
        
        # 记录forward前的weight
        weight_before = wn_linear.weight.clone()
        
        # 执行forward
        y = wn_linear(x)
        
        # 记录forward后的weight
        weight_after = wn_linear.weight.clone()
        
        print(f"   Forward前weight: shape={weight_before.shape}, min={weight_before.min():.6f}, max={weight_before.max():.6f}")
        print(f"   Forward后weight: shape={weight_after.shape}, min={weight_after.min():.6f}, max={weight_after.max():.6f}")
        
        # 比较forward前后的weight
        diff_forward = torch.abs(weight_before - weight_after)
        max_diff_forward = torch.max(diff_forward)
        
        print(f"   Forward前后差异: {max_diff_forward:.6f}")
        
        if max_diff_forward < 1e-6:
            print(f"   ✅ Forward前后weight相同")
        else:
            print(f"   ❌ Forward前后weight不同")
            print(f"   💡 PyTorch在forward时动态重构了权重")
        
        # 检查weight_g和weight_v是否变化
        print(f"\n📊 检查weight_g和weight_v:")
        print(f"   weight_g: shape={wn_linear.weight_g.shape}, min={wn_linear.weight_g.min():.6f}, max={wn_linear.weight_g.max():.6f}")
        print(f"   weight_v: shape={wn_linear.weight_v.shape}, min={wn_linear.weight_v.min():.6f}, max={wn_linear.weight_v.max():.6f}")
        
        # 再次手动重构
        norm_v_after = torch.sqrt(torch.sum(wn_linear.weight_v**2, dim=1, keepdim=True))
        w_manual_after = wn_linear.weight_g * wn_linear.weight_v / (norm_v_after + 1e-8)
        
        print(f"\n📊 Forward后手动重构权重:")
        print(f"   w_manual_after: shape={w_manual_after.shape}, min={w_manual_after.min():.6f}, max={w_manual_after.max():.6f}")
        
        # 比较forward后的weight与手动重构
        diff_after = torch.abs(weight_after - w_manual_after)
        max_diff_after = torch.max(diff_after)
        
        print(f"   Forward后weight与手动重构差异: {max_diff_after:.6f}")
        
        if max_diff_after < 1e-6:
            print(f"   ✅ Forward后weight与手动重构一致")
        else:
            print(f"   ❌ Forward后weight与手动重构不一致")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_weight_norm_forward()
