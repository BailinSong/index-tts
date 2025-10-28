#!/usr/bin/env python3
"""
检查CFM权重结构
"""

import torch

def check_cfm_weights():
    """检查CFM权重结构"""
    print("🔍 检查CFM权重结构")
    print("="*60)
    
    try:
        pytorch_weights = torch.load("checkpoints/s2mel.pth", map_location='cpu')
        cfm_weights = pytorch_weights['net']['cfm']
        
        print(f"📊 CFM权重包含 {len(cfm_weights)} 个键:")
        
        # 查找包含'embedder'的键
        embedder_keys = [k for k in cfm_weights.keys() if 'embedder' in k]
        if embedder_keys:
            print(f"   找到embedder相关键: {embedder_keys}")
            for key in embedder_keys:
                value = cfm_weights[key]
                if isinstance(value, torch.Tensor):
                    print(f"     {key}: {value.shape} {value.dtype}")
                    print(f"        min={value.min():.6f}, max={value.max():.6f}, avg={value.mean():.6f}")
                else:
                    print(f"     {key}: {type(value)}")
        else:
            print(f"   CFM中没有embedder相关键")
        
        # 显示所有CFM键
        print(f"\n📊 CFM的所有键:")
        for i, key in enumerate(cfm_weights.keys()):
            value = cfm_weights[key]
            if isinstance(value, torch.Tensor):
                print(f"   {i+1:2d}. {key}: {value.shape} {value.dtype}")
            else:
                print(f"   {i+1:2d}. {key}: {type(value)}")
        
        # 查找包含'x'的键
        print(f"\n📊 包含'x'的键:")
        x_keys = [k for k in cfm_weights.keys() if 'x' in k]
        for key in x_keys:
            value = cfm_weights[key]
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape} {value.dtype}")
                print(f"      min={value.min():.6f}, max={value.max():.6f}, avg={value.mean():.6f}")
            else:
                print(f"   {key}: {type(value)}")
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_cfm_weights()
