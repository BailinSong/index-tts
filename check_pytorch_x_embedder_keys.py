#!/usr/bin/env python3
"""
检查PyTorch权重文件中的x_embedder相关键
"""

import torch

def check_pytorch_weights():
    """检查PyTorch权重文件"""
    print("🔍 检查PyTorch权重文件中的x_embedder相关键")
    print("="*60)
    
    try:
        pytorch_weights = torch.load("checkpoints/s2mel.pth", map_location='cpu')
        
        print("📊 所有包含'x_embedder'的键:")
        x_embedder_keys = [k for k in pytorch_weights.keys() if 'x_embedder' in k]
        for key in sorted(x_embedder_keys):
            weight = pytorch_weights[key]
            if isinstance(weight, torch.Tensor):
                print(f"   {key}: shape={weight.shape}, dtype={weight.dtype}")
                print(f"      min={weight.min():.6f}, max={weight.max():.6f}, avg={weight.mean():.6f}")
            else:
                print(f"   {key}: {type(weight)}")
        
        print(f"\n📊 总共找到 {len(x_embedder_keys)} 个x_embedder相关键")
        
        # 检查是否有直接的weight键
        direct_weight_keys = [k for k in x_embedder_keys if 'weight' in k and 'weight_g' not in k and 'weight_v' not in k]
        print(f"\n📊 直接的weight键: {direct_weight_keys}")
        
        # 检查是否有weight_g和weight_v
        weight_g_keys = [k for k in x_embedder_keys if 'weight_g' in k]
        weight_v_keys = [k for k in x_embedder_keys if 'weight_v' in k]
        print(f"📊 weight_g键: {weight_g_keys}")
        print(f"📊 weight_v键: {weight_v_keys}")
        
        if weight_g_keys and weight_v_keys:
            print(f"\n📊 验证weight_norm重构:")
            g = pytorch_weights[weight_g_keys[0]]
            v = pytorch_weights[weight_v_keys[0]]
            
            # 重构权重
            norm_v = torch.sqrt(torch.sum(v**2, dim=1, keepdim=True))
            w_reconstructed = g * v / (norm_v + 1e-8)
            
            print(f"   weight_g: shape={g.shape}, min={g.min():.6f}, max={g.max():.6f}")
            print(f"   weight_v: shape={v.shape}, min={v.min():.6f}, max={v.max():.6f}")
            print(f"   重构权重: shape={w_reconstructed.shape}, min={w_reconstructed.min():.6f}, max={w_reconstructed.max():.6f}")
            
            # 如果有直接的weight键，进行比较
            if direct_weight_keys:
                direct_weight = pytorch_weights[direct_weight_keys[0]]
                diff = torch.abs(direct_weight - w_reconstructed)
                max_diff = torch.max(diff)
                avg_diff = torch.mean(diff)
                
                print(f"\n📊 与直接权重差异:")
                print(f"   最大差异: {max_diff:.6f}")
                print(f"   平均差异: {avg_diff:.6f}")
                
                if max_diff < 1e-6:
                    print(f"   ✅ 权重重构正确")
                else:
                    print(f"   ❌ 权重重构有问题")
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_pytorch_weights()
