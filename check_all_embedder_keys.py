#!/usr/bin/env python3
"""
检查PyTorch权重文件中所有embedder相关的键
"""

import torch

def check_all_embedder_keys():
    """检查所有embedder相关的键"""
    print("🔍 检查PyTorch权重文件中所有embedder相关的键")
    print("="*60)
    
    try:
        pytorch_weights = torch.load("checkpoints/s2mel.pth", map_location='cpu')
        
        print("📊 所有包含'embedder'的键:")
        embedder_keys = [k for k in pytorch_weights.keys() if 'embedder' in k]
        for key in sorted(embedder_keys):
            weight = pytorch_weights[key]
            if isinstance(weight, torch.Tensor):
                print(f"   {key}: shape={weight.shape}, dtype={weight.dtype}")
                print(f"      min={weight.min():.6f}, max={weight.max():.6f}, avg={weight.mean():.6f}")
            else:
                print(f"   {key}: {type(weight)}")
        
        print(f"\n📊 总共找到 {len(embedder_keys)} 个embedder相关键")
        
        # 检查所有键，寻找可能的x_embedder
        print(f"\n📊 所有键名（前50个）:")
        all_keys = list(pytorch_weights.keys())
        for i, key in enumerate(all_keys[:50]):
            print(f"   {i+1:2d}. {key}")
        
        if len(all_keys) > 50:
            print(f"   ... 还有 {len(all_keys) - 50} 个键")
        
        # 搜索包含'x'和'embed'的键
        print(f"\n📊 包含'x'和'embed'的键:")
        x_embed_keys = [k for k in all_keys if 'x' in k and 'embed' in k]
        for key in x_embed_keys:
            weight = pytorch_weights[key]
            if isinstance(weight, torch.Tensor):
                print(f"   {key}: shape={weight.shape}, dtype={weight.dtype}")
                print(f"      min={weight.min():.6f}, max={weight.max():.6f}, avg={weight.mean():.6f}")
            else:
                print(f"   {key}: {type(weight)}")
        
        # 搜索包含'cfm'的键
        print(f"\n📊 包含'cfm'的键（前20个）:")
        cfm_keys = [k for k in all_keys if 'cfm' in k][:20]
        for key in cfm_keys:
            weight = pytorch_weights[key]
            if isinstance(weight, torch.Tensor):
                print(f"   {key}: shape={weight.shape}, dtype={weight.dtype}")
            else:
                print(f"   {key}: {type(weight)}")
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_all_embedder_keys()
