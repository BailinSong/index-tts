#!/usr/bin/env python3
"""
检查PyTorch权重文件的完整结构
"""

import torch

def check_pytorch_weight_structure():
    """检查PyTorch权重文件结构"""
    print("🔍 检查PyTorch权重文件结构")
    print("="*60)
    
    try:
        pytorch_weights = torch.load("checkpoints/s2mel.pth", map_location='cpu')
        
        print("📊 权重文件顶层键:")
        for key in pytorch_weights.keys():
            value = pytorch_weights[key]
            print(f"   {key}: {type(value)}")
            
            # 如果是字典，显示子键
            if isinstance(value, dict):
                print(f"     子键数量: {len(value)}")
                if len(value) <= 10:
                    for subkey in value.keys():
                        subvalue = value[subkey]
                        if isinstance(subvalue, torch.Tensor):
                            print(f"       {subkey}: {subvalue.shape} {subvalue.dtype}")
                        else:
                            print(f"       {subkey}: {type(subvalue)}")
                else:
                    print(f"       (显示前10个子键)")
                    for i, subkey in enumerate(list(value.keys())[:10]):
                        subvalue = value[subkey]
                        if isinstance(subvalue, torch.Tensor):
                            print(f"       {subkey}: {subvalue.shape} {subvalue.dtype}")
                        else:
                            print(f"       {subkey}: {type(subvalue)}")
        
        # 检查net键（通常是模型权重）
        if 'net' in pytorch_weights:
            print(f"\n📊 'net'键的内容:")
            net = pytorch_weights['net']
            if isinstance(net, dict):
                print(f"   net是字典，包含 {len(net)} 个键")
                
                # 查找包含'embedder'的键
                embedder_keys = [k for k in net.keys() if 'embedder' in k]
                if embedder_keys:
                    print(f"   找到embedder相关键: {embedder_keys}")
                    for key in embedder_keys:
                        value = net[key]
                        if isinstance(value, dict):
                            print(f"     {key}: 字典，包含 {len(value)} 个子键")
                            for subkey in value.keys():
                                subvalue = value[subkey]
                                if isinstance(subvalue, torch.Tensor):
                                    print(f"       {subkey}: {subvalue.shape} {subvalue.dtype}")
                                else:
                                    print(f"       {subkey}: {type(subvalue)}")
                        elif isinstance(value, torch.Tensor):
                            print(f"     {key}: {value.shape} {value.dtype}")
                        else:
                            print(f"     {key}: {type(value)}")
                else:
                    print(f"   net中没有embedder相关键")
                    
                    # 显示net的所有键
                    print(f"   net的所有键:")
                    for key in list(net.keys())[:20]:
                        value = net[key]
                        if isinstance(value, dict):
                            print(f"     {key}: 字典")
                        elif isinstance(value, torch.Tensor):
                            print(f"     {key}: {value.shape} {value.dtype}")
                        else:
                            print(f"     {key}: {type(value)}")
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_pytorch_weight_structure()
