#!/usr/bin/env python3
"""
专门分析conv2差异的脚本
"""

import pickle
import numpy as np
import torch

def analyze_conv2_differences():
    """分析conv2的具体差异"""
    
    # 加载调试数据
    with open('cfm_debug_outputs/complete_dit_debug_analysis.pkl', 'rb') as f:
        debug_data = pickle.load(f)
    
    print("🔍 Conv2 差异详细分析")
    print("=" * 60)
    
    # 获取conv2数据
    pytorch_data = debug_data['pytorch_data']
    mlx_data = debug_data['mlx_data']
    
    # 查找conv2相关的数据
    conv2_pytorch = None
    conv2_mlx = None
    
    for step_key, step_data in pytorch_data.items():
        if 'conv2_output' in step_data:
            conv2_pytorch = step_data['conv2_output']['conv2_out']
            print(f"📊 PyTorch Conv2 数据:")
            print(f"   形状: {conv2_pytorch.shape}")
            print(f"   数据类型: {type(conv2_pytorch)}")
            print(f"   数值范围: [{conv2_pytorch.min():.6f}, {conv2_pytorch.max():.6f}]")
            print(f"   均值: {conv2_pytorch.mean():.6f}")
            print(f"   标准差: {conv2_pytorch.std():.6f}")
            break
    
    for step_key, step_data in mlx_data.items():
        if 'conv2_output' in step_data:
            conv2_mlx = step_data['conv2_output']['conv2_out']
            print(f"\n📊 MLX Conv2 数据:")
            print(f"   形状: {conv2_mlx.shape}")
            print(f"   数据类型: {type(conv2_mlx)}")
            print(f"   数值范围: [{conv2_mlx.min():.6f}, {conv2_mlx.max():.6f}]")
            print(f"   均值: {conv2_mlx.mean():.6f}")
            print(f"   标准差: {conv2_mlx.std():.6f}")
            break
    
    if conv2_pytorch is not None and conv2_mlx is not None:
        # 转换为numpy进行比较
        if isinstance(conv2_pytorch, torch.Tensor):
            conv2_pytorch_np = conv2_pytorch.detach().cpu().numpy()
        else:
            conv2_pytorch_np = conv2_pytorch
        
        if hasattr(conv2_mlx, 'item'):  # MLX array
            conv2_mlx_np = np.array(conv2_mlx)
        else:
            conv2_mlx_np = conv2_mlx
        
        print(f"\n🔍 差异分析:")
        print(f"   形状差异: PyTorch {conv2_pytorch_np.shape} vs MLX {conv2_mlx_np.shape}")
        
        if conv2_pytorch_np.shape == conv2_mlx_np.shape:
            diff = np.abs(conv2_pytorch_np - conv2_mlx_np)
            print(f"   最大差异: {diff.max():.6f}")
            print(f"   平均差异: {diff.mean():.6f}")
            print(f"   差异标准差: {diff.std():.6f}")
            
            # 检查前几个元素
            print(f"\n🔍 前5个元素对比:")
            for i in range(min(5, conv2_pytorch_np.size)):
                pytorch_val = conv2_pytorch_np.flat[i]
                mlx_val = conv2_mlx_np.flat[i]
                diff_val = abs(pytorch_val - mlx_val)
                print(f"   [{i}] PyTorch: {pytorch_val:.6f}, MLX: {mlx_val:.6f}, 差异: {diff_val:.6f}")
        else:
            print(f"   ❌ 形状不匹配，无法直接比较")
    
    # 检查final_layer的输出
    print(f"\n🔍 Final Layer 输出分析:")
    for step_key, step_data in pytorch_data.items():
        if 'final_layer' in step_data and 'final_layer_output' in step_data['final_layer']:
            final_pytorch = step_data['final_layer']['final_layer_output']
            print(f"📊 PyTorch Final Layer 输出:")
            print(f"   形状: {final_pytorch.shape}")
            print(f"   数值范围: [{final_pytorch.min():.6f}, {final_pytorch.max():.6f}]")
            break
    
    for step_key, step_data in mlx_data.items():
        if 'final_layer' in step_data and 'final_layer_output' in step_data['final_layer']:
            final_mlx = step_data['final_layer']['final_layer_output']
            print(f"📊 MLX Final Layer 输出:")
            print(f"   形状: {final_mlx.shape}")
            print(f"   数值范围: [{final_mlx.min():.6f}, {final_mlx.max():.6f}]")
            break

if __name__ == "__main__":
    analyze_conv2_differences()

