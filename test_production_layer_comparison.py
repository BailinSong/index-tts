#!/usr/bin/env python3
"""
测试MLX和PyTorch CFM的逐层对比
使用前级缓存数据直接从CFM开始测试
"""

import os
import sys
import pickle
import torch
import mlx.core as mx
import numpy as np

# 添加项目路径
sys.path.append('.')

from indextts.infer_v2 import IndexTTS2

def load_cached_inputs():
    """加载前级缓存的输入数据"""
    cache_files = [
        'cfm_inputs_mlx.pkl',
        'cfm_inputs_torch.pkl', 
        'cfm_outputs_mlx.pkl',
        'cfm_outputs_torch.pkl'
    ]
    
    cached_data = {}
    for file in cache_files:
        if os.path.exists(file):
            try:
                with open(file, 'rb') as f:
                    cached_data[file.replace('.pkl', '')] = pickle.load(f)
                print(f"✓ 加载缓存文件: {file}")
            except Exception as e:
                print(f"✗ 加载缓存文件失败 {file}: {e}")
        else:
            print(f"✗ 缓存文件不存在: {file}")
    
    return cached_data

def compare_tensor_values(torch_tensor, mlx_array, name, tolerance=1e-5):
    """对比PyTorch tensor和MLX array的数值"""
    # 转换MLX array为numpy
    if hasattr(mlx_array, 'numpy'):
        mlx_np = mlx_array.numpy()
    else:
        mlx_np = np.array(mlx_array)
    
    # 转换PyTorch tensor为numpy
    torch_np = torch_tensor.detach().cpu().numpy()
    
    # 确保形状一致
    if torch_np.shape != mlx_np.shape:
        print(f"❌ {name}: 形状不匹配 - PyTorch: {torch_np.shape}, MLX: {mlx_np.shape}")
        return False
    
    # 计算差异
    diff = np.abs(torch_np - mlx_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"📊 {name}:")
    print(f"   形状: {torch_np.shape}")
    print(f"   PyTorch范围: [{torch_np.min():.6f}, {torch_np.max():.6f}]")
    print(f"   MLX范围: [{mlx_np.min():.6f}, {mlx_np.max():.6f}]")
    print(f"   最大差异: {max_diff:.10f}")
    print(f"   平均差异: {mean_diff:.10f}")
    
    if max_diff < tolerance:
        print(f"   ✅ 差异在容忍范围内 (< {tolerance})")
        return True
    else:
        print(f"   ❌ 差异超出容忍范围 (>= {tolerance})")
        return False

def test_layer_by_layer_comparison():
    """测试逐层对比功能"""
    print("="*70)
    print("🧪 测试MLX和PyTorch CFM逐层对比")
    print("="*70)
    
    # 初始化IndexTTS2
    print("\n📦 初始化IndexTTS2...")
    tts = IndexTTS2(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        use_mlx=True,
        diffusion_steps=25
    )
    
    # 检查是否有缓存数据
    cached_data = load_cached_inputs()
    if not cached_data:
        print("❌ 没有找到缓存数据，请先运行推理生成缓存")
        return
    
    # 进行对比
    print("\n🔍 开始逐层对比...")
    try:
        mlx_output, pytorch_output = tts.compare_mlx_pytorch_cfm(cached_data)
        
        if mlx_output is not None and pytorch_output is not None:
            print("\n✅ 对比完成")
            print(f"MLX输出形状: {mlx_output.shape}")
            print(f"PyTorch输出形状: {pytorch_output.shape}")
            
            # 最终输出对比
            print("\n📊 最终输出对比:")
            compare_tensor_values(pytorch_output, mlx_output, "CFM最终输出", tolerance=1e-5)
        else:
            print("❌ 对比失败")
            
    except Exception as e:
        print(f"❌ 对比过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_layer_by_layer_comparison()
