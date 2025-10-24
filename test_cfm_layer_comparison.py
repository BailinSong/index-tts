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

def test_cfm_layer_comparison():
    """测试CFM逐层对比功能"""
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
    
    # 检查是否有MLX CFM
    if not hasattr(tts, 'mlx_s2mel_cfm') or tts.mlx_s2mel_cfm is None:
        print("❌ MLX CFM未初始化")
        return
    
    # 检查是否有PyTorch CFM
    if not hasattr(tts, 's2mel') or tts.s2mel is None:
        print("❌ PyTorch S2MEL未初始化")
        return
    
    print("\n🔍 开始逐层对比...")
    
    try:
        # 使用MLX CFM进行对比
        if 'cfm_inputs_mlx' in cached_data:
            inputs = cached_data['cfm_inputs_mlx']
            cat_condition = inputs['cat_condition']
            x_lens = inputs['x_lens']
            ref_mel = inputs['ref_mel']
            style = inputs['style']
            
            print("\n📊 使用缓存输入进行对比:")
            print(f"   cat_condition: {cat_condition.shape}")
            print(f"   x_lens: {x_lens.shape}")
            print(f"   ref_mel: {ref_mel.shape}")
            print(f"   style: {style.shape}")
            
            # 转换输入格式
            from indextts.utils.mlx_utils import torch_to_mlx
            cat_condition_mlx = torch_to_mlx(cat_condition)
            x_lens_mlx = torch_to_mlx(x_lens)
            ref_mel_mlx = torch_to_mlx(ref_mel)
            style_mlx = torch_to_mlx(style)
            
            print("\n📊 输入对比:")
            compare_tensor_values(cat_condition, cat_condition_mlx, "cat_condition", 1e-5)
            compare_tensor_values(x_lens, x_lens_mlx, "x_lens", 1e-5)
            compare_tensor_values(ref_mel, ref_mel_mlx, "ref_mel", 1e-5)
            compare_tensor_values(style, style_mlx, "style", 1e-5)
            
            # 对比estimator
            print("\n🔍 Estimator对比:")
            if hasattr(tts.mlx_s2mel_cfm, 'estimator') and hasattr(tts.s2mel.cfm, 'estimator'):
                mlx_estimator = tts.mlx_s2mel_cfm.estimator
                pytorch_estimator = tts.s2mel.cfm.estimator
                
                # 对比权重
                if hasattr(mlx_estimator, 'weight') and hasattr(pytorch_estimator, 'weight'):
                    mlx_weight = mlx_estimator.weight
                    pytorch_weight = pytorch_estimator.weight.detach().cpu()
                    compare_tensor_values(pytorch_weight, mlx_weight, "Estimator权重", 1e-5)
                
                # 对比偏置
                if hasattr(mlx_estimator, 'bias') and hasattr(pytorch_estimator, 'bias'):
                    mlx_bias = mlx_estimator.bias
                    pytorch_bias = pytorch_estimator.bias.detach().cpu()
                    compare_tensor_values(pytorch_bias, mlx_bias, "Estimator偏置", 1e-5)
            
            # 对比输出
            print("\n📊 输出对比:")
            mlx_output = tts.mlx_s2mel_cfm.inference(
                cat_condition_mlx, x_lens_mlx, ref_mel_mlx, style_mlx
            )
            pytorch_output = tts.s2mel.cfm.inference(
                cat_condition, x_lens, ref_mel, style
            )
            
            compare_tensor_values(pytorch_output, mlx_output, "CFM输出", 1e-5)
            
            print("\n✅ 对比完成")
            return mlx_output, pytorch_output
        else:
            print("❌ 没有找到MLX CFM输入缓存")
            return None, None
            
    except Exception as e:
        print(f"❌ 对比过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    test_cfm_layer_comparison()


