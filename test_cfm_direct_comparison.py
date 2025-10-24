#!/usr/bin/env python3
"""
直接使用缓存数据进行CFM对比测试
跳过前级处理，直接从CFM开始测试
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

def test_cfm_direct_comparison():
    """直接使用缓存数据进行CFM对比测试"""
    print("="*70)
    print("🧪 直接使用缓存数据进行CFM对比测试")
    print("="*70)
    
    # 加载缓存数据
    cached_data = load_cached_inputs()
    if not cached_data:
        print("❌ 没有找到缓存数据，请先运行推理生成缓存")
        return
    
    # 初始化IndexTTS2
    print("\n📦 初始化IndexTTS2...")
    tts = IndexTTS2(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        use_mlx=True,
        diffusion_steps=25
    )
    
    # 检查是否有MLX CFM
    if not hasattr(tts, 'mlx_s2mel_cfm') or tts.mlx_s2mel_cfm is None:
        print("❌ MLX CFM未初始化")
        return
    
    print("\n🔍 开始直接对比...")
    
    try:
        # 使用缓存数据进行对比
        if 'cfm_inputs_mlx' in cached_data and 'cfm_outputs_mlx' in cached_data:
            inputs = cached_data['cfm_inputs_mlx']
            outputs = cached_data['cfm_outputs_mlx']
            
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
            
            # 使用MLX CFM进行推理
            print("\n🔍 使用MLX CFM进行推理...")
            # 添加缺失的参数
            f0 = mx.zeros((1, 1, 1))  # 默认f0
            n_timesteps = 25  # 默认时间步数
            mlx_output = tts.mlx_s2mel_cfm.inference(
                cat_condition_mlx, x_lens_mlx, ref_mel_mlx, style_mlx, f0, n_timesteps
            )
            
            print(f"MLX CFM输出形状: {mlx_output.shape}")
            print(f"MLX CFM输出范围: [{mlx_output.min():.6f}, {mlx_output.max():.6f}]")
            
            # 对比输出
            if 'cfm_outputs_torch' in cached_data:
                torch_outputs = cached_data['cfm_outputs_torch']
                torch_output = torch_outputs['vc_target']
                
                print("\n📊 输出对比:")
                print(f"PyTorch输出形状: {torch_output.shape}")
                print(f"PyTorch输出范围: [{torch_output.min():.6f}, {torch_output.max():.6f}]")
                
                compare_tensor_values(torch_output, mlx_output, "CFM输出", 1e-5)
            
            # 对比estimator内部状态
            print("\n🔍 Estimator内部状态对比:")
            if hasattr(tts.mlx_s2mel_cfm, 'estimator'):
                mlx_estimator = tts.mlx_s2mel_cfm.estimator
                
                # 检查estimator的属性
                print(f"MLX Estimator类型: {type(mlx_estimator)}")
                print(f"MLX Estimator属性: {dir(mlx_estimator)}")
                
                # 检查是否有权重
                if hasattr(mlx_estimator, 'weight'):
                    print(f"MLX Estimator权重形状: {mlx_estimator.weight.shape}")
                    print(f"MLX Estimator权重范围: [{mlx_estimator.weight.min():.6f}, {mlx_estimator.weight.max():.6f}]")
                
                if hasattr(mlx_estimator, 'bias'):
                    print(f"MLX Estimator偏置形状: {mlx_estimator.bias.shape}")
                    print(f"MLX Estimator偏置范围: [{mlx_estimator.bias.min():.6f}, {mlx_estimator.bias.max():.6f}]")
            
            print("\n✅ 直接对比完成")
            return mlx_output
        else:
            print("❌ 没有找到MLX CFM输入/输出缓存")
            return None
            
    except Exception as e:
        print(f"❌ 对比过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_cfm_direct_comparison()
