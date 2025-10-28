#!/usr/bin/env python3
"""
调试 mu 长度不一致问题
检查 PyTorch 和 MLX 在 CFM 推理过程中 mu 长度的差异
"""

import torch
import mlx.core as mx
import numpy as np
from indextts.infer_v2 import IndexTTS2

def debug_mu_length_issue():
    """调试 mu 长度问题"""
    print("🔍 调试 mu 长度不一致问题")
    print("="*60)
    
    # 加载 PyTorch 和 MLX 模型
    print("📦 加载模型...")
    pytorch_tts = IndexTTS2(use_mlx=False)
    mlx_tts = IndexTTS2(use_mlx=True)
    
    print("✅ 模型加载完成")
    
    # 使用相同的输入参数
    print("\n🔍 检查 CFM 输入参数...")
    
    # 从一致测试数据中加载参数
    import pickle
    with open('cfm_consistent_test/consistent_test_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    pytorch_inputs = data['pytorch_inputs']
    mlx_inputs = data['mlx_inputs']
    
    print(f"PyTorch 输入形状:")
    for k, v in pytorch_inputs.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape}")
    
    print(f"MLX 输入形状:")
    for k, v in mlx_inputs.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape}")
    
    # 检查 CFM 模型接收到的参数
    print("\n🔍 检查 CFM 模型接收到的参数...")
    
    # PyTorch CFM
    pytorch_cfm = pytorch_tts.s2mel.models['cfm']
    print(f"PyTorch CFM 类型: {type(pytorch_cfm)}")
    
    # MLX CFM  
    mlx_cfm = mlx_tts.mlx_s2mel_cfm
    print(f"MLX CFM 类型: {type(mlx_cfm)}")
    
    # 检查 CFM 的 inference 方法签名
    print(f"\nPyTorch CFM inference 方法: {pytorch_cfm.inference}")
    print(f"MLX CFM inference 方法: {mlx_cfm.inference}")
    
    # 尝试调用 CFM inference 并检查参数
    print("\n🔍 尝试调用 CFM inference...")
    
    try:
        # 准备参数
        mu_pt = pytorch_inputs['mu']
        x_lens_pt = pytorch_inputs['x_lens']
        prompt_pt = pytorch_inputs['prompt']
        style_pt = pytorch_inputs['style']
        f0 = pytorch_inputs['f0']
        n_timesteps = pytorch_inputs['n_timesteps']
        temperature = pytorch_inputs['temperature']
        inference_cfg_rate = pytorch_inputs['inference_cfg_rate']
        
        print(f"调用 PyTorch CFM inference...")
        print(f"  输入参数形状:")
        print(f"    mu: {mu_pt.shape}")
        print(f"    x_lens: {x_lens_pt.shape}")
        print(f"    prompt: {prompt_pt.shape}")
        print(f"    style: {style_pt.shape}")
        
        # 调用 PyTorch CFM
        pytorch_output = pytorch_cfm.inference(
            mu_pt, x_lens_pt, prompt_pt, style_pt, f0, n_timesteps, 
            temperature, inference_cfg_rate, pytorch_tts.unified_random
        )
        
        print(f"  PyTorch CFM 输出形状: {pytorch_output.shape}")
        
    except Exception as e:
        print(f"❌ PyTorch CFM 调用失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # 准备 MLX 参数
        mu_mx = mlx_inputs['mu']
        x_lens_mx = mlx_inputs['x_lens']
        prompt_mx = mlx_inputs['prompt']
        style_mx = mlx_inputs['style']
        f0_mx = mlx_inputs['f0']
        n_timesteps_mx = mlx_inputs['n_timesteps']
        temperature_mx = mlx_inputs['temperature']
        inference_cfg_rate_mx = mlx_inputs['inference_cfg_rate']
        
        print(f"\n调用 MLX CFM inference...")
        print(f"  输入参数形状:")
        print(f"    mu: {mu_mx.shape}")
        print(f"    x_lens: {x_lens_mx.shape}")
        print(f"    prompt: {prompt_mx.shape}")
        print(f"    style: {style_mx.shape}")
        
        # 调用 MLX CFM
        mlx_output = mlx_cfm.inference(
            mu_mx, x_lens_mx, prompt_mx, style_mx, f0_mx, n_timesteps_mx,
            temperature_mx, inference_cfg_rate_mx, mlx_tts.unified_random
        )
        
        print(f"  MLX CFM 输出形状: {mlx_output.shape}")
        
    except Exception as e:
        print(f"❌ MLX CFM 调用失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_mu_length_issue()
