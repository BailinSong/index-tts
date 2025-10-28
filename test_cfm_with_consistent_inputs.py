#!/usr/bin/env python3
"""
使用一致的输入数据测试 PyTorch 和 MLX CFM 推理
"""

import os
import pickle
import numpy as np
import torch
import mlx.core as mx
from typing import Dict, Any
import time

def load_consistent_test_data():
    """加载一致的测试数据"""
    test_file = "cfm_consistent_test/consistent_test_data.pkl"
    
    if not os.path.exists(test_file):
        print(f"❌ 测试数据文件不存在: {test_file}")
        print("💡 请先运行 analyze_cfm_consistency.py 生成一致的测试数据")
        return None, None
    
    with open(test_file, 'rb') as f:
        data = pickle.load(f)
    
    pytorch_inputs = data['pytorch_inputs']
    mlx_inputs = data['mlx_inputs']
    
    print("✅ 加载一致的测试数据")
    pytorch_shapes = [f'{k}: {v.shape if hasattr(v, "shape") else v}' for k, v in pytorch_inputs.items()]
    mlx_shapes = [f'{k}: {v.shape if hasattr(v, "shape") else v}' for k, v in mlx_inputs.items()]
    print(f"   PyTorch 输入形状: {pytorch_shapes}")
    print(f"   MLX 输入形状: {mlx_shapes}")
    
    return pytorch_inputs, mlx_inputs

def test_pytorch_cfm(pytorch_inputs):
    """测试 PyTorch CFM"""
    print("\n🔍 测试 PyTorch CFM...")
    
    try:
        # 使用 IndexTTS2 进行 PyTorch CFM 测试
        from indextts.infer_v2 import IndexTTS2
        
        # 创建 IndexTTS2 实例
        tts = IndexTTS2()
        
        # 准备输入数据，确保所有张量在正确的设备上
        device = next(tts.s2mel.models['cfm'].parameters()).device
        mu = pytorch_inputs['mu'].to(device)
        x_lens = pytorch_inputs['x_lens'].to(device)
        prompt = pytorch_inputs['prompt'].to(device)
        style = pytorch_inputs['style'].to(device)
        f0 = pytorch_inputs['f0']
        n_timesteps = pytorch_inputs['n_timesteps']
        temperature = pytorch_inputs['temperature']
        inference_cfg_rate = pytorch_inputs['inference_cfg_rate']
        
        print(f"   📊 输入数据:")
        print(f"      mu: {mu.shape}")
        print(f"      x_lens: {x_lens}")
        print(f"      prompt: {prompt.shape}")
        print(f"      style: {style.shape}")
        print(f"      n_timesteps: {n_timesteps}")
        print(f"      temperature: {temperature}")
        print(f"      inference_cfg_rate: {inference_cfg_rate}")
        
        # 执行 CFM 推理
        with torch.no_grad():
            start_time = time.time()
            output = tts.s2mel.models['cfm'].inference(
                mu=mu,
                x_lens=x_lens,
                prompt=prompt,
                style=style,
                f0=f0,
                n_timesteps=n_timesteps,
                temperature=temperature,
                inference_cfg_rate=inference_cfg_rate
            )
            end_time = time.time()
        
        print(f"   ✅ PyTorch CFM 推理完成")
        print(f"   📊 输出形状: {output.shape}")
        print(f"   ⏱️  推理时间: {end_time - start_time:.3f} 秒")
        print(f"   📈 输出统计: min={output.min().item():.6f}, max={output.max().item():.6f}, mean={output.mean().item():.6f}")
        
        return output
        
    except Exception as e:
        print(f"   ❌ PyTorch CFM 推理失败: {e}")
        return None

def test_mlx_cfm(mlx_inputs):
    """测试 MLX CFM"""
    print("\n🔍 测试 MLX CFM...")
    
    try:
        # 使用 IndexTTS2 进行 MLX CFM 测试
        from indextts.infer_v2 import IndexTTS2
        
        # 创建 IndexTTS2 实例，指定使用 MLX
        tts = IndexTTS2(use_mlx=True)
        
        # 准备输入数据
        mu = mlx_inputs['mu']
        x_lens = mlx_inputs['x_lens']
        prompt = mlx_inputs['prompt']
        style = mlx_inputs['style']
        f0 = mlx_inputs['f0']
        n_timesteps = mlx_inputs['n_timesteps']
        temperature = mlx_inputs['temperature']
        inference_cfg_rate = mlx_inputs['inference_cfg_rate']
        
        print(f"   📊 输入数据:")
        print(f"      mu: {mu.shape}")
        print(f"      x_lens: {x_lens}")
        print(f"      prompt: {prompt.shape}")
        print(f"      style: {style.shape}")
        print(f"      n_timesteps: {n_timesteps}")
        print(f"      temperature: {temperature}")
        print(f"      inference_cfg_rate: {inference_cfg_rate}")
        
        # 执行 MLX CFM 推理
        start_time = time.time()
        output = tts.s2mel.models['cfm'].inference(
            mu=mu,
            x_lens=x_lens,
            prompt=prompt,
            style=style,
            f0=f0,
            n_timesteps=n_timesteps,
            temperature=temperature,
            inference_cfg_rate=inference_cfg_rate
        )
        end_time = time.time()
        
        print(f"   ✅ MLX CFM 推理完成")
        print(f"   📊 输出形状: {output.shape}")
        print(f"   ⏱️  推理时间: {end_time - start_time:.3f} 秒")
        print(f"   📈 输出统计: min={mx.min(output):.6f}, max={mx.max(output):.6f}, mean={mx.mean(output):.6f}")
        
        return output
        
    except Exception as e:
        print(f"   ❌ MLX CFM 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_outputs(pytorch_output, mlx_output):
    """比较 PyTorch 和 MLX 输出"""
    print("\n🔍 输出比较:")
    
    if pytorch_output is None or mlx_output is None:
        print("   ❌ 无法比较，因为某个输出为 None")
        return
    
    # 转换为 numpy 进行比较
    pytorch_np = pytorch_output.detach().cpu().numpy()
    mlx_np = np.array(mlx_output)
    
    print(f"   PyTorch 输出形状: {pytorch_np.shape}")
    print(f"   MLX 输出形状: {mlx_np.shape}")
    
    if pytorch_np.shape != mlx_np.shape:
        print("   ❌ 输出形状不匹配")
        return
    
    # 计算差异
    diff = np.abs(pytorch_np - mlx_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"   📊 差异统计:")
    print(f"      最大差异: {max_diff:.8f}")
    print(f"      平均差异: {mean_diff:.8f}")
    
    # 判断一致性
    tolerance = 1e-6
    if max_diff < tolerance:
        print(f"   ✅ 输出完全一致 (差异 < {tolerance})")
    else:
        print(f"   ⚠️  输出存在差异 (差异 >= {tolerance})")
        
        # 分析差异分布
        diff_percentiles = np.percentile(diff, [50, 90, 95, 99, 99.9])
        print(f"   📈 差异分布:")
        print(f"      50%: {diff_percentiles[0]:.8f}")
        print(f"      90%: {diff_percentiles[1]:.8f}")
        print(f"      95%: {diff_percentiles[2]:.8f}")
        print(f"      99%: {diff_percentiles[3]:.8f}")
        print(f"      99.9%: {diff_percentiles[4]:.8f}")

def main():
    """主函数"""
    print("🔧 CFM 一致性测试工具")
    print("=" * 50)
    
    # 加载一致的测试数据
    pytorch_inputs, mlx_inputs = load_consistent_test_data()
    if pytorch_inputs is None or mlx_inputs is None:
        return
    
    # 测试 PyTorch CFM
    pytorch_output = test_pytorch_cfm(pytorch_inputs)
    
    # 测试 MLX CFM
    mlx_output = test_mlx_cfm(mlx_inputs)
    
    # 比较输出
    compare_outputs(pytorch_output, mlx_output)
    
    print("\n🎉 测试完成!")

if __name__ == "__main__":
    main()
