#!/usr/bin/env python3
"""
CFM 输出全零问题调试

分析为什么 CFM 输出全为零，即使输入数据正常
"""

import pickle
import numpy as np
import torch
import mlx.core as mx
from pathlib import Path
import sys
import os

# 添加项目路径
sys.path.append('/Users/bailin/index-tts')


def debug_cfm_zero_output():
    """调试 CFM 输出全零问题"""
    print("🔍 Debugging CFM Zero Output Issue")
    print("="*60)
    
    # 加载 CFM 前级缓存数据
    cfm_cache_file = Path("cfm_debug_outputs/cfm_input_cache.pkl")
    
    if not cfm_cache_file.exists():
        print("❌ CFM input cache file not found!")
        return
    
    with open(cfm_cache_file, 'rb') as f:
        cfm_cache = pickle.load(f)
    
    print(f"✅ CFM input cache loaded from: {cfm_cache_file}")
    
    # 分析输入数据
    print("\n📊 Input Data Analysis:")
    analyze_input_data(cfm_cache)
    
    # 分析 CFM 处理逻辑
    print("\n🔍 CFM Processing Logic Analysis:")
    analyze_cfm_processing_logic(cfm_cache)
    
    # 手动执行 CFM 步骤
    print("\n🔍 Manual CFM Step Execution:")
    manual_cfm_execution(cfm_cache)
    
    print("\n🎉 CFM Zero Output Debug completed!")


def analyze_input_data(cfm_cache):
    """分析输入数据"""
    original_inputs = cfm_cache['original_inputs']
    
    print("   📋 Original Inputs:")
    for key, value in original_inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"     {key}: {value.shape}, range: [{value.min():.6f}, {value.max():.6f}]")
        elif isinstance(value, np.ndarray):
            print(f"     {key}: {value.shape}, range: [{value.min():.6f}, {value.max():.6f}]")
        else:
            print(f"     {key}: {value}")
    
    # 检查关键输入
    x = original_inputs['x']
    prompt_x = original_inputs['prompt_x']
    x_lens = original_inputs['x_lens']
    
    print(f"\n   🔍 Key Input Analysis:")
    print(f"     x (noise): shape={x.shape}, range=[{x.min():.6f}, {x.max():.6f}]")
    print(f"     prompt_x: shape={prompt_x.shape}, range=[{prompt_x.min():.6f}, {prompt_x.max():.6f}]")
    print(f"     x_lens: {x_lens}")
    
    # 检查 prompt 处理
    prompt_len = original_inputs['prompt_len']
    print(f"     prompt_len: {prompt_len}")
    
    if prompt_len == x.shape[-1]:
        print(f"     ⚠️ Problem: prompt_len ({prompt_len}) == x.shape[-1] ({x.shape[-1]})")
        print(f"     💡 This means the entire x will be set to zero!")


def analyze_cfm_processing_logic(cfm_cache):
    """分析 CFM 处理逻辑"""
    print("   📋 CFM Processing Logic:")
    
    original_inputs = cfm_cache['original_inputs']
    x = original_inputs['x']
    prompt_x = original_inputs['prompt_x']
    prompt_len = original_inputs['prompt_len']
    
    print(f"     1. Original x: range=[{x.min():.6f}, {x.max():.6f}]")
    print(f"     2. Original prompt_x: range=[{prompt_x.min():.6f}, {prompt_x.max():.6f}]")
    print(f"     3. prompt_len: {prompt_len}")
    
    # 模拟 CFM 的 prompt 处理
    print(f"\n     🔍 CFM Prompt Processing Simulation:")
    
    # CFM 的处理逻辑：
    # x[..., :prompt_len] = 0  # 将 prompt 区域设为 0
    # prompt_x[..., :prompt_len] = prompt[..., :prompt_len]  # 设置 prompt
    
    x_processed = x.clone()
    x_processed[..., :prompt_len] = 0
    
    prompt_x_processed = torch.zeros_like(x)
    prompt_x_processed[..., :prompt_len] = prompt_x[..., :prompt_len]
    
    print(f"     After CFM processing:")
    print(f"       x_processed: range=[{x_processed.min():.6f}, {x_processed.max():.6f}]")
    print(f"       prompt_x_processed: range=[{prompt_x_processed.min():.6f}, {prompt_x_processed.max():.6f}]")
    
    # 检查是否全为零
    if x_processed.min() == 0.0 and x_processed.max() == 0.0:
        print(f"     ❌ Problem: x_processed is all zeros!")
        print(f"     💡 This explains why CFM output is all zeros")
    
    if prompt_x_processed.min() == 0.0 and prompt_x_processed.max() == 0.0:
        print(f"     ❌ Problem: prompt_x_processed is all zeros!")
        print(f"     💡 This means no prompt information is preserved")


def manual_cfm_execution(cfm_cache):
    """手动执行 CFM 步骤"""
    try:
        # 导入 PyTorch CFM
        from indextts.s2mel.modules.flow_matching import CFM
        
        # 使用缓存中的配置
        config_dict = cfm_cache['config']
        
        # 创建 SimpleConfig 类
        class SimpleConfig:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    if isinstance(value, dict):
                        setattr(self, key, SimpleConfig(value))
                    else:
                        setattr(self, key, value)
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        config = SimpleConfig(config_dict)
        
        # 初始化 PyTorch CFM
        pytorch_cfm = CFM(config)
        pytorch_cfm.eval()
        
        # 初始化 caches
        original_inputs = cfm_cache['original_inputs']
        batch_size = original_inputs['batch_size']
        seq_len = original_inputs['seq_len']
        pytorch_cfm.estimator.setup_caches(max_batch_size=batch_size, max_seq_length=seq_len)
        
        # 使用缓存中的输入数据
        x = original_inputs['x']
        prompt_x = original_inputs['prompt_x']
        x_lens = original_inputs['x_lens']
        style = original_inputs['style']
        mu = original_inputs['mu']
        t_span = cfm_cache['t_span']
        inference_cfg_rate = cfm_cache['inference_cfg_rate']
        
        print("   📋 Manual CFM Execution:")
        print(f"     Input shapes: x={x.shape}, prompt_x={prompt_x.shape}, x_lens={x_lens.shape}")
        print(f"     Input shapes: style={style.shape}, mu={mu.shape}")
        
        # 检查 CFM 的 solve_euler 方法
        print(f"\n     🔍 CFM solve_euler Method Analysis:")
        
        # 手动执行 CFM 的关键步骤
        print(f"     1. CFM 内部处理:")
        print(f"        - 应用 prompt: x[..., :prompt_len] = 0")
        print(f"        - 设置 prompt: prompt_x[..., :prompt_len] = prompt[..., :prompt_len]")
        
        # 检查时间跨度
        print(f"     2. Time span: {t_span}")
        print(f"        - 时间步数: {len(t_span) - 1}")
        print(f"        - 时间范围: [{t_span[0]:.6f}, {t_span[-1]:.6f}]")
        
        # 检查 CFG 设置
        print(f"     3. CFG settings:")
        print(f"        - inference_cfg_rate: {inference_cfg_rate}")
        print(f"        - CFG disabled: {inference_cfg_rate == 0.0}")
        
        # 运行 CFM 推理
        print(f"\n     🔍 Running CFM Inference:")
        
        with torch.no_grad():
            # 启用调试模式
            pytorch_cfm._debug_layers = True
            
            # 运行 CFM 推理
            output = pytorch_cfm.solve_euler(
                x=x,
                x_lens=x_lens,
                prompt=prompt_x,
                mu=mu,
                style=style,
                f0=None,
                t_span=t_span,
                inference_cfg_rate=inference_cfg_rate,
                debug_layers=True
            )
            
            print(f"     CFM output shape: {output.shape}")
            print(f"     CFM output range: [{output.min():.6f}, {output.max():.6f}]")
            
            # 检查输出
            if output.min() == 0.0 and output.max() == 0.0:
                print(f"     ❌ Problem: CFM output is all zeros!")
                print(f"     💡 This indicates a fundamental issue with CFM processing")
            else:
                print(f"     ✅ CFM output has non-zero values")
        
        # 分析可能的原因
        print(f"\n     🔍 Possible Causes Analysis:")
        print(f"     1. Prompt length issue:")
        print(f"        - prompt_len: {original_inputs['prompt_len']}")
        print(f"        - x.shape[-1]: {x.shape[-1]}")
        print(f"        - If prompt_len == x.shape[-1], entire x becomes zero")
        
        print(f"     2. CFM flow matching issue:")
        print(f"        - CFM starts with x and flows to target")
        print(f"        - If x is all zeros, flow result will be zeros")
        
        print(f"     3. Euler integration issue:")
        print(f"        - Time steps: {len(t_span) - 1}")
        print(f"        - If time steps are too few, integration may fail")
        
        print(f"     4. Estimator issue:")
        print(f"        - DiT estimator may not be working correctly")
        print(f"        - Check if estimator output is meaningful")
        
    except Exception as e:
        print(f"     ❌ Manual CFM execution failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("🔍 CFM Zero Output Debug")
    print("="*60)
    
    try:
        # 调试 CFM 输出全零问题
        debug_cfm_zero_output()
        
        print("\n🎉 CFM Zero Output Debug completed!")
        print("\n📋 Key Findings:")
        print("1. Input data is normal (not all zeros)")
        print("2. CFM processing logic may have issues")
        print("3. Prompt length may be causing problems")
        print("4. CFM flow matching may not be working correctly")
        
        print("\n💡 Recommendations:")
        print("1. Check prompt length vs sequence length")
        print("2. Verify CFM flow matching logic")
        print("3. Test with different input data")
        print("4. Check DiT estimator functionality")
        
    except Exception as e:
        print(f"\n❌ Error during debug: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


