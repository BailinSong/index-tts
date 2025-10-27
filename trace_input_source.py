#!/usr/bin/env python3
"""
追踪analyze_cfm_differences_fixed.py中实际使用的输入数据来源
"""

import sys
import os
import torch
import numpy as np
import pickle
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from indextts.infer_v2 import IndexTTS2
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch
from unified_random_generator import UnifiedRandomGenerator

def trace_input_source():
    """追踪输入数据来源"""
    print("=== 追踪analyze_cfm_differences_fixed.py中的输入数据来源 ===\n")
    
    # 初始化TTS系统
    print("1. 初始化TTS系统...")
    tts = IndexTTS2()
    
    # 设置统一随机数生成器
    unified_rng = UnifiedRandomGenerator(42)
    
    # 模拟analyze_cfm_differences_fixed.py的推理过程
    print("2. 模拟analyze_cfm_differences_fixed.py的推理过程...")
    
    # 生成测试文本
    text = "今天天气真不错"
    
    # 进行MLX推理
    print("3. 进行MLX推理...")
    try:
        # 使用与analyze_cfm_differences_fixed.py相同的参数
        output_path = "trace_input_test.wav"
        
        # 调用MLX推理
        result = tts.infer(
            text=text,
            spk_audio_prompt="examples/voice_01.wav",
            output_path=output_path,
            use_mlx=True
        )
        
        print(f"   MLX推理完成，输出: {output_path}")
        
    except Exception as e:
        print(f"   MLX推理失败: {e}")
        return
    
    # 检查是否有缓存文件
    print("4. 检查缓存文件...")
    cache_files = [
        "cfm_inputs.pkl",
        "cfm_outputs_mlx.pkl", 
        "cfm_outputs_pytorch.pkl"
    ]
    
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            print(f"   找到缓存文件: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                print(f"   缓存数据键: {list(cached_data.keys())}")
                
                # 分析x数据
                if 'x' in cached_data:
                    x_data = cached_data['x']
                    print(f"   x数据形状: {x_data.shape}")
                    print(f"   x数据范围: [{x_data.min():.6f}, {x_data.max():.6f}]")
                    print(f"   x数据类型: {type(x_data)}")
                    
                    # 检查是否与analyze_cfm_differences_fixed.py输出一致
                    print(f"   与analyze_cfm_differences_fixed.py对比:")
                    print(f"   analyze_cfm_differences_fixed.py MLX x: [-4.462968, 3.760155]")
                    print(f"   当前缓存x: [{x_data.min():.6f}, {x_data.max():.6f}]")
                    
                    if abs(x_data.min() - (-4.462968)) < 0.001 and abs(x_data.max() - 3.760155) < 0.001:
                        print("   ✅ 数据一致！")
                    else:
                        print("   ❌ 数据不一致")
                        
            except Exception as e:
                print(f"   读取缓存文件失败: {e}")
        else:
            print(f"   未找到缓存文件: {cache_file}")
    
    # 检查TTS系统内部状态
    print("5. 检查TTS系统内部状态...")
    if hasattr(tts, 'mlx_s2mel_cfm') and tts.mlx_s2mel_cfm is not None:
        print("   MLX S2MEL CFM已加载")
        
        # 检查CFM的输入数据
        if hasattr(tts.mlx_s2mel_cfm, '_last_inputs'):
            last_inputs = tts.mlx_s2mel_cfm._last_inputs
            print(f"   CFM最后输入: {list(last_inputs.keys())}")
            
            if 'x' in last_inputs:
                x_input = last_inputs['x']
                print(f"   CFM x输入形状: {x_input.shape}")
                print(f"   CFM x输入范围: [{x_input.min():.6f}, {x_input.max():.6f}]")
    else:
        print("   MLX S2MEL CFM未加载")
    
    # 分析可能的输入差异来源
    print("\n6. 分析输入差异来源...")
    print("   从analyze_cfm_differences_fixed.py输出分析:")
    print("   PyTorch x: [-4.465604, 4.479084]")
    print("   MLX x: [-4.462968, 3.760155]")
    print("   差异:")
    print("   - 最小值差异: 0.002636 (很小)")
    print("   - 最大值差异: 0.718929 (较大)")
    print("   可能原因:")
    print("   1. 不同的推理步骤或时间点")
    print("   2. 不同的输入预处理")
    print("   3. 不同的随机数生成时机")
    print("   4. 不同的缓存加载时机")
    
    # 检查推理过程中的中间状态
    print("\n7. 检查推理过程中的中间状态...")
    if hasattr(tts, '_last_inference_data'):
        last_data = tts._last_inference_data
        print(f"   最后推理数据: {list(last_data.keys())}")
    else:
        print("   未找到推理中间数据")
    
    return True

if __name__ == "__main__":
    try:
        success = trace_input_source()
        if success:
            print("\n✅ 输入来源追踪完成")
        else:
            print("\n❌ 输入来源追踪失败")
    except Exception as e:
        print(f"\n❌ 追踪失败: {e}")
        import traceback
        traceback.print_exc()
