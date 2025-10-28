#!/usr/bin/env python3
"""
CFM 输入数据缓存基准测试
在 PyTorch 版本数据输入 CFM 前将其缓存，用于验证 MLX 版本的一致性
"""

import os
import sys
import pickle
import time
import statistics
import numpy as np
import torch
import mlx.core as mx
from typing import Dict, Any, Optional

# 添加项目路径
sys.path.insert(0, '.')

from indextts.infer_v2 import IndexTTS2
from cfm_data_cache import CFMDataCache

# 全局缓存管理器
cfm_cache = CFMDataCache("cfm_baseline_cache")

def patch_pytorch_cfm_for_caching():
    """
    为 PyTorch CFM 添加输入缓存功能
    """
    from indextts.s2mel.modules.flow_matching import BASECFM
    
    # 保存原始 inference 方法
    original_inference = BASECFM.inference
    
    def cached_inference(self, mu, x_lens, prompt, style, f0, n_timesteps, 
                        temperature=1.0, inference_cfg_rate=0.5, unified_random=None):
        """
        带缓存的 PyTorch CFM inference
        """
        # 缓存输入数据
        cache_data = {
            'mu': mu,
            'x_lens': x_lens, 
            'prompt': prompt,
            'style': style,
            'f0': f0,
            'n_timesteps': n_timesteps,
            'temperature': temperature,
            'inference_cfg_rate': inference_cfg_rate,
            'unified_random_seed': unified_random.seed if unified_random else None,
            'timestamp': time.time(),
            'model_type': 'pytorch'
        }
        
        # 保存到缓存
        filename = f"cfm_pytorch_inputs_{int(time.time() * 1000)}.pkl"
        cfm_cache.save_torch_inputs(cache_data, filename)
        
        print(f"🔍 Cached PyTorch CFM inputs: {filename}")
        print(f"   mu: {mu.shape}, prompt: {prompt.shape}, style: {style.shape}")
        
        # 调用原始 inference 方法
        result = original_inference(
            self, mu, x_lens, prompt, style, f0, n_timesteps,
            temperature, inference_cfg_rate, unified_random
        )
        
        # 缓存输出结果
        output_cache = {
            'output': result,
            'timestamp': time.time(),
            'model_type': 'pytorch'
        }
        
        output_filename = f"cfm_pytorch_output_{int(time.time() * 1000)}.pkl"
        cfm_cache.save_torch_inputs(output_cache, output_filename)
        
        print(f"🔍 Cached PyTorch CFM output: {output_filename}")
        print(f"   output: {result.shape}")
        
        return result
    
    # 应用补丁
    BASECFM.inference = cached_inference
    print("✅ PyTorch CFM patched for input/output caching")


def patch_mlx_cfm_for_caching():
    """
    为 MLX CFM 添加输入缓存功能
    """
    from indextts.s2mel.modules.mlx_flow_matching import MLXCFM
    
    # 保存原始 inference 方法
    original_inference = MLXCFM.inference
    
    def cached_inference(self, mu, x_lens, prompt, style, f0, n_timesteps,
                        temperature=1.0, inference_cfg_rate=0.5, unified_random=None):
        """
        带缓存的 MLX CFM inference
        """
        # 缓存输入数据
        cache_data = {
            'mu': mu,
            'x_lens': x_lens,
            'prompt': prompt, 
            'style': style,
            'f0': f0,
            'n_timesteps': n_timesteps,
            'temperature': temperature,
            'inference_cfg_rate': inference_cfg_rate,
            'unified_random_seed': unified_random.seed if unified_random else None,
            'timestamp': time.time(),
            'model_type': 'mlx'
        }
        
        # 保存到缓存
        filename = f"cfm_mlx_inputs_{int(time.time() * 1000)}.pkl"
        cfm_cache.save_mlx_inputs(cache_data, filename)
        
        print(f"🔍 Cached MLX CFM inputs: {filename}")
        print(f"   mu: {mu.shape}, prompt: {prompt.shape}, style: {style.shape}")
        
        # 调用原始 inference 方法
        result = original_inference(
            self, mu, x_lens, prompt, style, f0, n_timesteps,
            temperature, inference_cfg_rate, unified_random
        )
        
        # 缓存输出结果
        output_cache = {
            'output': result,
            'timestamp': time.time(),
            'model_type': 'mlx'
        }
        
        output_filename = f"cfm_mlx_output_{int(time.time() * 1000)}.pkl"
        cfm_cache.save_mlx_inputs(output_cache, output_filename)
        
        print(f"🔍 Cached MLX CFM output: {output_filename}")
        print(f"   output: {result.shape}")
        
        return result
    
    # 应用补丁
    MLXCFM.inference = cached_inference
    print("✅ MLX CFM patched for input/output caching")


def run_cfm_caching_benchmark():
    """
    运行 CFM 缓存基准测试
    """
    print("=" * 80)
    print("CFM 输入数据缓存基准测试")
    print("=" * 80)
    
    # 配置
    CONFIG = {
        "texts": [
            "今天天气真不错",
            "到底应该吃什么", 
            "你为什么不愿意",
        ],
        "voice": "examples/zh_vo_Main_Linaxita_2_4_24_6.wav",
        "seed": 42,
        "num_runs": 3,
    }
    
    print(f"\n测试配置:")
    print(f"  文本: {CONFIG['texts']}")
    print(f"  Voice: {CONFIG['voice']}")
    print(f"  Seed: {CONFIG['seed']} (固定)")
    print(f"  运行次数: {CONFIG['num_runs']}")
    
    # 检查语音文件
    if not os.path.exists(CONFIG['voice']):
        print(f"❌ 语音文件不存在: {CONFIG['voice']}")
        return
    
    # 应用补丁
    print(f"\n应用 CFM 缓存补丁...")
    patch_pytorch_cfm_for_caching()
    patch_mlx_cfm_for_caching()
    
    # 运行 PyTorch 基准测试
    print(f"\n" + "="*50)
    print("PyTorch CFM 缓存测试")
    print("="*50)
    
    pytorch_results = []
    pytorch_tts = IndexTTS2(use_mlx=False)
    
    for i in range(CONFIG['num_runs']):
        run_num = i + 1
        text = CONFIG['texts'][i]
        
        print(f"\n[PyTorch] Run {run_num}: {text}")
        
        start_time = time.perf_counter()
        
        try:
            pytorch_tts.infer(
                spk_audio_prompt=CONFIG['voice'],
                text=text,
                output_path=f"outputs/pytorch_cfm_cache_{run_num}.wav",
                seed=CONFIG['seed']
            )
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            pytorch_results.append(duration)
            
            print(f"✅ PyTorch Run {run_num} completed: {duration:.2f}s")
            
        except Exception as e:
            print(f"❌ PyTorch Run {run_num} failed: {e}")
            continue
    
    # 运行 MLX 基准测试
    print(f"\n" + "="*50)
    print("MLX CFM 缓存测试")
    print("="*50)
    
    mlx_results = []
    mlx_tts = IndexTTS2(use_mlx=True)
    
    for i in range(CONFIG['num_runs']):
        run_num = i + 1
        text = CONFIG['texts'][i]
        
        print(f"\n[MLX] Run {run_num}: {text}")
        
        start_time = time.perf_counter()
        
        try:
            mlx_tts.infer(
                spk_audio_prompt=CONFIG['voice'],
                text=text,
                output_path=f"outputs/mlx_cfm_cache_{run_num}.wav",
                seed=CONFIG['seed']
            )
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            mlx_results.append(duration)
            
            print(f"✅ MLX Run {run_num} completed: {duration:.2f}s")
            
        except Exception as e:
            print(f"❌ MLX Run {run_num} failed: {e}")
            continue
    
    # 显示结果
    print(f"\n" + "="*80)
    print("CFM 缓存基准测试结果")
    print("="*80)
    
    if pytorch_results:
        pytorch_avg = statistics.mean(pytorch_results)
        pytorch_std = statistics.stdev(pytorch_results) if len(pytorch_results) > 1 else 0
        print(f"\nPyTorch CFM 缓存测试:")
        print(f"  成功运行: {len(pytorch_results)}/{CONFIG['num_runs']}")
        print(f"  平均时间: {pytorch_avg:.2f}s")
        print(f"  标准差:   {pytorch_std:.2f}s")
        print(f"  详细结果: {[f'{r:.2f}s' for r in pytorch_results]}")
    
    if mlx_results:
        mlx_avg = statistics.mean(mlx_results)
        mlx_std = statistics.stdev(mlx_results) if len(mlx_results) > 1 else 0
        print(f"\nMLX CFM 缓存测试:")
        print(f"  成功运行: {len(mlx_results)}/{CONFIG['num_runs']}")
        print(f"  平均时间: {mlx_avg:.2f}s")
        print(f"  标准差:   {mlx_std:.2f}s")
        print(f"  详细结果: {[f'{r:.2f}s' for r in mlx_results]}")
    
    # 显示缓存文件
    print(f"\n📁 CFM 缓存文件:")
    cfm_cache.list_cache_files()
    
    # 性能比较
    if pytorch_results and mlx_results:
        speedup = pytorch_avg / mlx_avg
        print(f"\n🚀 性能比较:")
        print(f"  MLX 相对 PyTorch 加速: {speedup:.2f}x")
        if speedup > 1:
            print(f"  MLX 比 PyTorch 快: {((speedup-1)*100):.1f}%")
        else:
            print(f"  PyTorch 比 MLX 快: {((1/speedup-1)*100):.1f}%")


def analyze_cached_data():
    """
    分析缓存的 CFM 数据
    """
    print(f"\n" + "="*50)
    print("CFM 缓存数据分析")
    print("="*50)
    
    # 列出缓存文件
    files = cfm_cache.list_cache_files()
    
    pytorch_input_files = [f for f in files if 'pytorch_inputs' in f]
    mlx_input_files = [f for f in files if 'mlx_inputs' in f]
    pytorch_output_files = [f for f in files if 'pytorch_output' in f]
    mlx_output_files = [f for f in files if 'mlx_output' in f]
    
    print(f"\n📊 缓存文件统计:")
    print(f"  PyTorch 输入文件: {len(pytorch_input_files)}")
    print(f"  MLX 输入文件:     {len(mlx_input_files)}")
    print(f"  PyTorch 输出文件: {len(pytorch_output_files)}")
    print(f"  MLX 输出文件:     {len(mlx_output_files)}")
    
    # 分析输入数据
    if pytorch_input_files and mlx_input_files:
        print(f"\n🔍 输入数据分析:")
        
        # 加载最新的 PyTorch 输入
        latest_pytorch_input = pytorch_input_files[-1]
        pytorch_data = cfm_cache.load_torch_inputs(latest_pytorch_input)
        
        # 加载最新的 MLX 输入
        latest_mlx_input = mlx_input_files[-1]
        mlx_data = cfm_cache.load_mlx_inputs(latest_mlx_input)
        
        print(f"  PyTorch 输入形状:")
        for key, value in pytorch_data.items():
            if isinstance(value, torch.Tensor):
                print(f"    {key}: {value.shape}")
        
        print(f"  MLX 输入形状:")
        for key, value in mlx_data.items():
            if isinstance(value, mx.array):
                print(f"    {key}: {value.shape}")
    
    # 分析输出数据
    if pytorch_output_files and mlx_output_files:
        print(f"\n🔍 输出数据分析:")
        
        # 加载最新的 PyTorch 输出
        latest_pytorch_output = pytorch_output_files[-1]
        pytorch_output_data = cfm_cache.load_torch_inputs(latest_pytorch_output)
        
        # 加载最新的 MLX 输出
        latest_mlx_output = mlx_output_files[-1]
        mlx_output_data = cfm_cache.load_mlx_inputs(latest_mlx_output)
        
        if 'output' in pytorch_output_data and 'output' in mlx_output_data:
            pytorch_output = pytorch_output_data['output']
            mlx_output = mlx_output_data['output']
            
            # 比较输出
            is_consistent = cfm_cache.compare_outputs(
                pytorch_output, mlx_output, tolerance=1e-6, name="CFM Output"
            )
            
            if is_consistent:
                print("✅ CFM 输出数据一致!")
            else:
                print("❌ CFM 输出数据不一致!")
                print("🔧 建议检查:")
                print("   - 随机种子设置")
                print("   - 权重加载")
                print("   - 数值精度")


def main():
    """主函数"""
    print("🔧 CFM 输入数据缓存基准测试")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs("outputs", exist_ok=True)
    
    # 运行基准测试
    run_cfm_caching_benchmark()
    
    # 分析缓存数据
    analyze_cached_data()
    
    print(f"\n🎉 CFM 缓存基准测试完成!")
    print(f"📁 缓存文件保存在: cfm_baseline_cache/")
    print(f"🎵 输出音频保存在: outputs/")


if __name__ == "__main__":
    main()
