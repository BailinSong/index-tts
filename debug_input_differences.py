#!/usr/bin/env python3
"""
分析PyTorch和MLX版本输入差异的来源
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

def analyze_input_differences():
    """分析输入差异的来源"""
    print("=== 分析PyTorch和MLX版本输入差异来源 ===\n")
    
    # 初始化TTS系统
    print("1. 初始化TTS系统...")
    tts = IndexTTS2()
    
    # 设置统一随机数生成器
    unified_rng = UnifiedRandomGenerator(42)
    
    # 生成测试输入
    print("2. 生成测试输入...")
    batch_size = 1
    seq_len = 80
    cond_len = 502
    
    # 生成随机输入
    x = unified_rng.generate_noise((batch_size, seq_len, cond_len))
    prompt_x = unified_rng.generate_noise((batch_size, seq_len, cond_len)) * 2 - 5
    t = unified_rng.generate_noise((batch_size,)) * 0.5 + 0.5  # 0-1范围
    style = unified_rng.generate_noise((batch_size, 192))
    cond = unified_rng.generate_noise((batch_size, cond_len, 512)) * 0.1
    
    print(f"   原始输入形状:")
    print(f"   x: {x.shape}")
    print(f"   prompt_x: {prompt_x.shape}")
    print(f"   t: {t.shape}")
    print(f"   style: {style.shape}")
    print(f"   cond: {cond.shape}")
    
    # 转换为PyTorch张量
    print("\n3. 转换为PyTorch张量...")
    x_torch = torch.tensor(x, dtype=torch.float32, device='mps')
    prompt_x_torch = torch.tensor(prompt_x, dtype=torch.float32, device='mps')
    t_torch = torch.tensor(t, dtype=torch.float32, device='mps')
    style_torch = torch.tensor(style, dtype=torch.float32, device='mps')
    cond_torch = torch.tensor(cond, dtype=torch.float32, device='mps')
    
    print(f"   PyTorch张量范围:")
    print(f"   x_torch: [{x_torch.min():.6f}, {x_torch.max():.6f}]")
    print(f"   prompt_x_torch: [{prompt_x_torch.min():.6f}, {prompt_x_torch.max():.6f}]")
    print(f"   t_torch: [{t_torch.min():.6f}, {t_torch.max():.6f}]")
    print(f"   style_torch: [{style_torch.min():.6f}, {style_torch.max():.6f}]")
    print(f"   cond_torch: [{cond_torch.min():.6f}, {cond_torch.max():.6f}]")
    
    # 转换为MLX数组
    print("\n4. 转换为MLX数组...")
    x_mlx = torch_to_mlx(x_torch)
    prompt_x_mlx = torch_to_mlx(prompt_x_torch)
    t_mlx = torch_to_mlx(t_torch)
    style_mlx = torch_to_mlx(style_torch)
    cond_mlx = torch_to_mlx(cond_torch)
    
    print(f"   MLX数组范围:")
    print(f"   x_mlx: [{x_mlx.min():.6f}, {x_mlx.max():.6f}]")
    print(f"   prompt_x_mlx: [{prompt_x_mlx.min():.6f}, {prompt_x_mlx.max():.6f}]")
    print(f"   t_mlx: [{t_mlx.min():.6f}, {t_mlx.max():.6f}]")
    print(f"   style_mlx: [{style_mlx.min():.6f}, {style_mlx.max():.6f}]")
    print(f"   cond_mlx: [{cond_mlx.min():.6f}, {cond_mlx.max():.6f}]")
    
    # 分析转换差异
    print("\n5. 分析转换差异...")
    x_diff = np.abs(x_torch.cpu().numpy() - mlx_to_torch(x_mlx).cpu().numpy())
    prompt_x_diff = np.abs(prompt_x_torch.cpu().numpy() - mlx_to_torch(prompt_x_mlx).cpu().numpy())
    t_diff = np.abs(t_torch.cpu().numpy() - mlx_to_torch(t_mlx).cpu().numpy())
    style_diff = np.abs(style_torch.cpu().numpy() - mlx_to_torch(style_mlx).cpu().numpy())
    cond_diff = np.abs(cond_torch.cpu().numpy() - mlx_to_torch(cond_mlx).cpu().numpy())
    
    print(f"   转换差异:")
    print(f"   x_diff: 最大 {x_diff.max():.10f}, 平均 {x_diff.mean():.10f}")
    print(f"   prompt_x_diff: 最大 {prompt_x_diff.max():.10f}, 平均 {prompt_x_diff.mean():.10f}")
    print(f"   t_diff: 最大 {t_diff.max():.10f}, 平均 {t_diff.mean():.10f}")
    print(f"   style_diff: 最大 {style_diff.max():.10f}, 平均 {style_diff.mean():.10f}")
    print(f"   cond_diff: 最大 {cond_diff.max():.10f}, 平均 {cond_diff.mean():.10f}")
    
    # 检查是否有缓存输入数据
    print("\n6. 检查缓存输入数据...")
    cache_file = "cfm_inputs.pkl"
    if os.path.exists(cache_file):
        print(f"   找到缓存文件: {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        
        print(f"   缓存数据键: {list(cached_data.keys())}")
        
        # 比较缓存数据与当前输入
        if 'x' in cached_data:
            cached_x = cached_data['x']
            print(f"   缓存x形状: {cached_x.shape}")
            print(f"   缓存x范围: [{cached_x.min():.6f}, {cached_x.max():.6f}]")
            
            # 计算与当前输入的差异
            if cached_x.shape == x.shape:
                current_x_diff = np.abs(cached_x - x)
                print(f"   当前输入与缓存x差异: 最大 {current_x_diff.max():.10f}, 平均 {current_x_diff.mean():.10f}")
            else:
                print(f"   形状不匹配: 缓存 {cached_x.shape} vs 当前 {x.shape}")
    else:
        print(f"   未找到缓存文件: {cache_file}")
    
    # 分析analyze_cfm_differences_fixed.py中的输入差异
    print("\n7. 分析analyze_cfm_differences_fixed.py中的输入差异...")
    print("   从analyze_cfm_differences_fixed.py输出中看到的差异:")
    print("   PyTorch x: [-4.465604, 4.479084]")
    print("   MLX x: [-4.462968, 3.760155]")
    print("   差异分析:")
    print("   - 最小值差异: -4.465604 vs -4.462968 = 0.002636")
    print("   - 最大值差异: 4.479084 vs 3.760155 = 0.718929")
    print("   - 这表明MLX版本的输入范围被压缩了")
    
    # 检查是否来自不同的推理步骤
    print("\n8. 检查推理步骤差异...")
    print("   可能的原因:")
    print("   1. 不同的随机数生成")
    print("   2. 不同的输入预处理")
    print("   3. 不同的缓存加载")
    print("   4. 不同的推理步骤")
    
    return {
        'x_torch': x_torch,
        'x_mlx': x_mlx,
        'x_diff': x_diff,
        'prompt_x_torch': prompt_x_torch,
        'prompt_x_mlx': prompt_x_mlx,
        'prompt_x_diff': prompt_x_diff
    }

if __name__ == "__main__":
    try:
        results = analyze_input_differences()
        print("\n✅ 输入差异分析完成")
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
