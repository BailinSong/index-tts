#!/usr/bin/env python3
"""
DiT 前向传播逐步分析工具
逐步比较 PyTorch 和 MLX DiT 的前向传播过程
"""

import os
import pickle
import torch
import mlx.core as mx
import numpy as np
from typing import Dict, Any

def load_dit_input_data():
    """加载 DiT 输入数据"""
    import glob
    
    # 找到最新的 DiT 输入缓存
    pytorch_files = glob.glob("cfm_production_cache/cfm_pytorch_step_*_dit_input_*.pkl")
    mlx_files = glob.glob("cfm_production_cache/cfm_mlx_step_*_dit_input_*.pkl")
    
    if not pytorch_files or not mlx_files:
        print("❌ 未找到 DiT 输入缓存文件")
        return None, None
        
    # 加载最新的文件
    pytorch_file = max(pytorch_files, key=os.path.getmtime)
    mlx_file = max(mlx_files, key=os.path.getmtime)
    
    with open(pytorch_file, 'rb') as f:
        pytorch_data = pickle.load(f)
    with open(mlx_file, 'rb') as f:
        mlx_data = pickle.load(f)
        
    return pytorch_data, mlx_data

def to_numpy(tensor):
    """转换为 numpy 数组"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, mx.array):
        return np.array(tensor)
    else:
        return np.array(tensor)

def compare_tensors(name, pt_tensor, mlx_tensor, tolerance=1e-6):
    """比较两个张量"""
    pt_np = to_numpy(pt_tensor)
    mlx_np = to_numpy(mlx_tensor)
    
    print(f"   {name}:")
    print(f"     PyTorch: {pt_np.shape}, min={pt_np.min():.6f}, max={pt_np.max():.6f}, mean={pt_np.mean():.6f}")
    print(f"     MLX:     {mlx_np.shape}, min={mlx_np.min():.6f}, max={mlx_np.max():.6f}, mean={mlx_np.mean():.6f}")
    
    if pt_np.shape == mlx_np.shape:
        diff = np.abs(pt_np - mlx_np)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"     差异: max={max_diff:.6f}, mean={mean_diff:.6f}")
        
        if max_diff < tolerance:
            print(f"     ✅ 一致")
            return True
        else:
            print(f"     ❌ 不一致")
            return False
    else:
        print(f"     ❌ 形状不匹配: {pt_np.shape} vs {mlx_np.shape}")
        return False

def analyze_timestep_embedding():
    """分析时间步嵌入"""
    print(f"\n{'='*60}")
    print(f"🔍 时间步嵌入分析")
    print(f"{'='*60}")
    
    # 加载输入数据
    pt_data, mlx_data = load_dit_input_data()
    if pt_data is None or mlx_data is None:
        return
    
    # 提取时间步
    pt_t = pt_data['t']
    mlx_t = mlx_data['t']
    
    print(f"输入时间步:")
    compare_tensors("t", pt_t, mlx_t)
    
    # 模拟时间步嵌入计算
    print(f"\n时间步嵌入计算:")
    
    # PyTorch 时间步嵌入
    if isinstance(pt_t, torch.Tensor):
        pt_t_scalar = pt_t[0].item() if pt_t.numel() > 1 else pt_t.item()
    else:
        pt_t_scalar = float(pt_t[0]) if len(pt_t) > 1 else float(pt_t)
    
    # MLX 时间步嵌入
    if isinstance(mlx_t, mx.array):
        mlx_t_scalar = float(mlx_t[0]) if mlx_t.size > 1 else float(mlx_t)
    else:
        mlx_t_scalar = float(mlx_t[0]) if len(mlx_t) > 1 else float(mlx_t)
    
    print(f"   PyTorch t_scalar: {pt_t_scalar:.6f}")
    print(f"   MLX t_scalar:     {mlx_t_scalar:.6f}")
    
    # 检查时间步是否一致
    if abs(pt_t_scalar - mlx_t_scalar) < 1e-6:
        print(f"   ✅ 时间步一致")
    else:
        print(f"   ❌ 时间步不一致")

def analyze_x_embedding():
    """分析 x 嵌入"""
    print(f"\n{'='*60}")
    print(f"🔍 X 嵌入分析")
    print(f"{'='*60}")
    
    # 加载输入数据
    pt_data, mlx_data = load_dit_input_data()
    if pt_data is None or mlx_data is None:
        return
    
    # 提取 x
    pt_x = pt_data['x']
    mlx_x = mlx_data['x']
    
    print(f"输入 x:")
    compare_tensors("x", pt_x, mlx_x)
    
    # 检查 x 的转置需求
    pt_np = to_numpy(pt_x)
    mlx_np = to_numpy(mlx_x)
    
    print(f"\nX 形状分析:")
    print(f"   PyTorch x: {pt_np.shape} (B, C, T)")
    print(f"   MLX x:     {mlx_np.shape} (B, C, T)")
    
    # 检查是否需要转置
    if pt_np.shape == mlx_np.shape:
        print(f"   ✅ 形状一致，无需转置")
    else:
        print(f"   ❌ 形状不一致")

def analyze_cond_projection():
    """分析条件投影"""
    print(f"\n{'='*60}")
    print(f"🔍 条件投影分析")
    print(f"{'='*60}")
    
    # 加载输入数据
    pt_data, mlx_data = load_dit_input_data()
    if pt_data is None or mlx_data is None:
        return
    
    # 提取 mu (条件)
    pt_mu = pt_data['mu']
    mlx_mu = mlx_data['mu']
    
    print(f"输入 mu (条件):")
    compare_tensors("mu", pt_mu, mlx_mu)
    
    # 检查 mu 的形状
    pt_np = to_numpy(pt_mu)
    mlx_np = to_numpy(mlx_mu)
    
    print(f"\nMu 形状分析:")
    print(f"   PyTorch mu: {pt_np.shape} (B, T, C)")
    print(f"   MLX mu:     {mlx_np.shape} (B, T, C)")
    
    if pt_np.shape == mlx_np.shape:
        print(f"   ✅ 形状一致")
    else:
        print(f"   ❌ 形状不一致")

def analyze_style_embedding():
    """分析风格嵌入"""
    print(f"\n{'='*60}")
    print(f"🔍 风格嵌入分析")
    print(f"{'='*60}")
    
    # 加载输入数据
    pt_data, mlx_data = load_dit_input_data()
    if pt_data is None or mlx_data is None:
        return
    
    # 提取 style
    pt_style = pt_data['style']
    mlx_style = mlx_data['style']
    
    print(f"输入 style:")
    compare_tensors("style", pt_style, mlx_style)
    
    # 检查 style 的形状
    pt_np = to_numpy(pt_style)
    mlx_np = to_numpy(mlx_style)
    
    print(f"\nStyle 形状分析:")
    print(f"   PyTorch style: {pt_np.shape} (B, C)")
    print(f"   MLX style:     {mlx_np.shape} (B, C)")
    
    if pt_np.shape == mlx_np.shape:
        print(f"   ✅ 形状一致")
    else:
        print(f"   ❌ 形状不一致")

def analyze_prompt_x():
    """分析 prompt_x"""
    print(f"\n{'='*60}")
    print(f"🔍 Prompt X 分析")
    print(f"{'='*60}")
    
    # 加载输入数据
    pt_data, mlx_data = load_dit_input_data()
    if pt_data is None or mlx_data is None:
        return
    
    # 提取 prompt_x
    pt_prompt_x = pt_data['prompt_x']
    mlx_prompt_x = mlx_data['prompt_x']
    
    print(f"输入 prompt_x:")
    compare_tensors("prompt_x", pt_prompt_x, mlx_prompt_x)
    
    # 检查 prompt_x 的形状
    pt_np = to_numpy(pt_prompt_x)
    mlx_np = to_numpy(mlx_prompt_x)
    
    print(f"\nPrompt X 形状分析:")
    print(f"   PyTorch prompt_x: {pt_np.shape} (B, C, T)")
    print(f"   MLX prompt_x:     {mlx_np.shape} (B, C, T)")
    
    if pt_np.shape == mlx_np.shape:
        print(f"   ✅ 形状一致")
    else:
        print(f"   ❌ 形状不一致")

def analyze_x_lens():
    """分析 x_lens"""
    print(f"\n{'='*60}")
    print(f"🔍 X Lengths 分析")
    print(f"{'='*60}")
    
    # 加载输入数据
    pt_data, mlx_data = load_dit_input_data()
    if pt_data is None or mlx_data is None:
        return
    
    # 提取 x_lens
    pt_x_lens = pt_data['x_lens']
    mlx_x_lens = mlx_data['x_lens']
    
    print(f"输入 x_lens:")
    compare_tensors("x_lens", pt_x_lens, mlx_x_lens)
    
    # 检查 x_lens 的形状和值
    pt_np = to_numpy(pt_x_lens)
    mlx_np = to_numpy(mlx_x_lens)
    
    print(f"\nX Lengths 分析:")
    print(f"   PyTorch x_lens: {pt_np.shape}, values={pt_np}")
    print(f"   MLX x_lens:     {mlx_np.shape}, values={mlx_np}")
    
    if pt_np.shape == mlx_np.shape and np.allclose(pt_np, mlx_np):
        print(f"   ✅ 形状和值一致")
    else:
        print(f"   ❌ 形状或值不一致")

def main():
    print("🔍 DiT 前向传播逐步分析工具")
    print("="*60)
    
    # 分析各个输入组件
    analyze_timestep_embedding()
    analyze_x_embedding()
    analyze_cond_projection()
    analyze_style_embedding()
    analyze_prompt_x()
    analyze_x_lens()
    
    print(f"\n{'='*60}")
    print(f"📊 总结")
    print(f"{'='*60}")
    print("所有输入组件分析完成。如果所有输入都一致，")
    print("那么问题可能在于 DiT 内部的计算逻辑差异。")
    print("建议下一步分析 DiT 内部的具体计算步骤。")

if __name__ == "__main__":
    main()







