#!/usr/bin/env python3
"""
简化的CFM模块对比调试工具
只对比能够正常工作的模块
"""

import torch
import mlx.core as mx
import numpy as np
import pickle
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from indextts.infer_v2 import IndexTTS2
from indextts.utils.mlx_utils import mlx_to_torch, torch_to_mlx
from unified_random_generator import UnifiedRandomGenerator

def debug_simple_cfm_modules():
    """主函数：简化对比CFM中能够正常工作的模块"""
    
    print("=== 简化CFM模块输入输出对比调试 ===")
    
    # 初始化TTS系统
    print("\n1. 初始化TTS系统...")
    tts = IndexTTS2()
    
    # 生成测试输入数据
    print("\n2. 生成测试输入数据...")
    
    # 设置随机种子确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 生成测试数据
    batch_size = 1
    seq_len = 100
    in_channels = 80
    hidden_dim = 512
    style_dim = 192
    
    # 获取设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    x = torch.randn(batch_size, in_channels, seq_len, device=device)
    prompt_x = torch.randn(batch_size, in_channels, seq_len, device=device)
    x_lens = torch.tensor([seq_len], device=device)
    t = torch.zeros(batch_size, device=device)
    style = torch.randn(batch_size, style_dim, device=device)
    cond = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    print(f"✅ 生成测试数据:")
    print(f"输入形状:")
    print(f"  x: {x.shape}")
    print(f"  prompt_x: {prompt_x.shape}")
    print(f"  x_lens: {x_lens.shape}")
    print(f"  t: {t.shape}")
    print(f"  style: {style.shape}")
    print(f"  cond: {cond.shape}")
    
    # 设置统一随机数生成器
    unified_rng = UnifiedRandomGenerator(42)
    
    print("\n3. 对比基础模块...")
    
    # 获取PyTorch CFM estimator
    pytorch_cfm = tts.s2mel.models.cfm
    pytorch_estimator = pytorch_cfm.estimator
    
    # 手动创建MLX CFM
    from indextts.s2mel.modules.mlx_cfm import MLXCFM
    mlx_cfm = MLXCFM(tts.cfg.s2mel)
    mlx_estimator = mlx_cfm.estimator
    
    print(f"PyTorch CFM estimator类型: {type(pytorch_estimator)}")
    print(f"MLX CFM estimator类型: {type(mlx_estimator)}")
    
    # 准备MLX输入
    mlx_x = torch_to_mlx(x)
    mlx_prompt_x = torch_to_mlx(prompt_x)
    mlx_x_lens = torch_to_mlx(x_lens)
    mlx_t = torch_to_mlx(t)
    mlx_style = torch_to_mlx(style)
    mlx_cond = torch_to_mlx(cond)
    
    print("\n--- 基础模块对比 ---")
    
    # 1. x_embedder对比
    print("\n📊 x_embedder对比:")
    with torch.no_grad():
        pytorch_x_emb = pytorch_estimator.x_embedder(x.transpose(1, 2))
        print(f"PyTorch: {pytorch_x_emb.shape}, min={pytorch_x_emb.min():.6f}, max={pytorch_x_emb.max():.6f}, mean={pytorch_x_emb.mean():.6f}")
    
    mlx_x_emb = mlx_estimator.x_embedder(mlx_x.transpose(0, 2, 1))
    print(f"MLX: {mlx_x_emb.shape}, min={mlx_x_emb.min():.6f}, max={mlx_x_emb.max():.6f}, mean={mlx_x_emb.mean():.6f}")
    
    # 对比x_embedder
    mlx_x_emb_torch = mlx_to_torch(mlx_x_emb)
    if pytorch_x_emb.device != mlx_x_emb_torch.device:
        mlx_x_emb_torch = mlx_x_emb_torch.to(pytorch_x_emb.device)
    
    diff = torch.abs(pytorch_x_emb - mlx_x_emb_torch)
    print(f"差异: 最大={diff.max():.6f}, 平均={diff.mean():.6f}, 标准差={diff.std():.6f}")
    
    # 2. cond_embedder对比
    print("\n📊 cond_embedder对比:")
    with torch.no_grad():
        pytorch_cond_emb = pytorch_estimator.cond_embedder(cond.long())
        print(f"PyTorch: {pytorch_cond_emb.shape}, min={pytorch_cond_emb.min():.6f}, max={pytorch_cond_emb.max():.6f}, mean={pytorch_cond_emb.mean():.6f}")
    
    mlx_cond_emb = mlx_estimator.cond_embedder(mlx_cond.astype(mx.int32))
    print(f"MLX: {mlx_cond_emb.shape}, min={mlx_cond_emb.min():.6f}, max={mlx_cond_emb.max():.6f}, mean={mlx_cond_emb.mean():.6f}")
    
    # 对比cond_embedder
    mlx_cond_emb_torch = mlx_to_torch(mlx_cond_emb)
    if pytorch_cond_emb.device != mlx_cond_emb_torch.device:
        mlx_cond_emb_torch = mlx_cond_emb_torch.to(pytorch_cond_emb.device)
    
    diff = torch.abs(pytorch_cond_emb - mlx_cond_emb_torch)
    print(f"差异: 最大={diff.max():.6f}, 平均={diff.mean():.6f}, 标准差={diff.std():.6f}")
    
    # 3. cond_projection对比
    print("\n📊 cond_projection对比:")
    with torch.no_grad():
        pytorch_cond_proj = pytorch_estimator.cond_projection(pytorch_cond_emb)
        print(f"PyTorch: {pytorch_cond_proj.shape}, min={pytorch_cond_proj.min():.6f}, max={pytorch_cond_proj.max():.6f}, mean={pytorch_cond_proj.mean():.6f}")
    
    mlx_cond_proj = mlx_estimator.cond_projection(mlx_cond_emb)
    print(f"MLX: {mlx_cond_proj.shape}, min={mlx_cond_proj.min():.6f}, max={mlx_cond_proj.max():.6f}, mean={mlx_cond_proj.mean():.6f}")
    
    # 对比cond_projection
    mlx_cond_proj_torch = mlx_to_torch(mlx_cond_proj)
    if pytorch_cond_proj.device != mlx_cond_proj_torch.device:
        mlx_cond_proj_torch = mlx_cond_proj_torch.to(pytorch_cond_proj.device)
    
    diff = torch.abs(pytorch_cond_proj - mlx_cond_proj_torch)
    print(f"差异: 最大={diff.max():.6f}, 平均={diff.mean():.6f}, 标准差={diff.std():.6f}")
    
    # 4. t_embedder对比
    print("\n📊 t_embedder对比:")
    with torch.no_grad():
        pytorch_t_emb = pytorch_estimator.t_embedder(t)
        print(f"PyTorch: {pytorch_t_emb.shape}, min={pytorch_t_emb.min():.6f}, max={pytorch_t_emb.max():.6f}, mean={pytorch_t_emb.mean():.6f}")
    
    mlx_t_emb = mlx_estimator.t_embedder(mlx_t)
    print(f"MLX: {mlx_t_emb.shape}, min={mlx_t_emb.min():.6f}, max={mlx_t_emb.max():.6f}, mean={mlx_t_emb.mean():.6f}")
    
    # 对比t_embedder
    mlx_t_emb_torch = mlx_to_torch(mlx_t_emb)
    if pytorch_t_emb.device != mlx_t_emb_torch.device:
        mlx_t_emb_torch = mlx_t_emb_torch.to(pytorch_t_emb.device)
    
    diff = torch.abs(pytorch_t_emb - mlx_t_emb_torch)
    print(f"差异: 最大={diff.max():.6f}, 平均={diff.mean():.6f}, 标准差={diff.std():.6f}")
    
    # 5. 输入预处理对比
    print("\n📊 输入预处理对比:")
    
    # PyTorch输入预处理
    with torch.no_grad():
        x_t = x.transpose(1, 2)  # (batch, seq_len, in_channels)
        prompt_x_t = prompt_x.transpose(1, 2)
        pytorch_x_in = torch.cat([x_t, prompt_x_t], dim=-1)  # (batch, seq_len, in_channels*2)
        
        # 添加style
        if pytorch_estimator.transformer_style_condition:
            style_broadcast = style.unsqueeze(1).expand(-1, pytorch_x_in.shape[1], -1)
            pytorch_x_in = torch.cat([pytorch_x_in, style_broadcast], dim=-1)
        
        print(f"PyTorch x_in: {pytorch_x_in.shape}, min={pytorch_x_in.min():.6f}, max={pytorch_x_in.max():.6f}, mean={pytorch_x_in.mean():.6f}")
    
    # MLX输入预处理
    x_t = mlx_x.transpose(0, 2, 1)  # (batch, seq_len, in_channels)
    prompt_x_t = mlx_prompt_x.transpose(0, 2, 1)
    mlx_x_in = mx.concatenate([x_t, prompt_x_t], axis=-1)  # (batch, seq_len, in_channels*2)
    
    # 添加style
    if mlx_estimator.transformer_style_condition:
        style_broadcast = mx.expand_dims(mlx_style, 1)
        style_broadcast = mx.broadcast_to(style_broadcast, (mlx_x_in.shape[0], mlx_x_in.shape[1], style_broadcast.shape[-1]))
        mlx_x_in = mx.concatenate([mlx_x_in, style_broadcast], axis=-1)
    
    print(f"MLX x_in: {mlx_x_in.shape}, min={mlx_x_in.min():.6f}, max={mlx_x_in.max():.6f}, mean={mlx_x_in.mean():.6f}")
    
    # 对比输入预处理
    mlx_x_in_torch = mlx_to_torch(mlx_x_in)
    if pytorch_x_in.device != mlx_x_in_torch.device:
        mlx_x_in_torch = mlx_x_in_torch.to(pytorch_x_in.device)
    
    diff = torch.abs(pytorch_x_in - mlx_x_in_torch)
    print(f"差异: 最大={diff.max():.6f}, 平均={diff.mean():.6f}, 标准差={diff.std():.6f}")
    
    # 6. cond_x_merge_linear对比
    print("\n📊 cond_x_merge_linear对比:")
    
    # 检查输入维度并修复
    expected_input_dim = pytorch_estimator.cond_x_merge_linear.weight.shape[1]
    actual_input_dim = pytorch_x_in.shape[-1]
    print(f"输入维度检查: 期望={expected_input_dim}, 实际={actual_input_dim}")
    
    if actual_input_dim != expected_input_dim:
        missing_dim = expected_input_dim - actual_input_dim
        if missing_dim > 0:
            padding = torch.zeros(pytorch_x_in.shape[0], pytorch_x_in.shape[1], missing_dim, device=pytorch_x_in.device)
            pytorch_x_in = torch.cat([pytorch_x_in, padding], dim=-1)
            print(f"PyTorch添加零填充后: {pytorch_x_in.shape}")
    
    # 检查MLX输入维度
    mlx_expected_input_dim = mlx_estimator.cond_x_merge_linear.weight.shape[1]
    mlx_actual_input_dim = mlx_x_in.shape[-1]
    print(f"MLX输入维度检查: 期望={mlx_expected_input_dim}, 实际={mlx_actual_input_dim}")
    
    if mlx_actual_input_dim != mlx_expected_input_dim:
        missing_dim = mlx_expected_input_dim - mlx_actual_input_dim
        if missing_dim > 0:
            padding = mx.zeros((mlx_x_in.shape[0], mlx_x_in.shape[1], missing_dim))
            mlx_x_in = mx.concatenate([mlx_x_in, padding], axis=-1)
            print(f"MLX添加零填充后: {mlx_x_in.shape}")
    
    # 执行cond_x_merge_linear
    with torch.no_grad():
        pytorch_merged = pytorch_estimator.cond_x_merge_linear(pytorch_x_in)
        print(f"PyTorch merged: {pytorch_merged.shape}, min={pytorch_merged.min():.6f}, max={pytorch_merged.max():.6f}, mean={pytorch_merged.mean():.6f}")
    
    mlx_merged = mlx_estimator.cond_x_merge_linear(mlx_x_in)
    print(f"MLX merged: {mlx_merged.shape}, min={mlx_merged.min():.6f}, max={mlx_merged.max():.6f}, mean={mlx_merged.mean():.6f}")
    
    # 对比cond_x_merge_linear
    mlx_merged_torch = mlx_to_torch(mlx_merged)
    if pytorch_merged.device != mlx_merged_torch.device:
        mlx_merged_torch = mlx_merged_torch.to(pytorch_merged.device)
    
    diff = torch.abs(pytorch_merged - mlx_merged_torch)
    print(f"差异: 最大={diff.max():.6f}, 平均={diff.mean():.6f}, 标准差={diff.std():.6f}")
    
    print("\n=== 总结 ===")
    print("✅ 基础模块对比完成")
    print("📊 详细差异分析已输出")

if __name__ == "__main__":
    debug_simple_cfm_modules()

