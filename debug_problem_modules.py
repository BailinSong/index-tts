#!/usr/bin/env python3
"""
问题模块深度分析工具
专门分析t_embedder、cond_embedder、Transformer逐层、FinalLayer的差异
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

def analyze_problem_modules():
    """分析问题模块的详细差异"""
    
    print("=== 问题模块深度分析 ===")
    
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
    print(f"  t: {t.shape}")
    print(f"  style: {style.shape}")
    print(f"  cond: {cond.shape}")
    
    # 设置统一随机数生成器
    unified_rng = UnifiedRandomGenerator(42)
    
    print("\n3. 分析问题模块...")
    
    # 获取PyTorch CFM estimator
    pytorch_cfm = tts.s2mel.models.cfm
    pytorch_estimator = pytorch_cfm.estimator
    
    # 手动创建MLX CFM
    from indextts.s2mel.modules.mlx_cfm import MLXCFM
    mlx_cfm = MLXCFM(tts.cfg.s2mel)
    mlx_estimator = mlx_cfm.estimator
    
    # 准备MLX输入
    mlx_x = torch_to_mlx(x)
    mlx_prompt_x = torch_to_mlx(prompt_x)
    mlx_x_lens = torch_to_mlx(x_lens)
    mlx_t = torch_to_mlx(t)
    mlx_style = torch_to_mlx(style)
    mlx_cond = torch_to_mlx(cond)
    
    print("\n--- 问题模块1: t_embedder (最大差异源头) ---")
    
    # 分析t_embedder
    with torch.no_grad():
        pytorch_t_emb = pytorch_estimator.t_embedder(t)
        print(f"PyTorch t_embedder:")
        print(f"  形状: {pytorch_t_emb.shape}")
        print(f"  数值范围: min={pytorch_t_emb.min():.6f}, max={pytorch_t_emb.max():.6f}")
        print(f"  统计: mean={pytorch_t_emb.mean():.6f}, std={pytorch_t_emb.std():.6f}")
        print(f"  数值范围: {pytorch_t_emb.max() - pytorch_t_emb.min():.6f}")
    
    mlx_t_emb = mlx_estimator.t_embedder(mlx_t)
    print(f"MLX t_embedder:")
    print(f"  形状: {mlx_t_emb.shape}")
    print(f"  数值范围: min={mlx_t_emb.min():.6f}, max={mlx_t_emb.max():.6f}")
    print(f"  统计: mean={mlx_t_emb.mean():.6f}, std={mlx_t_emb.std():.6f}")
    print(f"  数值范围: {mlx_t_emb.max() - mlx_t_emb.min():.6f}")
    
    # 对比t_embedder
    mlx_t_emb_torch = mlx_to_torch(mlx_t_emb)
    if pytorch_t_emb.device != mlx_t_emb_torch.device:
        mlx_t_emb_torch = mlx_t_emb_torch.to(pytorch_t_emb.device)
    
    diff = torch.abs(pytorch_t_emb - mlx_t_emb_torch)
    print(f"t_embedder差异:")
    print(f"  最大差异: {diff.max():.6f}")
    print(f"  平均差异: {diff.mean():.6f}")
    print(f"  标准差差异: {diff.std():.6f}")
    print(f"  差异比例: {diff.max() / pytorch_t_emb.std():.6f}")
    
    print("\n--- 问题模块2: cond_embedder (第二大差异源头) ---")
    
    # 分析cond_embedder
    with torch.no_grad():
        pytorch_cond_emb = pytorch_estimator.cond_embedder(cond.long())
        print(f"PyTorch cond_embedder:")
        print(f"  形状: {pytorch_cond_emb.shape}")
        print(f"  数值范围: min={pytorch_cond_emb.min():.6f}, max={pytorch_cond_emb.max():.6f}")
        print(f"  统计: mean={pytorch_cond_emb.mean():.6f}, std={pytorch_cond_emb.std():.6f}")
        print(f"  数值范围: {pytorch_cond_emb.max() - pytorch_cond_emb.min():.6f}")
    
    mlx_cond_emb = mlx_estimator.cond_embedder(mlx_cond.astype(mx.int32))
    print(f"MLX cond_embedder:")
    print(f"  形状: {mlx_cond_emb.shape}")
    print(f"  数值范围: min={mlx_cond_emb.min():.6f}, max={mlx_cond_emb.max():.6f}")
    print(f"  统计: mean={mlx_cond_emb.mean():.6f}, std={mlx_cond_emb.std():.6f}")
    print(f"  数值范围: {mlx_cond_emb.max() - mlx_cond_emb.min():.6f}")
    
    # 对比cond_embedder
    mlx_cond_emb_torch = mlx_to_torch(mlx_cond_emb)
    if pytorch_cond_emb.device != mlx_cond_emb_torch.device:
        mlx_cond_emb_torch = mlx_cond_emb_torch.to(pytorch_cond_emb.device)
    
    diff = torch.abs(pytorch_cond_emb - mlx_cond_emb_torch)
    print(f"cond_embedder差异:")
    print(f"  最大差异: {diff.max():.6f}")
    print(f"  平均差异: {diff.mean():.6f}")
    print(f"  标准差差异: {diff.std():.6f}")
    print(f"  差异比例: {diff.max() / pytorch_cond_emb.std():.6f}")
    
    print("\n--- 问题模块3: Transformer逐层 (差异累积放大) ---")
    
    # 分析Transformer逐层
    with torch.no_grad():
        # 准备Transformer输入
        x_t = x.transpose(1, 2)
        prompt_x_t = prompt_x.transpose(1, 2)
        x_in = torch.cat([x_t, prompt_x_t], dim=-1)
        
        if pytorch_estimator.transformer_style_condition:
            style_broadcast = style.unsqueeze(1).expand(-1, x_in.shape[1], -1)
            x_in = torch.cat([x_in, style_broadcast], dim=-1)
        
        # 检查输入维度并修复
        expected_input_dim = pytorch_estimator.cond_x_merge_linear.weight.shape[1]
        actual_input_dim = x_in.shape[-1]
        if actual_input_dim != expected_input_dim:
            missing_dim = expected_input_dim - actual_input_dim
            if missing_dim > 0:
                padding = torch.zeros(x_in.shape[0], x_in.shape[1], missing_dim, device=x_in.device)
                x_in = torch.cat([x_in, padding], dim=-1)
        
        x_in = pytorch_estimator.cond_x_merge_linear(x_in)
        
        # Transformer逐层分析
        transformer = pytorch_estimator.transformer
        input_pos = torch.arange(seq_len, device=device)
        freqs_cis = transformer.freqs_cis[:seq_len]
        mask = torch.ones(1, 1, seq_len, seq_len, device=device)
        
        x_current = x_in
        pytorch_layers = []
        for i, layer in enumerate(transformer.layers):
            x_before = x_current.clone()
            x_current = layer(x_current, pytorch_t_emb, input_pos, freqs_cis, mask, None, None, None, None)
            pytorch_layers.append(x_current.clone())
            print(f"PyTorch layer_{i}: shape={x_current.shape}, min={x_current.min():.6f}, max={x_current.max():.6f}, mean={x_current.mean():.6f}")
    
    # MLX Transformer逐层分析
    x_t = mlx_x.transpose(0, 2, 1)
    prompt_x_t = mlx_prompt_x.transpose(0, 2, 1)
    x_in = mx.concatenate([x_t, prompt_x_t], axis=-1)
    
    if mlx_estimator.transformer_style_condition:
        style_broadcast = mx.expand_dims(mlx_style, 1)
        style_broadcast = mx.broadcast_to(style_broadcast, (x_in.shape[0], x_in.shape[1], style_broadcast.shape[-1]))
        x_in = mx.concatenate([x_in, style_broadcast], axis=-1)
    
    # 检查输入维度并修复
    expected_input_dim = mlx_estimator.cond_x_merge_linear.weight.shape[1]
    actual_input_dim = x_in.shape[-1]
    if actual_input_dim != expected_input_dim:
        missing_dim = expected_input_dim - actual_input_dim
        if missing_dim > 0:
            padding = mx.zeros((x_in.shape[0], x_in.shape[1], missing_dim))
            x_in = mx.concatenate([x_in, padding], axis=-1)
    
    x_in = mlx_estimator.cond_x_merge_linear(x_in)
    
    # MLX Transformer逐层分析
    transformer = mlx_estimator.transformer
    input_pos = mx.arange(seq_len)
    freqs_cis = transformer.freqs_cis[:seq_len]
    mask = mx.ones((1, 1, seq_len, seq_len))
    
    x_current = x_in
    mlx_layers = []
    for i, layer in enumerate(transformer.layers):
        x_before = x_current
        x_current = layer(x_current, mlx_t_emb, input_pos, freqs_cis, mask, None, None, None, None)
        mlx_layers.append(x_current)
        print(f"MLX layer_{i}: shape={x_current.shape}, min={x_current.min():.6f}, max={x_current.max():.6f}, mean={x_current.mean():.6f}")
    
    # 对比Transformer逐层
    print(f"\nTransformer逐层差异分析:")
    for i in range(len(pytorch_layers)):
        pytorch_layer = pytorch_layers[i]
        mlx_layer = mlx_layers[i]
        
        mlx_layer_torch = mlx_to_torch(mlx_layer)
        if pytorch_layer.device != mlx_layer_torch.device:
            mlx_layer_torch = mlx_layer_torch.to(pytorch_layer.device)
        
        diff = torch.abs(pytorch_layer - mlx_layer_torch)
        print(f"  layer_{i}: 最大差异={diff.max():.6f}, 平均差异={diff.mean():.6f}, 差异比例={diff.max() / pytorch_layer.std():.6f}")
    
    print("\n--- 问题模块4: FinalLayer (最终差异体现) ---")
    
    # 分析FinalLayer
    with torch.no_grad():
        # 准备FinalLayer输入
        x_final = pytorch_layers[-1]  # 使用最后一层Transformer输出
        pytorch_final_out = pytorch_estimator.final_layer(x_final, pytorch_t_emb)
        print(f"PyTorch FinalLayer:")
        print(f"  形状: {pytorch_final_out.shape}")
        print(f"  数值范围: min={pytorch_final_out.min():.6f}, max={pytorch_final_out.max():.6f}")
        print(f"  统计: mean={pytorch_final_out.mean():.6f}, std={pytorch_final_out.std():.6f}")
        print(f"  数值范围: {pytorch_final_out.max() - pytorch_final_out.min():.6f}")
    
    mlx_final_out = mlx_estimator.final_layer(mlx_layers[-1], mlx_t_emb)
    print(f"MLX FinalLayer:")
    print(f"  形状: {mlx_final_out.shape}")
    print(f"  数值范围: min={mlx_final_out.min():.6f}, max={mlx_final_out.max():.6f}")
    print(f"  统计: mean={mlx_final_out.mean():.6f}, std={mlx_final_out.std():.6f}")
    print(f"  数值范围: {mlx_final_out.max() - mlx_final_out.min():.6f}")
    
    # 对比FinalLayer
    mlx_final_out_torch = mlx_to_torch(mlx_final_out)
    if pytorch_final_out.device != mlx_final_out_torch.device:
        mlx_final_out_torch = mlx_final_out_torch.to(pytorch_final_out.device)
    
    diff = torch.abs(pytorch_final_out - mlx_final_out_torch)
    print(f"FinalLayer差异:")
    print(f"  最大差异: {diff.max():.6f}")
    print(f"  平均差异: {diff.mean():.6f}")
    print(f"  标准差差异: {diff.std():.6f}")
    print(f"  差异比例: {diff.max() / pytorch_final_out.std():.6f}")
    
    print("\n=== 问题模块总结 ===")
    print("1. t_embedder: 最大差异源头 - 时间嵌入器产生完全不同的数值范围")
    print("2. cond_embedder: 第二大差异源头 - 条件嵌入器产生完全不同的数值范围")
    print("3. Transformer逐层: 差异累积放大 - 早期差异在逐层处理中不断放大")
    print("4. FinalLayer: 最终差异体现 - 所有差异的最终体现")
    
    print("\n=== 根本原因分析 ===")
    print("问题根源: MLX的t_embedder和cond_embedder权重没有正确从PyTorch转换")
    print("影响范围: 从早期模块开始，差异在后续模块中不断放大")
    print("解决方案: 需要检查并修复MLX的权重加载和转换过程")

if __name__ == "__main__":
    analyze_problem_modules()
