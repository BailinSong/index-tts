#!/usr/bin/env python3
"""
简化的 DiT 组件分析
专注于可以正常工作的组件比较
"""

import os
import torch
import mlx.core as mx
import numpy as np
from omegaconf import OmegaConf
from unified_random_generator import UnifiedRandomGenerator
from indextts.s2mel.modules.mlx_flow_matching import MLXCFM
from indextts.infer_v2 import IndexTTS2 as TorchIndexTTS2

def main():
    print("🔍 简化 DiT 组件分析")
    print("="*60)
    
    # 创建模型
    print("🔧 初始化模型...")
    
    # PyTorch 模型
    pytorch_tts = TorchIndexTTS2(use_mlx=False)
    pytorch_dit = pytorch_tts.s2mel.models['cfm'].estimator
    
    # MLX 模型
    args = OmegaConf.load("checkpoints/config.yaml")
    mlx_cfm = MLXCFM(args.s2mel)
    npz_path = "checkpoints/mlx/s2mel.npz"
    with np.load(npz_path, allow_pickle=True) as data:
        cache_dict = {k: data[k] for k in data.files}
    mlx_cfm.load_from_cache(cache_dict)
    mlx_dit = mlx_cfm.estimator
    
    print("✅ 模型初始化完成")
    
    # 准备测试数据
    print("\n🔧 准备测试数据...")
    unified_random = UnifiedRandomGenerator(seed=42)
    
    batch_size = 2
    seq_len = 415
    in_channels = 80
    content_dim = 512
    style_dim = 192
    
    # 生成一致的测试数据
    x_pt = unified_random.generate_noise((batch_size, in_channels, seq_len), device='cpu')
    x_mlx = unified_random.generate_noise_mlx((batch_size, in_channels, seq_len))
    
    prompt_x_pt = torch.randn(batch_size, in_channels, seq_len)
    prompt_x_mlx = mx.array(prompt_x_pt.numpy())
    
    x_lens_pt = torch.tensor([415, 415])
    x_lens_mlx = mx.array([415, 415])
    
    t_pt = torch.tensor([0.0, 0.0])
    t_mlx = mx.array([0.0, 0.0])
    
    style_pt = torch.randn(batch_size, style_dim)
    style_mlx = mx.array(style_pt.numpy())
    
    mu_pt = torch.randn(batch_size, seq_len, content_dim)
    mu_mlx = mx.array(mu_pt.numpy())
    
    # 获取 PyTorch 模型的设备并移动数据
    pytorch_device = next(pytorch_dit.parameters()).device
    print(f"   📊 PyTorch 设备: {pytorch_device}")
    
    x_pt = x_pt.to(pytorch_device)
    prompt_x_pt = prompt_x_pt.to(pytorch_device)
    x_lens_pt = x_lens_pt.to(pytorch_device)
    t_pt = t_pt.to(pytorch_device)
    style_pt = style_pt.to(pytorch_device)
    mu_pt = mu_pt.to(pytorch_device)
    
    print("✅ 测试数据准备完成")
    
    # 分析各个组件
    print("\n🔍 分析 DiT 各组件")
    print("="*50)
    
    # 1. 分析嵌入层
    print("\n📊 1. 嵌入层分析...")
    
    # x_embedder (PyTorch)
    with torch.no_grad():
        x_emb_pt = pytorch_dit.x_embedder(x_pt.transpose(1, 2))
    print(f"   PyTorch x_embedder: shape={x_emb_pt.shape}, min={x_emb_pt.min():.6f}, max={x_emb_pt.max():.6f}")
    
    # cond_projection
    with torch.no_grad():
        cond_proj_pt = pytorch_dit.cond_projection(mu_pt)
    cond_proj_mlx = mlx_dit.cond_projection(mu_mlx)
    print(f"   PyTorch cond_projection: shape={cond_proj_pt.shape}, min={cond_proj_pt.min():.6f}, max={cond_proj_pt.max():.6f}")
    print(f"   MLX cond_projection:     shape={cond_proj_mlx.shape}, min={float(cond_proj_mlx.min()):.6f}, max={float(cond_proj_mlx.max()):.6f}")
    
    # 比较 cond_projection
    diff_cond = np.abs(cond_proj_pt.detach().cpu().numpy() - np.array(cond_proj_mlx))
    print(f"   cond_projection 差异: max={np.max(diff_cond):.6f}, avg={np.mean(diff_cond):.6f}")
    
    # t_embedder
    with torch.no_grad():
        t_emb_pt = pytorch_dit.t_embedder(t_pt)
    t_emb_mlx = mlx_dit.t_embedder(t_mlx)
    print(f"   PyTorch t_embedder: shape={t_emb_pt.shape}, min={t_emb_pt.min():.6f}, max={t_emb_pt.max():.6f}")
    print(f"   MLX t_embedder:     shape={t_emb_mlx.shape}, min={float(t_emb_mlx.min()):.6f}, max={float(t_emb_mlx.max()):.6f}")
    
    # 比较 t_embedder
    diff_t = np.abs(t_emb_pt.detach().cpu().numpy() - np.array(t_emb_mlx))
    print(f"   t_embedder 差异: max={np.max(diff_t):.6f}, avg={np.mean(diff_t):.6f}")
    
    # 2. 分析条件与输入合并
    print("\n📊 2. 条件与输入合并分析...")
    
    # 构建合并输入
    prompt_x_expanded_pt = torch.nn.functional.interpolate(
        prompt_x_pt, size=x_pt.shape[-1], mode='linear', align_corners=False
    )
    x_t_pt = x_pt.transpose(1, 2)  # (batch, seq_len, in_channels)
    prompt_x_t_pt = prompt_x_expanded_pt.transpose(1, 2)  # (batch, seq_len, in_channels)
    style_expanded_pt = style_pt[:, None, :].repeat(1, x_t_pt.shape[1], 1)  # (batch, seq_len, style_dim)
    
    x_cond_pt = torch.cat([x_t_pt, prompt_x_t_pt, cond_proj_pt, style_expanded_pt], dim=-1)
    
    # MLX 版本 - 使用 resize 函数
    prompt_x_expanded_mlx = mx.resize(prompt_x_mlx, (prompt_x_mlx.shape[0], prompt_x_mlx.shape[1], x_mlx.shape[-1]))
    x_t_mlx = x_mlx.transpose(0, 2, 1)  # (batch, seq_len, in_channels)
    prompt_x_t_mlx = prompt_x_expanded_mlx.transpose(0, 2, 1)  # (batch, seq_len, in_channels)
    style_expanded_mlx = mx.broadcast_to(style_mlx[:, None, :], (style_mlx.shape[0], x_t_mlx.shape[1], style_mlx.shape[1]))
    
    x_cond_mlx = mx.concatenate([x_t_mlx, prompt_x_t_mlx, cond_proj_mlx, style_expanded_mlx], axis=-1)
    
    print(f"   PyTorch x_cond: shape={x_cond_pt.shape}")
    print(f"   MLX x_cond:     shape={x_cond_mlx.shape}")
    
    # 合并层
    with torch.no_grad():
        merged_pt = pytorch_dit.cond_x_merge_linear(x_cond_pt)
    merged_mlx = mlx_dit.cond_x_merge_linear(x_cond_mlx)
    
    print(f"   PyTorch merged: shape={merged_pt.shape}, min={merged_pt.min():.6f}, max={merged_pt.max():.6f}")
    print(f"   MLX merged:     shape={merged_mlx.shape}, min={float(merged_mlx.min()):.6f}, max={float(merged_mlx.max()):.6f}")
    
    # 比较合并层
    diff_merged = np.abs(merged_pt.detach().cpu().numpy() - np.array(merged_mlx))
    print(f"   merged 差异: max={np.max(diff_merged):.6f}, avg={np.mean(diff_merged):.6f}")
    
    # 3. 分析 Transformer 层
    print("\n📊 3. Transformer 层分析...")
    
    # 获取 Transformer 层信息
    pytorch_transformer = pytorch_dit.transformer
    mlx_transformer = mlx_dit.transformer
    
    print(f"   PyTorch Transformer 层数: {len(pytorch_transformer.layers)}")
    print(f"   MLX Transformer 层数: {len(mlx_transformer.layers)}")
    
    # 分析第一层 Transformer
    print("\n   📊 第 1 层 Transformer 分析...")
    with torch.no_grad():
        layer1_output_pt = pytorch_transformer.layers[0](merged_pt, t_emb_pt, x_lens_pt)
    layer1_output_mlx = mlx_transformer.layers[0](merged_mlx, t_emb_mlx, x_lens_mlx)
    
    print(f"   PyTorch layer1: shape={layer1_output_pt.shape}, min={layer1_output_pt.min():.6f}, max={layer1_output_pt.max():.6f}")
    print(f"   MLX layer1:     shape={layer1_output_mlx.shape}, min={float(layer1_output_mlx.min()):.6f}, max={float(layer1_output_mlx.max()):.6f}")
    
    # 比较第一层
    diff_layer1 = np.abs(layer1_output_pt.detach().cpu().numpy() - np.array(layer1_output_mlx))
    print(f"   layer1 差异: max={np.max(diff_layer1):.6f}, avg={np.mean(diff_layer1):.6f}")
    
    # 4. 分析最终层
    print("\n📊 4. 最终层分析...")
    
    # 获取 Transformer 输出
    with torch.no_grad():
        transformer_output_pt = pytorch_dit.transformer(merged_pt, t_emb_pt, x_lens_pt)
    transformer_output_mlx = mlx_dit.transformer(merged_mlx, t_emb_mlx, x_lens_mlx)
    
    print(f"   PyTorch transformer: shape={transformer_output_pt.shape}, min={transformer_output_pt.min():.6f}, max={transformer_output_pt.max():.6f}")
    print(f"   MLX transformer:     shape={transformer_output_mlx.shape}, min={float(transformer_output_mlx.min()):.6f}, max={float(transformer_output_mlx.max()):.6f}")
    
    # 比较 Transformer 输出
    diff_transformer = np.abs(transformer_output_pt.detach().cpu().numpy() - np.array(transformer_output_mlx))
    print(f"   transformer 差异: max={np.max(diff_transformer):.6f}, avg={np.mean(diff_transformer):.6f}")
    
    # final_norm
    with torch.no_grad():
        norm_output_pt = pytorch_dit.final_norm(transformer_output_pt)
    norm_output_mlx = mlx_dit.final_norm(transformer_output_mlx)
    
    print(f"   PyTorch final_norm: shape={norm_output_pt.shape}, min={norm_output_pt.min():.6f}, max={norm_output_pt.max():.6f}")
    print(f"   MLX final_norm:     shape={norm_output_mlx.shape}, min={float(norm_output_mlx.min()):.6f}, max={float(norm_output_mlx.max()):.6f}")
    
    # 比较 final_norm
    diff_norm = np.abs(norm_output_pt.detach().cpu().numpy() - np.array(norm_output_mlx))
    print(f"   final_norm 差异: max={np.max(diff_norm):.6f}, avg={np.mean(diff_norm):.6f}")
    
    # final_layer (WaveNet)
    with torch.no_grad():
        final_output_pt = pytorch_dit.final_layer(norm_output_pt)
    final_output_mlx = mlx_dit.final_layer(norm_output_mlx)
    
    print(f"   PyTorch final_layer: shape={final_output_pt.shape}, min={final_output_pt.min():.6f}, max={final_output_pt.max():.6f}")
    print(f"   MLX final_layer:     shape={final_output_mlx.shape}, min={float(final_output_mlx.min()):.6f}, max={float(final_output_mlx.max()):.6f}")
    
    # 比较 final_layer
    diff_final = np.abs(final_output_pt.detach().cpu().numpy() - np.array(final_output_mlx))
    print(f"   final_layer 差异: max={np.max(diff_final):.6f}, avg={np.mean(diff_final):.6f}")
    
    # 5. 总结分析
    print("\n📊 5. 差异总结分析")
    print("="*50)
    
    print("各组件差异统计:")
    print(f"   cond_projection: max={np.max(diff_cond):.6f}, avg={np.mean(diff_cond):.6f}")
    print(f"   t_embedder:      max={np.max(diff_t):.6f}, avg={np.mean(diff_t):.6f}")
    print(f"   merged:          max={np.max(diff_merged):.6f}, avg={np.mean(diff_merged):.6f}")
    print(f"   layer1:          max={np.max(diff_layer1):.6f}, avg={np.mean(diff_layer1):.6f}")
    print(f"   transformer:     max={np.max(diff_transformer):.6f}, avg={np.mean(diff_transformer):.6f}")
    print(f"   final_norm:      max={np.max(diff_norm):.6f}, avg={np.mean(diff_norm):.6f}")
    print(f"   final_layer:     max={np.max(diff_final):.6f}, avg={np.mean(diff_final):.6f}")
    
    # 找出差异最大的组件
    components = {
        'cond_projection': np.max(diff_cond),
        't_embedder': np.max(diff_t),
        'merged': np.max(diff_merged),
        'layer1': np.max(diff_layer1),
        'transformer': np.max(diff_transformer),
        'final_norm': np.max(diff_norm),
        'final_layer': np.max(diff_final)
    }
    
    max_diff_component = max(components, key=components.get)
    print(f"\n💡 最大差异组件: {max_diff_component} (差异: {components[max_diff_component]:.6f})")
    
    print("\n🎯 组件分析完成")

if __name__ == "__main__":
    main()
