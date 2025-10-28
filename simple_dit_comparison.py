#!/usr/bin/env python3
"""
简化的 DiT 比较分析
直接比较 PyTorch 和 MLX DiT 的最终输出，跳过内部步骤分析
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
    print("🔍 简化 DiT 比较分析")
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
    
    # 执行推理
    print("\n🔧 执行 DiT 推理...")
    
    # PyTorch 推理
    print("   📊 PyTorch DiT 推理...")
    with torch.no_grad():
        pytorch_output = pytorch_dit(x_pt, prompt_x_pt, x_lens_pt, t_pt, style_pt, mu_pt)
    print(f"      PyTorch 输出: shape={pytorch_output.shape}, min={pytorch_output.min():.6f}, max={pytorch_output.max():.6f}")
    
    # MLX 推理
    print("   📊 MLX DiT 推理...")
    mlx_output = mlx_dit(x_mlx, prompt_x_mlx, x_lens_mlx, t_mlx, style_mlx, mu_mlx)
    print(f"      MLX 输出: shape={mlx_output.shape}, min={float(mlx_output.min()):.6f}, max={float(mlx_output.max()):.6f}")
    
    # 比较结果
    print("\n📊 DiT 输出比较分析")
    print("="*60)
    
    # 转换为 numpy 进行比较
    pytorch_np = pytorch_output.cpu().numpy()
    mlx_np = np.array(mlx_output)
    
    # 比较形状
    if pytorch_np.shape == mlx_np.shape:
        print(f"✅ 形状一致: {pytorch_np.shape}")
        
        # 比较数值
        diff = np.abs(pytorch_np - mlx_np)
        max_diff = np.max(diff)
        avg_diff = np.mean(diff)
        
        print(f"📊 PyTorch: min={np.min(pytorch_np):.6f}, max={np.max(pytorch_np):.6f}, avg={np.mean(pytorch_np):.6f}")
        print(f"📊 MLX:     min={np.min(mlx_np):.6f}, max={np.max(mlx_np):.6f}, avg={np.mean(mlx_np):.6f}")
        print(f"📊 差异:    max={max_diff:.6f}, avg={avg_diff:.6f}")
        
        if max_diff < 1e-6:
            print("✅ DiT 输出完全一致")
        else:
            print("❌ DiT 输出存在差异")
            # 找出差异最大的位置
            max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"💡 最大差异位置: {max_diff_idx}, 差异值: {max_diff:.6f}")
            
            # 分析差异分布
            print(f"💡 差异统计:")
            print(f"   - 差异 > 1.0: {np.sum(diff > 1.0)} 个元素")
            print(f"   - 差异 > 0.1: {np.sum(diff > 0.1)} 个元素")
            print(f"   - 差异 > 0.01: {np.sum(diff > 0.01)} 个元素")
    else:
        print(f"❌ 形状不匹配: PyTorch={pytorch_np.shape}, MLX={mlx_np.shape}")
    
    print("\n🎯 分析完成")

if __name__ == "__main__":
    main()
