#!/usr/bin/env python3
"""
简单的 CFM 一致性测试
使用相同的输入数据测试 PyTorch 和 MLX CFM
"""

import torch
import mlx.core as mx
import numpy as np
from unified_random_generator import UnifiedRandomGenerator

def test_cfm_consistency():
    """测试 CFM 一致性"""
    print("🔍 CFM 一致性测试")
    print("=" * 60)
    
    # 创建统一的随机数生成器
    unified_random = UnifiedRandomGenerator(seed=42)
    
    # 定义测试参数
    batch_size = 1
    seq_len = 415  # 固定序列长度
    in_channels = 80
    content_dim = 512
    style_dim = 192
    n_timesteps = 25
    temperature = 1.0
    inference_cfg_rate = 0.7
    
    print(f"📊 测试参数:")
    print(f"   batch_size: {batch_size}")
    print(f"   seq_len: {seq_len}")
    print(f"   in_channels: {in_channels}")
    print(f"   content_dim: {content_dim}")
    print(f"   style_dim: {style_dim}")
    print(f"   n_timesteps: {n_timesteps}")
    print(f"   temperature: {temperature}")
    print(f"   inference_cfg_rate: {inference_cfg_rate}")
    
    # 生成相同的输入数据
    print(f"\n🔄 生成测试输入数据...")
    
    # 初始化噪声 (x)
    x_pytorch = unified_random.generate_noise((batch_size, in_channels, seq_len)) * temperature
    x_mlx = unified_random.generate_noise_mlx((batch_size, in_channels, seq_len)) * temperature
    
    # 语义条件 (mu)
    mu_pytorch = torch.randn(batch_size, seq_len, content_dim)
    mu_mlx = mx.array(mu_pytorch.numpy())
    
    # 参考音频 (prompt)
    prompt_len = 243
    prompt_pytorch = torch.randn(batch_size, in_channels, prompt_len)
    prompt_mlx = mx.array(prompt_pytorch.numpy())
    
    # 风格向量 (style)
    style_pytorch = torch.randn(batch_size, style_dim)
    style_mlx = mx.array(style_pytorch.numpy())
    
    # 序列长度
    x_lens_pytorch = torch.tensor([seq_len])
    x_lens_mlx = mx.array([seq_len])
    
    # F0 (未使用)
    f0 = None
    
    print(f"✅ 输入数据生成完成:")
    print(f"   x: PyTorch {x_pytorch.shape}, MLX {x_mlx.shape}")
    print(f"   mu: PyTorch {mu_pytorch.shape}, MLX {mu_mlx.shape}")
    print(f"   prompt: PyTorch {prompt_pytorch.shape}, MLX {prompt_mlx.shape}")
    print(f"   style: PyTorch {style_pytorch.shape}, MLX {style_mlx.shape}")
    print(f"   x_lens: PyTorch {x_lens_pytorch.shape}, MLX {x_lens_mlx.shape}")
    
    # 验证输入数据一致性
    print(f"\n🔍 验证输入数据一致性...")
    
    # 检查 x 的一致性
    x_diff = torch.abs(x_pytorch - torch.from_numpy(np.array(x_mlx)))
    x_max_diff = torch.max(x_diff).item()
    x_mean_diff = torch.mean(x_diff).item()
    print(f"   x 差异: max={x_max_diff:.8f}, mean={x_mean_diff:.8f}")
    
    # 检查 mu 的一致性
    mu_diff = torch.abs(mu_pytorch - torch.from_numpy(np.array(mu_mlx)))
    mu_max_diff = torch.max(mu_diff).item()
    mu_mean_diff = torch.mean(mu_diff).item()
    print(f"   mu 差异: max={mu_max_diff:.8f}, mean={mu_mean_diff:.8f}")
    
    # 检查 prompt 的一致性
    prompt_diff = torch.abs(prompt_pytorch - torch.from_numpy(np.array(prompt_mlx)))
    prompt_max_diff = torch.max(prompt_diff).item()
    prompt_mean_diff = torch.mean(prompt_diff).item()
    print(f"   prompt 差异: max={prompt_max_diff:.8f}, mean={prompt_mean_diff:.8f}")
    
    # 检查 style 的一致性
    style_diff = torch.abs(style_pytorch - torch.from_numpy(np.array(style_mlx)))
    style_max_diff = torch.max(style_diff).item()
    style_mean_diff = torch.mean(style_diff).item()
    print(f"   style 差异: max={style_max_diff:.8f}, mean={style_mean_diff:.8f}")
    
    if all([x_max_diff < 1e-6, mu_max_diff < 1e-6, prompt_max_diff < 1e-6, style_max_diff < 1e-6]):
        print(f"   ✅ 输入数据完全一致")
    else:
        print(f"   ❌ 输入数据不一致")
        return False
    
    # 加载模型
    print(f"\n🔥 加载 CFM 模型...")
    
    try:
        from indextts.infer_v2 import IndexTTS2
        
        # 加载 PyTorch CFM
        pytorch_tts = IndexTTS2(use_mlx=False)
        pytorch_cfm = pytorch_tts.s2mel.models['cfm']
        print(f"   ✅ PyTorch CFM 加载成功")
        
        # 加载 MLX CFM
        mlx_tts = IndexTTS2(use_mlx=True)
        mlx_cfm = mlx_tts.mlx_s2mel_cfm
        print(f"   ✅ MLX CFM 加载成功")
        
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        return False
    
    # 测试 PyTorch CFM
    print(f"\n🔥 测试 PyTorch CFM...")
    try:
        pytorch_output = pytorch_cfm.inference(
            mu_pytorch,
            x_lens_pytorch,
            prompt_pytorch,
            style_pytorch,
            f0,
            n_timesteps,
            inference_cfg_rate=inference_cfg_rate,
            unified_random=unified_random
        )
        print(f"   ✅ PyTorch CFM 输出: {pytorch_output.shape}")
    except Exception as e:
        print(f"   ❌ PyTorch CFM 推理失败: {e}")
        return False
    
    # 测试 MLX CFM
    print(f"\n🔥 测试 MLX CFM...")
    try:
        mlx_output = mlx_cfm.inference(
            mu_mlx,
            x_lens_mlx,
            prompt_mlx,
            style_mlx,
            f0,
            n_timesteps,
            temperature=temperature,
            inference_cfg_rate=inference_cfg_rate,
            unified_random=unified_random
        )
        print(f"   ✅ MLX CFM 输出: {mlx_output.shape}")
    except Exception as e:
        print(f"   ❌ MLX CFM 推理失败: {e}")
        return False
    
    # 比较输出
    print(f"\n🔍 比较输出结果...")
    
    # 转换为相同格式
    if isinstance(mlx_output, mx.array):
        mlx_output_torch = torch.from_numpy(np.array(mlx_output))
    else:
        mlx_output_torch = mlx_output
    
    print(f"   PyTorch 输出: {pytorch_output.shape}")
    print(f"   MLX 输出:     {mlx_output_torch.shape}")
    
    if pytorch_output.shape != mlx_output_torch.shape:
        print(f"   ❌ 输出形状不匹配")
        return False
    
    # 计算差异
    output_diff = torch.abs(pytorch_output - mlx_output_torch)
    max_diff = torch.max(output_diff).item()
    mean_diff = torch.mean(output_diff).item()
    
    print(f"   最大差异: {max_diff:.8f}")
    print(f"   平均差异: {mean_diff:.8f}")
    
    # 分析输出统计
    pytorch_stats = {
        'min': torch.min(pytorch_output).item(),
        'max': torch.max(pytorch_output).item(),
        'mean': torch.mean(pytorch_output).item(),
        'std': torch.std(pytorch_output).item()
    }
    
    mlx_stats = {
        'min': torch.min(mlx_output_torch).item(),
        'max': torch.max(mlx_output_torch).item(),
        'mean': torch.mean(mlx_output_torch).item(),
        'std': torch.std(mlx_output_torch).item()
    }
    
    print(f"\n📊 输出统计:")
    print(f"   PyTorch: min={pytorch_stats['min']:.6f}, max={pytorch_stats['max']:.6f}, mean={pytorch_stats['mean']:.6f}, std={pytorch_stats['std']:.6f}")
    print(f"   MLX:     min={mlx_stats['min']:.6f}, max={mlx_stats['max']:.6f}, mean={mlx_stats['mean']:.6f}, std={mlx_stats['std']:.6f}")
    
    # 判断一致性
    if max_diff < 1e-6:
        print(f"   ✅ CFM 输出完全一致")
        return True
    elif max_diff < 0.1:
        print(f"   ⚠️  CFM 输出基本一致 (差异较小)")
        return True
    else:
        print(f"   ❌ CFM 输出不一致 (差异较大)")
        return False

if __name__ == "__main__":
    success = test_cfm_consistency()
    
    if success:
        print(f"\n🎉 测试通过! PyTorch 和 MLX CFM 输出一致")
    else:
        print(f"\n❌ 测试失败! PyTorch 和 MLX CFM 输出不一致")
