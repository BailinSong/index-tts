#!/usr/bin/env python3
"""
使用缓存输入直接测试PyTorch和MLX版本的CFM推理音质差异
"""

import sys
import os
sys.path.append('.')

import torch
import numpy as np
import pickle
import librosa
import soundfile as sf
from indextts.infer_v2 import IndexTTS2
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def load_cached_inputs():
    """加载缓存的CFM输入"""
    
    cache_file = 'cfm_inputs_mlx.pkl'
    if not os.path.exists(cache_file):
        print(f"❌ 缓存文件不存在: {cache_file}")
        return None
    
    print(f"📁 加载缓存文件: {cache_file}")
    with open(cache_file, 'rb') as f:
        cached_data = pickle.load(f)
    
    print("📊 缓存数据内容:")
    for key, value in cached_data.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}, dtype={value.dtype}, device={value.device}")
        else:
            print(f"  {key}: {type(value)} = {value}")
    
    return cached_data

def analyze_cfm_inference_differences():
    """分析CFM推理差异"""
    
    print("=== 使用缓存输入分析CFM推理差异 ===")
    
    # 加载缓存输入
    cached_data = load_cached_inputs()
    if cached_data is None:
        return
    
    # 初始化TTS
    tts = IndexTTS2()
    
    # 提取缓存数据
    cat_condition = cached_data['cat_condition']
    x_lens = cached_data['x_lens']
    ref_mel = cached_data['ref_mel']
    style = cached_data['style']
    diffusion_steps = cached_data['diffusion_steps']
    inference_cfg_rate = cached_data['inference_cfg_rate']
    
    print(f"\n📋 输入数据统计:")
    print(f"  cat_condition: {cat_condition.shape}, min={cat_condition.min():.6f}, max={cat_condition.max():.6f}")
    print(f"  x_lens: {x_lens}")
    print(f"  ref_mel: {ref_mel.shape}, min={ref_mel.min():.6f}, max={ref_mel.max():.6f}")
    print(f"  style: {style.shape}, min={style.min():.6f}, max={style.max():.6f}")
    print(f"  diffusion_steps: {diffusion_steps}")
    print(f"  inference_cfg_rate: {inference_cfg_rate}")
    
    # 确保所有张量在正确设备上
    device = 'mps'
    cat_condition = cat_condition.to(device)
    x_lens = x_lens.to(device)
    ref_mel = ref_mel.to(device)
    style = style.to(device)
    
    # 构造CFM输入
    mu = cat_condition  # (batch, seq_len, 512)
    prompt = ref_mel    # (batch, 80, prompt_len)
    
    # 调整prompt序列长度匹配mu的序列长度
    target_seq_len = mu.shape[1]  # 523
    if prompt.shape[-1] != target_seq_len:
        if prompt.shape[-1] < target_seq_len:
            # 重复prompt到目标长度
            repeat_factor = target_seq_len // prompt.shape[-1] + 1
            prompt = prompt.repeat(1, 1, repeat_factor)[:, :, :target_seq_len]
        else:
            # 截断prompt到目标长度
            prompt = prompt[:, :, :target_seq_len]
    
    # prompt现在应该是 (batch, 80, seq_len) 格式
    
    # 生成随机噪声x
    B, T = mu.size(0), mu.size(1)
    torch.manual_seed(42)
    x = torch.randn(B, 80, T, device=device)
    
    print(f"\n🔧 调整后的输入:")
    print(f"  mu: {mu.shape}")
    print(f"  prompt: {prompt.shape}")
    print(f"  x: {x.shape}")
    print(f"  x_lens: {x_lens}")
    print(f"  style: {style.shape}")
    
    # 测试PyTorch CFM
    print(f"\n🔥 测试PyTorch CFM推理...")
    try:
        pytorch_cfm = tts.s2mel.models.cfm
        pytorch_output = pytorch_cfm.inference(
            mu=mu, x_lens=x_lens, prompt=prompt, style=style, f0=None,
            n_timesteps=diffusion_steps, temperature=1.0, inference_cfg_rate=inference_cfg_rate,
            unified_random=tts.unified_random
        )
        print(f"✅ PyTorch CFM输出: {pytorch_output.shape}")
        print(f"   范围: [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]")
        print(f"   均值: {pytorch_output.mean():.6f}, 标准差: {pytorch_output.std():.6f}")
    except Exception as e:
        print(f"❌ PyTorch CFM失败: {e}")
        return
    
    # 测试MLX CFM
    print(f"\n🔥 测试MLX CFM推理...")
    try:
        # 转换为MLX格式
        mu_mlx = torch_to_mlx(mu)
        x_lens_mlx = torch_to_mlx(x_lens)
        prompt_mlx = torch_to_mlx(prompt)
        style_mlx = torch_to_mlx(style)
        
        # 确保MLX CFM已初始化
        if tts.mlx_s2mel_cfm is None:
            print("🔧 初始化MLX CFM...")
            from indextts.s2mel.modules.mlx_cfm import MLXCFM
            tts.mlx_s2mel_cfm = MLXCFM(tts.cfg.s2mel)
            tts.mlx_s2mel_cfm.load_weights_from_pytorch(tts.s2mel.models.cfm.state_dict())
        
        mlx_cfm = tts.mlx_s2mel_cfm
        mlx_output = mlx_cfm.inference(
            mu=mu_mlx, x_lens=x_lens_mlx, prompt=prompt_mlx, style=style_mlx, f0=None,
            n_timesteps=diffusion_steps, temperature=1.0, inference_cfg_rate=inference_cfg_rate,
            unified_random=tts.unified_random
        )
        
        # 转换回PyTorch格式
        mlx_output_torch = mlx_to_torch(mlx_output, device=device)
        
        print(f"✅ MLX CFM输出: {mlx_output_torch.shape}")
        print(f"   范围: [{mlx_output_torch.min():.6f}, {mlx_output_torch.max():.6f}]")
        print(f"   均值: {mlx_output_torch.mean():.6f}, 标准差: {mlx_output_torch.std():.6f}")
    except Exception as e:
        print(f"❌ MLX CFM失败: {e}")
        return
    
    # 分析输出差异
    print(f"\n📊 输出差异分析:")
    
    # 基本统计差异
    diff = torch.abs(pytorch_output - mlx_output_torch)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    std_diff = diff.std().item()
    
    print(f"  最大差异: {max_diff:.6f}")
    print(f"  平均差异: {mean_diff:.6f}")
    print(f"  差异标准差: {std_diff:.6f}")
    
    # 相对差异
    rel_diff = diff / (torch.abs(pytorch_output) + 1e-8)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    print(f"  最大相对差异: {max_rel_diff:.6f}")
    print(f"  平均相对差异: {mean_rel_diff:.6f}")
    
    # 频谱分析
    print(f"\n🎵 频谱分析:")
    
    # 转换为numpy进行频谱分析
    pytorch_np = pytorch_output.cpu().numpy()
    mlx_np = mlx_output_torch.cpu().numpy()
    
    # 计算频谱
    stft_pytorch = librosa.stft(pytorch_np.flatten(), n_fft=1024, hop_length=256)
    stft_mlx = librosa.stft(mlx_np.flatten(), n_fft=1024, hop_length=256)
    
    mag_pytorch = np.abs(stft_pytorch)
    mag_mlx = np.abs(stft_mlx)
    
    # 频谱差异
    mag_diff = np.mean(np.abs(mag_pytorch - mag_mlx))
    print(f"  幅度谱差异: {mag_diff:.6f}")
    
    # 计算信噪比
    signal_power = np.mean(pytorch_np ** 2)
    noise_power = np.mean((pytorch_np - mlx_np) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    print(f"  信噪比: {snr:.2f} dB")
    
    # 保存输出用于进一步分析
    print(f"\n💾 保存输出文件:")
    
    # 保存为音频文件
    sf.write("cfm_output_pytorch.wav", pytorch_np.flatten(), 22050)
    sf.write("cfm_output_mlx.wav", mlx_np.flatten(), 22050)
    print("  ✅ 音频文件已保存: cfm_output_pytorch.wav, cfm_output_mlx.wav")
    
    # 保存数值数据
    output_data = {
        'pytorch_output': pytorch_output.cpu(),
        'mlx_output': mlx_output_torch.cpu(),
        'diff': diff.cpu(),
        'snr': snr,
        'mag_diff': mag_diff,
        'max_diff': max_diff,
        'mean_diff': mean_diff
    }
    
    with open('cfm_inference_analysis.pkl', 'wb') as f:
        pickle.dump(output_data, f)
    print("  ✅ 分析数据已保存: cfm_inference_analysis.pkl")
    
    # 总结
    print(f"\n📋 总结:")
    if snr > 30:
        print("  ✅ 音质差异很小，MLX实现良好")
    elif snr > 20:
        print("  ⚠️  音质差异中等，建议进一步优化")
    else:
        print("  ❌ 音质差异较大，需要修复MLX实现")
    
    return output_data

def main():
    """主函数"""
    
    print("开始使用缓存输入分析CFM推理音质差异...")
    
    # 分析CFM推理差异
    result = analyze_cfm_inference_differences()
    
    if result:
        print(f"\n🎉 分析完成！")
        print(f"信噪比: {result['snr']:.2f} dB")
        print(f"最大差异: {result['max_diff']:.6f}")
        print(f"平均差异: {result['mean_diff']:.6f}")

if __name__ == "__main__":
    main()
