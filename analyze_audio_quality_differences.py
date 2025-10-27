#!/usr/bin/env python3
"""
分析MLX和PyTorch CFM推理的音质差异
"""

import sys
import os
sys.path.append('.')

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa
from indextts.infer_v2 import IndexTTS2
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def analyze_audio_spectral_differences(audio1, audio2, sr=22050, name1="PyTorch", name2="MLX"):
    """分析两个音频的频谱差异"""
    
    # 计算频谱
    stft1 = librosa.stft(audio1, n_fft=1024, hop_length=256)
    stft2 = librosa.stft(audio2, n_fft=1024, hop_length=256)
    
    # 幅度谱
    mag1 = np.abs(stft1)
    mag2 = np.abs(stft2)
    
    # 相位谱
    phase1 = np.angle(stft1)
    phase2 = np.angle(stft2)
    
    # 计算差异
    mag_diff = np.mean(np.abs(mag1 - mag2))
    phase_diff = np.mean(np.abs(phase1 - phase2))
    
    # 计算信噪比
    signal_power = np.mean(audio1 ** 2)
    noise_power = np.mean((audio1 - audio2) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    print(f"\n=== {name1} vs {name2} 频谱分析 ===")
    print(f"幅度谱差异: {mag_diff:.6f}")
    print(f"相位谱差异: {phase_diff:.6f}")
    print(f"信噪比: {snr:.2f} dB")
    
    return {
        'mag_diff': mag_diff,
        'phase_diff': phase_diff,
        'snr': snr,
        'mag1': mag1,
        'mag2': mag2,
        'phase1': phase1,
        'phase2': phase2
    }

def analyze_cfm_step_differences():
    """分析CFM推理过程中每一步的差异"""
    
    print("=== 分析CFM推理步骤差异 ===")
    
    # 初始化TTS
    tts = IndexTTS2()
    
    # 设置测试参数
    text = "你好，这是一个测试音频。"
    spk_audio_prompt = "examples/voice_01.wav"
    
    # 生成PyTorch版本
    print("生成PyTorch版本...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        wav_torch = tts.infer(
            spk_audio_prompt=spk_audio_prompt,
            text=text,
            output_path="temp_torch.wav",
            n_timesteps=25,
            inference_cfg_rate=0.5
        )
        print(f"PyTorch版本生成成功: {wav_torch.shape}")
    except Exception as e:
        print(f"PyTorch版本生成失败: {e}")
        return
    
    # 生成MLX版本
    print("生成MLX版本...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        wav_mlx = tts.infer(
            spk_audio_prompt=spk_audio_prompt,
            text=text,
            output_path="temp_mlx.wav",
            use_mlx=True,
            n_timesteps=25,
            inference_cfg_rate=0.5
        )
        print(f"MLX版本生成成功: {wav_mlx.shape}")
    except Exception as e:
        print(f"MLX版本生成失败: {e}")
        return
    
    # 转换为numpy数组进行分析
    if isinstance(wav_torch, torch.Tensor):
        wav_torch_np = wav_torch.cpu().numpy()
    else:
        wav_torch_np = wav_torch
    
    if isinstance(wav_mlx, torch.Tensor):
        wav_mlx_np = wav_mlx.cpu().numpy()
    else:
        wav_mlx_np = wav_mlx
    
    # 确保长度一致
    min_len = min(len(wav_torch_np), len(wav_mlx_np))
    wav_torch_np = wav_torch_np[:min_len]
    wav_mlx_np = wav_mlx_np[:min_len]
    
    # 分析频谱差异
    spectral_analysis = analyze_audio_spectral_differences(
        wav_torch_np, wav_mlx_np, 
        name1="PyTorch", name2="MLX"
    )
    
    # 保存音频文件
    import soundfile as sf
    sf.write("audio_torch.wav", wav_torch_np, 22050)
    sf.write("audio_mlx.wav", wav_mlx_np, 22050)
    print("音频文件已保存: audio_torch.wav, audio_mlx.wav")
    
    return spectral_analysis

def analyze_numerical_precision():
    """分析数值精度差异"""
    
    print("\n=== 数值精度分析 ===")
    
    # 测试不同精度下的计算
    test_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    for val in test_values:
        # PyTorch计算
        torch_val = torch.tensor(val, dtype=torch.float32)
        torch_result = torch.sin(torch_val) + torch.cos(torch_val)
        
        # MLX计算
        mlx_val = torch_to_mlx(torch_val)
        mlx_result = mlx_to_torch(mlx_val.sin() + mlx_val.cos())
        
        diff = torch.abs(torch_result - mlx_result).item()
        print(f"值 {val}: PyTorch={torch_result.item():.8f}, MLX={mlx_result.item():.8f}, 差异={diff:.2e}")

def analyze_random_generation():
    """分析随机数生成差异"""
    
    print("\n=== 随机数生成分析 ===")
    
    # 设置相同的种子
    torch.manual_seed(42)
    
    # PyTorch随机数
    torch_rand = torch.randn(10)
    print(f"PyTorch随机数: {torch_rand[:5].tolist()}")
    
    # MLX随机数
    import mlx.core as mx
    mx.random.seed(42)
    mlx_rand = mx.random.normal((10,))
    mlx_rand_torch = mlx_to_torch(mlx_rand)
    print(f"MLX随机数: {mlx_rand_torch[:5].tolist()}")
    
    # 计算差异
    diff = torch.abs(torch_rand - mlx_rand_torch)
    print(f"随机数差异: {diff.mean().item():.6f}")

def main():
    """主函数"""
    
    print("开始分析MLX和PyTorch CFM推理的音质差异...")
    
    # 1. 分析数值精度
    analyze_numerical_precision()
    
    # 2. 分析随机数生成
    analyze_random_generation()
    
    # 3. 分析CFM推理差异
    spectral_analysis = analyze_cfm_step_differences()
    
    if spectral_analysis:
        print(f"\n=== 总结 ===")
        print(f"信噪比: {spectral_analysis['snr']:.2f} dB")
        if spectral_analysis['snr'] < 20:
            print("⚠️  音质差异较大，需要优化")
        else:
            print("✅ 音质差异在可接受范围内")

if __name__ == "__main__":
    main()
