#!/usr/bin/env python3
"""
严格按照生产环境调用方式分析PyTorch和MLX版本的音质差异
参考benchmark_v1_baseline.py的调用方式
"""

import sys
import os
sys.path.append('.')

import torch
import numpy as np
import librosa
import soundfile as sf
import time
from indextts.infer_v2 import IndexTTS2

def analyze_production_inference_differences():
    """分析生产环境推理的音质差异"""
    
    print("=== 严格按照生产环境调用方式分析音质差异 ===")
    
    # 配置（参考benchmark_v1_baseline.py）
    CONFIG = {
        "text": "今天天气真不错",
        "voice": "examples/zh_vo_Main_Linaxita_2_4_24_6.wav",
        "seed": 42,
        "num_beams": 1,
    }
    
    print(f"\n📋 测试配置:")
    print(f"  文本: '{CONFIG['text']}'")
    print(f"  Voice: {CONFIG['voice']}")
    print(f"  Seed: {CONFIG['seed']}")
    print(f"  Num beams: {CONFIG['num_beams']}")
    
    # 测试PyTorch版本
    print(f"\n🔥 测试PyTorch版本...")
    
    # 初始化PyTorch TTS
    tts_pytorch = IndexTTS2(use_mlx=False)
    
    # 设置随机种子
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # 生成PyTorch版本音频
    pytorch_start = time.time()
    pytorch_output = tts_pytorch.infer(
        spk_audio_prompt=CONFIG['voice'],
        text=CONFIG['text'],
        output_path="production_pytorch.wav",
        num_beams=CONFIG['num_beams'],
        seed=CONFIG['seed']
    )
    pytorch_time = time.time() - pytorch_start
    
    print(f"✅ PyTorch版本完成: {pytorch_time:.2f}s")
    
    # 测试MLX版本
    print(f"\n🔥 测试MLX版本...")
    
    # 初始化MLX TTS
    tts_mlx = IndexTTS2(use_mlx=True)
    
    # 设置随机种子
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # 生成MLX版本音频
    mlx_start = time.time()
    mlx_output = tts_mlx.infer(
        spk_audio_prompt=CONFIG['voice'],
        text=CONFIG['text'],
        output_path="production_mlx.wav",
        num_beams=CONFIG['num_beams'],
        seed=CONFIG['seed']
    )
    mlx_time = time.time() - mlx_start
    
    print(f"✅ MLX版本完成: {mlx_time:.2f}s")
    
    # 分析音频文件
    print(f"\n📊 分析生成的音频文件...")
    
    # 加载音频文件
    pytorch_audio, sr = librosa.load("production_pytorch.wav", sr=22050)
    mlx_audio, sr = librosa.load("production_mlx.wav", sr=22050)
    
    print(f"  PyTorch音频: {len(pytorch_audio)} 样本, {len(pytorch_audio)/sr:.2f} 秒")
    print(f"  MLX音频: {len(mlx_audio)} 样本, {len(mlx_audio)/sr:.2f} 秒")
    
    # 确保长度一致
    min_len = min(len(pytorch_audio), len(mlx_audio))
    pytorch_audio = pytorch_audio[:min_len]
    mlx_audio = mlx_audio[:min_len]
    
    # 基本统计
    print(f"\n📈 基本统计:")
    print(f"  PyTorch: min={pytorch_audio.min():.6f}, max={pytorch_audio.max():.6f}, mean={pytorch_audio.mean():.6f}")
    print(f"  MLX: min={mlx_audio.min():.6f}, max={mlx_audio.max():.6f}, mean={mlx_audio.mean():.6f}")
    
    # 计算差异
    diff = np.abs(pytorch_audio - mlx_audio)
    max_diff = diff.max()
    mean_diff = diff.mean()
    std_diff = diff.std()
    
    print(f"\n🔍 差异分析:")
    print(f"  最大差异: {max_diff:.6f}")
    print(f"  平均差异: {mean_diff:.6f}")
    print(f"  差异标准差: {std_diff:.6f}")
    
    # 计算信噪比
    signal_power = np.mean(pytorch_audio ** 2)
    noise_power = np.mean((pytorch_audio - mlx_audio) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    print(f"  信噪比: {snr:.2f} dB")
    
    # 频谱分析
    print(f"\n🎵 频谱分析:")
    
    # 计算频谱
    stft_pytorch = librosa.stft(pytorch_audio, n_fft=1024, hop_length=256)
    stft_mlx = librosa.stft(mlx_audio, n_fft=1024, hop_length=256)
    
    mag_pytorch = np.abs(stft_pytorch)
    mag_mlx = np.abs(stft_mlx)
    
    # 频谱差异
    mag_diff = np.mean(np.abs(mag_pytorch - mag_mlx))
    print(f"  幅度谱差异: {mag_diff:.6f}")
    
    # 频谱质心
    centroid_pytorch = librosa.feature.spectral_centroid(y=pytorch_audio, sr=sr)[0]
    centroid_mlx = librosa.feature.spectral_centroid(y=mlx_audio, sr=sr)[0]
    centroid_diff = np.mean(np.abs(centroid_pytorch - centroid_mlx))
    print(f"  频谱质心差异: {centroid_diff:.2f} Hz")
    
    # 频谱带宽
    bandwidth_pytorch = librosa.feature.spectral_bandwidth(y=pytorch_audio, sr=sr)[0]
    bandwidth_mlx = librosa.feature.spectral_bandwidth(y=mlx_audio, sr=sr)[0]
    bandwidth_diff = np.mean(np.abs(bandwidth_pytorch - bandwidth_mlx))
    print(f"  频谱带宽差异: {bandwidth_diff:.2f} Hz")
    
    # 性能对比
    print(f"\n⚡ 性能对比:")
    print(f"  PyTorch时间: {pytorch_time:.2f}s")
    print(f"  MLX时间: {mlx_time:.2f}s")
    print(f"  速度提升: {pytorch_time/mlx_time:.2f}x")
    
    # 总结
    print(f"\n📋 总结:")
    if snr > 30:
        print("  ✅ 音质差异很小，MLX实现良好")
    elif snr > 20:
        print("  ⚠️  音质差异中等，建议进一步优化")
    else:
        print("  ❌ 音质差异较大，需要修复MLX实现")
    
    if pytorch_time > mlx_time:
        print(f"  ✅ MLX版本速度提升: {pytorch_time/mlx_time:.2f}x")
    else:
        print(f"  ⚠️  MLX版本速度未提升")
    
    # 保存分析结果
    analysis_result = {
        'pytorch_time': pytorch_time,
        'mlx_time': mlx_time,
        'snr': snr,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'mag_diff': mag_diff,
        'centroid_diff': centroid_diff,
        'bandwidth_diff': bandwidth_diff,
        'config': CONFIG
    }
    
    import pickle
    with open('production_analysis.pkl', 'wb') as f:
        pickle.dump(analysis_result, f)
    
    print(f"\n💾 分析结果已保存到: production_analysis.pkl")
    
    return analysis_result

def main():
    """主函数"""
    
    print("开始严格按照生产环境调用方式分析音质差异...")
    
    # 分析生产环境推理差异
    result = analyze_production_inference_differences()
    
    if result:
        print(f"\n🎉 分析完成！")
        print(f"信噪比: {result['snr']:.2f} dB")
        print(f"最大差异: {result['max_diff']:.6f}")
        print(f"平均差异: {result['mean_diff']:.6f}")
        print(f"速度提升: {result['pytorch_time']/result['mlx_time']:.2f}x")

if __name__ == "__main__":
    main()
