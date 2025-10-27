#!/usr/bin/env python3
"""
分析现有音频文件的音质差异
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from scipy import signal
import os

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
    
    # 计算频谱质心
    centroid1 = librosa.feature.spectral_centroid(y=audio1, sr=sr)[0]
    centroid2 = librosa.feature.spectral_centroid(y=audio2, sr=sr)[0]
    centroid_diff = np.mean(np.abs(centroid1 - centroid2))
    
    # 计算频谱带宽
    bandwidth1 = librosa.feature.spectral_bandwidth(y=audio1, sr=sr)[0]
    bandwidth2 = librosa.feature.spectral_bandwidth(y=audio2, sr=sr)[0]
    bandwidth_diff = np.mean(np.abs(bandwidth1 - bandwidth2))
    
    print(f"频谱质心差异: {centroid_diff:.2f} Hz")
    print(f"频谱带宽差异: {bandwidth_diff:.2f} Hz")
    
    return {
        'mag_diff': mag_diff,
        'phase_diff': phase_diff,
        'snr': snr,
        'centroid_diff': centroid_diff,
        'bandwidth_diff': bandwidth_diff,
        'mag1': mag1,
        'mag2': mag2,
        'phase1': phase1,
        'phase2': phase2
    }

def analyze_existing_audio_files():
    """分析现有的音频文件"""
    
    print("=== 分析现有音频文件 ===")
    
    # 查找PyTorch和MLX版本的音频文件
    pytorch_files = []
    mlx_files = []
    
    for file in os.listdir('.'):
        if file.endswith('.wav'):
            if 'pytorch' in file.lower():
                pytorch_files.append(file)
            elif 'mlx' in file.lower():
                mlx_files.append(file)
    
    print(f"找到PyTorch文件: {pytorch_files}")
    print(f"找到MLX文件: {mlx_files}")
    
    # 分析一些典型的文件对
    test_pairs = [
        ('test_pytorch.wav', 'test_mlx.wav'),
        ('test_pytorch_hello.wav', 'test_mlx_hello.wav'),
        ('test_pytorch_weather.wav', 'test_mlx_weather.wav'),
        ('test_output_pytorch.wav', 'test_output_mlx.wav')
    ]
    
    results = []
    
    for pytorch_file, mlx_file in test_pairs:
        if os.path.exists(pytorch_file) and os.path.exists(mlx_file):
            print(f"\n分析文件对: {pytorch_file} vs {mlx_file}")
            
            try:
                # 加载音频
                audio1, sr1 = librosa.load(pytorch_file, sr=22050)
                audio2, sr2 = librosa.load(mlx_file, sr=22050)
                
                # 确保长度一致
                min_len = min(len(audio1), len(audio2))
                audio1 = audio1[:min_len]
                audio2 = audio2[:min_len]
                
                # 分析频谱差异
                result = analyze_audio_spectral_differences(
                    audio1, audio2, sr1, 
                    pytorch_file.replace('.wav', ''), 
                    mlx_file.replace('.wav', '')
                )
                
                results.append({
                    'pytorch_file': pytorch_file,
                    'mlx_file': mlx_file,
                    'snr': result['snr'],
                    'mag_diff': result['mag_diff'],
                    'phase_diff': result['phase_diff'],
                    'centroid_diff': result['centroid_diff'],
                    'bandwidth_diff': result['bandwidth_diff']
                })
                
            except Exception as e:
                print(f"分析失败: {e}")
    
    return results

def analyze_specific_files():
    """分析特定的文件"""
    
    print("\n=== 分析特定文件 ===")
    
    # 分析最新的测试文件
    specific_files = [
        'test_pytorch_fresh.wav',
        'test_mlx_fixed_random_seed_final.wav',
        'rewritten_vs_pytorch_test.wav',
        'test_output_mlx_pytorch_cfm.wav'
    ]
    
    for file in specific_files:
        if os.path.exists(file):
            print(f"\n分析文件: {file}")
            
            try:
                audio, sr = librosa.load(file, sr=22050)
                
                # 基本统计
                print(f"  长度: {len(audio)} 样本 ({len(audio)/sr:.2f} 秒)")
                print(f"  采样率: {sr} Hz")
                print(f"  幅度范围: [{audio.min():.4f}, {audio.max():.4f}]")
                print(f"  RMS: {np.sqrt(np.mean(audio**2)):.4f}")
                
                # 频谱分析
                stft = librosa.stft(audio, n_fft=1024, hop_length=256)
                mag = np.abs(stft)
                
                print(f"  频谱范围: [{mag.min():.6f}, {mag.max():.6f}]")
                print(f"  平均频谱能量: {np.mean(mag):.6f}")
                
                # 检查是否有异常值
                if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
                    print("  ⚠️  发现NaN或Inf值")
                
                if np.max(np.abs(audio)) > 1.0:
                    print("  ⚠️  音频幅度超过1.0，可能被削波")
                
            except Exception as e:
                print(f"分析失败: {e}")

def main():
    """主函数"""
    
    print("开始分析音频文件音质差异...")
    
    # 1. 分析现有文件对
    results = analyze_existing_audio_files()
    
    # 2. 分析特定文件
    analyze_specific_files()
    
    # 3. 总结
    if results:
        print(f"\n=== 总结 ===")
        avg_snr = np.mean([r['snr'] for r in results])
        avg_mag_diff = np.mean([r['mag_diff'] for r in results])
        avg_phase_diff = np.mean([r['phase_diff'] for r in results])
        
        print(f"平均信噪比: {avg_snr:.2f} dB")
        print(f"平均幅度谱差异: {avg_mag_diff:.6f}")
        print(f"平均相位谱差异: {avg_phase_diff:.6f}")
        
        if avg_snr < 20:
            print("⚠️  音质差异较大，需要优化MLX实现")
        elif avg_snr < 30:
            print("⚠️  音质差异中等，建议进一步优化")
        else:
            print("✅ 音质差异在可接受范围内")

if __name__ == "__main__":
    main()
