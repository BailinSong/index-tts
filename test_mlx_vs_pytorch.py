#!/usr/bin/env python3
"""
对比测试 PyTorch vs MLX Hybrid 模式
"""

import subprocess
import torchaudio
import time

def run_inference(text, voice, use_mlx=False):
    """运行推理并返回结果"""
    cmd = [
        "python", "-m", "indextts.cli",
        text,
        "-v", voice,
        "--force"
    ]
    if use_mlx:
        cmd.append("--mlx")
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start
    
    # 解析输出
    for line in result.stdout.split('\n'):
        if 'Total inference time:' in line:
            inference_time = float(line.split(':')[1].strip().split()[0])
        if 'Generated audio length:' in line:
            audio_length = float(line.split(':')[1].strip().split()[0])
        if 'RTF:' in line:
            rtf = float(line.split(':')[1].strip())
    
    # 加载音频
    wav, sr = torchaudio.load('gen.wav')
    
    return {
        'inference_time': inference_time,
        'audio_length': audio_length,
        'rtf': rtf,
        'wav_std': wav.std().item(),
        'wav_range': (wav.min().item(), wav.max().item()),
        'total_duration': duration
    }

def main():
    text = "今天"
    voice = "examples/voice_01.wav"
    
    print("="*70)
    print("IndexTTS2 性能对比测试")
    print("="*70)
    print(f"文本: {text}")
    print(f"参考音频: {voice}")
    print()
    
    # PyTorch测试
    print(">>> 测试 PyTorch 模式...")
    pt_result = run_inference(text, voice, use_mlx=False)
    subprocess.run(["mv", "gen.wav", "gen_pytorch_test.wav"])
    
    print(f"  推理时间: {pt_result['inference_time']:.2f}s")
    print(f"  音频时长: {pt_result['audio_length']:.2f}s")
    print(f"  RTF: {pt_result['rtf']:.2f}")
    print(f"  音频Std: {pt_result['wav_std']:.4f}")
    print()
    
    # MLX测试
    print(">>> 测试 MLX Hybrid 模式...")
    mlx_result = run_inference(text, voice, use_mlx=True)
    subprocess.run(["mv", "gen.wav", "gen_mlx_test.wav"])
    
    print(f"  推理时间: {mlx_result['inference_time']:.2f}s")
    print(f"  音频时长: {mlx_result['audio_length']:.2f}s")
    print(f"  RTF: {mlx_result['rtf']:.2f}")
    print(f"  音频Std: {mlx_result['wav_std']:.4f}")
    print()
    
    # 对比
    print("="*70)
    print("对比结果")
    print("="*70)
    speedup = (pt_result['inference_time'] - mlx_result['inference_time']) / pt_result['inference_time'] * 100
    rtf_improve = (pt_result['rtf'] - mlx_result['rtf']) / pt_result['rtf'] * 100
    
    print(f"推理时间加速: {speedup:+.1f}%")
    print(f"RTF改进: {rtf_improve:+.1f}%")
    print(f"音频时长差异: {abs(pt_result['audio_length'] - mlx_result['audio_length']):.2f}s")
    print(f"音频质量: PyTorch Std={pt_result['wav_std']:.4f}, MLX Std={mlx_result['wav_std']:.4f}")
    print()
    
    if speedup > 0:
        print("✅ MLX Hybrid模式更快！")
    else:
        print("⚠️  PyTorch模式更快")
    
    print()
    print("生成的音频文件:")
    print(f"  - gen_pytorch_test.wav")
    print(f"  - gen_mlx_test.wav")
    print("  - gen_base.wav (baseline)")

if __name__ == "__main__":
    main()

