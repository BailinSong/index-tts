#!/usr/bin/env python3
"""
测试 torch.compile 对 Length Regulator 的加速效果
"""
import sys
sys.path.insert(0, '/Users/bailin/index-tts')

import torch
import time

def test_compile():
    print("="*80)
    print("🚀 测试 torch.compile 优化 Length Regulator")
    print("="*80)
    
    from indextts.infer_v2 import IndexTTS2
    
    # Load model
    print("\n加载模型...")
    model = IndexTTS2(device="mps", use_mlx=True)
    
    # Test sentences
    test_texts = [
        "今天天气真不错",
        "我们一起去看电影吧",
        "人工智能技术发展迅速",
        "测试一下修复效果如何"
    ]
    
    # Test without compile
    print("\n" + "="*80)
    print("测试 1: 无优化 (baseline)")
    print("="*80)
    
    times_baseline = []
    for text in test_texts:
        t0 = time.perf_counter()
        wav = model.infer(
            text,
            spk_audio_prompt="examples/voice_01.wav",
            verbose=False
        )
        t1 = time.perf_counter()
        times_baseline.append(t1 - t0)
        print(f"  '{text}': {t1-t0:.2f}s")
    
    baseline_mean = sum(times_baseline) / len(times_baseline)
    print(f"\n平均时间: {baseline_mean:.2f}s")
    
    # Test with compile
    print("\n" + "="*80)
    print("测试 2: torch.compile 优化")
    print("="*80)
    
    # Apply compile to length_regulator
    print("\n应用 torch.compile...")
    try:
        model.s2mel.models['length_regulator'] = torch.compile(
            model.s2mel.models['length_regulator'],
            mode='reduce-overhead'  # Focus on reducing overhead
        )
        print("✅ torch.compile 应用成功")
    except Exception as e:
        print(f"❌ torch.compile 失败: {e}")
        return
    
    # Warmup
    print("\nWarming up...")
    for _ in range(2):
        _ = model.infer(
            "测试",
            spk_audio_prompt="examples/voice_01.wav",
            verbose=False
        )
    
    times_compiled = []
    for text in test_texts:
        t0 = time.perf_counter()
        wav = model.infer(
            text,
            spk_audio_prompt="examples/voice_01.wav",
            verbose=False
        )
        t1 = time.perf_counter()
        times_compiled.append(t1 - t0)
        print(f"  '{text}': {t1-t0:.2f}s")
    
    compiled_mean = sum(times_compiled) / len(times_compiled)
    print(f"\n平均时间: {compiled_mean:.2f}s")
    
    # Summary
    print("\n" + "="*80)
    print("📊 对比总结")
    print("="*80)
    
    speedup = baseline_mean / compiled_mean
    improvement = (baseline_mean - compiled_mean) / baseline_mean * 100
    
    print(f"\nBaseline:  {baseline_mean:.2f}s")
    print(f"Compiled:  {compiled_mean:.2f}s")
    print(f"Speedup:   {speedup:.2f}x")
    print(f"Improvement: {improvement:.1f}%")
    
    if speedup > 1.2:
        print("\n✅ 显著加速！torch.compile 有效")
    elif speedup > 1.05:
        print("\n⚠️  有小幅加速")
    else:
        print("\n❌ 加速不明显")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    test_compile()


