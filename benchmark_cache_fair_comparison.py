#!/usr/bin/env python3
"""
公平对比：使用相同长度的文本测试缓存效果
"""

from indextts.infer_v2 import IndexTTS2
import time

print("=" * 80)
print("Conditioning缓存性能测试（公平对比 - 相同文本长度）")
print("=" * 80)

# 初始化
tts = IndexTTS2(use_mlx=True)

voice = "examples/zh_vo_Main_Linaxita_2_4_24_6.wav"
# 使用相同长度的文本（7个字）
texts = ["今天天气真不错", "你们好吗朋友", "早上好啊大家"]
seeds = [42, 43, 44]

results = []

for i, (text, seed) in enumerate(zip(texts, seeds)):
    print(f"\n[Run {i+1}/3] 文本='{text}' (seed={seed})")
    print("-" * 80)
    
    t0 = time.time()
    tts.infer(
        spk_audio_prompt=voice,
        text=text,
        output_path=f"gen_fair_{i+1}.wav",
        num_beams=1,
        seed=seed
    )
    t = time.time() - t0
    results.append(t)
    
    cache_status = "❌ 未命中" if i == 0 else "✅ 命中"
    print(f"✅ Run {i+1}完成: {t:.2f}s ({cache_status})")

# 分析
print("\n" + "=" * 80)
print("性能分析")
print("=" * 80)

baseline = results[0]
cached_avg = sum(results[1:]) / len(results[1:])

print(f"\nBaseline (缓存未命中):")
print(f"  Run 1: {results[0]:.2f}s")

print(f"\n优化后 (缓存命中):")
print(f"  Run 2: {results[1]:.2f}s")
print(f"  Run 3: {results[2]:.2f}s")
print(f"  平均: {cached_avg:.2f}s")

if cached_avg < baseline:
    speedup = (baseline - cached_avg) / baseline * 100
    print(f"\n✅ 性能提升:")
    print(f"   绝对: {baseline - cached_avg:.2f}s")
    print(f"   相对: {speedup:.1f}%")
    print(f"   RTF: ~{baseline/2.53:.2f} → ~{cached_avg/2.53:.2f}")

# 保存结果
with open('CACHE_FAIR_COMPARISON.md', 'w') as f:
    f.write("# Conditioning缓存性能测试（公平对比）\n\n")
    f.write("## 测试配置\n\n")
    f.write("- 文本长度: 相同（7个字）\n")
    f.write(f"- 文本: {texts}\n")
    f.write(f"- Voice: {voice}\n\n")
    f.write("## 结果\n\n")
    f.write(f"| Run | 文本 | 时间 | 缓存 |\n")
    f.write(f"|-----|------|------|------|\n")
    for i, (text, t) in enumerate(zip(texts, results), 1):
        cache = "❌ 未命中" if i == 1 else "✅ 命中"
        f.write(f"| {i} | {text} | {t:.2f}s | {cache} |\n")
    if cached_avg < baseline:
        speedup = (baseline - cached_avg) / baseline * 100
        f.write(f"\n## 性能提升\n\n")
        f.write(f"- 提速: **{speedup:.1f}%**\n")
        f.write(f"- 节省: **{baseline - cached_avg:.2f}s**\n")

print("\n✅ 结果已保存")

