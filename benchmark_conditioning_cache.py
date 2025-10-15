#!/usr/bin/env python3
"""
Conditioning缓存性能对比测试
对比缓存命中vs未命中的实际性能差异
"""

from indextts.infer_v2 import IndexTTS2
import time
import re

def extract_timing(tts_instance):
    """从最后一次推理中提取时间数据（如果有的话）"""
    # 这个方法假设我们能访问内部状态，实际可能需要从输出解析
    return {}

print("=" * 80)
print("Conditioning缓存性能对比测试")
print("=" * 80)
print("\n测试配置:")
print("  Voice: examples/zh_vo_Main_Linaxita_2_4_24_6.wav")
print("  Seed: 42 (固定)")
print("  Num runs: 3次缓存命中测试")
print()

# 初始化
print("初始化 IndexTTS2...")
tts = IndexTTS2(use_mlx=True)

voice = "examples/zh_vo_Main_Linaxita_2_4_24_6.wav"

# ============================================================================
# 阶段1: 缓存未命中（baseline）
# ============================================================================
print("\n" + "=" * 80)
print("阶段1: 缓存未命中测试（Baseline）")
print("=" * 80)

print("\n[Baseline] 首次推理 - 文本='今天天气真不错'")
print("-" * 80)
t0 = time.time()
tts.infer(
    spk_audio_prompt=voice,
    text="今天天气真不错",
    output_path="gen_baseline.wav",
    num_beams=1,
    seed=42
)
t_baseline = time.time() - t0
print(f"✅ Baseline完成: {t_baseline:.2f}s")

# ============================================================================
# 阶段2: 缓存命中（优化后）
# ============================================================================
print("\n" + "=" * 80)
print("阶段2: 缓存命中测试（优化后）")
print("=" * 80)

cache_times = []

for i in range(3):
    text = ["你好世界", "早上好", "晚安"][i]
    seed = 43 + i
    
    print(f"\n[Run {i+1}/3] 相同voice，文本='{text}'")
    print("-" * 80)
    t0 = time.time()
    tts.infer(
        spk_audio_prompt=voice,
        text=text,
        output_path=f"gen_cached_{i+1}.wav",
        num_beams=1,
        seed=seed
    )
    t = time.time() - t0
    cache_times.append(t)
    print(f"✅ Run {i+1}完成: {t:.2f}s")

# ============================================================================
# 阶段3: 性能对比
# ============================================================================
print("\n" + "=" * 80)
print("性能对比结果")
print("=" * 80)

avg_cached = sum(cache_times) / len(cache_times)
min_cached = min(cache_times)
max_cached = max(cache_times)

print(f"\n📊 Baseline（缓存未命中）:")
print(f"   时间: {t_baseline:.2f}s")

print(f"\n📊 优化后（缓存命中，3次运行）:")
print(f"   Run 1: {cache_times[0]:.2f}s")
print(f"   Run 2: {cache_times[1]:.2f}s")
print(f"   Run 3: {cache_times[2]:.2f}s")
print(f"   平均: {avg_cached:.2f}s")
print(f"   最快: {min_cached:.2f}s")
print(f"   最慢: {max_cached:.2f}s")

# 计算提速
if avg_cached < t_baseline:
    speedup_abs = t_baseline - avg_cached
    speedup_pct = speedup_abs / t_baseline * 100
    
    print(f"\n✅ 性能提升:")
    print(f"   绝对提速: {speedup_abs:.2f}s")
    print(f"   相对提速: {speedup_pct:.1f}%")
    print(f"   RTF改善: {t_baseline/2.53:.2f} → {avg_cached/2.53:.2f}")
    
    print(f"\n📈 批量生成收益（100句话，相同voice）:")
    total_no_cache = t_baseline * 100
    total_with_cache = t_baseline + avg_cached * 99
    saving = total_no_cache - total_with_cache
    print(f"   无缓存: {total_no_cache:.0f}s = {total_no_cache/60:.1f}分钟")
    print(f"   有缓存: {total_with_cache:.0f}s = {total_with_cache/60:.1f}分钟")
    print(f"   节省: {saving:.0f}s = {saving/60:.1f}分钟 ({saving/total_no_cache*100:.1f}%)")
else:
    print(f"\n❌ 未观察到提速")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)

# 保存结果
with open('CACHE_PERFORMANCE_TEST.md', 'w') as f:
    f.write("# Conditioning缓存性能测试结果\n\n")
    f.write(f"## 测试配置\n\n")
    f.write(f"- Voice: {voice}\n")
    f.write(f"- Seed: 42 (baseline), 43-45 (cached runs)\n")
    f.write(f"- 测试次数: 3次缓存命中\n\n")
    f.write(f"## 性能数据\n\n")
    f.write(f"| 运行 | 时间 | 缓存状态 |\n")
    f.write(f"|------|------|----------|\n")
    f.write(f"| Baseline | {t_baseline:.2f}s | ❌ 未命中 |\n")
    for i, t in enumerate(cache_times, 1):
        f.write(f"| Run {i} | {t:.2f}s | ✅ 命中 |\n")
    f.write(f"| **平均（缓存）** | **{avg_cached:.2f}s** | ✅ 命中 |\n\n")
    
    if avg_cached < t_baseline:
        speedup_abs = t_baseline - avg_cached
        speedup_pct = speedup_abs / t_baseline * 100
        f.write(f"## 性能提升\n\n")
        f.write(f"- 绝对提速: **{speedup_abs:.2f}s**\n")
        f.write(f"- 相对提速: **{speedup_pct:.1f}%**\n")
        f.write(f"- RTF: {t_baseline/2.53:.2f} → {avg_cached/2.53:.2f}\n\n")

print("\n✅ 结果已保存到: CACHE_PERFORMANCE_TEST.md")

