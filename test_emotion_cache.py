#!/usr/bin/env python3
"""
测试情感文本缓存功能
"""
import time
from indextts.infer_v2 import IndexTTS2

print("=" * 70)
print("测试情感文本缓存功能")
print("=" * 70)

# 初始化
print("\n初始化 IndexTTS2...")
tts = IndexTTS2()

voice = "examples/voice_01.wav"
emo_text = "非常开心"

# 测试1: 首次使用情感文本（会调用 Qwen LLM）
print("\n" + "=" * 70)
print("测试 1: 首次使用情感文本（应该调用 Qwen LLM）")
print("=" * 70)
t0 = time.time()
tts.infer(
    spk_audio_prompt=voice,
    text="你好，这是第一次测试。",
    output_path="test_emo_1.wav",
    use_emo_text=True,
    emo_text=emo_text
)
t1 = time.time() - t0
print(f"\n✅ 测试1完成: {t1:.2f}s")

# 测试2: 相同情感文本（应该使用缓存）
print("\n" + "=" * 70)
print("测试 2: 相同情感文本（应该使用缓存，快很多）")
print("=" * 70)
t0 = time.time()
tts.infer(
    spk_audio_prompt=voice,
    text="你好，这是第二次测试，不同的文本。",
    output_path="test_emo_2.wav",
    use_emo_text=True,
    emo_text=emo_text  # 相同的情感文本
)
t2 = time.time() - t0
print(f"\n✅ 测试2完成: {t2:.2f}s")

# 测试3: 不同情感文本（会重新调用 LLM）
print("\n" + "=" * 70)
print("测试 3: 不同情感文本（应该重新分析）")
print("=" * 70)
t0 = time.time()
tts.infer(
    spk_audio_prompt=voice,
    text="你好，这是第三次测试。",
    output_path="test_emo_3.wav",
    use_emo_text=True,
    emo_text="有点悲伤"  # 不同的情感文本
)
t3 = time.time() - t0
print(f"\n✅ 测试3完成: {t3:.2f}s")

# 结果对比
print("\n" + "=" * 70)
print("缓存效果对比")
print("=" * 70)
print(f"\n测试1（首次，无缓存）: {t1:.2f}s")
print(f"测试2（缓存命中）:     {t2:.2f}s")
print(f"测试3（不同文本）:     {t3:.2f}s")

if t2 < t1:
    speedup = ((t1 - t2) / t1) * 100
    saved_time = t1 - t2
    print(f"\n✅ 缓存命中加速: {speedup:.1f}% (节省 {saved_time:.2f}s)")
else:
    print(f"\n⚠️  缓存可能未生效")

print("\n" + "=" * 70)

