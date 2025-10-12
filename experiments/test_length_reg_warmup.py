#!/usr/bin/env python3
"""
测试 Length Regulator 的 warm-up 效应
"""

import torch
import time
import sys
sys.path.insert(0, '/Users/bailin/index-tts')

from indextts.infer_v2 import IndexTTS2


def test_length_reg_warmup():
    print("=" * 60)
    print("Length Regulator Warm-up 测试")
    print("=" * 60)
    
    # 初始化模型
    print("\n>> 初始化模型...")
    model = IndexTTS2(device='mps', use_mlx=True)
    
    # 准备测试数据
    ref_audio_path = 'examples/voice_01.wav'
    
    # 测试句子
    sentences = [
        "今天天气真不错",
        "我们一起去看电影吧",
        "人工智能技术发展迅速",
        "测试",
        "今天天气真不错",  # 重复
    ]
    
    print("\n>> 开始测试...")
    print("-" * 60)
    
    for i, text in enumerate(sentences, 1):
        print(f"\n[Test {i}/{len(sentences)}] {text}")
        
        # 生成音频
        t0 = time.perf_counter()
        wav, sr = model.inference_v2(
            text=text,
            ref_audio_path=ref_audio_path,
            ref_text=None,
            emotion="happy",
            verbose=False
        )
        total_time = time.perf_counter() - t0
        
        audio_length = len(wav) / sr
        rtf = total_time / audio_length
        
        print(f"   Total: {total_time:.2f}s, Audio: {audio_length:.2f}s, RTF: {rtf:.2f}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    print("\n如果后续调用明显更快，说明有 warm-up 效应")


if __name__ == '__main__':
    test_length_reg_warmup()

