#!/usr/bin/env python3
"""
诊断英文文本处理和 tokenization 问题
"""

import sys
sys.path.insert(0, '/Users/bailin/index-tts')

from indextts.infer_v2 import IndexTTS2
import torch


def diagnose_english_text():
    print("=" * 70)
    print("英文文本处理诊断")
    print("=" * 70)
    
    # 初始化模型
    print("\n>> 初始化模型...")
    model = IndexTTS2(device='mps', use_mlx=True, diffusion_steps=20)
    
    # 问题文本
    test_text = "Let's discover San Diego's delightful sugar options together; interested?"
    
    print(f"\n>> 原始文本: \"{test_text}\"")
    print(f">> 文本长度: {len(test_text)} 字符")
    
    # 检查文本预处理
    print("\n" + "-" * 70)
    print("1. 文本预处理检查")
    print("-" * 70)
    
    # 使用 model 的 text_normalizer
    if hasattr(model, 'text_normalizer') and model.text_normalizer:
        normalized = model.text_normalizer.normalize_text(test_text)
        print(f"标准化后: \"{normalized}\"")
    else:
        normalized = test_text
        print("未进行标准化")
    
    # 检查 BPE tokenization
    print("\n" + "-" * 70)
    print("2. BPE Tokenization 检查")
    print("-" * 70)
    
    if hasattr(model, 'bpe_model') and model.bpe_model:
        # BPE encode
        tokens = model.bpe_model.encode(normalized)
        print(f"BPE tokens: {tokens}")
        print(f"Token 数量: {len(tokens)}")
        
        # Decode 回来检查
        decoded = model.bpe_model.decode(tokens)
        print(f"Decode 回来: \"{decoded}\"")
        
        # 检查首尾是否有损失
        if not decoded.startswith("Let"):
            print("⚠️  WARNING: 首部 'Let' 丢失或变化！")
        if not decoded.endswith("interested") and not decoded.endswith("?"):
            print("⚠️  WARNING: 尾部 'interested' 丢失或变化！")
        
        # 打印每个 token 对应的文本
        print("\n逐个 token 解码:")
        for i, token_id in enumerate(tokens):
            token_text = model.bpe_model.decode([token_id])
            print(f"  Token {i:2d}: {token_id:4d} -> \"{token_text}\"")
    else:
        print("BPE model 未加载")
    
    # 检查生成的 token IDs
    print("\n" + "-" * 70)
    print("3. GPT Input Token IDs 检查")
    print("-" * 70)
    
    # 使用 model 内部的 tokenization
    from indextts.utils.front import Front
    
    front = Front()
    text_tokens = front.text2tokens(normalized)
    print(f"Text tokens: {text_tokens}")
    print(f"Token 数量: {len(text_tokens)}")
    print(f"首个 token: {text_tokens[0] if text_tokens else 'N/A'}")
    print(f"最后 token: {text_tokens[-1] if text_tokens else 'N/A'}")
    
    # 测试生成
    print("\n" + "-" * 70)
    print("4. 测试生成（使用问题文本）")
    print("-" * 70)
    
    ref_audio = 'examples/voice_01.wav'
    
    try:
        print(f">> 开始生成...")
        wav, sr = model.infer(
            spk_audio_prompt=ref_audio,
            text=test_text,
            output_path='experiments/english_test_output.wav',
            verbose=True
        )
        print(f">> 生成完成: experiments/english_test_output.wav")
        print(f">> 请听音频，检查首尾是否有吞音")
        
    except Exception as e:
        print(f">> 生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 对比测试：简单英文文本
    print("\n" + "-" * 70)
    print("5. 对比测试：简单英文文本")
    print("-" * 70)
    
    simple_texts = [
        "Hello world",
        "Let's go",
        "Interested?",
        "Sugar options"
    ]
    
    for text in simple_texts:
        print(f"\n文本: \"{text}\"")
        if hasattr(model, 'bpe_model') and model.bpe_model:
            tokens = model.bpe_model.encode(text)
            decoded = model.bpe_model.decode(tokens)
            print(f"  Tokens: {tokens}")
            print(f"  Decoded: \"{decoded}\"")
            if text != decoded:
                print(f"  ⚠️  不一致！原文 != 解码")


if __name__ == '__main__':
    diagnose_english_text()

