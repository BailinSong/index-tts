#!/usr/bin/env python3
"""
测试文本首尾边界处理
检查是否是 padding/truncation 或 attention mask 问题
"""

import sys
sys.path.insert(0, '/Users/bailin/index-tts')

from sentencepiece import SentencePieceProcessor
from indextts.utils.front import Front


def test_text_processing():
    print("=" * 70)
    print("文本边界处理测试")
    print("=" * 70)
    
    # 加载工具
    bpe_path = 'checkpoints/bpe.model'
    bpe = SentencePieceProcessor()
    bpe.load(bpe_path)
    
    front = Front()
    front.load_model(bpe_path)
    
    # 测试文本
    test_texts = [
        "今天天气真不错",
        "Let's discover San Diego's delightful sugar options together; interested?",
        "Hello world",
        "a",  # 单字符
        "ab",  # 双字符
    ]
    
    for text in test_texts:
        print(f"\n{'='*70}")
        print(f"原始文本: \"{text}\"")
        print(f"长度: {len(text)} 字符")
        
        # 1. BPE tokenization
        tokens = bpe.encode(text)
        print(f"\nBPE Tokens ({len(tokens)}): {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        
        # 2. Front.text2tokens
        try:
            text_tokens = front.text2tokens(text)
            print(f"Front Tokens ({len(text_tokens)}): {text_tokens[:10]}{'...' if len(text_tokens) > 10 else ''}")
            
            # 检查首尾 token
            if len(text_tokens) > 0:
                print(f"  首个 token: {text_tokens[0]}")
                print(f"  最后 token: {text_tokens[-1]}")
                
            # 检查是否有特殊的 padding/start/stop tokens
            if 0 in text_tokens:
                print(f"  ⚠️  包含 token 0 (padding?)")
            if 1 in text_tokens:
                print(f"  ⚠️  包含 token 1 (start/stop?)")
            if 2 in text_tokens:
                print(f"  ⚠️  包含 token 2 (unknown?)")
                
        except Exception as e:
            print(f"  ❌ Front 处理失败: {e}")
        
        # 3. 检查是否有截断
        max_length = 600  # 根据代码中的 max text pos embeddings
        if len(text_tokens) > max_length:
            print(f"  ⚠️  文本过长！{len(text_tokens)} > {max_length}")
            print(f"  可能被截断，导致尾部丢失")


def test_attention_mask():
    """测试是否是 attention mask 导致首尾被忽略"""
    print("\n" + "=" * 70)
    print("Attention Mask 检查")
    print("=" * 70)
    
    import torch
    
    # 模拟序列
    seq_lengths = [10, 20, 50]
    
    for seq_len in seq_lengths:
        print(f"\n序列长度: {seq_len}")
        
        # Causal mask (GPT style)
        # 每个位置只能看到自己和之前的位置
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        
        # 检查首尾位置的 attention 情况
        print(f"  第 0 个位置能看到的位置数: {causal_mask[0].sum().item()}")
        print(f"  第 {seq_len-1} 个位置能看到的位置数: {causal_mask[seq_len-1].sum().item()}")
        
        # 首个位置只能看到自己 -> 可能导致首个 token 信息不足
        if causal_mask[0].sum() == 1:
            print(f"  ⚠️  首个位置只能自己看自己（self-attention only）")


def test_positional_encoding():
    """测试 positional encoding 边界"""
    print("\n" + "=" * 70)
    print("Positional Encoding 边界检查")
    print("=" * 70)
    
    # 根据代码：text_pos_embedding 最大 600
    max_text_pos = 600
    
    test_lengths = [1, 2, 5, 10, 50, 100, 500, 600, 650]
    
    for length in test_lengths:
        if length <= max_text_pos:
            print(f"  长度 {length:3d}: ✅ 正常")
        else:
            print(f"  长度 {length:3d}: ❌ 超出范围！可能导致问题")


if __name__ == '__main__':
    test_text_processing()
    test_attention_mask()
    test_positional_encoding()

