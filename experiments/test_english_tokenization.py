#!/usr/bin/env python3
"""
快速测试英文文本 tokenization
"""

import sys
sys.path.insert(0, '/Users/bailin/index-tts')

from sentencepiece import SentencePieceProcessor


def test_tokenization():
    print("=" * 70)
    print("BPE Tokenization 测试")
    print("=" * 70)
    
    # 加载 BPE model
    bpe_path = 'checkpoints/bpe.model'
    bpe = SentencePieceProcessor()
    bpe.load(bpe_path)
    
    # 问题文本
    texts = [
        "Let's discover San Diego's delightful sugar options together; interested?",
        "Let's go",
        "interested?",
        "sugar options",
        " Let's discover",  # 前面加空格
        "Let's discover ",  # 后面加空格
    ]
    
    for text in texts:
        print(f"\n{'='*70}")
        print(f"文本: \"{text}\"")
        print(f"长度: {len(text)} 字符")
        
        # Encode
        tokens = bpe.encode(text)
        print(f"Tokens ({len(tokens)}): {tokens}")
        
        # Decode
        decoded = bpe.decode(tokens)
        print(f"Decoded: \"{decoded}\"")
        
        # 检查差异
        if text != decoded:
            print(f"⚠️  不一致！")
            print(f"  原文开头: \"{text[:20]}...\"")
            print(f"  解码开头: \"{decoded[:20]}...\"")
            print(f"  原文结尾: \"...{text[-20:]}\"")
            print(f"  解码结尾: \"...{decoded[-20:]}\"")
        else:
            print("✅ 一致")
        
        # 逐个 token 解码
        print("逐个 token:")
        for i, token_id in enumerate(tokens):
            token_text = bpe.decode([token_id])
            print(f"  [{i:2d}] {token_id:5d} -> \"{token_text}\"")


if __name__ == '__main__':
    test_tokenization()

