#!/usr/bin/env python3
"""
对比 PyTorch 和 MLX 在英文文本上的表现
确认是 MLX 特有问题还是通用问题
"""

import sys
sys.path.insert(0, '/Users/bailin/index-tts')

from sentencepiece import SentencePieceProcessor


def analyze_bpe():
    print("=" * 70)
    print("重新分析 BPE Tokenization")
    print("=" * 70)
    
    # 加载 BPE model
    bpe_path = 'checkpoints/bpe.model'
    bpe = SentencePieceProcessor()
    bpe.load(bpe_path)
    
    print(f"\nBPE Vocabulary Size: {bpe.vocab_size()}")
    print(f"BOS ID: {bpe.bos_id()}")
    print(f"EOS ID: {bpe.eos_id()}")
    print(f"UNK ID: {bpe.unk_id()}")
    print(f"PAD ID: {bpe.pad_id()}")
    
    # 问题文本
    text = "Let's discover San Diego's delightful sugar options together; interested?"
    
    print(f"\n{'='*70}")
    print(f"问题文本: \"{text}\"")
    
    # 使用不同的 encode 选项
    print("\n1. 标准 encode:")
    tokens_normal = bpe.encode(text)
    print(f"   Tokens: {tokens_normal}")
    print(f"   数量: {len(tokens_normal)}")
    
    print("\n2. encode_as_pieces (查看实际的 pieces):")
    pieces = bpe.encode_as_pieces(text)
    print(f"   Pieces: {pieces}")
    print(f"   数量: {len(pieces)}")
    
    print("\n3. 逐个 piece 对应的 token ID:")
    for i, piece in enumerate(pieces):
        token_id = bpe.piece_to_id(piece)
        print(f"   [{i:2d}] \"{piece}\" -> Token ID: {token_id}")
    
    # 检查 unknown token
    unk_count = tokens_normal.count(bpe.unk_id())
    print(f"\n4. Unknown tokens 数量: {unk_count} / {len(tokens_normal)}")
    
    if unk_count > len(tokens_normal) * 0.5:
        print("   ⚠️  超过 50% 的 tokens 是 unknown！")
        print("   这解释了为什么英文支持差")
    
    # 测试简单的中文
    print(f"\n{'='*70}")
    print("对比：中文文本")
    
    chinese_text = "今天天气真不错"
    print(f"文本: \"{chinese_text}\"")
    
    tokens_cn = bpe.encode(chinese_text)
    pieces_cn = bpe.encode_as_pieces(chinese_text)
    
    print(f"Tokens: {tokens_cn}")
    print(f"Pieces: {pieces_cn}")
    
    unk_count_cn = tokens_cn.count(bpe.unk_id())
    print(f"Unknown tokens: {unk_count_cn} / {len(tokens_cn)}")
    
    # 关键发现
    print(f"\n{'='*70}")
    print("关键发现")
    print("="*70)
    
    print(f"\n英文 unknown 比例: {unk_count / len(tokens_normal) * 100:.1f}%")
    print(f"中文 unknown 比例: {unk_count_cn / len(tokens_cn) * 100:.1f}%")
    
    # 但是！即使是 unknown tokens，模型可能也能处理
    print(f"\n⚠️  重要：即使大量 unknown tokens，")
    print(f"    模型在训练中可能见过类似的 token 序列模式")
    print(f"    所以可能还是能生成合理的语音")
    print(f"\n    这意味着：")
    print(f"    - BPE 虽然无法正确 decode")
    print(f"    - 但 token IDs 本身可能包含了足够信息")
    print(f"    - 首尾吞音可能是**其他原因**！")


if __name__ == '__main__':
    analyze_bpe()

