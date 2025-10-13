#!/usr/bin/env python3
"""
对比 PyTorch 和 MLX 生成的 mel tokens
找出首尾吞音的根本原因
"""

import sys
sys.path.insert(0, '/Users/bailin/index-tts')

import torch
from indextts.infer_v2 import IndexTTS2


def compare_generated_tokens():
    print("=" * 70)
    print("对比 PyTorch 和 MLX 生成的 Mel Tokens")
    print("=" * 70)
    
    text = "Let's discover San Diego's delightful sugar options together; interested?"
    ref_audio = 'examples/zh_vo_Main_Linaxita_2_4_24_6.wav'
    
    # 1. PyTorch 版本
    print("\n>> 初始化 PyTorch 模型...")
    model_pytorch = IndexTTS2(device='mps', use_mlx=False)
    
    print(f"\n>> PyTorch 推理...")
    # 捕获生成的 tokens
    # 需要修改 model 来返回 tokens
    
    # 2. MLX 版本
    print("\n>> 初始化 MLX 模型...")
    model_mlx = IndexTTS2(device='mps', use_mlx=True)
    
    print(f"\n>> MLX 推理...")
    
    print("\n" + "=" * 70)
    print("分析建议")
    print("=" * 70)
    print("\n需要在 indextts/gpt/mlx_model.py 中添加 debug 输出：")
    print("1. 打印生成的每个 mel token")
    print("2. 打印首个和最后几个 token 的 logits")
    print("3. 对比 PyTorch 和 MLX 的 token 序列")
    
    print("\n建议修改位置：")
    print("  - simple_forward() 方法")
    print("  - 在生成循环中添加 verbose 参数")
    print("  - 打印前 5 个和后 5 个 tokens")


if __name__ == '__main__':
    compare_generated_tokens()

