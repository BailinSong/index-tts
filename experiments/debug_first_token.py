#!/usr/bin/env python3
"""
诊断首单词吞音问题
对比 PyTorch 和 MLX 生成的前几个 mel tokens
"""

import sys
sys.path.insert(0, '/Users/bailin/index-tts')

print("""
首单词吞音诊断
==================

问题：
- 确定性模式：尾单词改善 ✅，首单词无改善 ❌
- 说明：首单词吞音不是随机采样的问题

可能原因：
1. Start mel token 的初始化或位置编码
2. Conditioning + Text 的拼接边界
3. 第一个 mel token 生成时的 context 不够
4. Text positional encoding 的起始位置错误

需要对比：
- PyTorch 和 MLX 生成的前 10 个 mel tokens
- Start mel token 的位置
- Context 长度和内容

诊断步骤：
1. 修改 MLX simple_forward() 添加详细日志
2. 打印前 10 个生成的 tokens
3. 对比 PyTorch 的 tokens
4. 分析差异位置

修改位置：
indextts/gpt/mlx_model.py - simple_forward()
添加：
  print(f"[DEBUG] First 10 tokens: {generated[:10]}")
  print(f"[DEBUG] Context length: {context_len}")
  print(f"[DEBUG] Text tokens: {text_tokens}")
""")


if __name__ == '__main__':
    print("\n建议手动修改代码添加 debug 输出")
    print("或使用 --deterministic 参数测试")

