#!/bin/bash
# 测试 PyTorch 版本是否也有英文首尾吞音问题

echo "测试 PyTorch 版本（不使用 --mlx）"
python -m indextts.cli \
    --text "Let's discover San Diego's delightful sugar options together; interested?" \
    --prompt examples/voice_01.wav \
    --output test_pytorch_english.wav \
    --device mps

echo ""
echo "✅ 生成完成: test_pytorch_english.wav"
echo "请听音频，检查是否也有首尾吞音问题"

