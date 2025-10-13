#!/bin/bash
# 详细的 Debug 输出测试

TEXT="Let's discover San Diego's delightful sugar options together; interested?"
VOICE="examples/zh_vo_Main_Linaxita_2_4_24_6.wav"

echo "======================================================================="
echo "MLX Debug 模式 - 详细输出"
echo "======================================================================="
echo ""
echo "测试文本: \"$TEXT\""
echo ""

conda run -n indextts2 python -m indextts.cli \
    "$TEXT" \
    -v "$VOICE" \
    --mlx \
    --deterministic \
    --debug \
    --force

echo ""
echo "======================================================================="
echo "完成！请检查输出中的："
echo "1. [DEBUG] Text tokens (first 10) - 文本 token"
echo "2. [DEBUG] First token logits top 10 - 首个 token 的 logits"
echo "3. [DEBUG] First 15 generated tokens - 前 15 个生成的 tokens"
echo "======================================================================="

