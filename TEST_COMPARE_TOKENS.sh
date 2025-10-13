#!/bin/bash
# 对比 PyTorch 和 MLX 生成的 Mel Tokens

TEXT="Let's discover San Diego's delightful sugar options together; interested?"
VOICE="examples/zh_vo_Main_Linaxita_2_4_24_6.wav"

echo "======================================================================="
echo "对比 PyTorch 和 MLX 生成的 Mel Tokens"
echo "======================================================================="
echo ""
echo "测试文本: \"$TEXT\""
echo ""

echo "-----------------------------------------------------------------------"
echo "1. PyTorch 版本（基准）"
echo "-----------------------------------------------------------------------"
conda run -n indextts2 python -m indextts.cli "$TEXT" -v "$VOICE" --force 2>&1 | grep -E "(First token|Generated.*mel tokens|first.*tokens)"

echo ""
echo "-----------------------------------------------------------------------"
echo "2. MLX 版本（确定性模式 + Debug）"
echo "-----------------------------------------------------------------------"
conda run -n indextts2 python -m indextts.cli "$TEXT" -v "$VOICE" --mlx --deterministic --debug --force 2>&1 | grep -E "(First token|Generated.*mel tokens|first.*tokens|DEBUG)"

echo ""
echo "======================================================================="
echo "分析说明"
echo "======================================================================="
echo ""
echo "对比重点："
echo "1. 第一个 token 是否相同？"
echo "2. 前 15 个 tokens 的差异模式"
echo "3. 总 token 数量是否接近？"
echo ""
echo "如果首个 token 不同 → 问题在第一步生成"
echo "如果首个相同但后续不同 → 问题在 autoregressive 循环"
echo ""

