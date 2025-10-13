#!/bin/bash
# 测试固定随机种子 - 运行两次看是否完全一致

TEXT="Let's discover San Diego's delightful sugar options together; interested?"
VOICE="examples/zh_vo_Main_Linaxita_2_4_24_6.wav"

echo "======================================================================="
echo "测试固定随机种子 - 验证可复现性"
echo "======================================================================="
echo ""
echo "使用固定种子 42，运行两次，检查 First token 是否相同"
echo ""

echo "-----------------------------------------------------------------------"
echo "第 1 次运行"
echo "-----------------------------------------------------------------------"
export MLX_FIXED_SEED=42
conda run -n indextts2 python -m indextts.cli \
    "$TEXT" \
    -v "$VOICE" \
    --mlx --deterministic --debug --force 2>&1 \
    | grep -E "(DEBUG.*Random seed|First token:|First 15)"

echo ""
echo "-----------------------------------------------------------------------"
echo "第 2 次运行（相同种子）"
echo "-----------------------------------------------------------------------"
export MLX_FIXED_SEED=42
conda run -n indextts2 python -m indextts.cli \
    "$TEXT" \
    -v "$VOICE" \
    --mlx --deterministic --debug --force 2>&1 \
    | grep -E "(DEBUG.*Random seed|First token:|First 15)"

echo ""
echo "======================================================================="
echo "分析"
echo "======================================================================="
echo ""
echo "如果两次的 First token 相同 → 问题是随机种子导致的"
echo "如果两次的 First token 不同 → 还有其他随机性来源"
echo ""

