#!/bin/bash
# 测试固定随机种子是否能稳定输出

TEXT="Let's discover San Diego's delightful sugar options together; interested?"
VOICE="examples/zh_vo_Main_Linaxita_2_4_24_6.wav"

echo "======================================================================="
echo "测试固定随机种子"
echo "======================================================================="
echo ""

# 临时修改代码，使用固定种子
# 需要在 simple_forward() 中修改：
# seed = 42  # 固定种子，而不是时间戳

echo "需要临时修改代码："
echo "在 indextts/gpt/mlx_model.py 的 simple_forward() 中："
echo ""
echo "  # 原代码："
echo "  seed = int(time.time() * 1000000) % (2**32)"
echo ""
echo "  # 改为："
echo "  seed = 42  # 固定种子测试"
echo ""
echo "然后运行两次测试，看 First token 是否相同"

