#!/bin/bash
# benchmark_m4_quick.sh - M4 快速性能测试脚本

set -e

echo "🍎 IndexTTS2 M4 Quick Performance Test"
echo "======================================"
echo ""

# 检查是否在正确的目录
if [ ! -f "benchmark_baseline.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# 设置 MPS fallback（避免某些操作失败）
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 创建输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="outputs/m4_quick_${TIMESTAMP}"

echo "📋 Test Configuration:"
echo "   Device: MPS (Apple Silicon)"
echo "   Platform: Apple M4"
echo "   FP16: Disabled (recommended for MPS)"
echo "   Output: ${OUTPUT_BASE}"
echo ""

# 测试 1: 基线配置
echo "📊 Test 1/3: Baseline Configuration"
echo "   (Standard settings, FP32)"
python benchmark_baseline.py \
    --device mps \
    --output_dir "${OUTPUT_BASE}/baseline" \
    --max_tokens 120 \
    --skip_env_check \
    > "${OUTPUT_BASE}/baseline.log" 2>&1

if [ $? -eq 0 ]; then
    echo "   ✅ Baseline test completed"
else
    echo "   ❌ Baseline test failed (see ${OUTPUT_BASE}/baseline.log)"
fi

# 测试 2: 优化分句
echo ""
echo "📊 Test 2/3: Optimized Segmentation"
echo "   (Larger segments: 150 tokens)"
python benchmark_baseline.py \
    --device mps \
    --output_dir "${OUTPUT_BASE}/opt_segments" \
    --max_tokens 150 \
    --skip_env_check \
    > "${OUTPUT_BASE}/opt_segments.log" 2>&1

if [ $? -eq 0 ]; then
    echo "   ✅ Optimized segmentation test completed"
else
    echo "   ❌ Test failed (see ${OUTPUT_BASE}/opt_segments.log)"
fi

# 测试 3: 激进优化
echo ""
echo "📊 Test 3/3: Aggressive Optimization"
echo "   (Max segments: 180 tokens)"
python benchmark_baseline.py \
    --device mps \
    --output_dir "${OUTPUT_BASE}/opt_aggressive" \
    --max_tokens 180 \
    --skip_env_check \
    > "${OUTPUT_BASE}/opt_aggressive.log" 2>&1

if [ $? -eq 0 ]; then
    echo "   ✅ Aggressive optimization test completed"
else
    echo "   ❌ Test failed (see ${OUTPUT_BASE}/opt_aggressive.log)"
fi

echo ""
echo "======================================"
echo "📊 Performance Comparison"
echo "======================================"
echo ""

# 提取和对比关键指标
for config in baseline opt_segments opt_aggressive; do
    echo "Configuration: $config"
    if [ -f "${OUTPUT_BASE}/${config}/benchmark_baseline_"*.json ]; then
        python -c "
import json
import sys
import glob

files = glob.glob('${OUTPUT_BASE}/${config}/benchmark_baseline_*.json')
if not files:
    sys.exit(1)

with open(files[0]) as f:
    data = json.load(f)

summary = data.get('summary', {})
rtf = summary.get('rtf', {})
total_time = summary.get('total_time', {})
gpu_mem = summary.get('gpu_memory_mb', {})

print(f\"  Mean RTF: {rtf.get('mean', 0):.4f}\")
print(f\"  Mean Time: {total_time.get('mean', 0):.3f}s\")
if gpu_mem.get('max', 0) > 0:
    print(f\"  Peak Memory: {gpu_mem.get('max', 0):.1f} MB\")
" 2>/dev/null || echo "  ⚠️  No data available"
    fi
    echo ""
done

echo "======================================"
echo ""
echo "✅ Quick test completed!"
echo ""
echo "📁 Results saved to: ${OUTPUT_BASE}/"
echo "📄 Logs: ${OUTPUT_BASE}/*.log"
echo ""
echo "💡 Next steps:"
echo "   1. Review detailed results:"
echo "      python analyze_optimization.py ${OUTPUT_BASE}/baseline/benchmark_*.json"
echo ""
echo "   2. Compare configurations:"
echo "      python analyze_optimization.py ${OUTPUT_BASE}/opt_aggressive/benchmark_*.json"
echo ""
echo "   3. Read M4-specific optimization guide:"
echo "      cat docs/optimization_apple_silicon_mps.md"
echo ""


