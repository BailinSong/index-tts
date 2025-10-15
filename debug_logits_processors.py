#!/usr/bin/env python3
"""
调试logits processors性能和卡住问题
"""
import time
import mlx.core as mx
import numpy as np
from indextts.gpt.mlx_logits_processors import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
)
from indextts.gpt.mlx_logits_processors_optimized import (
    CombinedLogitsProcessor,
    RepetitionPenaltyLogitsProcessorOptimized,
)

print("=" * 80)
print("Logits Processors调试测试")
print("=" * 80)

# 模拟真实场景
batch_size = 1
vocab_size = 6152  # 实际vocab大小
seq_len = 50  # 已生成的token数量

print(f"\n测试配置:")
print(f"  batch_size: {batch_size}")
print(f"  vocab_size: {vocab_size}")
print(f"  seq_len: {seq_len}")

# 创建测试数据
print(f"\n创建测试数据...")
input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))
logits = mx.random.normal((batch_size, vocab_size))

print(f"  input_ids shape: {input_ids.shape}")
print(f"  logits shape: {logits.shape}")

# ========================================
# 测试1: 原始TemperatureLogitsWarper
# ========================================
print(f"\n{'='*80}")
print(f"[Test 1] TemperatureLogitsWarper (原始版本)")
print(f"{'='*80}")

processor = TemperatureLogitsWarper(temperature=0.8)
t0 = time.time()
result = processor(input_ids, logits)
mx.eval(result)  # 强制计算
elapsed = time.time() - t0
print(f"✅ 完成: {elapsed:.4f}s")
print(f"  结果shape: {result.shape}")

# ========================================
# 测试2: 原始RepetitionPenaltyLogitsProcessor
# ========================================
print(f"\n{'='*80}")
print(f"[Test 2] RepetitionPenaltyLogitsProcessor (原始版本)")
print(f"{'='*80}")

processor = RepetitionPenaltyLogitsProcessor(penalty=10.0)
print(f"开始测试...")
t0 = time.time()
try:
    result = processor(input_ids, logits)
    mx.eval(result)
    elapsed = time.time() - t0
    print(f"✅ 完成: {elapsed:.4f}s")
    print(f"  结果shape: {result.shape}")
except Exception as e:
    print(f"❌ 失败: {e}")

# ========================================
# 测试3: 优化版RepetitionPenalty (逐步测试)
# ========================================
print(f"\n{'='*80}")
print(f"[Test 3] RepetitionPenaltyLogitsProcessorOptimized (逐步测试)")
print(f"{'='*80}")

processor_opt = RepetitionPenaltyLogitsProcessorOptimized(penalty=10.0)

print(f"\n步骤1: 转换为numpy...")
t0 = time.time()
logits_np = np.array(logits)
input_ids_np = np.array(input_ids)
print(f"  ✅ 完成: {time.time()-t0:.4f}s")
print(f"  logits_np shape: {logits_np.shape}")
print(f"  input_ids_np shape: {input_ids_np.shape}")

print(f"\n步骤2: 获取unique tokens...")
t0 = time.time()
batch_input_ids = input_ids_np[0]
unique_ids = np.unique(batch_input_ids)
print(f"  ✅ 完成: {time.time()-t0:.4f}s")
print(f"  unique tokens数量: {len(unique_ids)}")

print(f"\n步骤3: 过滤有效tokens...")
t0 = time.time()
valid_mask = (unique_ids >= 0) & (unique_ids < vocab_size)
unique_ids_valid = unique_ids[valid_mask]
print(f"  ✅ 完成: {time.time()-t0:.4f}s")
print(f"  有效tokens数量: {len(unique_ids_valid)}")

print(f"\n步骤4: 应用penalty...")
t0 = time.time()
batch_logits = logits_np[0].copy()
for token_id in unique_ids_valid[:10]:  # 只测试前10个
    score = batch_logits[token_id]
    if score < 0:
        batch_logits[token_id] = score * 10.0
    else:
        batch_logits[token_id] = score / 10.0
print(f"  ✅ 完成（前10个tokens）: {time.time()-t0:.4f}s")

print(f"\n步骤5: 完整应用penalty（所有unique tokens）...")
t0 = time.time()
batch_logits = logits_np[0].copy()
for token_id in unique_ids_valid:
    score = batch_logits[token_id]
    if score < 0:
        batch_logits[token_id] = score * 10.0
    else:
        batch_logits[token_id] = score / 10.0
print(f"  ✅ 完成（{len(unique_ids_valid)}个tokens）: {time.time()-t0:.4f}s")

print(f"\n步骤6: 转换回MLX...")
t0 = time.time()
result_mlx = mx.array(logits_np)
mx.eval(result_mlx)
print(f"  ✅ 完成: {time.time()-t0:.4f}s")

print(f"\n步骤7: 完整processor调用...")
t0 = time.time()
try:
    result = processor_opt(input_ids, logits)
    print(f"  processor调用完成，等待eval...")
    mx.eval(result)
    elapsed = time.time() - t0
    print(f"  ✅ 完成: {elapsed:.4f}s")
except Exception as e:
    print(f"  ❌ 失败: {e}")
    import traceback
    traceback.print_exc()

# ========================================
# 测试4: CombinedLogitsProcessor
# ========================================
print(f"\n{'='*80}")
print(f"[Test 4] CombinedLogitsProcessor")
print(f"{'='*80}")

processor_combined = CombinedLogitsProcessor(temperature=0.8, repetition_penalty=10.0)
print(f"开始测试...")
t0 = time.time()
try:
    result = processor_combined(input_ids, logits)
    print(f"  processor调用完成，等待eval...")
    mx.eval(result)
    elapsed = time.time() - t0
    print(f"  ✅ 完成: {elapsed:.4f}s")
except Exception as e:
    print(f"  ❌ 失败: {e}")
    import traceback
    traceback.print_exc()

# ========================================
# 测试5: 压力测试（更长序列）
# ========================================
print(f"\n{'='*80}")
print(f"[Test 5] 压力测试（seq_len=100）")
print(f"{'='*80}")

seq_len_long = 100
input_ids_long = mx.random.randint(0, vocab_size, (batch_size, seq_len_long))
logits_long = mx.random.normal((batch_size, vocab_size))

print(f"  input_ids shape: {input_ids_long.shape}")

print(f"\n原始RepetitionPenalty...")
processor_orig = RepetitionPenaltyLogitsProcessor(penalty=10.0)
t0 = time.time()
try:
    result = processor_orig(input_ids_long, logits_long)
    mx.eval(result)
    elapsed = time.time() - t0
    print(f"  ✅ 完成: {elapsed:.4f}s")
except Exception as e:
    print(f"  ❌ 失败: {e}")

print(f"\n优化版RepetitionPenalty...")
processor_opt = RepetitionPenaltyLogitsProcessorOptimized(penalty=10.0)
t0 = time.time()
try:
    result = processor_opt(input_ids_long, logits_long)
    mx.eval(result)
    elapsed = time.time() - t0
    print(f"  ✅ 完成: {elapsed:.4f}s")
except Exception as e:
    print(f"  ❌ 失败: {e}")

# ========================================
# 总结
# ========================================
print(f"\n{'='*80}")
print(f"测试完成")
print(f"{'='*80}")
print(f"\n如果所有测试都通过，说明processors本身没问题。")
print(f"如果某个测试卡住，说明问题在该processor的实现。")

