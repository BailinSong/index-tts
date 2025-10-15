# WIP: Logits Processor优化

**状态**: 🚧 进行中  
**优先级**: P0  
**预期收益**: 0.5-1s

---

## 🎯 目标

优化MLX logits processors以减少GPT生成时间（当前占54.3%）。

## 📊 当前瓶颈

```
GPT Generation: 4.11s (平均)
  └─ Logits Processing: 每个token约30-40ms
     ├─ TemperatureLogitsWarper: ~5ms/token
     ├─ RepetitionPenaltyLogitsProcessor: ~20ms/token ⚠️ 主要瓶颈
     └─ Other processors: ~5ms/token
```

**问题分析**：
1. RepetitionPenalty使用Python循环遍历unique tokens
2. 频繁的MLX↔numpy转换
3. 每个processor独立处理，没有合并优化

## ✅ 已完成

1. **创建优化版本** (`mlx_logits_processors_optimized.py`)
   - `CombinedLogitsProcessor`: 合并Temperature + RepetitionPenalty
   - `RepetitionPenaltyLogitsProcessorOptimized`: 向量化处理
   - `LogitsProcessorList.create_optimized()`: 智能选择最优组合

2. **集成到mlx_model.py**
   - 添加优化版本加载逻辑
   - 提供原始版本回退机制

## ⚠️ 当前问题

**执行卡住**：
- 测试时程序在generation循环中卡住（30s超时）
- 可能原因：
  1. MLX索引赋值操作导致的死锁
  2. `mx.unique()`或`mx.where()`的性能问题
  3. numpy转换开销反而更大

## 🔧 下一步行动

### 方案A：深度调试（预计1-2天）
1. 逐个测试processor，找出卡住的具体位置
2. 使用profiler分析MLX操作开销
3. 重新设计避免问题操作的实现

### 方案B：替代优化（预计0.5天）
1. **减少processor调用频率**
   - 只在前N个tokens应用RepetitionPenalty
   - 减少unique tokens查找频率

2. **简化RepetitionPenalty**
   - 只惩罚最近K个tokens，而非全部历史
   - 使用固定penalty而非条件判断

3. **使用Greedy Decoding**
   - num_beams=1时禁用sampling
   - 完全跳过logits processors

## 📝 当前实现（已禁用）

```python
# mlx_model.py line 1186
use_optimized = False  # ⚠️ 暂时禁用
```

优化代码保留在`mlx_logits_processors_optimized.py`，等待调试完成后启用。

## 🚀 临时解决方案

### 优先测试方案B-3：Greedy Decoding

**实现**（已在代码中）：
```python
# num_beams=1时强制greedy
if num_beams == 1:
    do_sample = False  # PyTorch
    use_sampling = False  # MLX
```

**预期收益**：
- 跳过所有logits processors
- 直接使用argmax，速度最快
- 预计节省 0.5-1s

**风险**：
- 输出多样性降低
- 可能影响质量（需要测试）

## 📈 性能目标

| 优化 | 当前 | 目标 | 方法 |
|------|------|------|------|
| RepetitionPenalty | ~20ms/token | ~5ms/token | 向量化/简化 |
| 总Logits Processing | ~30ms/token | ~10ms/token | 合并processors |
| GPT Generation | 4.11s | 3.5s | -0.6s |

## 📁 相关文件

- `indextts/gpt/mlx_logits_processors_optimized.py` - 优化版本（WIP）
- `indextts/gpt/mlx_logits_processors.py` - 原始版本（当前使用）
- `indextts/gpt/mlx_model.py` - 集成点

## 🔍 调试建议

1. **隔离测试**：
   ```python
   # 测试单个processor性能
   python test_single_processor.py
   ```

2. **Profiling**：
   ```python
   import time
   for processor in logits_processor:
       t0 = time.time()
       logits = processor(input_ids, logits)
       print(f"{type(processor).__name__}: {time.time()-t0:.3f}s")
   ```

3. **逐步启用**：
   - 先测试CombinedLogitsProcessor（温度+惩罚）
   - 再测试Top-p/Top-k优化版本

---

**更新时间**: 2024-10-15  
**负责人**: AI Assistant  
**预计完成**: TBD（需要深度调试）

