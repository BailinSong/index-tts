# MLX性能优化进度跟踪

**分支**: mlx-performance-opt  
**目标**: 达到或超过PyTorch性能  
**更新**: 2024-10-15

---

## 📊 性能演进

| 版本 | 时间 | RTF | vs PyTorch | 改进 | 状态 |
|------|------|-----|-----------|------|------|
| V0 (Pure MLX) | 16.62s | 6.57 | +183% | 基线 | ✅ 完成 |
| V1 (+ Conditioning Cache) | 7.57s | 2.99 | +29% | -54% | ✅ 完成 |
| **V2 (+ Logits Opt)** | **6.83s** | **2.70** | **+16%** | **-59%** | ✅ 完成 |
| PyTorch参考 | 5.87s | 2.32 | - | - | 参考 |

**当前差距**: MLX比PyTorch慢 **16%** (0.96s)

---

## ✅ 已完成的优化

### 1. Conditioning缓存 (V0→V1)
**收益**: -9.05s (-54%)  
**实现**: 
- 双缓存策略（MLX + PyTorch格式）
- 完整conds_mlx缓存（34 tokens）
- PyTorch conditioning保留音色

**关键修复**:
- 初始BUG：dummy tensor导致音色丢失
- 修复：缓存完整conditioning（speech+emotion+duration）

**文件**:
- `indextts/infer_v2.py` - 缓存逻辑
- `indextts/gpt/mlx_model.py` - 返回完整conditioning

### 2. Logits Processor优化 (V1→V2)
**收益**: -0.74s (-9.8%)  
**实现**:
- `RepetitionPenaltyLogitsProcessorOptimized`: numpy向量化，58x加速
- `CombinedLogitsProcessor`: 合并Temperature+RepetitionPenalty
- 智能选择最优processor组合

**关键修复**:
- 初始BUG：使用`mx.unique()`导致程序卡住（MLX无此API）
- 修复：改用numpy，性能反而更好

**文件**:
- `indextts/gpt/mlx_logits_processors_optimized.py` - 优化实现
- `debug_logits_processors.py` - 调试工具

---

## 📈 详细性能分析（V2）

### V2基准数据（3次运行平均）
```
平均值: 6.83s (RTF=2.70)
中位数: 6.32s
标准差: ±0.93s (13.7%)
范围: 6.26s - 7.91s

详细:
  Run 1: 6.26s (到底应该吃什么)
  Run 2: 7.91s (你为什么不愿意)
  Run 3: 6.32s (今天天气真不错)
```

### 模块时间分布（估算）
```
Total: 6.83s
├─ gpt_gen: 3.50s (51.2%) ⚠️ 主要瓶颈
│  ├─ emovec: ~0.01s (0.3%)
│  └─ MLX inference: ~3.49s (99.7%)
│     ├─ Conditioning: 0s (缓存命中) ✅
│     ├─ Generation loop: ~3.3s
│     └─ Conversion: ~0.2s
├─ s2mel: 2.42s (35.4%)
│  ├─ gpt_layer: 0.00s
│  ├─ length_reg: 0.44s (平均)
│  └─ cfm: 2.00s (20 steps)
├─ bigvgan: 0.63s (9.2%)
└─ 其他: ~0.28s (4.1%)
```

---

## 🎯 下一步优化方向

### Phase 2 优化（剩余差距：0.96s）

#### 优先级P0（高收益）

1. **Diffusion Steps优化**
   - 当前：20 steps
   - 目标：15 steps
   - 预期收益：-0.5s
   - 风险：需验证音质
   - 难度：低（配置修改）

2. **Generation Loop优化**
   - 当前：3.3s (127 tokens, ~26ms/token)
   - 瓶颈：每token的logits处理
   - 预期收益：-0.3s
   - 优化点：
     - 减少mx.eval()调用
     - 批量处理logits
     - JIT编译关键路径

#### 优先级P1（中等收益）

3. **Length Regulator优化**
   - 当前：0.44s（平均，波动大）
   - 预期收益：-0.2s
   - 优化点：MLX化或简化算法

4. **数据转换优化**
   - torch↔mlx转换：~0.2s
   - 预期收益：-0.1s
   - 优化点：批量转换，避免重复

#### 优先级P2（长期）

5. **CFM MLX化**
   - 当前：2.0s (PyTorch)
   - 预期收益：-0.3s
   - 难度：高

---

## 📁 关键文档

### 性能基准
- `BASELINE_V1_OPTIMIZED.md` - V1/V2基准数据
- `BASELINE_V1_PYTORCH.md` - PyTorch对比
- `MLX_V1_PERFORMANCE_ANALYSIS.md` - 详细分析

### 里程碑
- `MILESTONE_CONDITIONING_CACHE_FIX.md` - V0→V1里程碑
- `MILESTONE_LOGITS_PROCESSOR_OPT.md` - V1→V2里程碑（待创建）

### 技术文档
- `WIP_LOGITS_PROCESSOR_OPTIMIZATION.md` - Logits优化过程
- `OPTIMIZATION_SUMMARY.md` - 总体优化总结

### 测试工具
- `benchmark_v1_baseline.py` - MLX基准测试
- `benchmark_v1_baseline_pytorch.py` - PyTorch对比
- `debug_logits_processors.py` - Logits调试工具

---

## 🔬 技术亮点

### 1. 双缓存策略
解决了缓存与音色保留的矛盾：
- MLX缓存：用于快速generation
- PyTorch缓存：用于保留音色特征

### 2. Numpy vs MLX权衡
发现numpy在某些场景下比MLX更快：
- unique操作：MLX不支持
- 向量化索引：numpy更成熟
- 数组赋值：numpy直接支持

### 3. 微基准驱动优化
通过隔离测试找出具体瓶颈：
- `debug_logits_processors.py`
- 58x加速的RepetitionPenalty

---

## 📊 对比PyTorch

### MLX优势
✅ Conditioning缓存（PyTorch不支持）  
✅ Apple Silicon优化（MPS backend）  
✅ 低内存占用

### MLX劣势
❌ Generation loop慢16% (3.5s vs 2.8s)  
❌ S2MEL稍慢12% (2.4s vs 2.2s)  
❌ 生态不如PyTorch成熟

### 差距来源
主要在**GPT Generation循环**：
- PyTorch: 2.83s (平均)
- MLX: 3.50s (平均)
- 差距: 0.67s (24%)

这0.67s就是核心优化目标！

---

## 🚀 V3优化计划

**目标**: 6.83s → **5.5s** (RTF=2.2)

| 优化 | 预期收益 | 难度 | 优先级 |
|------|---------|------|--------|
| Diffusion steps 20→15 | -0.5s | 低 | P0 ⭐⭐⭐⭐⭐ |
| Generation loop优化 | -0.3s | 中 | P0 ⭐⭐⭐⭐ |
| Length regulator优化 | -0.2s | 中 | P1 ⭐⭐⭐ |
| 数据转换优化 | -0.1s | 低 | P1 ⭐⭐ |

**预计总收益**: -1.1s  
**预计V3性能**: 5.7s (RTF=2.3) - **接近PyTorch！**

---

## ✅ 提交历史

```
3bc12a5 ✅ Logits Processor优化完成 - 性能提升9.8%
e2cfbb3 WIP: Add logits processor optimization (disabled)
21fbca4 Add milestone: Conditioning cache fix completed
ae3a202 🔥 Critical fix: Complete conditioning cache for voice preservation
```

---

**当前状态**: ✅ V2完成，性能提升59% vs V0  
**下一目标**: V3优化，目标接近PyTorch性能  
**预计时间**: 1-2天（P0优化）

