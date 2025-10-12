# Diffusion Steps 对比测试结果

## 📊 性能对比（10 vs 15 vs 25 步）

| Diffusion Steps | CFM Time | Length Reg | Total RTF | 改进 (vs 25步) |
|-----------------|----------|-----------|-----------|---------------|
| **25 steps (baseline)** | 2.90-3.31s | 7.41s | 7.48 | - |
| **15 steps (recommended)** | 1.71-2.30s | 6.69-8.75s | 4.90-7.56 | ↓ 21-37% |
| **10 steps (aggressive)** | 1.13-1.44s | 6.38-9.09s | 5.66-6.57 | ↓ 12-24% |

## 🎯 结论

### CFM 性能
- **25→15步**：CFM 时间减少 **32-42%** ✅
- **15→10步**：CFM 时间减少额外 **30-37%** ✅
- **总改进（25→10步）**：CFM 时间减少 **61-67%** 🎉

### 总体 RTF
- **25→15步**：RTF 改进 **21-37%** ✅
- **15→10步**：RTF 改进 **有限**（因为 Length Reg 占主导）⚠️
- **瓶颈**：Length Regulator 首次调用开销（6-9s）是主要限制因素

## 🎧 音质测试音频

### 25 Steps (Baseline)
- 未生成（baseline 参考）

### 15 Steps (Balanced)
1. `test_optimized_15steps.wav` - "今天天气真不错"
2. `test_opt_medium.wav` - "我们一起去看电影吧"
3. `test_opt_ai.wav` - "人工智能技术发展"
4. `test_final_opt1.wav` - "今天天气真不错"
5. `test_final_opt2.wav` - "我们一起去看电影吧"
6. `test_final_opt3.wav` - "人工智能技术发展迅速"

### 10 Steps (Aggressive)
1. `test_10steps_1.wav` - "今天天气真不错"
2. `test_10steps_2.wav` - "我们一起去看电影吧"
3. `test_10steps_3.wav` - "人工智能技术发展迅速"

## 💡 建议

### 推荐配置：15 Steps ⭐
**原因：**
- ✅ CFM 性能改进显著（32-42%）
- ✅ 音质预期较好（25→15的质量下降应该较小）
- ✅ RTF 改进明显（4.9-7.5x，平均 ~6x）
- ✅ 平衡点：性能与质量的最佳折衷

### 激进配置：10 Steps ⚡
**原因：**
- ✅ CFM 性能最佳（61-67% 改进 vs baseline）
- ⚠️ 音质可能有明显下降（需听测验证）
- ⚠️ RTF 改进有限（5.7-6.6x，因为 Length Reg 占主导）
- ⚠️ 风险：音质下降可能不值得

### 保守配置：20 Steps 🛡️
**原因：**
- ✅ 音质接近 baseline
- ⚠️ 性能改进有限（未测试，预计 RTF ~7x）
- 适用场景：音质优先，性能次要

## 📈 详细测试数据

### 15 Steps 测试
```
Test 1: "今天天气真不错"
  - CFM: 1.71-1.92s
  - Length Reg: 6.80-7.29s
  - RTF: 6.84

Test 2: "我们一起去看电影吧"
  - CFM: 1.70-1.78s
  - Length Reg: 6.69-9.81s
  - RTF: 5.75-5.97

Test 3: "人工智能技术发展迅速"
  - CFM: 1.94-2.30s
  - Length Reg: 8.75s
  - RTF: 4.90
```

### 10 Steps 测试
```
Test 1: "今天天气真不错"
  - CFM: 1.27s
  - Length Reg: 6.38s
  - RTF: 6.57

Test 2: "我们一起去看电影吧"
  - CFM: 1.44s
  - Length Reg: 6.52s
  - RTF: 5.90

Test 3: "人工智能技术发展迅速"
  - CFM: 1.13s
  - Length Reg: 9.09s
  - RTF: 5.66
```

## 🔍 关键发现

### 1. Length Regulator 是真正的瓶颈
无论 diffusion steps 是 10、15 还是 25，Length Regulator 的首次调用都需要 6-9s。这占据了总推理时间的 **60-75%**。

**原因：**
- MPS 设备首次调用需要编译 Metal 内核
- 复杂的操作组合（Conv1d + GroupNorm + F.interpolate）
- Batch size = 1，无法充分利用并行计算

**解决方案：**
- 模型预热（warm-up）
- 批处理推理（batch_size > 1）
- 优化 MPS 内核调用
- 或接受现状（首次调用的固有开销）

### 2. Diffusion Steps 优化收益递减
- 25→15步：CFM ↓32%，RTF ↓21-37% ✅ 高收益
- 15→10步：CFM ↓30%，RTF ↓有限 ⚠️ 低收益（因为 Length Reg 占主导）

**结论：**
进一步减少 diffusion steps 对总体 RTF 的改善有限，除非同时解决 Length Regulator 瓶颈。

### 3. 达成目标评估
**目标：** RTF 3-4x  
**当前（15 steps）：** RTF 4.9-7.5x（平均 ~6x）  
**差距：** ↑ 50-88%

**要达成 RTF 3-4x，需要：**
- Length Regulator 优化（6-9s → 1-2s）：**必须**
- CFM 进一步优化（1.8s → 0.5-1s）：可选
- 或组合其他优化（批处理、torch.compile、MLX化）

## 🎤 用户决策点

请听测以下音频并反馈：

1. **15 steps vs 10 steps 音质对比**
   - 哪个更自然？
   - 是否有丢字现象？
   - 音色和韵律是否接近参考样本？

2. **下一步优化方向**
   - **选项 A**：接受 15 steps，RTF ~6x，专注音质改进
   - **选项 B**：激进优化到 10 steps，追求 RTF ~5.5x，可接受质量下降
   - **选项 C**：攻克 Length Regulator 瓶颈，追求 RTF 3-4x
   - **选项 D**：回退到 20-25 steps，优先保证音质

---

**测试时间：** 2024-10-12  
**配置：** M4 Max, MPS device, Pure MLX mode  
**代码：** `indextts/infer_v2.py` line 797

