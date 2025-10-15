# IndexTTS2 MLX性能基线（详细计时，Seed=42）

## 🎯 标准测试命令

```bash
python -m indextts.cli "今天天气真不错" \
  -v examples/zh_vo_Main_Linaxita_2_4_24_6.wav \
  --force --mlx --num-beams 1 --seed 42
```

---

## 📊 详细性能分解（单次运行示例）

### 总览

```
Total inference time: 20.36s (RTF=8.05, 首次运行)
Total inference time: 16.15s (RTF=6.38, 预热后)
音频长度: 2.53s (固定seed)
```

### 详细分解（预热后示例）

```
Total: 16.15s (100%)
├─ gpt_gen_time: 9.04s (56.0%)
│  ├─ emovec: 0.10s (1.1% of gpt_gen)
│  └─ MLX inference: 8.94s (98.9% of gpt_gen) ⬅️ 主要时间
│     ├─ Conditioning (Conformer+Perceiver): ~2-3s (估算)
│     ├─ Autoregressive generation: ~6-7s (估算)
│     └─ 数据转换 (torch↔mlx): ~0.5s (估算)
├─ gpt_forward_time: 0.07s (0.4%)
├─ s2mel_time: 5.14s (31.8%)
│  ├─ gpt_layer: 0.00s
│  ├─ vq2emb: 0.01s
│  ├─ prepare: 0.00s
│  ├─ length_reg: 2.68s (52.1% of s2mel)
│  └─ cfm: 2.45s (47.7% of s2mel)
├─ bigvgan_time: 0.66s (4.1%)
└─ 其他: ~1.24s (7.7%)
```

---

## 🎯 当前计时状态

### ✅ 已实现的计时

#### 1级拆分（总时间）
- ✅ Total inference time
- ✅ gpt_gen_time
- ✅ gpt_forward_time  
- ✅ s2mel_time
- ✅ bigvgan_time

#### 2级拆分（GPT）
- ✅ emovec计算时间
- ✅ MLX inference时间 (generation)

#### 2级拆分（S2MEL）
- ✅ gpt_layer
- ✅ vq2emb
- ✅ prepare
- ✅ length_reg
- ✅ cfm

### ⚠️ 待细化的计时

#### MLX inference内部（8.94s）

当前是黑盒，需要拆分为：
1. **数据准备（torch→mlx）**: ~0.5s
2. **Conformer + Perceiver计算**: ~2-3s
3. **自回归生成循环**: ~6-7s
4. **结果转换（mlx→torch）**: ~0.5s

**如何添加：**
需要修改`indextts/gpt/mlx_model.py`的`inference_speech`方法，
在关键位置添加计时并通过某种方式传回infer_v2.py

**复杂度：** 中等（需要修改MLX模型内部）

---

## 📈 性能瓶颈优先级（基于详细计时）

### 按时间占比排序

| 瓶颈 | 时间 | 占比 | 优化难度 | ROI |
|------|------|------|---------|-----|
| 1. **MLX inference** | 8.94s | 55.4% | 中-高 | ⭐⭐⭐⭐ |
| 2. **S2MEL length_reg** | 2.68s | 16.6% | 高 | ⭐⭐⭐ |
| 3. **S2MEL cfm** | 2.45s | 15.2% | 中 | ⭐⭐⭐ |
| 4. **BigVGAN** | 0.66s | 4.1% | 中 | ⭐⭐ |
| 5. **emovec** | 0.10s | 0.6% | 低 | ⭐ |

### MLX inference内部预估（需验证）

| 子模块 | 预估时间 | 占比 | 优化方向 |
|--------|---------|------|----------|
| 数据准备 | ~0.5s | 5.6% | 批量转换、缓存 |
| Conditioning | ~2.5s | 28.0% | 缓存结果 |
| 自回归生成 | ~6.5s | 72.7% | 并行、更优采样 |
| 结果转换 | ~0.5s | 5.6% | 批量转换 |

---

## 💡 优化建议（基于详细计时）

### P0 - 立即可做

1. **Conditioning缓存** ⭐⭐⭐⭐⭐
   - 相同voice prompt可缓存Conditioning结果
   - 预计节省: ~2.5s (15%)
   - 工作量: 1天

2. **批量数据转换**
   - 减少torch↔mlx转换次数
   - 预计节省: ~0.5s (3%)
   - 工作量: 0.5天

### P1 - 短期考虑

3. **减少diffusion steps**
   - 从20步→15步
   - 预计节省: ~0.6s (4%)
   - 工作量: 0.1天（仅改参数）

4. **优化自回归生成**
   - 当前: 每步都有torch↔mlx转换开销
   - 优化: 保持在MLX域内
   - 预计节省: ~1-2s (6-12%)
   - 工作量: 2-3天

### P2 - 长期

5. **S2MEL MLX实现**
   - 需要解决Conv1d转换问题
   - 预计节省: ~2-3s (12-18%)
   - 工作量: 5-7天

---

## 📊 优化路线图

### Phase 1: 缓存优化（1-2天）
```
当前: 16.15s
目标: 13-14s
提升: 13-19%
方法: Conditioning缓存 + 批量转换
```

### Phase 2: 采样优化（1周）
```
当前: 13-14s
目标: 11-12s
提升: 15-23%
方法: 优化自回归生成 + 减少diffusion steps
```

### Phase 3: 深度优化（1个月）
```
当前: 11-12s
目标: 8-10s
提升: 27-38%
方法: S2MEL部分MLX化
```

---

## ✅ 总结

### 当前状态

**GPT生成时间已拆分：**
```
gpt_gen_time: 9.04s
├─ emovec: 0.10s (1.1%)  ← 很快，无需优化
└─ MLX inference: 8.94s (98.9%)  ← 主要瓶颈
```

**MLX inference内部（黑盒）：**
- 包含：Conditioning + 自回归生成 + 转换
- 需要进一步细化（可选）

**最大优化机会：**
1. Conditioning缓存（~2.5s）
2. 自回归生成优化（~1-2s）
3. S2MEL length_reg MLX化（~2.5s，但有技术难度）

**下一步：实施Conditioning缓存优化**

