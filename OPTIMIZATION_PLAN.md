# MLX性能优化计划

## 基线版本
- **Tag**: v1.0.0-mlx-baseline
- **分支**: mlx-performance-opt
- **当前性能**: 比PyTorch慢30%
- **目标**: 提速50%+，超越PyTorch

---

## 已识别的瓶颈

### 🐌 主要瓶颈 (按影响排序)

| # | 瓶颈 | 开销 | 影响 | 优先级 |
|---|------|------|------|--------|
| 1 | **torch↔mlx转换** | ~2s | 30% | P0 🔥 |
| 2 | **Conformer计算2次** | ~4s | 60% | P1 |
| 3 | **JIT编译**(首次) | ~0.5s | 8% | P2 |
| 4 | **缺少mx.eval()** | ??? | ? | P0 |

---

## 优化方案

### P0 - 立即优化 (预期30%提速)

#### ✅ Task 1: 减少torch↔mlx转换

**当前问题**：
```python
# indextts/gpt/mlx_model.py:1705-1713
speech_condition_mlx = torch_to_mlx(speech_condition.cpu())  # 🐌 慢!
```

**优化方案**：
```python
# 方案A: 在infer_v2.py中直接生成MLX tensor
# 避免PyTorch → MLX转换

# 方案B: 批量转换，一次性转换所有需要的数据
```

**预期**: 节省 ~1-2s (**15-30%提速**)

---

#### ✅ Task 2: 添加mx.eval()强制计算

**当前问题**：
MLX使用lazy evaluation，计算可能被延迟

**优化方案**：
```python
# 在关键路径添加mx.eval()
hidden = self.gpt_ln_f(hidden)
hidden = self.final_norm(hidden)
logits = self.mel_head(hidden)
mx.eval(logits)  # 🔥 强制立即计算
next_token = mx.argmax(logits, axis=-1)
mx.eval(next_token)  # 🔥 确保token已计算
```

**预期**: 节省 ~0.5s (**5-10%提速**)

---

### P1 - 重要优化 (预期20%提速)

#### ✅ Task 3: JIT预热

**问题**: 第一次forward有0.5s编译开销

**方案**：
```python
def warmup_jit(self):
    """模型初始化后预热JIT编译"""
    dummy_seq = mx.zeros((1, 10, self.model_dim))
    
    # 预热transformer
    for block in self.transformer_blocks:
        dummy_seq, _ = block(dummy_seq, causal_mask=self.causal_mask, use_cache=True)
    
    # 预热LayerNorm和mel_head
    dummy_seq = self.gpt_ln_f(dummy_seq)
    dummy_seq = self.final_norm(dummy_seq)
    dummy_logits = self.mel_head(dummy_seq)
    
    mx.eval(dummy_logits)
    print(">> JIT warmup completed")
```

**预期**: 节省首次forward的 ~0.5s

---

#### ✅ Task 4: 真正的批处理beam search

**问题**: 当前"批处理"还是逐个处理beam

**当前代码**：
```python
for i, beam_idx in enumerate(active_beam_indices):
    h = batch_hidden[i:i+1]  # 逐个处理 🐌
    h, kv = block(h, ...)
```

**优化方案**：
```python
# 一次处理所有beams (真正的批处理)
batch_hidden = mx.concatenate([所有beams], axis=0)  # (num_beams, 1, D)
batch_hidden, batch_kvs = block(batch_hidden, ...)  # 一次forward
```

**预期**: Beam search提速 **10-15x**

---

### P2 - 进阶优化

#### Task 5: Conditioning结果缓存
- 相同voice缓存Conformer输出
- 预期: 节省 ~4s (重复使用时)

#### Task 6: 融合操作
- 使用MLX的融合kernel
- LayerNorm + Linear融合

---

## 实施计划

### Sprint 1: P0任务 (1-2天)
- [ ] 添加mx.eval()到所有关键路径
- [ ] 减少inference_speech中的转换
- [ ] 性能测试验证

**目标**: 达到与PyTorch相当的速度

### Sprint 2: P1任务 (2-3天)
- [ ] 实现JIT预热
- [ ] 重写批处理beam search
- [ ] 性能测试验证

**目标**: 超越PyTorch 20%+

### Sprint 3: P2任务 (可选)
- [ ] Conditioning缓存
- [ ] 探索融合操作

**目标**: 充分发挥Apple Silicon优势

---

## 成功指标

| 指标 | 基线 | P0目标 | P1目标 | 最终目标 |
|------|------|--------|--------|---------|
| **总推理时间** | 12s | 9s | 7s | 5s |
| **vs PyTorch** | +30% | 0% | -20% | -40% |
| **RTF** | 4-5x | 6-7x | 8-10x | 12-15x |

---

## 测试方法

```bash
# 性能测试命令
conda run -n indextts2 python -m indextts.cli \
  "今天天气真不错，让我们出去走走吧" \
  -v examples/zh_vo_Main_Linaxita_2_4_24_6.wav \
  --force --mlx --num-beams 1

# 对比PyTorch
conda run -n indextts2 python -m indextts.cli \
  "今天天气真不错，让我们出去走走吧" \
  -v examples/zh_vo_Main_Linaxita_2_4_24_6.wav \
  --force --num-beams 1
```

---

**分支**: mlx-performance-opt  
**基于**: v1.0.0-mlx-baseline  
**状态**: 🚀 Ready to start!
