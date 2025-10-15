# MLX Beam Search 支持说明

## 为什么 MLX 当前不支持 Beam Search？

### 技术背景

**Beam Search 是什么？**
- 一种解码策略，维护 `num_beams` 个候选序列
- 每一步扩展所有候选，选择累积分数最高的 top-k 个
- 比贪婪搜索（argmax）更全局，比随机采样更稳定

**PyTorch 实现：**
```python
# PyTorch 默认参数
num_beams = 3  # 维护 3 个候选序列
temperature = 0.8
top_p = 0.8
top_k = 30
```

### 当前 MLX 实现的采样策略

我们的 MLX 实现目前支持两种策略：

1. **贪婪采样（Argmax）- 确定性**
   ```python
   next_token = mx.argmax(logits)  # 选择概率最高的 token
   ```
   - ✅ 优点：完全确定性，稳定
   - ❌ 缺点：可能陷入局部最优

2. **随机采样（Categorical）- 随机性**
   ```python
   probs = mx.softmax(logits / temperature)
   next_token = mx.random.categorical(mx.log(probs))
   ```
   - ✅ 优点：多样性，探索性强
   - ❌ 缺点：不稳定，每次结果不同（导致丢字问题）

### 为什么还没实现 Beam Search？

**1. 实现复杂度高**

Beam search 需要：
```python
# 伪代码示意
beams = [initial_sequence] * num_beams
scores = [0.0] * num_beams

for step in range(max_length):
    all_candidates = []
    
    # 对每个 beam 扩展
    for beam_id in range(num_beams):
        logits = model(beams[beam_id])
        top_k_tokens, top_k_scores = topk(logits, k=num_beams)
        
        for token, score in zip(top_k_tokens, top_k_scores):
            candidate = beams[beam_id] + [token]
            candidate_score = scores[beam_id] + score
            all_candidates.append((candidate, candidate_score))
    
    # 选择 top num_beams 个候选
    beams, scores = select_top_k(all_candidates, k=num_beams)
```

**关键挑战：**
- 需要维护 `num_beams` 个序列的 KV cache
- 每一步需要扩展 `num_beams × vocab_size` 个候选
- MLX 的 KV cache 需要动态扩展（目前是单序列）

**2. 开发优先级**

我们的优化路线：
- ✅ Phase 1: 基础功能正确性（Conditioning, Transformer）
- ✅ Phase 2: 修复关键 bugs（英文吞音、内存泄漏、丢字）
- 🚧 Phase 3: 性能优化（S2MEL, BigVGAN MLX 化）
- ⏸️  Phase 4: 高级采样策略（Beam Search, Top-p, Top-k）

Beam search 在 Phase 4，优先级较低。

**3. 当前的权衡方案**

使用 **贪婪采样（argmax）** 作为 beam search 的近似：
- Beam search (num_beams=3) ≈ 选择 top-3 路径中的最佳
- Argmax ≈ 选择 top-1 路径（单一最佳）
- 质量略有下降，但稳定性大幅提升

---

## 临时解决方案：确定性采样

**当前配置（已实现）：**
```python
# indextts/gpt/mlx_model.py
use_sampling = kwargs.get('use_sampling', False)  # 默认 False（确定性）

if use_sampling and temperature > 0:
    # 随机采样（不稳定）
    next_token = mx.random.categorical(...)
else:
    # 贪婪采样（稳定）
    next_token = mx.argmax(logits)
```

**效果对比：**

| 策略 | PyTorch (beam search) | MLX (argmax) | MLX (sampling) |
|------|----------------------|--------------|----------------|
| 稳定性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| 质量 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 多样性 | ⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| 速度 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**结论：** MLX argmax 是当前最佳折衷方案。

---

## 未来实现 Beam Search 的路线图

### Phase 1: 基础 Beam Search (预计工作量: 2-3 天)

```python
def beam_search_forward(self, text_tokens, conditioning, num_beams=3, max_length=1500):
    """
    实现基础 beam search
    
    挑战：
    1. KV cache 需要支持 beam expansion
    2. 每一步需要处理 num_beams 个序列
    3. 需要实现 beam scoring 和 pruning
    """
    batch_size = text_tokens.shape[0]
    
    # 初始化 beams
    beams = mx.repeat(initial_sequence, num_beams, axis=0)  # (num_beams, seq_len)
    beam_scores = mx.zeros(num_beams)
    
    for step in range(max_length):
        # Forward pass for all beams
        logits = self.forward(beams)  # (num_beams, seq_len, vocab_size)
        
        # Compute scores for all candidates
        log_probs = mx.log_softmax(logits[:, -1, :], axis=-1)  # (num_beams, vocab_size)
        
        # Expand and score
        candidate_scores = beam_scores.reshape(-1, 1) + log_probs  # (num_beams, vocab_size)
        candidate_scores_flat = candidate_scores.reshape(-1)  # (num_beams * vocab_size)
        
        # Select top num_beams candidates
        top_indices = mx.argsort(candidate_scores_flat)[-num_beams:]
        beam_indices = top_indices // vocab_size
        token_indices = top_indices % vocab_size
        
        # Update beams
        beams = mx.concatenate([beams[beam_indices], token_indices.reshape(-1, 1)], axis=1)
        beam_scores = candidate_scores_flat[top_indices]
        
        # Check if all beams finished
        if mx.all(beams[:, -1] == stop_token):
            break
    
    # Return best beam
    best_beam_idx = mx.argmax(beam_scores)
    return beams[best_beam_idx]
```

**关键问题：**
- KV cache 的 beam expansion
- 内存效率（num_beams × seq_len × d_model）
- MLX 的动态形状支持

### Phase 2: 优化 Beam Search (预计工作量: 1-2 天)

1. **Length Normalization**
   ```python
   normalized_score = beam_score / (sequence_length ** length_penalty)
   ```

2. **Diverse Beam Search**
   - 引入 diversity penalty
   - 鼓励不同 beams 之间的差异

3. **Early Stopping**
   - 当最佳候选的分数远超其他候选时提前停止

### Phase 3: 集成到推理流程 (预计工作量: 1 天)

```python
# indextts/cli.py
parser.add_argument('--num-beams', type=int, default=3, help='Beam search beams')
parser.add_argument('--length-penalty', type=float, default=1.0, help='Length penalty')

# indextts/gpt/mlx_model.py
if num_beams > 1:
    codes = self.beam_search_forward(text_tokens, conditioning, num_beams=num_beams)
else:
    codes = self.simple_forward(text_tokens, conditioning)  # 当前的 greedy/sampling
```

---

## 估算总工作量

| 任务 | 工作量 | 优先级 |
|------|--------|--------|
| 基础 Beam Search 实现 | 2-3 天 | 中 |
| KV Cache Beam Expansion | 1-2 天 | 中 |
| 优化和测试 | 1-2 天 | 中 |
| 集成和文档 | 1 天 | 中 |
| **总计** | **5-8 天** | - |

---

## 当前建议

**对于用户：**
1. ✅ 使用默认的确定性采样（argmax）- 已经很稳定
2. ⚠️  避免使用随机采样（`use_sampling=True`）- 可能丢字
3. 📊 质量略低于 PyTorch beam search，但在可接受范围内

**对于开发：**
1. 优先完成 S2MEL 和 BigVGAN 的 MLX 化（更大性能收益）
2. Beam search 可以在性能优化完成后再实现
3. 或者用户可以贡献 PR！

---

## 技术细节：为什么 Argmax 不够稳定？

**问题根源：**
```python
# 确定性采样 (argmax)
next_token = mx.argmax(logits)

# 问题：如果 conditioning 或 随机种子 略有不同
# → logits 分布略有变化
# → argmax 可能选择不同的 token
# → 级联效应：一个错误 token 导致后续全部错误
```

**Beam search 为什么更稳定？**
```python
# Beam search 维护多个候选
beams = [candidate1, candidate2, candidate3]

# 即使 candidate1 略有偏差
# candidate2 或 candidate3 可能仍然正确
# → 选择累积分数最高的
# → 更鲁棒
```

---

**更新时间：** 2025-10-13  
**状态：** Beam Search 未实现，使用 Argmax 作为替代  
**下一步：** 完成 S2MEL 优化后考虑实现 Beam Search


