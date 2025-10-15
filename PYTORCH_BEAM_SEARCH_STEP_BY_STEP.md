# PyTorch Beam Search 逐步详细分析

## 前置条件
- `num_beams = 3`
- `do_sample = True`
- `vocab_size = 8194`
- `batch_size = 1`

---

## 第一部分：初始化阶段

### 1.1 输入准备（transformers_generation_utils.py:2270-2276）

```python
# 原始 input_ids: (1, seq_len) - 只有1个序列
input_ids, model_kwargs = self._expand_inputs_for_generation(
    input_ids=input_ids,
    expand_size=generation_config.num_beams,  # expand_size = 3
    is_encoder_decoder=self.config.is_encoder_decoder,
    **model_kwargs,
)
# 结果: input_ids: (3, seq_len) - 复制为3个序列
#   [seq_0]
#   [seq_0]  (相同)
#   [seq_0]  (相同)
```

### 1.2 Beam Scores 初始化（transformers_generation_utils.py:3438-3442）

```python
# 创建 beam_scores
beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
# beam_scores = [[0.0, 0.0, 0.0]]  shape: (1, 3)

beam_scores[:, 1:] = -1e9  # 只有第一个beam为0，其他为-1e9
# beam_scores = [[0.0, -1e9, -1e9]]

beam_scores = beam_scores.view((batch_size * num_beams,))
# beam_scores = [0.0, -1e9, -1e9]  shape: (3,)
```

**关键点：为什么这样初始化？**
> 注释：This makes sure that only tokens of the first beam are considered to avoid sampling the exact same tokens across all beams.
> 
> 只考虑第一个beam的tokens，避免所有beams采样到相同的tokens。

---

## 第二部分：Step 0 - 第一步扩展

### 2.1 模型前向传播（3个beams并行）

```python
# input_ids: (3, seq_len)
outputs = self(**model_inputs, return_dict=True)

# 获取最后一个token的logits
next_token_logits = outputs.logits[:, -1, :]  # (3, vocab_size)
# 每个beam都有一个vocab_size维的logits向量

next_token_scores = nn.functional.log_softmax(next_token_logits, dim=-1)
# next_token_scores: (3, 8194) - log概率
```

**关键：3个beams的logits此时完全相同**（因为input_ids相同）

```python
# next_token_scores 示例：
# Beam 0: [log_prob_token_0, log_prob_token_1, ..., log_prob_token_8193]
# Beam 1: [log_prob_token_0, log_prob_token_1, ..., log_prob_token_8193]  (相同)
# Beam 2: [log_prob_token_0, log_prob_token_1, ..., log_prob_token_8193]  (相同)
```

### 2.2 Logits Processor（包含repetition penalty）

```python
next_token_scores_processed = logits_processor(input_ids, next_token_scores)
# 应用 RepetitionPenaltyLogitsProcessor 等
# 结果: (3, 8194)
```

### 2.3 添加 Beam Scores

```python
next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores_processed)

# beam_scores[:, None]:
#   [[0.0],
#    [-1e9],
#    [-1e9]]  shape: (3, 1)

# expand_as -> (3, 8194):
#   [[0.0, 0.0, 0.0, ..., 0.0],         # Beam 0: 所有scores保持不变
#    [-1e9, -1e9, -1e9, ..., -1e9],     # Beam 1: 所有scores -= 1e9
#    [-1e9, -1e9, -1e9, ..., -1e9]]     # Beam 2: 所有scores -= 1e9

# 相加后:
# next_token_scores: (3, 8194)
#   Beam 0: [log_prob_0, log_prob_1, ..., log_prob_8193]  (保持原值)
#   Beam 1: [log_prob_0-1e9, log_prob_1-1e9, ...]         (所有极小)
#   Beam 2: [log_prob_0-1e9, log_prob_1-1e9, ...]         (所有极小)
```

**关键理解：此时只有Beam 0的scores是正常值，Beam 1和2的所有候选都是极小值！**

### 2.4 Reshape 为全局候选池

```python
vocab_size = next_token_scores.shape[-1]  # 8194
next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
# next_token_scores: (1, 24582) - 合并所有beams的候选

# 结构：[Beam0_token0, Beam0_token1, ..., Beam0_token8193, 
#        Beam1_token0, ..., Beam1_token8193,
#        Beam2_token0, ..., Beam2_token8193]

# 其中：
#   - 前8194个（Beam 0）：正常的log_prob值
#   - 后16388个（Beam 1和2）：极小值（~-1e9）
```

### 2.5 Multinomial 采样

```python
n_tokens_to_keep = max(2, 1 + n_eos_tokens) * num_beams
# n_tokens_to_keep = 2 * 3 = 6

# Softmax
probs = nn.functional.softmax(next_token_scores, dim=-1)  # (1, 24582)

# 由于Beam 1和2的scores都是-1e9，它们的概率接近0
# Beam 0的8194个候选占据了几乎所有概率质量

# Multinomial 采样 6 个
next_tokens = torch.multinomial(probs, num_samples=n_tokens_to_keep)
# next_tokens: (1, 6) - 6个flat indices
# 例如: [[123, 456, 789, 1024, 2048, 4096]]
```

**关键：由于Beam 1和2的概率接近0，这6个采样几乎肯定都来自Beam 0的8194个候选！**

```python
# 示例采样结果（假设）:
# next_tokens = [[7932, 286, 5942, 6505, 1782, 3889]]
# 这些都是 < 8194 的索引，即都来自Beam 0
```

### 2.6 获取对应的 Scores 并排序

```python
next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
# 获取采样的6个token对应的scores

next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
next_tokens = torch.gather(next_tokens, -1, _indices)

# 按score降序排序
# 例如:
# next_tokens = [[7932, 286, 5942, 6505, 1782, 3889]]  (已排序)
# next_token_scores = [[-3.5412, -3.6379, -4.4758, -5.0323, -5.6863, -11.8342]]
```

### 2.7 计算 Beam Index 和 Token ID

```python
next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
# next_indices: (1, 6)
# 7932 // 8194 = 0
# 286 // 8194 = 0
# ... 都是 0
# next_indices = [[0, 0, 0, 0, 0, 0]]  (都来自Beam 0)

next_tokens = next_tokens % vocab_size
# next_tokens: (1, 6)
# next_tokens = [[7932, 286, 5942, 6505, 1782, 3889]]
```

### 2.8 BeamSearchScorer.process

```python
beam_outputs = beam_scorer.process(
    input_ids,           # (3, seq_len)
    next_token_scores,   # (1, 6) = [[-3.5412, -3.6379, -4.4758, -5.0323, -5.6863, -11.8342]]
    next_tokens,         # (1, 6) = [[7932, 286, 5942, 6505, 1782, 3889]]
    next_indices,        # (1, 6) = [[0, 0, 0, 0, 0, 0]]
    ...
)
```

#### BeamSearchScorer.process 内部逻辑

```python
# 初始化输出
next_beam_scores = torch.zeros((batch_size, self.group_size))  # (1, 3)
next_beam_tokens = torch.zeros((batch_size, self.group_size))  # (1, 3)
next_beam_indices = torch.zeros((batch_size, self.group_size)) # (1, 3)

batch_idx = 0  # 只有1个batch
beam_idx = 0   # 计数已选择的beams

# 遍历6个候选（已按score排序）
for beam_token_rank, (next_token, next_score, next_index) in enumerate(
    zip(next_tokens[0], next_token_scores[0], next_indices[0])
):
    # Iteration 0: token=7932, score=-3.5412, next_index=0
    batch_beam_idx = 0 * 3 + 0 = 0
    
    if next_token not in eos_token_id:  # 不是stop token
        # 加入活跃beams
        next_beam_scores[0, 0] = -3.5412
        next_beam_tokens[0, 0] = 7932
        next_beam_indices[0, 0] = 0  # 来自原始的beam 0
        beam_idx += 1  # beam_idx = 1
    
    # Iteration 1: token=286, score=-3.6379, next_index=0
    batch_beam_idx = 0
    next_beam_scores[0, 1] = -3.6379
    next_beam_tokens[0, 1] = 286
    next_beam_indices[0, 1] = 0
    beam_idx += 1  # beam_idx = 2
    
    # Iteration 2: token=5942, score=-4.4758, next_index=0
    batch_beam_idx = 0
    next_beam_scores[0, 2] = -5942
    next_beam_tokens[0, 2] = 5942
    next_beam_indices[0, 2] = 0
    beam_idx += 1  # beam_idx = 3
    
    if beam_idx == self.group_size:  # beam_idx == 3
        break  # 已经凑够3个beams，停止
}

# 结果:
# next_beam_scores = [[-3.5412, -3.6379, -4.4758]]
# next_beam_tokens = [[7932, 286, 5942]]
# next_beam_indices = [[0, 0, 0]]  (都来自beam 0)
```

返回：
```python
return {
    "next_beam_scores": next_beam_scores.view(-1),  # [3] = [-3.5412, -3.6379, -4.4758]
    "next_beam_tokens": next_beam_tokens.view(-1),  # [3] = [7932, 286, 5942]
    "next_beam_indices": next_beam_indices.view(-1), # [3] = [0, 0, 0]
}
```

### 2.9 更新状态

```python
beam_scores = beam_outputs["next_beam_scores"]
# beam_scores = [-3.5412, -3.6379, -4.4758]  shape: (3,)

beam_next_tokens = beam_outputs["next_beam_tokens"]
# beam_next_tokens = [7932, 286, 5942]

beam_idx = beam_outputs["next_beam_indices"]
# beam_idx = [0, 0, 0]

# 根据beam_idx重新排列input_ids（选择parent beam）
input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

# beam_idx = [0, 0, 0] 意味着：
#   新beam 0 = 旧beam 0 + token 7932
#   新beam 1 = 旧beam 0 + token 286
#   新beam 2 = 旧beam 0 + token 5942

# input_ids:
#   [[seq_0, 7932],
#    [seq_0, 286],
#    [seq_0, 5942]]
```

**Step 0 结束状态：**
```
beam_codes:
  Beam 0: [start_token, 7932]   score = -3.5412
  Beam 1: [start_token, 286]    score = -3.6379
  Beam 2: [start_token, 5942]   score = -4.4758
```

---

## 第三部分：Step 1 - 三个不同beams扩展

### 3.1 模型前向传播

```python
# input_ids: 
#   [[seq_0, 7932],   # Beam 0
#    [seq_0, 286],    # Beam 1
#    [seq_0, 5942]]   # Beam 2

outputs = self(**model_inputs, return_dict=True)

next_token_logits = outputs.logits[:, -1, :]  # (3, 8194)
next_token_scores = nn.functional.log_softmax(next_token_logits, dim=-1)

# 此时3个beams的logits不同（因为输入不同）
# Beam 0: 基于 [seq_0, 7932] 预测的log_probs
# Beam 1: 基于 [seq_0, 286] 预测的log_probs
# Beam 2: 基于 [seq_0, 5942] 预测的log_probs
```

### 3.2 Logits Processor + 添加 Beam Scores

```python
next_token_scores_processed = logits_processor(input_ids, next_token_scores)

# 添加累积的beam_scores
next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(...)

# beam_scores = [-3.5412, -3.6379, -4.4758]
# beam_scores[:, None] = [[-3.5412], [-3.6379], [-4.4758]]

# expand后：
#   Beam 0的8194个候选 += -3.5412
#   Beam 1的8194个候选 += -3.6379
#   Beam 2的8194个候选 += -4.4758

# next_token_scores: (3, 8194)
#   Beam 0: [log_prob_0-3.5412, log_prob_1-3.5412, ..., log_prob_8193-3.5412]
#   Beam 1: [log_prob_0-3.6379, log_prob_1-3.6379, ..., log_prob_8193-3.6379]
#   Beam 2: [log_prob_0-4.4758, log_prob_1-4.4758, ..., log_prob_8193-4.4758]
```

### 3.3 Reshape 为全局候选池

```python
next_token_scores = next_token_scores.view(1, 24582)

# 结构：[Beam0的8194个候选, Beam1的8194个候选, Beam2的8194个候选]

# 假设每个beam的最好token的log_prob都是-2.5（简化示例）：
# Beam 0最好候选: -2.5 + (-3.5412) = -6.0412
# Beam 1最好候选: -2.5 + (-3.6379) = -6.1379
# Beam 2最好候选: -2.5 + (-4.4758) = -6.9758
```

### 3.4 Multinomial 采样

```python
probs = nn.functional.softmax(next_token_scores, dim=-1)  # (1, 24582)

# 关键：Beam 0的候选有最高的scores，所以有最高的概率
# 但是！24582个候选中，即使Beam 0的最好token概率最高，
# multinomial采样6个时，它仍可能不被选中！

next_tokens = torch.multinomial(probs, num_samples=6)

# 可能的结果（取决于随机性）：
# 情况A（理想）: 从Beam 0采样3个，Beam 1采样2个，Beam 2采样1个
# 情况B（可能）: 从Beam 0采样4个，Beam 1采样2个，Beam 2采样0个
# 情况C（不太可能但可能）: 所有6个都来自Beam 1（如果Beam 1某些token概率异常高）
```

### 3.5 关键问题：Multinomial 的随机性

**PyTorch的multinomial如何工作：**

```python
# 简化版multinomial算法（实际更复杂）
def multinomial(probs, num_samples):
    # 1. 归一化概率
    probs = probs / probs.sum()
    
    # 2. 计算累积分布 CDF
    cdf = cumsum(probs)
    
    # 3. 生成随机数并采样
    samples = []
    for _ in range(num_samples):
        r = random.uniform(0, 1)
        # 找到第一个 cdf[i] > r 的位置
        idx = binary_search(cdf, r)
        samples.append(idx)
        # 如果 replace=False，需要调整概率并重新计算CDF
    
    return samples
```

**关键洞察：**
- 即使某个候选概率最高，在采样少量样本时，仍可能不被选中
- 例如：probs[123] = 0.001（最高），但采样6次，每次miss的概率是 (1-0.001)^6 ≈ 99.4%

### 3.6 排序、解码、更新

```python
# 排序、解码beam_idx和token_id
next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
next_tokens = torch.gather(next_tokens, -1, _indices)

next_indices = next_tokens // vocab_size  # beam index
next_tokens = next_tokens % vocab_size    # token id

# 假设采样结果（示例）：
# next_tokens = [[4432, 6065, 329, 1994, 7113, 476]]
# next_indices = [[0, 0, 1, 1, 2, 0]]
#   - 2个来自Beam 0
#   - 2个来自Beam 1
#   - 1个来自Beam 2
#   - 1个来自Beam 0

# BeamSearchScorer.process选择top-3（非EOS）:
beam_outputs = {
    "next_beam_scores": [score_of_4432, score_of_6065, score_of_329],
    "next_beam_tokens": [4432, 6065, 329],
    "next_beam_indices": [0, 0, 1]  # parent beam
}

# 更新input_ids:
# 新Beam 0 = 旧Beam 0 + token 4432 = [seq_0, 7932, 4432]
# 新Beam 1 = 旧Beam 0 + token 6065 = [seq_0, 7932, 6065]
# 新Beam 2 = 旧Beam 1 + token 329  = [seq_0, 286, 329]

# 注意：旧Beam 2被丢弃了！
```

---

## 第四部分：关键机制总结

### 4.1 Beam Diversity 如何保证？

**Step 0:**
- 只有Beam 0的scores正常，强制所有6个采样来自Beam 0的不同tokens
- 结果：3个不同的beams

**Step 1 onwards:**
- 所有beams的候选进入全局池
- Multinomial从全局池采样
- **Diversity来源于随机性！**

### 4.2 为什么PyTorch能工作？

**假设1：Random Seed的稳定性**
- `torch.manual_seed(seed)` 确保multinomial的随机性可复现
- 可能在大多数情况下，采样结果能覆盖所有beams

**假设2：概率分布的特性**
- 真实的TTS模型输出可能不像我们想的那么极端
- Beam 0, 1, 2的最好候选概率可能相近
- 导致multinomial采样时有较高概率从每个beam都采到

**假设3：PyTorch可能有特殊处理（需验证）**
- 可能有额外的逻辑确保diversity？
- 或者transformers库的某个版本有特殊优化？

### 4.3 MLX vs PyTorch 的差异

**可能的差异点：**

1. **Random number generation:**
   ```python
   # PyTorch
   torch.manual_seed(42)
   torch.multinomial(probs, 6)
   
   # MLX/NumPy
   np.random.seed(42)
   np.random.choice(range(len(probs)), 6, p=probs, replace=False)
   ```
   → RNG算法不同，相同seed可能产生不同随机数序列

2. **Floating point precision:**
   - PyTorch: float32
   - MLX: float32 or bfloat16?
   - Softmax的微小差异导致probs不同

3. **Multinomial implementation:**
   - PyTorch: CUDA/CPU优化的multinomial
   - NumPy: 可能使用不同的采样算法

---

## 第五部分：验证方案

### 验证1：打印PyTorch Step 1的采样详情

在 `transformers_generation_utils.py:3547` 添加：
```python
if step == 1:
    print(f"[PyTorch] Step 1 multinomial sampling:")
    print(f"  next_tokens (flat indices): {next_tokens}")
    print(f"  next_indices (beam sources): {next_indices}")
    for i, (token, idx) in enumerate(zip(next_tokens[0], next_indices[0])):
        print(f"    Candidate {i}: beam={idx.item()}, token={token.item()}")
```

### 验证2：对比相同输入下的scores分布

保存PyTorch step 1的：
- `next_token_scores` (before softmax)
- `probs` (after softmax)
- `next_tokens` (sampled indices)

对比MLX的相同数据。

### 验证3：测试Top-K vs Multinomial

修改PyTorch代码，强制使用topk：
```python
# next_tokens = torch.multinomial(probs, num_samples=n_tokens_to_keep)
next_token_scores, next_tokens = torch.topk(next_token_scores, n_tokens_to_keep, dim=1, largest=True, sorted=True)
```

看结果是否更稳定。

---

## 结论

### PyTorch Beam Search的核心机制

1. **Step 0**: 通过beam_scores初始化 `[0, -1e9, -1e9]` 强制第一步扩展产生diversity
2. **Step 1+**: 依赖multinomial的随机性维持diversity
3. **最终选择**: 按normalized score (sum_logprobs / generated_len^length_penalty) 排序

### 可能导致MLX问题的原因

1. **RNG差异**: `torch.multinomial` vs `np.random.choice` 在相同seed下产生不同序列
2. **采样偏差**: MLX的某些步骤multinomial没有采样到最好的beams
3. **概率分布差异**: 浮点精度或softmax实现导致probs不同

### 下一步行动

1. **立即测试**: 在MLX中用Top-K替换multinomial，验证是否解决问题
2. **详细对比**: 添加debug输出，逐步对比PyTorch和MLX的中间结果
3. **考虑混合策略**: Top-K + Multinomial，既保证质量又保持diversity

