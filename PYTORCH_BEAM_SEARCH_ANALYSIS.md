# PyTorch Beam Search 详细分析

## 1. 初始化阶段

### 1.1 输入扩展
```python
# transformers_generation_utils.py:2270-2276
# 将 input_ids 从 (batch_size, seq_len) 扩展到 (batch_size * num_beams, seq_len)
input_ids, model_kwargs = self._expand_inputs_for_generation(
    input_ids=input_ids,
    expand_size=generation_config.num_beams,
    is_encoder_decoder=self.config.is_encoder_decoder,
    **model_kwargs,
)
```

### 1.2 Beam Scores 初始化
```python
# transformers_generation_utils.py:3438-3442
# 关键：只有第一个beam的score是0，其他都是-1e9
# 这确保第一步只考虑第一个beam的候选，避免所有beams采样相同tokens
beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
beam_scores[:, 1:] = -1e9  # 第2到num_beams个beam设为-1e9
beam_scores = beam_scores.view((batch_size * num_beams,))
# 结果：beam_scores = [0.0, -1e9, -1e9, ...] (长度 = batch_size * num_beams)
```

## 2. 主循环 - 每一步的处理

### 2.1 模型前向传播
```python
# transformers_generation_utils.py:3490
outputs = self(**model_inputs, return_dict=True)

# 3505-3509: 获取logits并转为log_softmax
next_token_logits = outputs.logits[:, -1, :].clone().float()  # (batch_beam_size, vocab_size)
next_token_scores = nn.functional.log_softmax(next_token_logits, dim=-1)
```

### 2.2 Logits Processor 处理
```python
# transformers_generation_utils.py:3511
# logits_processor 包含: RepetitionPenaltyLogitsProcessor 等
# 注意：这里处理的是 log_softmax 后的 scores，不是原始 logits
next_token_scores_processed = logits_processor(input_ids, next_token_scores)
```

#### RepetitionPenaltyLogitsProcessor 逻辑
```python
# RepetitionPenaltyLogitsProcessor.__call__
# 对 input_ids 中所有出现过的 token 进行惩罚
for token_id in unique_tokens_in_input_ids:
    if score[token_id] < 0:
        score[token_id] = score[token_id] * penalty  # 负分更负
    else:
        score[token_id] = score[token_id] / penalty  # 正分更小
```

### 2.3 添加 Beam Scores
```python
# transformers_generation_utils.py:3512-3514
# 将处理后的scores与累积的beam_scores相加
next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores_processed)
# 形状: (batch_size * num_beams, vocab_size)
```

### 2.4 Reshape 为 Beam Search 格式
```python
# transformers_generation_utils.py:3536-3537
vocab_size = next_token_scores.shape[-1]
next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
# 关键：将所有beams的候选合并到一个大的候选池
# 形状: (batch_size, num_beams * vocab_size)
```

### 2.5 候选采样 (do_sample=True)
```python
# transformers_generation_utils.py:3542-3550
n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
n_tokens_to_keep = max(2, 1 + n_eos_tokens) * num_beams

if do_sample:
    # 对合并后的大分布做 softmax
    probs = nn.functional.softmax(next_token_scores, dim=-1)
    # 从这个大分布中 multinomial 采样 n_tokens_to_keep 个
    next_tokens = torch.multinomial(probs, num_samples=n_tokens_to_keep)
    # 获取对应的scores
    next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
    # 按score降序排序
    next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
    next_tokens = torch.gather(next_tokens, -1, _indices)
```

**重要：PyTorch beam search with sampling 的关键**
- 不是对每个beam独立采样
- 而是对所有beams的候选合并后做 softmax 和 multinomial 采样
- 这样可以在全局范围内选择最有希望的候选

### 2.6 计算 Beam Index 和 Token ID
```python
# transformers_generation_utils.py:3558-3559
# 从 flat index 计算出来自哪个beam
next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
# 计算实际的token id
next_tokens = next_tokens % vocab_size
```

### 2.7 Beam Scorer Process
```python
# transformers_generation_utils.py:3562-3571
beam_outputs = beam_scorer.process(
    input_ids,
    next_token_scores,
    next_tokens,
    next_indices,
    pad_token_id=pad_token_id,
    eos_token_id=eos_token_id,
    beam_indices=beam_indices,
    decoder_prompt_len=decoder_prompt_len,
)
```

#### BeamSearchScorer.process 逻辑
```python
# transformers_beam_search.py:215-310
# 输入：
#   - input_ids: (batch_size * num_beams, seq_len)
#   - next_scores: (batch_size, num_beams) 已排序
#   - next_tokens: (batch_size, num_beams) 已排序
#   - next_indices: (batch_size, num_beams) 已排序，表示来自哪个beam

cur_len = input_ids.shape[-1] + 1  # 当前长度+1

for batch_idx in range(batch_size):
    beam_idx = 0
    # 遍历已排序的候选
    for beam_token_rank, (next_token, next_score, next_index) in enumerate(
        zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
    ):
        batch_beam_idx = batch_idx * self.group_size + next_index
        
        if next_token.item() in eos_token_id:
            # 如果是 EOS token，加入完成的假设
            if beam_token_rank < self.group_size:  # 只保留top-k个EOS
                self._beam_hyps[batch_group_idx].add(
                    input_ids[batch_beam_idx].clone(),
                    next_score.item(),
                    beam_indices=beam_index,
                    generated_len=cur_len - decoder_prompt_len,  # 关键：只计算生成的长度
                )
        else:
            # 否则加入活跃beams
            next_beam_scores[batch_idx, beam_idx] = next_score
            next_beam_tokens[batch_idx, beam_idx] = next_token
            next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
            beam_idx += 1
        
        # 凑够 num_beams 个就停止
        if beam_idx == self.group_size:
            break

return {
    "next_beam_scores": next_beam_scores.view(-1),  # 新的beam_scores
    "next_beam_tokens": next_beam_tokens.view(-1),
    "next_beam_indices": next_beam_indices.view(-1),
}
```

### 2.8 更新状态
```python
# transformers_generation_utils.py:3573-3579
beam_scores = beam_outputs["next_beam_scores"]
beam_next_tokens = beam_outputs["next_beam_tokens"]
beam_idx = beam_outputs["next_beam_indices"]

# 根据 beam_idx 重新排列 input_ids (选择对应的parent beam)
input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

# 更新 past_key_values (KV cache)
model_kwargs = self._update_model_kwargs_for_generation(...)
```

## 3. 最终选择阶段

### 3.1 BeamHypotheses 存储
```python
# transformers_beam_search.py:930-978
class BeamHypotheses:
    def __init__(self, num_beams, length_penalty, ...):
        self.beams = []  # 存储 (score, hypothesis, beam_indices) tuples
        
    def add(self, hyp, sum_logprobs, beam_indices, generated_len):
        # 关键：使用 generated_len 进行归一化
        score = sum_logprobs / (generated_len ** self.length_penalty)
        
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, beam_indices))
            if len(self) > self.num_beams:
                # 保持最多 num_beams 个假设
                sorted_next_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
```

### 3.2 最终 Finalize
```python
# transformers_beam_search.py:320-378
def finalize(self, input_ids, final_beam_scores, ...):
    # 将所有未完成的beams加入假设
    for batch_group_idx, beam_hyp in enumerate(self._beam_hyps):
        for index_per_group in range(self.group_size):
            batch_beam_idx = batch_group_idx * self.group_size + index_per_group
            final_score = final_beam_scores[batch_beam_idx].item()
            final_tokens = input_ids[batch_beam_idx]
            generated_len = final_tokens.shape[-1] - decoder_prompt_len
            beam_hyp.add(final_tokens, final_score, beam_indices=..., generated_len=generated_len)
    
    # 从所有假设中选择最佳的
    for i in range(batch_size):
        candidate_beams = [beam for beam_hyp in ... for beam in beam_hyp.beams]
        sorted_hyps = sorted(candidate_beams, key=lambda x: x[0])  # 按 normalized score 排序
        
        for j in range(self.num_beam_hyps_to_keep):
            best_hyp_tuple = sorted_hyps.pop()  # 最高分
            best_score = best_hyp_tuple[0]
            best_hyp = best_hyp_tuple[1]
            ...
```

## 4. 关键差异点总结

### 4.1 初始化
- **PyTorch**: `beam_scores = [0, -1e9, -1e9, ...]` 只有第一个beam有效
- **MLX**: 目前实现相同 ✅

### 4.2 候选采样方式
- **PyTorch**: 
  1. 合并所有beams: `(num_beams, vocab_size) -> (num_beams * vocab_size)`
  2. 对整个大分布做 softmax
  3. Multinomial 采样 `n_tokens_to_keep = 2 * num_beams` 个
  4. 排序选择
- **MLX**: 目前实现相同 ✅

### 4.3 Score 计算
- **PyTorch**: 
  1. log_softmax(logits) -> log_probs
  2. logits_processor(log_probs) -> processed_log_probs
  3. processed_log_probs + beam_scores -> next_token_scores
- **MLX**: 需要确认是否完全一致

### 4.4 Length Penalty
- **PyTorch**: `score = sum_logprobs / (generated_len ** length_penalty)`
  - `generated_len = cur_len - decoder_prompt_len` (不包括prompt)
- **MLX**: 已修复为使用 generated_len ✅

### 4.5 Repetition Penalty
- **PyTorch**: 应用于 input_ids 中的所有 unique tokens
  - `input_ids` 包括：fake_inputs (全是1) + start_mel_token + generated_tokens
- **MLX**: 目前实现应该相同 ✅

## 5. 潜在问题分析

### 问题1: 所有beams收敛到相同路径
**可能原因**:
1. ❓ Multinomial 采样的随机性不足
2. ❓ Score 计算存在细微差异导致某些路径始终占优
3. ❓ Top-p/Top-k filtering 未正确应用（PyTorch beam search 中没有 top-p/top-k）

### 问题2: 生成过长
**可能原因**:
1. ❓ Stop token 的概率被过度惩罚
2. ❓ Repetition penalty 应用方式不正确

## 6. 待验证的关键点

1. **Multinomial 采样是否真正随机**
   - 需要确认 np.random.seed() 是否正确工作
   - 验证采样结果的分布

2. **Score 数值精度**
   - PyTorch 使用 float32
   - MLX 默认精度是什么？

3. **Softmax 计算**
   - PyTorch: `nn.functional.softmax(scores, dim=-1)`
   - MLX: `mx.softmax(scores)` 是否等价？

4. **第一步之后的beam展开**
   - 第一步：3个不同的tokens (已确认 ✅)
   - 第二步：应该从这3个beams各自展开
   - 需要确认第2步及之后beam_scores的更新是否正确

