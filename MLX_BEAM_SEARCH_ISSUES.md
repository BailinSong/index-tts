# MLX Beam Search 问题分析和解决方案

## 当前问题

### 观察到的现象
1. ✅ Step 0: 3个beams生成了不同的tokens (7932, 286, 5942)
2. ❌ Step 1: 只有Beam 1和Beam 2的候选被选中，Beam 0被跳过
3. ❌ 最终: 所有beams收敛到类似的路径，长度都是242 tokens
4. ❌ 音频: 丢字，时长过长 (4.81s vs PyTorch 2.67s)

### Step 1 详细数据

**Beam状态 (Step 1开始时):**
```
Beam 0: tokens=[8192, 7932], beam_score=-3.5412  (最好)
Beam 1: tokens=[8192, 286],  beam_score=-3.6379
Beam 2: tokens=[8192, 5942], beam_score=-4.4758
```

**Score分布 (添加beam_score后):**
```
Total candidates: 24,582 (3 beams × 8,194 tokens)
Beam 0 range: [-280.76, -6.0411], top token score: -6.0411  (最好)
Beam 1 range: [-274.04, -6.3812], top token score: -6.3812
Beam 2 range: [-260.53, -7.1738], top token score: -7.1738
```

**Multinomial采样结果 (n=6):**
```
Candidate 0: source_beam=1, token=4432, score=-8.0321
Candidate 1: source_beam=2, token=329, score=-9.0405
Candidate 2: source_beam=1, token=6065, score=-9.0663
... (没有来自Beam 0的候选!)
```

## 根本原因分析

### 1. Multinomial采样的随机性

PyTorch和MLX都使用以下流程:
```python
# 1. 合并所有beams的scores: (num_beams * vocab_size,)
all_scores_flat = concatenate([beam0_scores, beam1_scores, beam2_scores])

# 2. Softmax得到概率分布
probs = softmax(all_scores_flat)

# 3. Multinomial采样n个 (n = 2 * num_beams = 6)
sampled_indices = multinomial(probs, n)
```

**问题:**
- 总候选数: 24,582
- 采样数: 6
- Beam 0最好token的概率: `exp(-6.0411) / sum(exp(all_scores))` ≈ 极小

即使Beam 0有最好的score，在24,582个候选中，单个token被采样到的概率仍然很小！

### 2. 为什么PyTorch能工作？

**关键假设需要验证:**

1. **Random seed差异**: `torch.manual_seed` vs `np.random.seed`
   - PyTorch的multinomial可能使用不同的RNG算法
   - 导致相同seed下采样结果不同

2. **数值精度**: PyTorch float32 vs MLX的默认精度
   - Softmax计算的微小差异可能影响采样分布

3. **PyTorch可能有特殊处理**:
   - 可能确保每个beam至少有一个候选？(待查证)
   - 或者使用了不同的候选选择策略？

## 对比参考: PyTorch的实际行为

根据 `PYTORCH_BEAM_SEARCH_ANALYSIS.md`:

```python
# PyTorch beam search with do_sample=True
probs = nn.functional.softmax(next_token_scores, dim=-1)
next_tokens = torch.multinomial(probs, num_samples=n_tokens_to_keep)
next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
next_tokens = torch.gather(next_tokens, -1, _indices)
```

MLX实现与此完全一致，但结果不同。

## 可能的解决方案

### 方案1: 使用确定性Top-K (推荐)

**修改思路:**
```python
# 不使用multinomial，改用topk
if do_sample:
    # 当前: multinomial采样
    sampled_indices = np.random.choice(...)
else:
    # 改为: 确定性topk
    top_indices = np.argsort(scores_np)[-n_tokens_to_keep:][::-1]
```

**优点:**
- 确定性: 每次运行结果一致
- 保证最好的候选被选中
- 更稳定

**缺点:**
- 不完全匹配PyTorch的do_sample=True行为
- 可能降低多样性

### 方案2: 分层采样

**修改思路:**
```python
# 从每个beam独立采样，而不是全局采样
candidates_per_beam = n_tokens_to_keep // num_beams  # 每个beam采样2个

all_candidates = []
for beam_idx in range(num_beams):
    beam_start = beam_idx * vocab_size
    beam_end = beam_start + vocab_size
    beam_scores = scores_np[beam_start:beam_end]
    beam_probs = softmax(beam_scores)
    
    # 从这个beam采样
    beam_sampled = np.random.choice(
        vocab_size,
        size=candidates_per_beam,
        replace=False,
        p=beam_probs
    )
    all_candidates.extend([(beam_idx, token) for token in beam_sampled])

# 按score排序选择top-k
...
```

**优点:**
- 保证每个beam都有代表
- 仍然有采样的随机性

**缺点:**
- 与PyTorch逻辑不完全一致

### 方案3: 调试PyTorch的实际行为

**步骤:**
1. 在PyTorch inference中添加debug print
2. 对比相同seed下，PyTorch step 1采样的候选
3. 看PyTorch是否也会跳过某些beams
4. 如果PyTorch也跳过，说明这是正常的；如果不跳过，找出差异点

**实施:**
```python
# 在 model_v2.py 的 generate 方法中添加:
if step == 1:
    print(f"[PyTorch Debug] Step 1 sampled indices: {next_tokens}")
    print(f"[PyTorch Debug] Step 1 beam sources: {next_indices}")
```

### 方案4: 混合策略

**修改思路:**
```python
# Top-50%用topk，Bottom-50%用multinomial
n_topk = n_tokens_to_keep // 2
n_sample = n_tokens_to_keep - n_topk

# 确定性选择top-k
top_indices = np.argsort(scores_np)[-n_topk:]

# 从剩余的multinomial采样
remaining_scores = scores_np.copy()
remaining_scores[top_indices] = -np.inf
remaining_probs = softmax(remaining_scores)
sampled_indices = np.random.choice(..., p=remaining_probs)

final_indices = np.concatenate([top_indices, sampled_indices])
```

**优点:**
- 平衡确定性和随机性
- 保证最好的候选被选中

## 立即可行的测试

### 测试1: 使用topk替换multinomial
```bash
# 在 mlx_model.py 的 beam_search_forward 中:
# 注释掉 multinomial 采样
# 改用 np.argsort

cd /Users/bailin/index-tts
export MLX_FIXED_SEED=42
python -m indextts.cli "今天天气真不错" -v examples/zh_vo_Main_Linaxita_2_4_24_6.wav --mlx --force -o test_topk.wav
```

### 测试2: 对比PyTorch step 1的采样
```bash
# 修改 model_v2.py 添加debug
python -m indextts.cli "今天天气真不错" -v examples/zh_vo_Main_Linaxita_2_4_24_6.wav --force -o test_pytorch_debug.wav
```

## 推荐行动

1. **首先**: 实施方案1 (Top-K)，测试是否能解决丢字问题
2. **如果有效**: 说明问题在multinomial的随机性，可以考虑混合策略
3. **如果无效**: 说明还有其他问题，需要更深入的对比分析

## 其他可能的问题点

### A. Repetition Penalty应用
当前实现在log_probs上应用repetition_penalty，需要确认:
- 是否对所有unique tokens (包括fake_inputs)
- 应用时机是否正确 (在加beam_scores之前)

### B. Stop Token选择
- 检查stop_token是否被过度惩罚
- 导致beam无法正常结束

### C. 长度归一化
已修复为使用generated_len，但需要确认:
- decoder_prompt_len = 1 是否正确
- 是否与PyTorch的计算一致

## 总结

**核心问题**: MLX beam search的multinomial采样没有选中Beam 0的候选，导致最好的beam被跳过。

**根本原因**: 在24,582个候选中采样6个时，即使最好的token概率最高，仍可能不被选中 (随机性)。

**下一步**: 实施Top-K替换multinomial，测试是否解决问题。

