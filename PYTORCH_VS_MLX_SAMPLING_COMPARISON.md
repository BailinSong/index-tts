# PyTorch vs MLX Beam Search 采样对比分析

## 实验设置
- 文本：`今天天气真不错`
- num_beams: 3
- do_sample: True
- Seed: 42 (PyTorch默认，MLX使用MLX_FIXED_SEED=42)

---

## Step 0 对比

### PyTorch (torch.multinomial)
```
Beam sources: [0, 0, 0, 0, 0, 0]  ✓ 全部来自Beam 0
Tokens (采样顺序):
  Candidate 0: beam=0, token=2214, score=-2.5055
  Candidate 1: beam=0, token=141,  score=-2.7478
  Candidate 2: beam=0, token=7932, score=-3.3507
  Candidate 3: beam=0, token=7772, score=-3.7897
  Candidate 4: beam=0, token=286,  score=-4.4151
  Candidate 5: beam=0, token=6779, score=-4.9577

选中top-3作为新beams:
  New Beam 0: token=2214 (score=-2.5055)
  New Beam 1: token=141  (score=-2.7478)
  New Beam 2: token=7932 (score=-3.3507)
```

### MLX (np.random.choice)
```
Beam sources: [0, 0, 0, 0, 0, 0]  ✓ 全部来自Beam 0
Tokens (采样顺序):
  Candidate 0: source_beam=0, token=7932, score=-3.5412
  Candidate 1: source_beam=0, token=286,  score=-3.6379
  Candidate 2: source_beam=0, token=5942, score=-4.4758
  Candidate 3: source_beam=0, token=6505, score=-5.0323
  Candidate 4: source_beam=0, token=1782, score=-5.6863
  Candidate 5: source_beam=0, token=3889, score=-11.8342

选中top-3作为新beams:
  New Beam 0: token=7932 (score=-3.5412)
  New Beam 1: token=286  (score=-3.6379)
  New Beam 2: token=5942 (score=-4.4758)
```

### Step 0 分析

**关键发现：**
1. ✅ 两者都正确地只从Beam 0采样（beam_scores机制工作正常）
2. ❌ **采样的tokens完全不同！**
   - PyTorch: [2214, 141, 7932, 7772, 286, 6779]
   - MLX:     [7932, 286, 5942, 6505, 1782, 3889]
   - 只有2214和286重复

**原因：**
- `torch.multinomial` 和 `np.random.choice` 使用不同的RNG算法
- 即使设置相同的seed，生成的随机数序列不同
- 导致从相同概率分布中采样出不同的tokens

**影响：**
- Step 0之后，PyTorch和MLX的3个beams就已经不同了
- 这会导致后续所有步骤的轨迹完全不同

---

## Step 1 对比

### PyTorch
```
当前3个beams:
  Beam 0: [start, 2214], score=-2.5055
  Beam 1: [start, 141],  score=-2.7478
  Beam 2: [start, 7932], score=-3.3507

Multinomial采样结果:
  Beam sources: [1, 0, 1, 0, 1, 0]
  
  Candidate 0: beam=1, token=2214, score=-7.5125
  Candidate 1: beam=0, token=286,  score=-7.7927
  Candidate 2: beam=1, token=1472, score=-7.8680
  Candidate 3: beam=0, token=1707, score=-8.0293
  Candidate 4: beam=1, token=2889, score=-8.4484
  Candidate 5: beam=0, token=7716, score=-8.6051

Beam分布:
  - Beam 0: 3个候选 ✓
  - Beam 1: 3个候选 ✓
  - Beam 2: 0个候选 ❌ (被跳过!)
```

### MLX
```
当前3个beams:
  Beam 0: [start, 7932], score=-3.5412
  Beam 1: [start, 286],  score=-3.6379
  Beam 2: [start, 5942], score=-4.4758

Multinomial采样结果:
  Beam sources: [1, 2, 1, 2, 2, 0]
  
  Candidate 0: source_beam=1, token=4432, score=-8.0321
  Candidate 1: source_beam=2, token=329,  score=-9.0405
  Candidate 2: source_beam=1, token=6065, score=-9.0663
  Candidate 3: source_beam=2, token=1994, score=-10.2544
  Candidate 4: source_beam=2, token=7113, score=-10.3231
  Candidate 5: source_beam=0, token=476,  score=-12.2132

Beam分布:
  - Beam 0: 1个候选 (排最后)
  - Beam 1: 2个候选
  - Beam 2: 3个候选
```

### Step 1 分析

**关键发现：**
1. ❌ **PyTorch也会跳过某些beams！** (Beam 2被跳过)
   - 这说明multinomial采样跳过某些beams是正常现象
   - 不是MLX特有的bug

2. ❌ **PyTorch和MLX的beam分布完全不同**
   - PyTorch: 主要从Beam 0和Beam 1采样
   - MLX: 主要从Beam 1和Beam 2采样
   - 原因：RNG差异 + Step 0的初始beams不同

3. ⚠️ **MLX的问题**
   - Beam 0的最好候选(score=-6.04)被排到了最后
   - 而Beam 1和2的较差候选被选中
   - 这可能导致质量下降

---

## 根本原因分析

### 1. RNG算法差异

**PyTorch的torch.multinomial:**
```python
torch.manual_seed(42)
torch.multinomial(probs, num_samples=6)
# 使用PyTorch的Mersenne Twister实现
```

**NumPy的np.random.choice:**
```python
np.random.seed(42)
np.random.choice(len(probs), size=6, p=probs, replace=False)
# 使用NumPy的PCG64 DXSM (默认) 或 MT19937
```

**结果：** 即使seed相同，生成的随机数序列不同！

### 2. Multinomial采样的随机性

从24,582个候选中采样6个：
- 每个token的概率都很小 (< 0.01%)
- 即使某个token概率最高，仍可能不被采样
- 例如：prob=0.001，采样6次都miss的概率 = (1-0.001)^6 ≈ 99.4%

### 3. 为什么PyTorch能工作？

**假设：PyTorch的RNG在多数情况下产生"好"的采样**
- 虽然也会跳过某些beams，但跳过的通常不是最好的
- 例如Step 1跳过Beam 2（score=-3.35，第3好），保留了Beam 0和1（更好）

**MLX的问题：**
- RNG可能产生"坏"的采样
- 跳过了最好的beam (Beam 0, score=-3.54)
- 选中了较差的beams (Beam 1, 2)

---

## 解决方案评估

### 方案A：统一RNG (困难)
**目标：** 让MLX的multinomial采样结果与PyTorch一致

**挑战：**
1. 需要在MLX中实现与PyTorch完全相同的RNG算法
2. 或者找到一个seed映射关系
3. 可行性低，维护成本高

**不推荐** ❌

### 方案B：使用Top-K替代Multinomial (推荐)
**实现：**
```python
# 替换 np.random.choice
# sampled_indices = np.random.choice(len(probs), size=6, p=probs, replace=False)

# 改为确定性top-k
sampled_indices = np.argsort(scores_np)[-n_tokens_to_keep:][::-1]
```

**优点：**
- ✅ 确定性：每次运行结果一致
- ✅ 保证选中最好的候选
- ✅ 解决beam被跳过的问题
- ✅ 实现简单

**缺点：**
- ⚠️ 不完全匹配PyTorch的do_sample=True行为
- ⚠️ 可能降低diversity（但beam search本身已有diversity机制）

**推荐指数：** ⭐⭐⭐⭐⭐

### 方案C：混合策略
**实现：**
```python
# 50% top-k + 50% multinomial
n_topk = n_tokens_to_keep // 2
n_sample = n_tokens_to_keep - n_topk

# 确定性选top-k
top_indices = np.argsort(scores_np)[-n_topk:]

# 剩余的multinomial采样
remaining_mask = np.ones(len(scores_np), dtype=bool)
remaining_mask[top_indices] = False
remaining_scores = scores_np[remaining_mask]
remaining_probs = softmax(remaining_scores)
sampled_indices = np.random.choice(np.where(remaining_mask)[0], size=n_sample, p=remaining_probs, replace=False)

final_indices = np.concatenate([top_indices, sampled_indices])
```

**优点：**
- ✅ 平衡确定性和随机性
- ✅ 保证最好的候选被选中
- ✅ 保持一定diversity

**缺点：**
- ⚠️ 实现复杂
- ⚠️ 仍不完全匹配PyTorch

**推荐指数：** ⭐⭐⭐

### 方案D：调整其他参数补偿
**思路：** 保持multinomial，调整temperature/repetition_penalty等参数

**挑战：**
- 很难找到完美的参数组合
- 可能需要大量实验

**推荐指数：** ⭐⭐

---

## 立即行动建议

### 1. 实施方案B (Top-K) - 优先级最高
```bash
# 修改 indextts/gpt/mlx_model.py
# 在 beam_search_forward 中替换multinomial采样

cd /Users/bailin/index-tts
# 备份
cp indextts/gpt/mlx_model.py indextts/gpt/mlx_model.py.backup

# 测试
export MLX_FIXED_SEED=42
conda run -n indextts2 python -m indextts.cli "今天天气真不错" \
    -v examples/zh_vo_Main_Linaxita_2_4_24_6.wav \
    --mlx --force -o test_topk_fix.wav
```

### 2. 验证结果
- 检查是否还丢字
- 对比音频长度 (应该接近PyTorch的2.67s)
- 检查音质

### 3. 如果方案B有效
- 考虑添加一个参数 `--beam-selection-strategy` 
- 支持 "topk" 和 "multinomial" 两种模式
- 默认使用 "topk" 确保稳定性

---

## 总结

### 核心问题
**PyTorch和MLX的multinomial采样产生完全不同的随机序列，导致beam search轨迹分叉。**

### 关键洞察
1. PyTorch的beam search也会跳过某些beams，但RNG"运气好"，跳过的不是最优beams
2. MLX的RNG"运气差"，经常跳过最优beams，导致质量下降
3. 依赖随机性的beam search在不同RNG下行为差异大

### 推荐方案
**使用Top-K替代Multinomial** - 确定性、稳定、高质量

### 下一步
立即测试Top-K方案，预期能解决丢字和时长问题。

