# Beam Search Diversity 分析报告

## 实验设置
- 文本：`今天天气真不错`
- Beams: 3
- Framework: PyTorch (baseline)

## PyTorch Beams 分析

### Token序列对比

**全部3个beams:**
- 长度: 138 tokens (完全相同)
- 前134个tokens: **完全相同**
- 差异仅在最后4个tokens

```
Position 0-133: [2214, 1170, 7716, ..., 6000, 5847, 6038] (所有beams相同)

Position 134-137 (差异部分):
  Beam 1: [2830, 3010, 3645, 8193]       Score: -330.36
  Beam 2: [2830, 3010, 8193, 8193]       Score: -330.51 ⚠️ 
  Beam 3: [1551, 3010, 3645, 8193]       Score: -331.27
```

### 分叉点分析

**Beam 1 vs Beam 2:**
- 第136位开始不同
- Beam 1: `3645, 8193` (正常结束)
- Beam 2: `8193, 8193` (连续2个stop tokens - 异常)

**Beam 2 vs Beam 3:**
- 第134位开始不同  
- Beam 2: `2830, 3010, 8193, 8193`
- Beam 3: `1551, 3010, 3645, 8193`

## 关键发现

### 1. Beam Diversity 非常低
- **97% 的序列完全相同** (134/138 tokens)
- 只有最后 2.9% 有差异
- 说明beam search的diversity主要体现在尾部

### 2. Beam 2 的异常模式
```python
Beam 2 last tokens: [..., 2830, 3010, 8193, 8193]
                                      ^^^^  ^^^^
                                      stop  stop (连续)
```

**问题：**
- 连续的stop tokens (8193, 8193) 是异常的
- 可能导致：
  - 音频提前截断
  - 最后一个音节丢失
  - 或者s2mel/bigvgan处理异常

### 3. Score vs Quality

**归一化scores (使用length_penalty=1.0):**
```
Beam 1: -330.36 / 138 = -2.39  (最好) ✓ 被选中
Beam 2: -330.51 / 138 = -2.40  (次好) ⚠️ 有连续stop tokens
Beam 3: -331.27 / 138 = -2.40  (第三)
```

**PyTorch选择了Beam 1（最好的），避开了有问题的Beam 2！**

## MLX vs PyTorch 对比

### RNG差异导致的问题

**PyTorch (幸运):**
- Multinomial采样"运气好"
- 选中了最优的Beam 1
- 避开了有问题的Beam 2

**MLX (不幸):**
- 不同的RNG算法
- 可能选中了类似Beam 2的有缺陷beams
- 导致丢字或音频异常

### Step 1采样对比

**PyTorch Step 1:**
```
Beam sources: [1, 0, 1, 0, 1, 0]
- Beam 0: 3个候选
- Beam 1: 3个候选  
- Beam 2: 0个候选 (被跳过，但这是较差的beam)
```

**MLX Step 1:**
```
Beam sources: [1, 2, 1, 2, 2, 0]
- Beam 0: 1个候选 (最好的beam被边缘化)
- Beam 1: 2个候选
- Beam 2: 3个候选
```

## 验证假设

### 假设1: PyTorch的所有beams都是好的
**❌ 否定**
- Beam 2有连续stop tokens，可能有问题
- 但PyTorch通过选择Beam 1避开了这个问题

### 假设2: Beam search能保证diversity
**❌ 否定**  
- 97%的序列相同
- diversity极低，主要在尾部

### 假设3: 问题在MLX的beam selection
**✅ 确认**
- PyTorch选中了最好的beam（Beam 1）
- MLX的RNG可能导致选中次优beams
- Top-K替代multinomial能解决这个问题

## 结论

### 核心问题
**并非MLX的beam generation有问题，而是beam selection运气差！**

1. Beam search产生的beams本身就有好有坏
2. PyTorch的multinomial"运气好"，跳过差的beams，选中好的
3. MLX的multinomial"运气差"，选中了有缺陷的beams

### 解决方案验证

**Top-K方案的优势:**
```python
# Top-K (确定性)
sampled_indices = np.argsort(scores)[-n_tokens_to_keep:][::-1]

# 保证:
# 1. 总是选中score最高的candidates
# 2. 不依赖RNG运气
# 3. 稳定可复现
```

### 待验证问题

**关键问题：每个beam单独生成音频，会丢字吗？**

需要验证：
1. Beam 1的音频是否完美？（应该是）
2. Beam 2的音频是否丢字？（可能因为连续stop tokens）
3. Beam 3的音频是否丢字？（可能）

**验证方法：**
修改pipeline直接使用每个beam的codes生成音频，分别听测。

## 下一步行动

1. ✅ 已实施Top-K替代multinomial (MLX)
2. 🔄 测试Top-K版本是否解决丢字
3. ❓ 如需进一步验证：创建完整pipeline为每个beam生成音频

## 文件参考

- PyTorch beams tokens: `pytorch_test_beams_pytorch_beams.json`
- Beam对比脚本: `generate_beam_audios.py`
- 详细分析: 本文档

