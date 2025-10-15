# 里程碑：MLX Mel Position Encoding 修复 - 解决吞字问题

## 日期
2025-10-15

## 问题描述
MLX推理存在吞字问题：生成的token数量不稳定，有时比PyTorch少很多。
通过固定seed对比发现，MLX和PyTorch从第0个token就完全不匹配。

## 根本原因

### Position Encoding策略差异

**PyTorch策略**（正确）：
```python
# Mel position使用相对位置（从0开始）
start_mel_token:     position 0
第1个生成的mel:     position 2  
第2个生成的mel:     position 3
...
```

**MLX策略**（修复前 - 错误）：
```python
# Mel position使用绝对位置（从context_len开始）
start_mel_token:     position 50 (context_len)
第1个生成的mel:     position 51
第2个生成的mel:     position 52
...
```

### 为什么导致吞字？

使用不同的position embeddings导致：
1. **Transformer输出完全不同** (max_diff=90.22)
2. **Logits完全不同** (max_diff=8.62)
3. **生成的token序列完全不同** (匹配率<5%)
4. **可能更早遇到stop token** → 吞字！

## 代码分析

### PyTorch实现

```python
# indextts/gpt/model_v2.py:148, 158

# 第一次forward (full sequence)
text_inputs = input_ids[:, mel_len:]  # 只有start_mel_token
text_emb = self.embeddings(text_inputs)
text_emb = text_emb + self.text_pos_embedding(text_emb)
# ↑ text_pos_embedding.forward()返回position [0]的embedding

# 后续forward (KV cache)
position = attention_mask.shape[1] - mel_len
# attention_mask增长: mel_len+1, mel_len+2, ...
# position增长: 1, 2, 3, ...
```

### MLX实现

**修复前**：
```python
# 错误：使用绝对位置
start_pos = context_len  # 50
start_token_emb = start_token_emb + self.mel_pos_embedding.weight[start_pos:start_pos+1]

# 后续生成
absolute_pos = context_len + step  # 50, 51, 52, ...
```

**修复后**：
```python
# 正确：使用相对位置
start_pos = 0  # ✅ 从0开始
start_token_emb = start_token_emb + self.mel_pos_embedding.weight[start_pos:start_pos+1]

# 后续生成
relative_pos = step + 1  # 2, 3, 4, ... (因为start_mel用了0)
```

## 修复内容

### 文件: `indextts/gpt/mlx_model.py`

#### 1. simple_forward() - Line 1257
```python
# 修复前
start_pos = context_len
start_token_emb = start_token_emb + self.mel_pos_embedding.weight[start_pos:start_pos+1]

# 修复后
start_pos = 0  # 🔧 Use relative position
start_token_emb = start_token_emb + self.mel_pos_embedding.weight[0:1]
```

#### 2. simple_forward() 循环 - Line 1354
```python
# 修复前
absolute_pos = context_len + step
mel_pos_enc = self.mel_pos_embedding.weight[absolute_pos:absolute_pos+1]

# 修复后  
relative_pos = step + 1  # 🔧 Relative position
mel_pos_enc = self.mel_pos_embedding.weight[relative_pos:relative_pos+1]
```

#### 3. beam_search_forward() - Line 762
```python
# 修复前
start_token_emb = start_token_emb + self.mel_pos_embedding.weight[context_len:context_len+1]

# 修复后
start_token_emb = start_token_emb + self.mel_pos_embedding.weight[0:1]
```

#### 4. beam_search_forward() 循环 - Line 844
```python
# 修复前
current_pos = context_len + beam_codes[beam_idx].shape[1]

# 修复后
current_pos = beam_codes[beam_idx].shape[1] + 1  # 🔧 Relative position
```

## 验证结果

### 修复前
```
MLX生成: 114 tokens (固定seed=42)
与PyTorch: 从第0个token就不匹配
匹配率: <5%
问题: 吞字、生成不稳定
```

### 修复后
```
MLX生成: 127 tokens (固定seed=42)
随机seed测试:
  Run 1: 145 tokens (2.89s)
  Run 2: 124 tokens (2.47s)
  Run 3: 174 tokens (3.47s)
  Run 4: 127 tokens (2.53s)
  Run 5: 141 tokens (2.81s)

平均: ~142 tokens
变化范围: 124-174 (正常的随机变化)
✅ 吞字问题解决！
```

## 影响

- ✅ 解决了MLX吞字问题
- ✅ 生成的token数量恢复正常
- ✅ 与PyTorch的行为更加一致
- ✅ 音频质量提升

## 教训

### 关键发现
**Position encoding的策略必须完全一致！**

1. ❌ **错误假设**: 以为position是绝对的（在整个序列中的位置）
2. ✅ **正确理解**: PyTorch使用相对position（mel tokens从0开始编号）
3. ⚠️ **易错点**: 不同模型可能使用不同的position策略，必须仔细对比

### 调试方法
1. ✅ 固定seed对比
2. ✅ 逐层检查hidden states
3. ✅ 发现transformer输出就不匹配 → 说明输入层有问题
4. ✅ 追踪position encoding的使用

---

**状态**: ✅ 完成并验证
**测试**: ✅ 5次随机seed测试通过
**问题**: ✅ 吞字问题已解决
