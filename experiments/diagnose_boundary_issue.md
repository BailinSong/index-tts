# 英文首尾吞音问题 - 重新诊断

## ✅ 您的观察是对的

**原始错误分析：** BPE 完全无法处理英文 → 所有语音应该是乱的
**实际情况：** 只有首尾单词有吞音，中间部分正常
**结论：** 首尾吞音有其他原因！

## 🔬 新发现

### BPE 分析结果

```
输入: "Let's discover San Diego's delightful sugar options together; interested?"

Pieces (正确识别):
['▁', 'L', 'et', "'", 's', '▁', 'discover', '▁', 'S', 'an', 
 '▁', 'D', 'iego', "'", 's', '▁', 'delightful', '▁', 'sugar', 
 '▁', 'options', '▁', 'together;', '▁', 'interested', '?']

Unknown 比例: 42.3% (11/26)
  - 但不是 100%！
  - 关键 pieces 如 "interested", "Let's" 被正确识别
  - 只是映射到 UNK token (ID 2)
```

**关键洞察：**
- 模型训练时可能见过 UNK token 的模式
- 即使 UNK，token 序列本身包含了结构信息
- 所以能生成合理的语音（中间部分正常）

---

## 🎯 可能的真正原因

### 1. Text-to-Mel 对齐问题 ⭐⭐⭐⭐⭐

**最可能的原因！**

TTS 模型需要将**文本 tokens** 对齐到 **mel tokens**：

```
Text tokens:  [conditioning] + [text_tokens] + [start_mel]
                   ↓
Mel tokens:   [mel_0, mel_1, mel_2, ..., mel_N, STOP]
```

**可能的问题：**
- 第一个 mel token 对应文本的哪个位置？
- 最后一个 mel token 对应文本的哪个位置？
- 如果对齐偏移，首尾就会被"吞"

### 2. Attention Mask 边界效应 ⭐⭐⭐⭐

**Causal Attention 的特性：**

```
Position 0 (首个):  只能看到自己
Position 1:         能看到 pos 0-1
Position 2:         能看到 pos 0-2
...
Position N (最后):  能看到 pos 0-N
```

**问题：**
- 首个位置 attention 信息最少 → 可能导致首词不准
- 但最后位置应该信息最多，为什么也有问题？

### 3. Start/Stop Token 处理 ⭐⭐⭐

**当前实现：**
```python
# Start token
start_token_ids = mx.full((batch_size, 1), self.start_mel_token, dtype=mx.int32)
start_token_emb = self.mel_embedding(start_token_ids)

# 生成第一个 mel token 是从 start_token 预测的
# 如果 start_token 的位置编码不对，可能影响首个 mel token
```

**Stop token 判断：**
```python
if token_val == self.stop_mel_token:
    break
```

**可能的问题：**
- Stop token 提前触发？
- 或者最后几个 token 受 stop token 概率影响？

### 4. Positional Encoding 边界 ⭐⭐

```python
# 文本位置编码
text_pos_emb = mx.stack([self.text_pos_embedding.weight[i] for i in range(text_seq_len)], axis=0)

# Mel 位置编码
absolute_pos = context_len + step
if absolute_pos < self.mel_pos_embedding.weight.shape[0]:
    mel_pos_enc = self.mel_pos_embedding.weight[absolute_pos:absolute_pos+1]
```

**可能的问题：**
- 边界位置的 positional encoding 质量差？
- 或者 conditioning + text 的拼接处有不连续？

### 5. 英文 vs 中文的不同 ⭐⭐

**中文特点：**
- 每个字是独立的语义单元
- BPE 分词准确
- Text-to-Mel 对齐相对简单

**英文特点：**
- 单词由多个字母组成
- BPE 分词质量差（42% UNK）
- Text-to-Mel 对齐更困难
- **首尾单词可能更容易出错**

---

## 🧪 验证方法

### 测试 1: 对比不同文本长度

```python
texts = [
    "Let's",                    # 极短 - 只有首词
    "Let's go",                 # 短
    "Let's go home",            # 中
    "Let's discover... interested?",  # 长
]
```

**预期：**
- 如果是对齐问题，短文本可能问题更明显
- 如果是 attention 问题，所有长度都有问题

### 测试 2: 对比 PyTorch 版本

```bash
# 用 PyTorch 生成同样的英文
python -m indextts.cli --text "Let's discover..." --output test_pytorch.wav

# 用 MLX 生成
python -m indextts.cli --text "Let's discover..." --output test_mlx.wav --mlx
```

**预期：**
- 如果 PyTorch 也有问题 → 模型本身的限制
- 如果只有 MLX 有问题 → MLX 实现的 bug

### 测试 3: 检查生成的 mel tokens

```python
# 在 simple_forward 中打印生成的 tokens
print(f"Generated mel tokens: {generated}")

# 检查：
# - 前几个 tokens 是什么？
# - 后几个 tokens 是什么？
# - 是否有异常的 token 值？
```

### 测试 4: 对比中文边界

```python
texts = [
    "今天天气",     # 短
    "今天天气真不错",  # 中
]
```

**预期：**
- 如果中文没有首尾吞音 → 问题是英文特有的
- 如果中文也有 → 是通用的边界问题

---

## 💡 建议的修复方向

### 短期（调试）

1. **添加详细的 generation 日志**
   ```python
   # 打印每个生成的 mel token
   # 打印 attention weights
   # 打印 first/last tokens 的 logits
   ```

2. **对比 PyTorch 和 MLX 的 generated tokens**
   - 看是否一致
   - 找出哪里开始分叉

### 中期（修复）

1. **调整 temperature**
   ```python
   # 当前 temperature = 0.8
   # 尝试降低（更确定性）或升高（更多样性）
   ```

2. **检查 start_mel_token 的位置编码**
   ```python
   # 确保 start_mel_token 的位置是正确的
   # 可能需要特殊处理第一个 mel token
   ```

3. **添加文本边界的特殊 token**
   ```python
   # 在文本开始和结束添加特殊标记
   # 帮助模型识别边界
   ```

---

## 🎯 立即行动

我现在帮您：
1. ✅ 对比 PyTorch 和 MLX 生成结果
2. ✅ 添加详细的 generation 日志
3. ✅ 测试不同长度的英文文本

选择哪个？或者您想先手动测试看看 PyTorch 版本是否也有同样的问题？

