# 首单词吞音调试指南

## 🎯 目标

定位并修复 MLX 版本首单词吞音问题

## 📊 已知信息

- ✅ PyTorch 版本：无吞音
- ❌ MLX 版本：首单词吞音
- ✅ 确定性模式（--deterministic）：尾单词已改善，首单词仍有问题

**结论：** 首单词问题不是随机采样导致的

---

## 🔬 调试工具

### 1. Debug 模式测试

```bash
# 激活环境
conda activate indextts2

# 运行带 debug 输出的测试
bash TEST_DEBUG_TOKENS.sh
```

**查看输出：**
- `[DEBUG] Text tokens (first 10)` - 输入的文本 tokens
- `[DEBUG] First token logits top 10` - 首个 mel token 的 top-10 logits
- `[DEBUG] First 15 generated tokens` - 前 15 个生成的 mel tokens

### 2. 对比 PyTorch vs MLX

```bash
# 对比测试
bash TEST_COMPARE_TOKENS.sh
```

**关键对比点：**
1. 第一个生成的 mel token 是否相同？
2. 如果不同，差异有多大？
3. 从第几个 token 开始分叉？

---

## 🔍 诊断步骤

### Step 1: 运行 Debug 测试

```bash
conda activate indextts2
cd /Users/bailin/index-tts

# MLX + Debug
python -m indextts.cli \
    "Let's discover San Diego's delightful sugar options together; interested?" \
    -v examples/zh_vo_Main_Linaxita_2_4_24_6.wav \
    --mlx --deterministic --debug --force
```

**记录输出：**
```
[DEBUG] Text tokens (first 10): [?, ?, ?, ...]
>> [MLX] First token: XXXX
[DEBUG] First token logits top 10:
  #1: token XXXX, logit YY.YYYY
  #2: token XXXX, logit YY.YYYY
  ...
[DEBUG] First 15 generated tokens: [?, ?, ?, ...]
```

### Step 2: 运行 PyTorch 基准测试

```bash
# PyTorch（无 --mlx）
python -m indextts.cli \
    "Let's discover San Diego's delightful sugar options together; interested?" \
    -v examples/zh_vo_Main_Linaxita_2_4_24_6.wav \
    --force
```

**注意：** PyTorch 版本没有 debug 输出，但会显示 "Generated XX mel tokens"

### Step 3: 对比分析

**问题定位：**

#### 情况 A: 第一个 token 就不同
→ 问题在初始生成阶段
→ 可能原因：
  - Conditioning 质量
  - Start mel token 处理
  - 位置编码

#### 情况 B: 第一个相同，后续分叉
→ 问题在 autoregressive 循环
→ 可能原因：
  - KV cache 问题
  - 位置编码递增问题
  - Attention 计算差异

#### 情况 C: Token 序列完全不同
→ 更根本的问题
→ 可能原因：
  - Conditioning 输出完全不同
  - Text embedding 问题
  - Conformer 输出质量

---

## 📝 下一步修复方向

### 如果是 Conditioning 问题

**检查：**
```python
# 在 get_conditioning_mlx() 添加输出
print(f"[DEBUG] Conditioning output shape: {conditioning.shape}")
print(f"[DEBUG] Conditioning first position mean: {conditioning[0, 0, :10]}")
```

**可能修复：**
- 调整 Conformer 参数
- 检查 Conv2d subsampling
- 验证 xscale 是否正确

### 如果是 Start Token 问题

**检查：**
```python
# 在 simple_forward() 添加
print(f"[DEBUG] Start token embedding mean: {start_token_emb[0, 0, :10]}")
print(f"[DEBUG] Start position: {start_pos}")
```

**可能修复：**
- 调整 start_mel_token 的位置编码
- 检查 context 拼接是否正确

### 如果是 Text Embedding 问题

**检查：**
```python
print(f"[DEBUG] Text embedding first position: {text_emb[0, 0, :10]}")
print(f"[DEBUG] Text positional encoding: {text_pos_emb[0, :10]}")
```

---

## 🧪 实验性修复

### 尝试 1: 调整首个 token 的 temperature

```python
# 在 simple_forward() 中
# 首个 token 使用更低的 temperature（更确定）
first_token_temperature = 0.5
if temperature > 0:
    logits = logits / first_token_temperature
```

### 尝试 2: 使用更多 context 信息

```python
# 使用最后 N 个位置的平均
N = 3
logits = self.mel_head(hidden[:, -N:, :].mean(dim=1, keepdim=True))
```

### 尝试 3: 添加额外的 LayerNorm

```python
# 在 mel_head 之前
hidden = self.extra_norm(hidden)
logits = self.mel_head(hidden[:, -1:, :])
```

---

## 💡 快速验证技巧

### 检查 Token 数量

```bash
# 如果 PyTorch 生成 120 tokens
# 而 MLX 生成 95 tokens
# → 说明提前停止或跳过了一些内容
```

### 听音频差异

```bash
# 生成两个版本
python -m indextts.cli "text" -v voice.wav --force -o pytorch.wav
python -m indextts.cli "text" -v voice.wav --mlx --deterministic --force -o mlx.wav

# 使用音频软件对比波形
# 看首部差异从第几秒开始
```

---

## 📞 需要的信息

请提供以下 debug 输出：

1. **MLX Debug 输出：**
   ```
   [DEBUG] Text tokens (first 10): [...]
   [DEBUG] First token logits top 10: [...]
   [DEBUG] First 15 generated tokens: [...]
   ```

2. **PyTorch Token 数量：**
   ```
   Generated XX mel tokens
   ```

3. **音频对比：**
   - 首单词具体丢失了什么？（例如："Let's" 变成 "et's"）
   - 从第几个音节/字母开始有问题？

有了这些信息，我们就能精确定位问题并修复！

