# 首单词吞音问题诊断报告

## 📊 测试结果

### 确定性模式测试
- ✅ **尾单词吞音**：有改善
- ❌ **首单词吞音**：无改善

**结论：** 首单词吞音不是随机采样的问题

---

## 🔍 可能的根本原因

### 1. MLX Conditioning 首部质量问题 ⭐⭐⭐⭐⭐

**假设：**
- MLX Conformer 输出与 PyTorch 虽然整体 correlation 0.98+
- 但序列**首部**的 correlation 可能更低
- 导致第一个 mel token 预测不准

**验证方法：**
```python
# 对比 PyTorch 和 MLX 的 conditioning 输出
# 重点检查首部 5-10 个位置的差异
```

### 2. Text-to-Mel 对齐偏移 ⭐⭐⭐⭐

**假设：**
- 第一个 mel token 应该对应第一个文字
- 但由于某种原因，对齐有偏移
- 导致第一个 mel token 实际编码了空白或其他信息

### 3. Start Mel Token 的初始化问题 ⭐⭐⭐

**假设：**
- start_mel_token 的 embedding 或位置编码不对
- 影响了后续第一个 mel token 的生成

---

## 🧪 建议的诊断步骤

### 步骤 1: 对比生成的 Mel Tokens

添加 debug 输出：
```python
# In simple_forward()
print(f"[DEBUG] First 10 generated tokens: {[int(t[0,0]) for t in generated[:10]]}")
```

**分别运行：**
- PyTorch 版本（不加 --mlx）
- MLX 版本（加 --mlx --deterministic）

**对比：**
- 前 10 个 tokens 是否相同？
- 从哪个位置开始不同？

### 步骤 2: 检查 Conditioning 首部

```python
# 在 get_conditioning_mlx() 返回前
conditioning_output = ...
print(f"[DEBUG] Conditioning first 3 positions mean: {conditioning_output[0, :3, :10]}")
```

对比 PyTorch 和 MLX 的首部输出。

### 步骤 3: 检查 Text Embedding

```python
# 在 simple_forward() 中
print(f"[DEBUG] Text tokens: {text_tokens[0, :10]}")
print(f"[DEBUG] Text embedding first 3: {text_emb[0, :3, :10]}")
```

---

## 💡 临时 Workaround

### 方法 1: 跳过首个字符

如果首单词总是有问题，可以：
```python
# 输入文本前添加一个空格或标点
text = " " + original_text  # 第一个 mel token 对应空格，丢了也无妨
```

### 方法 2: 使用 PyTorch Conditioning

```python
# 使用 Hybrid 模式
# PyTorch conditioning + MLX transformer
# 牺牲一些速度，换取准确性
```

### 方法 3: 后处理

```python
# 生成后裁剪/调整首部
# 或使用音频编辑工具修复
```

---

## 🎯 下一步行动

**选项 A: 深入调试（需要 2-3 小时）**
1. 添加详细 debug 输出
2. 对比 PyTorch 和 MLX 的每一步
3. 找到确切的差异位置
4. 针对性修复

**选项 B: 使用 Hybrid 模式（立即可用）**
```bash
# 使用 PyTorch conditioning（准确）+ MLX transformer（快速）
# 不需要修改，只需不启用 use_mlx_conditioning
```

**选项 C: 接受当前状态**
- 尾单词已修复（使用 --deterministic）
- 首单词问题：使用中文时基本不明显
- 英文使用场景较少，可以接受

---

## 📝 技术笔记

### MLX Conformer Correlation

之前测试显示整体 correlation 0.9867，但这是**平均值**。

序列不同位置的 correlation 可能不同：
- 中间部分：可能 > 0.99
- 首部：可能 < 0.95
- 尾部：可能 < 0.97

如果首部 correlation 较低，会直接影响首个 mel token。

### Causal Attention 特性

在 causal attention 中：
```
Position 0 (start_mel_token): 只能看到 conditioning + text
Position 1 (first mel token):  能看到 position 0
Position 2 (second mel token): 能看到 position 0-1
...
```

如果 position 0 (start_mel_token) 的表示不准确，会影响 position 1。

---

**建议：** 先使用 --deterministic 解决尾单词问题，首单词问题可以后续深入调试。

