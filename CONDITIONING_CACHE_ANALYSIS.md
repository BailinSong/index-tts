# Conditioning缓存可行性分析

## 问题：文本不同，Conditioning缓存有意义吗？

**答案：有意义！因为Conditioning只依赖voice prompt，不依赖文本。**

---

## 📊 Conditioning计算流程

### 当前流程分析

```python
# indextts/infer_v2.py

# ===== 阶段1: Semantic Features（已缓存✅）=====
if self.cache_spk_cond is None or self.cache_spk_audio_prompt != spk_audio_prompt:
    # 只有voice prompt改变时才重新计算
    audio_16k = load_audio(spk_audio_prompt)  # 加载音频
    input_features = extract_features(audio_16k)  # W2V-BERT特征提取
    spk_cond_emb = self.get_emb(input_features)  # Semantic features
    
    self.cache_spk_cond = spk_cond_emb  # ✅ 缓存semantic features
    self.cache_spk_audio_prompt = spk_audio_prompt
else:
    spk_cond_emb = self.cache_spk_cond  # ✅ 使用缓存

# ===== 阶段2: GPT Conditioning（每次都计算❌）=====
# 在每个segment的推理中：
codes = self.mlx_transformer.inference_speech(
    spk_cond_emb,  # ← 虽然这个是缓存的
    text_tokens,   # ← 这个每次不同
    emo_cond_emb,
    emo_vec,
    ...
)

# 在MLX模型内部（每次都执行）：
def inference_speech(self, speech_condition, text_inputs, ...):
    # ❌ 每次都重新计算，即使voice prompt相同！
    speech_conditioning_latent = self.get_conditioning_mlx(
        speech_condition,  # spk_cond_emb（虽然来自缓存）
        cond_lengths
    )
    # ↑ Conformer + Perceiver计算 (~2-3s)
    
    # 然后才用text生成
    codes = self.simple_forward(
        text_inputs,  # ← 文本在这里才用到
        conditioning=speech_conditioning_latent,
        ...
    )
```

---

## 🎯 关键发现

### Conditioning计算的两个阶段

| 阶段 | 输入 | 输出 | 是否缓存 | 耗时 | 依赖文本？ |
|------|------|------|----------|------|-----------|
| **Semantic** | Voice audio | spk_cond_emb | ✅ 已缓存 | ~2-3s | ❌ 否 |
| **GPT Cond** | spk_cond_emb | speech_conditioning_latent | ❌ 未缓存 | ~2-3s | ❌ 否 |
| **Generation** | text + cond_latent | mel_codes | 不可缓存 | ~6-7s | ✅ 是 |

### 发现

1. ✅ **Semantic features已缓存**
   - 输入：voice audio
   - 输出：spk_cond_emb
   - 依赖：只依赖voice prompt

2. ❌ **GPT Conditioning未缓存**
   - 输入：spk_cond_emb（来自缓存）
   - 输出：speech_conditioning_latent
   - 依赖：只依赖spk_cond_emb，**不依赖文本**！
   - 耗时：~2-3s（Conformer + Perceiver）
   
3. ⏭️ **Generation不可缓存**
   - 输入：text + speech_conditioning_latent
   - 输出：mel_codes
   - 依赖：**依赖文本**，每次不同

---

## 💡 优化价值分析

### 场景：相同voice prompt，不同文本

```
第1次推理："今天天气真不错" + voice.wav
  ├─ Semantic: 计算 → 缓存 (2-3s)
  ├─ GPT Cond: 计算 (2-3s) ❌ 可以缓存！
  └─ Generation: 计算 (6-7s)
  Total: ~11-13s

第2次推理："你好世界" + voice.wav (相同voice)
  ├─ Semantic: 缓存命中 (0.01s) ✅
  ├─ GPT Cond: 重新计算 (2-3s) ❌ 浪费！
  └─ Generation: 计算 (6-7s)
  Total: ~9-10s

如果缓存GPT Cond:
  ├─ Semantic: 缓存命中 (0.01s) ✅
  ├─ GPT Cond: 缓存命中 (0.01s) ✅
  └─ Generation: 计算 (6-7s)
  Total: ~6-7s  ← 节省2-3s！
```

### 使用场景

**非常有价值！**

1. **批量生成（相同voice，不同文本）**
   ```
   同一个speaker说100句话：
   - 第1句：全部计算（~11s）
   - 第2-100句：缓存命中，每句节省2-3s
   - 总节省：~200-300s（99句 × 2-3s）
   ```

2. **交互式对话系统**
   ```
   用户多轮对话，使用相同voice：
   - 每次回复节省2-3s
   - 用户体验提升显著
   ```

3. **A/B测试不同文本**
   ```
   同一个voice，测试不同prompt：
   - 节省大量重复计算
   ```

---

## 📋 实施方案

### 缓存策略

**缓存Key:**
```python
# 方法1：基于文件路径（简单）
cache_key = spk_audio_prompt  # 文件路径

# 方法2：基于内容hash（更可靠）
import hashlib
with open(spk_audio_prompt, 'rb') as f:
    audio_hash = hashlib.md5(f.read()).hexdigest()
cache_key = audio_hash
```

**缓存Value:**
```python
cached_conditioning = {
    'speech_conditioning_latent': speech_conditioning_latent,  # MLX或PyTorch
    'emo_conditioning_latent': emo_conditioning_latent,  # 如果计算了
    'spk_cond_emb': spk_cond_emb,  # 已有
    'emo_cond_emb': emo_cond_emb,  # 已有
}
```

### 代码修改

#### 1. 添加缓存字典（`__init__`）
```python
# indextts/infer_v2.py __init__
self.cache_gpt_conditioning = {}  # GPT Conditioning缓存
```

#### 2. 在推理时检查缓存
```python
# infer_generator中，在调用mlx_transformer.inference_speech前：

cache_key = spk_audio_prompt  # 或使用hash

if cache_key in self.cache_gpt_conditioning:
    # 缓存命中
    print(f">> [Cache Hit] Using cached GPT conditioning")
    cached = self.cache_gpt_conditioning[cache_key]
    speech_conditioning_latent = cached['speech_conditioning_latent']
    # 直接使用缓存的conditioning进行generation
    codes = self.mlx_transformer.simple_forward(
        text_mlx,
        conditioning=speech_conditioning_latent,  # 使用缓存
        ...
    )
else:
    # 缓存未命中，正常计算
    codes, speech_conditioning_latent = self.mlx_transformer.inference_speech(
        spk_cond_emb,  # 会计算Conditioning
        text_tokens,
        ...
    )
    # 缓存结果
    self.cache_gpt_conditioning[cache_key] = {
        'speech_conditioning_latent': speech_conditioning_latent
    }
    print(f">> [Cache Miss] Computed and cached GPT conditioning")
```

---

## 🚀 预期收益

### 性能提升

**首次推理（冷启动）：**
```
Total: 20.36s
├─ Semantic: 2-3s（计算并缓存）
├─ GPT Cond: 2-3s（计算并缓存）
└─ 其他: ~15s
```

**第2+次推理（相同voice）：**
```
Total: ~14-15s  ✅ 节省2-3s (10-15%)
├─ Semantic: 0.01s（缓存命中）
├─ GPT Cond: 0.01s（缓存命中）✨
└─ 其他: ~14-15s
```

### ROI分析

| 指标 | 评估 |
|------|------|
| 实施难度 | ⭐⭐ (简单) |
| 收益 | ⭐⭐⭐⭐ (中-高) |
| 风险 | ⭐ (低) |
| 工作量 | 1天 |

**ROI：⭐⭐⭐⭐⭐（极高）**

特别适合：
- 批量生成场景
- 交互式系统
- A/B测试

---

## ✅ 结论

### Conditioning缓存**非常有意义**！

**原因：**
1. ✅ Conditioning**完全不依赖文本**
   - 只依赖voice prompt
   - 可以安全缓存

2. ✅ 节省显著（2-3s per 推理）
   - 第2+次推理提速10-15%
   - 批量场景收益更大

3. ✅ 实施简单
   - 代码改动小
   - 风险低

4. ✅ 当前已有部分缓存
   - `cache_spk_cond`缓存了semantic features
   - 只需扩展到缓存GPT conditioning结果

**建议：立即实施！**

---

## 📋 实施计划

### Step 1: 添加缓存字典
```python
self.cache_gpt_conditioning = {}
```

### Step 2: 检查并使用缓存
```python
if spk_audio_prompt in self.cache_gpt_conditioning:
    # 使用缓存
    cond = self.cache_gpt_conditioning[spk_audio_prompt]
else:
    # 计算并缓存
    cond = compute_conditioning(...)
    self.cache_gpt_conditioning[spk_audio_prompt] = cond
```

### Step 3: 验证
- 相同voice，不同文本→缓存命中
- 不同voice→缓存未命中，重新计算

**预计工作量：4-6小时**

