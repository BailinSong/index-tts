# feat/memory-optimization 分支内存优化技术分析

## 概述

`feat/memory-optimization` 分支通过 **延迟加载（Lazy Loading）** 和 **即时卸载（Just-in-Time Unloading）** 技术，实现了显著的内存节省。

---

## 🎯 核心优化策略

### 1. Semantic Model 延迟加载与即时卸载 ✨ **节省 ~1.0GB**

#### 问题识别
Semantic Model (W2V-BERT) 仅在提取音频特征时使用，但传统实现在初始化时就加载，占用内存约 1.0GB。

#### 优化方案

**Before (主分支)**:
```python
# indextts/infer_v2.py __init__
# ❌ 立即加载，始终占用内存
self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
    os.path.join(self.model_dir, self.cfg.w2v_stat))
self.semantic_model = self.semantic_model.to(self.device)
self.semantic_model.eval()
```

**After (feat/memory-optimization)**:
```python
# 🎯 延迟加载标志
self.semantic_model = None
self.semantic_mean = None
self.semantic_std = None
self.semantic_model_loaded = False
self.semantic_stat_path = os.path.join(self.model_dir, self.cfg.w2v_stat)
print(">> Semantic Model (W2V-BERT): Lazy loading enabled (saves ~1.0GB)")

# 🎯 延迟加载函数
def _ensure_semantic_loaded(self):
    """延迟加载 Semantic Model（仅在提取特征时加载）"""
    if not self.semantic_model_loaded:
        print(">> Loading Semantic Model (W2V-BERT) for feature extraction...")
        from indextts.utils.maskgct_utils import build_semantic_model
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(self.semantic_stat_path)
        self.semantic_model = self.semantic_model.to(self.device)
        self.semantic_model.eval()
        self.semantic_mean = self.semantic_mean.to(self.device)
        self.semantic_std = self.semantic_std.to(self.device)
        self.semantic_model_loaded = True
        print(">> Semantic Model loaded (~1.0GB)")

# 🎯 即时卸载函数
def _unload_semantic(self):
    """卸载 Semantic Model，释放内存"""
    if self.semantic_model_loaded:
        print(">> Unloading Semantic Model...")
        del self.semantic_model
        del self.semantic_mean
        del self.semantic_std
        self.semantic_model = None
        self.semantic_mean = None
        self.semantic_std = None
        self.semantic_model_loaded = False
        
        # 清理内存
        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        print(">> Semantic Model unloaded (~1.0GB freed)")
```

#### 使用模式

```python
# 在 infer_generator 中，音频特征提取时
audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

# 1️⃣ 加载模型（按需）
self._ensure_semantic_loaded()

# 2️⃣ 提取特征
inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
input_features = inputs["input_features"].to(self.device)
attention_mask = inputs["attention_mask"].to(self.device)
spk_cond_emb = self.get_emb(input_features, attention_mask)

# 3️⃣ 量化特征（使用 semantic_codec）
_, S_ref = self.semantic_codec.quantize(spk_cond_emb)

# 4️⃣ 立即卸载，释放内存
self._unload_semantic()

# 5️⃣ 继续后续推理（不再需要 semantic model）
ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
# ... 剩余推理流程 ...
```

**关键点**：
- ✅ 仅在需要时加载（音频特征提取）
- ✅ 使用完立即卸载
- ✅ Semantic Codec 保留（推理时需要 `vq2emb` 查表）
- ✅ 使用 `gc.collect()` 强制垃圾回收
- ✅ M4/MPS 调用 `torch.mps.empty_cache()`

---

### 2. Qwen Emotion 延迟加载 ✨ **节省 ~1.2GB**

#### 问题识别
Qwen Emotion 模型仅在使用 `use_emo_text=True` 时需要，但传统实现总是加载，占用内存约 1.2GB。

#### 优化方案

**Before (主分支)**:
```python
# ❌ 立即加载，始终占用内存
self.qwen_emo = QwenEmotion(os.path.join(self.model_dir, self.cfg.qwen_emo_path))
```

**After (feat/memory-optimization)**:
```python
# 🎯 延迟加载
self.qwen_emo = None
self.qwen_emo_path = os.path.join(self.model_dir, self.cfg.qwen_emo_path)
print(">> Qwen Emotion: Lazy loading enabled (saves ~1.2GB)")

def _ensure_qwen_loaded(self):
    """延迟加载 Qwen Emotion 模型（仅在需要时加载）"""
    if self.qwen_emo is None:
        print(">> Loading Qwen Emotion model (first use)...")
        self.qwen_emo = QwenEmotion(self.qwen_emo_path)
        print(">> Qwen Emotion loaded (~1.2GB)")
```

#### 使用模式

```python
# 在 infer_generator 中
if use_emo_text:
    # 自动生成情感向量
    if emo_text is None:
        emo_text = text
    
    # 1️⃣ 延迟加载（仅在需要时）
    self._ensure_qwen_loaded()
    
    # 2️⃣ 使用模型
    emo_dict = self.qwen_emo.inference(emo_text)
    print(f"detected emotion vectors from text: {emo_dict}")
    emo_vector = list(emo_dict.values())
```

**关键点**：
- ✅ 仅在 `use_emo_text=True` 时加载
- ✅ 加载后保留（多次推理复用）
- ✅ 不需要时永远不加载

---

### 3. MLX 模式下的 PyTorch GPT 跳过 ✨ **节省 ~2.5GB**

#### 问题识别
MLX 模式下同时加载 PyTorch 和 MLX 的 GPT 模型，造成内存浪费。

#### 优化方案

**Before (混合模式)**:
```python
# ❌ 同时加载两个模型
self.gpt = UnifiedVoice(**self.cfg.gpt)  # PyTorch 版本
self.mlx_transformer = UnifiedVoiceMLX(**self.cfg.gpt)  # MLX 版本
```

**After (纯 MLX 模式)**:
```python
if self.use_mlx and self.mlx_available:
    # ✅ 只加载 MLX 模型
    from indextts.gpt.mlx_model import UnifiedVoiceMLX
    mlx_gpt_weights = self.mlx_cache.get_or_convert("gpt", self.gpt_path)
    self.mlx_transformer = UnifiedVoiceMLX(
        use_mlx_conditioning=True,
        **self.cfg.gpt
    )
    self.mlx_transformer.load_weights_from_dict(mlx_gpt_weights)
    self.gpt_is_mlx = True
    self.gpt = None  # 🎯 不加载 PyTorch 模型！节省 ~2.5GB
    print(">> ✓ Pure MLX GPT loaded successfully")
    print(">> ✓ PyTorch GPT skipped (saved ~2.5GB memory)")
else:
    # 非 MLX 模式：正常加载 PyTorch
    self.gpt = UnifiedVoice(**self.cfg.gpt)
    load_checkpoint(self.gpt, self.gpt_path)
    self.gpt = self.gpt.to(self.device).eval()
```

---

## 📊 内存节省对比

### 模型加载时内存占用

| 组件 | 主分支 | feat/memory-optimization | 节省 |
|------|--------|--------------------------|------|
| **Semantic Model** | 1.0GB (立即) | 0GB → 1.0GB (按需) → 0GB | ~1.0GB |
| **Qwen Emotion** | 1.2GB (立即) | 0GB (不用时) | ~1.2GB |
| **GPT (MLX模式)** | 5.0GB (双模型) | 2.5GB (单模型) | ~2.5GB |
| **总计** | ~7.2GB | ~2.5GB (MLX) 或 ~4.8GB (纯PyTorch) | **~2.4-4.7GB** |

### 推理时内存占用

```
主分支:
[初始化] 7.2GB (所有模型常驻)
[推理中] 7.2GB + 推理临时内存

feat/memory-optimization:
[初始化] 2.5GB (只有必需模型)
[音频特征提取] 2.5GB + 1.0GB (临时加载 Semantic) = 3.5GB
[特征提取完成] 2.5GB (立即卸载 Semantic)
[推理中] 2.5GB + 推理临时内存

✅ 峰值内存降低：7.2GB → 3.5GB (节省 51%)
```

---

## 🔑 关键技术点

### 1. 延迟加载模式（Lazy Loading Pattern）

```python
class LazyModel:
    def __init__(self):
        self.model = None
        self.model_path = "..."
        self.loaded = False
    
    def ensure_loaded(self):
        if not self.loaded:
            print("Loading model...")
            self.model = load_model(self.model_path)
            self.loaded = True
    
    def use_model(self, input):
        self.ensure_loaded()  # 使用前确保加载
        return self.model(input)
```

### 2. 即时卸载模式（Just-in-Time Unloading）

```python
def process_with_temporary_model(self, data):
    # 1. 加载
    self._load_temp_model()
    
    # 2. 使用
    result = self._use_temp_model(data)
    
    # 3. 立即卸载
    self._unload_temp_model()
    
    # 4. 强制清理
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    return result
```

### 3. 生命周期管理

```
Semantic Model 生命周期:
──────────────────────────────────────────────────
[初始化]    [提取特征]      [推理]
  None  →  Load → Use → Unload  →  None
         ↑                      ↑
      按需加载              立即释放
```

---

## 🍎 M4 平台特定优化

### M4 统一内存架构优势

```
传统 GPU (CUDA):
CPU Memory (8GB) ←→ PCIe ←→ GPU Memory (8GB)
     ↓                          ↓
  系统内存                   独立显存
  (总可用: 8GB)            (总可用: 8GB)

Apple M4 (统一内存):
     ┌────────────────────┐
     │ Unified Memory 32GB│
     │   (CPU + GPU 共享) │
     └────────────────────┘
           ↑         ↑
         CPU       GPU
    (总可用: 32GB)
```

**优势**：
- ✅ 延迟加载/卸载效果更明显（无需 CPU-GPU 拷贝）
- ✅ 内存释放立即可用（统一内存池）
- ✅ 更大的内存容量（vs. 独立显存限制）

### M4 内存清理

```python
def _unload_semantic(self):
    if self.semantic_model_loaded:
        del self.semantic_model
        # ... 其他清理 ...
        
        import gc
        gc.collect()  # 强制垃圾回收
        
        # M4 专用：清理 MPS 缓存
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()  # ⭐ M4 关键
```

---

## 💡 实际应用场景

### 场景 1：基础推理（不使用情感文本）

```python
tts = IndexTTS2(device="mps")  # M4

# ✅ Qwen Emotion 永远不加载 (节省 1.2GB)
tts.infer(
    spk_audio_prompt="voice.wav",
    text="你好",
    output_path="output.wav",
    # use_emo_text=False (默认)
)

# 内存占用：~2.5GB (vs. ~7.2GB 主分支)
```

### 场景 2：带情感文本推理

```python
tts = IndexTTS2(device="mps")

# ✅ Qwen Emotion 按需加载
tts.infer(
    spk_audio_prompt="voice.wav",
    text="你好",
    output_path="output.wav",
    use_emo_text=True,  # 触发加载
    emo_text="开心的"
)

# 内存占用：~3.7GB (临时) → ~2.5GB (Qwen保留)
```

### 场景 3：多次推理（缓存复用）

```python
tts = IndexTTS2(device="mps")

# 第一次推理
tts.infer("voice.wav", "文本1", "out1.wav")
# Semantic: 0GB → 1.0GB (加载) → 0GB (卸载)

# 第二次推理（相同参考音频，使用缓存）
tts.infer("voice.wav", "文本2", "out2.wav")
# Semantic: 不需要加载（使用缓存的 conditioning）

# 峰值内存：~3.5GB (vs. ~7.2GB 主分支)
```

---

## 🎯 适用于 PyTorch 性能基线测试

### 建议整合到主分支

这些优化技术**完全适用于纯 PyTorch 模式**（不依赖 MLX）：

1. **Semantic Model 延迟加载** ✅ 可直接移植
2. **Qwen Emotion 延迟加载** ✅ 可直接移植
3. **即时卸载机制** ✅ 可直接移植

### 整合建议

```python
# 在 indextts/infer_v2.py 主分支添加：

class IndexTTS2:
    def __init__(self, ..., enable_lazy_loading=True):
        """
        Args:
            enable_lazy_loading: 启用延迟加载优化（节省 ~2.2GB）
        """
        self.enable_lazy_loading = enable_lazy_loading
        
        if enable_lazy_loading:
            # 延迟加载模式
            self.qwen_emo = None
            self.semantic_model = None
            print(">> Lazy loading enabled (saves ~2.2GB)")
        else:
            # 传统模式（向后兼容）
            self.qwen_emo = QwenEmotion(...)
            self.semantic_model, ... = build_semantic_model(...)
```

### 性能基线测试整合

在 `benchmark_baseline.py` 中添加选项：

```bash
# 测试延迟加载效果
python benchmark_baseline.py \
    --device mps \
    --lazy_loading \  # 🆕 新参数
    --output_dir outputs/with_lazy_loading

# 对比传统加载
python benchmark_baseline.py \
    --device mps \
    --output_dir outputs/without_lazy_loading
```

---

## 📈 预期收益

### M4 平台

| 配置 | 初始内存 | 峰值内存 | 节省 |
|------|----------|----------|------|
| **主分支** | 7.2GB | 7.2GB | - |
| **+ Lazy Loading** | 2.5GB | 3.5GB | 3.7GB (51%) |
| **+ 即时卸载** | 2.5GB | 3.5GB | 3.7GB (51%) |

### CUDA 平台

| 配置 | 显存占用 | 系统内存 | 总节省 |
|------|----------|----------|--------|
| **主分支** | 4.5GB | 2.7GB | - |
| **+ Lazy Loading** | 2.3GB | 2.7GB | 2.2GB (31%) |

---

## 🔧 实施步骤

### 第1步：复制延迟加载函数

```python
# 从 feat/memory-optimization 复制到主分支
def _ensure_qwen_loaded(self): ...
def _ensure_semantic_loaded(self): ...
def _unload_semantic(self): ...
```

### 第2步：修改 `__init__`

```python
# 将立即加载改为延迟加载
self.qwen_emo = None  # 替代 QwenEmotion(...)
self.semantic_model = None  # 替代 build_semantic_model(...)
```

### 第3步：在使用点添加加载调用

```python
# 在需要时调用
self._ensure_semantic_loaded()  # 音频特征提取前
self._unload_semantic()  # 特征提取后
```

### 第4步：测试验证

```bash
# 对比内存占用
python benchmark_baseline.py --device mps --output_dir outputs/test
```

---

## 📚 相关文档

- `MEMORY_OPTIMIZATION_COMPLETE.md` - 内存优化完成总结
- `MEMORY_OPTIMIZATION_FINAL_REPORT.md` - 最终报告
- `MLX_MEMORY_OPTIMIZATION_PLAN.md` - MLX 内存优化计划

---

## 🎉 总结

`feat/memory-optimization` 分支通过以下技术实现了显著的内存节省：

### 核心技术
1. ✅ **延迟加载（Lazy Loading）** - 按需加载模型
2. ✅ **即时卸载（JIT Unloading）** - 用完立即释放
3. ✅ **生命周期管理** - 精确控制模型加载/卸载时机
4. ✅ **强制垃圾回收** - `gc.collect()` + `mps.empty_cache()`

### 内存节省
- **Semantic Model**: ~1.0GB
- **Qwen Emotion**: ~1.2GB
- **MLX 模式 GPT**: ~2.5GB
- **总计**: ~2.2-4.7GB (31-51%)

### 适用性
- ✅ 完全适用于 **纯 PyTorch 模式**
- ✅ **M4 平台效果最佳**（统一内存架构）
- ✅ 可直接整合到主分支和性能基线测试

**这些优化技术应该整合到主分支！** 🚀

---

**创建日期**: 2024-10-16  
**分析对象**: `feat/memory-optimization` 分支  
**适用平台**: PyTorch (CUDA/MPS/CPU)


