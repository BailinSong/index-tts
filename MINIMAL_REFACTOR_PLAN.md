# 最小侵入重构方案（最终版）

> 严格遵循"main分支原有文件最小改动"原则

**核心原则**: main 分支文件改动 < 100 行  
**策略**: 插件化 + 最小必要支持

---

## 🎯 改动量目标

### infer_v2.py 改动分解

| 改动类型 | 行数 | 必要性 | 说明 |
|---------|------|--------|------|
| __init__ 参数 | +2 | ✅ 必需 | 添加 use_mlx 参数 |
| 标志位初始化 | +3 | ✅ 必需 | self.gpt_is_mlx, mlx_transformer |
| infer 方法 MLX 分支 | +50 | ✅ 必需 | if self.gpt_is_mlx 最小分支 |
| 工厂函数 | +20 | ✅ 必需 | create_tts() |
| **总计** | **~75 行** | - | **可接受** |

**不需要在 infer_v2.py 中**:
- ❌ MLX 加载逻辑（→ IndexTTS2MLX）
- ❌ 内存优化逻辑（→ IndexTTS2MLX）
- ❌ MLX 初始化代码（→ IndexTTS2MLX）

---

## 📐 具体实施

### Step 1: 精简 infer_v2.py

#### 1.1 __init__ 最小改动

```python
class IndexTTS2:
    def __init__(
        self, cfg_path="...", model_dir="...", 
        use_fp16=False, device=None,
        use_cuda_kernel=None, use_deepspeed=False
        # ✅ 不添加 use_mlx 参数到这里！
    ):
        # === main 分支原有代码（完全不改）===
        # 设备检测
        # 加载配置
        # 加载 Qwen Emotion
        self.qwen_emo = QwenEmotion(...)
        
        # 加载 PyTorch GPT
        self.gpt = UnifiedVoice(...)
        load_checkpoint(self.gpt, ...)
        
        # 加载 Semantic Model
        self.semantic_model, ... = build_semantic_model(...)
        
        # === 仅新增标志位（+3 行）===
        self.mlx_transformer = None
        self.gpt_is_mlx = False
```

**改动**: +3 行

#### 1.2 infer 方法最小 MLX 支持

```python
def infer(self, ...):
    # ... 原有代码不变 ...
    
    # 生成代码部分
    # ✅ 添加 MLX 分支（+15 行）
    if self.gpt_is_mlx and self.mlx_transformer:
        codes, latent = self.mlx_transformer.inference_speech(...)
    else:
        codes, latent = self.gpt.inference_speech(...)  # 原有
    
    # ... 原有代码不变 ...
    
    # S2MEL forward 部分
    # ✅ 添加 MLX 分支（+10 行）
    if self.gpt_is_mlx and self.mlx_transformer:
        latent = self.mlx_transformer(...)
    else:
        latent = self.gpt(...)  # 原有
    
    # ... 原有代码不变 ...
```

**改动**: +25 行（仅关键分支）

#### 1.3 merge_emovec 支持

```python
# infer 方法中（+10 行）
if self.gpt_is_mlx and self.mlx_transformer:
    emovec = self.mlx_transformer.merge_emovec(...)
else:
    emovec = self.gpt.merge_emovec(...)  # 原有
```

**改动**: +10 行

#### 1.4 工厂函数

```python
# 文件末尾（+20 行）
def create_tts(use_mlx=False, **kwargs):
    if use_mlx:
        from indextts.mlx import IndexTTS2MLX
        return IndexTTS2MLX(**kwargs)
    else:
        return IndexTTS2(**kwargs)
```

**改动**: +20 行

### Step 2: IndexTTS2MLX 完整实现

**文件**: `indextts/mlx/infer_mlx.py`

```python
class IndexTTS2MLX(IndexTTS2):
    def __init__(self, **kwargs):
        # === MLX 预初始化 ===
        # 准备 MLX 环境
        from indextts.utils.mlx.utils import check_mlx_available
        self.mlx_available = check_mlx_available()
        
        # === 调用父类（加载 PyTorch 模型）===
        super().__init__(**kwargs)
        
        # === MLX 后处理：替换模型 ===
        if self.mlx_available:
            self._replace_with_mlx()
            self._apply_memory_optimizations()
    
    def _replace_with_mlx(self):
        """
        替换 PyTorch 模型为 MLX
        
        修改的是父类已经初始化的属性：
        - self.gpt → None
        - self.mlx_transformer → UnifiedVoiceMLX
        - self.gpt_is_mlx → True
        - self.qwen_emo → None (延迟加载)
        - self.semantic_model → None (按需加载)
        """
        # 删除 PyTorch GPT
        del self.gpt
        self.gpt = None
        
        # 加载 MLX GPT
        from .model_loader import MLXModelLoader
        loader = MLXModelLoader(self.model_dir, self.cfg, self.device)
        self.mlx_transformer, _ = loader.load_gpt(self.gpt_path)
        self.gpt_is_mlx = True
        
        print(">> ✓ PyTorch GPT replaced with MLX (-2.5GB)")
    
    def _apply_memory_optimizations(self):
        """应用内存优化"""
        from .memory_optimizer import MemoryOptimizer
        optimizer = MemoryOptimizer()
        
        # Qwen 延迟加载
        optimizer.optimize_qwen_emotion(self)
        
        # Semantic 按需加载
        optimizer.optimize_semantic_model(self)
```

---

## 📊 最终改动量

### infer_v2.py

**新增**:
```python
# Line 1: 导入
from typing import Optional  # +1 行

# __init__ 方法
self.mlx_transformer = None  # +3 行
self.gpt_is_mlx = False

# infer 方法
if self.gpt_is_mlx:  # +50 行
    # MLX 分支（3个位置，每个15行）
    
# 文件末尾
def create_tts(...):  # +20 行
    ...
```

**总计**: ~74 行（接近目标 <100 行）

### IndexTTS2MLX (indextts/mlx/infer_mlx.py)

**完整实现**: ~800 行
- __init__: 调用父类 + 替换模型
- _replace_with_mlx: GPT 替换逻辑
- _apply_memory_optimizations: 内存优化
- 重写 infer 的必要部分（如果需要）

---

## 🚀 立即执行

我现在需要：
1. 保留 indextts/mlx/ 插件模块
2. 精简 infer_v2.py，移除不必要的 MLX 代码
3. 完善 IndexTTS2MLX 实现

您同意这个方案吗？

