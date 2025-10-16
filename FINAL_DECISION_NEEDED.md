# 需要最终决策

> 当前方案 vs 最小侵入方案

**当前状态**: infer_v2.py 已包含完整 MLX 实现（+670/-84 = 586行）

---

## 两个方案对比

### 方案 A: 保持当前状态（+586 行）⭐⭐⭐⭐⭐

**现状**:
- infer_v2.py: 包含完整 PyTorch + MLX 实现
- 通过 `use_mlx` 参数控制
- 所有代码在一个类中

**优点**:
- ✅ **主团队更新自动继承**（最重要！）
- ✅ 功能完整且验证（6.43s, -4.7GB）
- ✅ 无代码重复
- ✅ 维护成本最低
- ✅ 立即可用

**缺点**:
- ⚠️ infer_v2.py 改动 586 行

**工作量**: 0 小时（已完成）

---

### 方案 B: 最小侵入重构（~75 行）⭐⭐⭐

**目标**:
- infer_v2.py: 只保留必要的 MLX 分支（~75 行）
- IndexTTS2MLX: 包含所有 MLX 加载和优化逻辑

#### infer_v2.py 精简后

```python
class IndexTTS2:
    def __init__(self, ...):
        # === main 分支原有代码（不改）===
        self.gpt = UnifiedVoice(...)
        self.qwen_emo = QwenEmotion(...)
        self.semantic_model = build_semantic_model(...)
        
        # === 仅新增标志位（+3行）===
        self.mlx_transformer = None
        self.gpt_is_mlx = False
    
    def infer(self, ...):
        # === 原有代码不变 ===
        ...
        
        # === 新增：最小 MLX 分支（+50行）===
        if self.gpt_is_mlx and self.mlx_transformer:
            codes, latent = self.mlx_transformer.inference_speech(...)
            emovec = self.mlx_transformer.merge_emovec(...)
            latent = self.mlx_transformer(...)
        else:
            codes, latent = self.gpt.inference_speech(...)  # 原有
            emovec = self.gpt.merge_emovec(...)  # 原有
            latent = self.gpt(...)  # 原有
        
        # === 原有代码不变 ===
        ...

# === 新增：工厂函数（+20行）===
def create_tts(use_mlx=False, **kwargs):
    if use_mlx:
        from indextts.mlx import IndexTTS2MLX
        return IndexTTS2MLX(**kwargs)
    return IndexTTS2(**kwargs)
```

**改动**: ~73 行

#### IndexTTS2MLX 需要实现

```python
class IndexTTS2MLX(IndexTTS2):
    def __init__(self, **kwargs):
        # 1. 调用父类（加载 PyTorch 模型）
        super().__init__(**kwargs)
        
        # 2. 删除 PyTorch 模型
        del self.gpt
        del self.qwen_emo
        del self.semantic_model
        
        # 3. 加载 MLX 模型
        loader = MLXModelLoader(...)
        self.mlx_transformer = loader.load_gpt()
        self.gpt = None
        self.gpt_is_mlx = True
        
        # 4. 应用内存优化
        optimizer = MemoryOptimizer()
        optimizer.optimize_qwen_emotion(self)
        optimizer.optimize_semantic_model(self)
    
    # ✅ 不需要重写 infer - 复用父类的 MLX 分支
```

**优点**:
- ✅ infer_v2.py 改动最小（~75 行）
- ✅ 主团队的 infer 更新仍然继承
- ✅ MLX 功能独立在插件模块

**缺点**:
- ⚠️ 需要完善 IndexTTS2MLX（~500 行代码）
- ⚠️ 需要完善 MLXModelLoader（~200 行）
- ⚠️ 需要完善 MemoryOptimizer（~300 行）
- ⚠️ 需要充分测试验证

**工作量**: 6-8 小时

---

## 🤔 关键权衡

### 代码量 vs 维护成本

**方案 A**: 
- 代码集中：586 行在 infer_v2.py
- 维护简单：一个文件
- **继承完整**：所有更新自动获得

**方案 B**:
- 代码分散：75 行在 infer_v2.py + 1000 行在插件
- 维护复杂：多个文件协同
- **继承部分**：infer 更新获得，但初始化更新需手动

### 主团队视角

**方案 A** (586行改动):
```
审查难度: ⭐⭐⭐ 中等
- 一个文件，改动集中
- if/else 分支清晰
- 可以一次性理解所有改动
```

**方案 B** (75行 + 插件):
```
审查难度: ⭐⭐⭐⭐ 较难  
- 需要理解插件架构
- 需要跨文件审查
- 需要理解继承关系
```

---

## 💡 我的建议

### 如果时间充足：方案 B（6-8小时）

**适合**:
- 重视代码组织
- 愿意投入时间重构
- 长期维护优先

### 如果希望快速推送：方案 A（0小时）

**适合**:
- 功能优先
- 已充分验证
- 立即可用

---

## 🎯 您的决策

**选项 1**: 保持方案 A
- 推送当前代码
- 创建 PR
- 0 小时

**选项 2**: 执行方案 B  
- 精简 infer_v2.py 到 ~75 行
- 完善插件模块
- 6-8 小时

**选项 3**: 折中方案
- 部分精简 infer_v2.py
- 降到 ~200 行改动
- 2-3 小时

---

**请选择**: 1, 2, 或 3？

我建议考虑：
1. 当前方案已经工作良好
2. 586 行改动虽多，但都是新增功能
3. 主团队继承价值 > 代码量

