# 重构进度检查点

> 方案 B 执行进度评估

**当前时间**: 已执行约 1 小时  
**目标**: infer_v2.py 改动 586 → 75 行

---

## ✅ 已完成 (60%)

### Step 1: MLXModelLoader ✅
- load_gpt: 完整 GPT 加载逻辑
- load_s2mel_mlx_modules: S2MEL MLX 模块
- cache_bigvgan: BigVGAN 缓存

### Step 2: MemoryOptimizer ✅
- LazySemanticModel: 完整实现
- LazyQwenEmotion: 完整实现
- optimize_semantic_model: 替换逻辑
- optimize_qwen_emotion: 替换逻辑

### Step 3: IndexTTS2MLX ⏳ 80%
- ✅ __init__: 正确的参数签名
- ✅ _apply_mlx_optimizations: 完整流程
- ✅ _replace_gpt_with_mlx: GPT 替换
- ✅ _load_s2mel_mlx_modules: S2MEL 模块
- ✅ _apply_memory_optimizations: 内存优化
- ✅ _print_memory_savings: 统计打印
- ✅ _print_mlx_summary: 摘要打印
- ⏳ infer 方法: 需要复用父类

---

## ⏳ 剩余工作 (40%)

### Step 4: 精简 infer_v2.py (最关键)

**需要做**:
1. 从 main 分支的干净版本开始
2. 添加最小的 MLX 分支支持

**关键认识**:
即使精简，infer 方法仍需要 ~50 行 MLX 分支：

```python
# infer 方法中的 3 个关键位置
if self.gpt_is_mlx and self.mlx_transformer:
    # 位置 1: merge_emovec (~10行)
    # 位置 2: inference_speech (~20行)
    # 位置 3: forward (~20行)
else:
    # 原有 PyTorch 逻辑
```

**预计改动**:
- __init__: +3 行（标志位）
- infer: +50 行（MLX 分支）
- 工厂函数: +20 行
- **总计**: ~73 行

### Step 5: 测试验证

**必须测试**:
- [ ] PyTorch 模式：正常工作
- [ ] MLX 模式：通过 IndexTTS2MLX
- [ ] benchmark_v1_baseline.py：性能 6.43s
- [ ] 音质验证

**预计时间**: 2 小时

---

## 🤔 重要发现

### 无法避免的事实

**即使完成方案 B，infer_v2.py 仍需 ~73 行改动**：

1. **标志位**（必需）:
   ```python
   self.mlx_transformer = None
   self.gpt_is_mlx = False
   ```

2. **infer 方法的 MLX 分支**（必需）:
   ```python
   if self.gpt_is_mlx:
       # MLX 推理路径（~50行）
   else:
       # PyTorch 路径（原有）
   ```

3. **工厂函数**（必需）:
   ```python
   def create_tts(use_mlx=False):
       ...
   ```

**为什么无法避免？**
- IndexTTS2MLX 继承 IndexTTS2
- IndexTTS2MLX.infer 调用 super().infer
- 父类的 infer 需要支持 MLX 分支
- 否则必须完全重写 infer（失去继承价值）

### 实际收益

**代码量对比**:
- 方案 A: infer_v2.py +586 行
- 方案 B: infer_v2.py +73 行 + 插件 ~1200 行 = **1273 行总计**

**审查工作量**:
- 方案 A: 586 行（集中）
- 方案 B: 1273 行（分散）

**方案 B 实际增加了代码总量和审查难度！**

---

## 💡 建议

### 选项 1: 停止方案 B，回到方案 A ⭐⭐⭐⭐⭐

**理由**:
1. 方案 A 代码总量更少（586 vs 1273）
2. 方案 A 审查更容易（集中 vs 分散）
3. 方案 A 已充分验证
4. 方案 B 实际收益不大（infer_v2.py 仍需 73 行）

**操作**:
```bash
git reset --hard 9bf94c7  # 回到简化前
```

**时间**: 0 小时

### 选项 2: 继续完成方案 B

**剩余工作**:
- 精简 infer_v2.py: 2-3 小时
- 测试验证: 2 小时
- **总计**: 4-5 小时

**收益**: infer_v2.py 改动 586 → 73 行（但总代码量增加）

---

## ❓ 您的决定

已执行 1 小时，剩余 4-5 小时。

**继续方案 B** 还是 **回到方案 A**？

我强烈建议回到方案 A，因为：
- 总代码量更少
- 审查更容易
- 已经验证
- 节省 4-5 小时

**请明确指示**: 继续 or 回退？

