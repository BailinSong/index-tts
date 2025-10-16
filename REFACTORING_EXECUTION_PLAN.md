# 无侵入式重构执行计划

> 详细的分步执行计划，将 MLX 功能重构为插件化架构

**基于**: REFACTORING_STRATEGY_V2.md  
**目标**: 将原有文件改动从 710 行降至 35 行（-95%）  
**预计时间**: 3 周

---

## 📋 总体目标

- ✅ 创建 `IndexTTS2MLX` 继承类
- ✅ 提取 `MLXModelLoader` 独立加载器
- ✅ 提取 `MemoryOptimizer` 独立优化器
- ✅ 最小化修改 `infer_v2.py`、`cli.py`、`webui.py`
- ✅ 保持完全向后兼容
- ✅ 通过所有测试

---

## 🗓️ Week 1: 基础架构（5 天）

### Day 1: 创建目录结构和基础框架

#### Task 1.1: 创建 MLX 模块目录
```bash
mkdir -p indextts/mlx
mkdir -p indextts/gpt/mlx
mkdir -p indextts/utils/mlx
```

#### Task 1.2: 移动现有 MLX 文件
```bash
# GPT MLX 实现
mv indextts/gpt/mlx_model.py indextts/gpt/mlx/model.py
mv indextts/gpt/mlx_conditioning.py indextts/gpt/mlx/conditioning.py
mv indextts/gpt/mlx_logits_processors_optimized.py indextts/gpt/mlx/logits_processors.py
mv indextts/gpt/mlx_subsampling.py indextts/gpt/mlx/subsampling.py

# 删除旧的 logits_processors（使用优化版）
rm indextts/gpt/mlx_logits_processors.py

# Utils
mv indextts/utils/mlx_cache.py indextts/utils/mlx/cache.py
mv indextts/utils/mlx_utils.py indextts/utils/mlx/utils.py

# S2MEL MLX
# (保持在原位置，已经独立)
```

#### Task 1.3: 创建 __init__.py 文件
```python
# indextts/mlx/__init__.py
# indextts/gpt/mlx/__init__.py
# indextts/utils/mlx/__init__.py
```

#### Task 1.4: 更新导入路径
修改所有文件中的导入语句，适配新的目录结构

**验证**: 确保 `python -c "import indextts.gpt.mlx"` 不报错

---

### Day 2: 创建 MLXModelLoader

#### Task 2.1: 创建 MLXModelLoader 基础框架

文件: `indextts/mlx/model_loader.py`

```python
"""MLX 模型加载器 - 统一管理所有 MLX 模型加载"""
from typing import Any, Optional, Dict
import os


class MLXModelLoader:
    """
    MLX 模型加载器
    
    职责:
    1. 加载所有 MLX 模型（GPT, S2MEL, BigVGAN等）
    2. 管理 MLX 缓存
    3. 处理加载失败和降级
    """
    
    def __init__(self, model_dir: str, config: Any):
        self.model_dir = model_dir
        self.config = config
        self._init_mlx_cache()
    
    def _init_mlx_cache(self):
        """初始化 MLX 缓存"""
        from indextts.utils.mlx.cache import MLXCache
        cache_dir = os.path.join(self.model_dir, "mlx")
        self.mlx_cache = MLXCache(cache_dir=cache_dir)
        print(f">> MLX Cache Directory: {cache_dir}")
    
    def load_gpt(self) -> Any:
        """加载 MLX GPT 模型"""
        print(">> [MLX Loader] Loading GPT model...")
        # 实现加载逻辑
        pass
    
    def load_s2mel(self) -> Any:
        """加载 MLX S2MEL 模型"""
        print(">> [MLX Loader] Loading S2MEL model...")
        pass
    
    def load_vocoder(self) -> Any:
        """加载 MLX BigVGAN 模型"""
        print(">> [MLX Loader] Loading BigVGAN model...")
        pass
```

#### Task 2.2: 实现 load_gpt 方法

从 `infer_v2.py` Line 119-169 提取 GPT 加载逻辑

#### Task 2.3: 实现 load_s2mel 方法

从 `infer_v2.py` 提取 S2MEL 加载逻辑

**验证**: 单元测试 `MLXModelLoader`

---

### Day 3: 创建 MemoryOptimizer

#### Task 3.1: 创建 MemoryOptimizer 基础框架

文件: `indextts/mlx/memory_optimizer.py`

```python
"""内存优化器 - 统一管理所有内存优化策略"""
import gc
import torch
from typing import Any


class MemoryOptimizer:
    """
    内存优化管理器
    
    职责:
    1. Semantic Model 按需加载/卸载
    2. Qwen Emotion 延迟加载
    3. 内存清理和监控
    """
    
    def __init__(self):
        self.semantic_wrapper = None
        self.qwen_wrapper = None
    
    def optimize_semantic_model(self, tts_instance):
        """优化 Semantic Model"""
        print(">> [Memory Opt] Optimizing Semantic Model...")
        pass
    
    def optimize_qwen_emotion(self, tts_instance):
        """优化 Qwen Emotion"""
        print(">> [Memory Opt] Optimizing Qwen Emotion...")
        pass
```

#### Task 3.2: 实现 LazySemanticModel

创建延迟加载的包装器

#### Task 3.3: 实现 LazyQwenEmotion

创建延迟加载的包装器

**验证**: 单元测试内存优化器

---

### Day 4: 创建 IndexTTS2MLX 继承类

#### Task 4.1: 创建基础框架

文件: `indextts/mlx/infer_mlx.py`

```python
"""MLX 优化的 IndexTTS2 实现"""
from typing import Optional
from ..infer_v2 import IndexTTS2
from .model_loader import MLXModelLoader
from .memory_optimizer import MemoryOptimizer


class IndexTTS2MLX(IndexTTS2):
    """
    MLX 优化版本的 IndexTTS2
    
    通过继承扩展，不修改原有代码
    """
    
    def __init__(
        self,
        model_dir: str,
        config_path: Optional[str] = None,
        use_mlx: bool = True,
        mlx_memory_optimization: bool = True,
        **kwargs
    ):
        """
        初始化 MLX 优化版本
        
        Args:
            use_mlx: 是否启用 MLX（默认 True）
            mlx_memory_optimization: 是否启用内存优化（默认 True）
        """
        self.use_mlx_flag = use_mlx
        self.mlx_memory_optimization = mlx_memory_optimization
        
        # 标记：阻止父类加载某些模型
        self._mlx_override_loading = use_mlx
        
        # 先调用父类初始化（加载配置等）
        super().__init__(
            model_dir=model_dir,
            config_path=config_path,
            use_mlx=False,  # 让父类以 PyTorch 模式初始化
            **kwargs
        )
        
        # 后处理：如果启用 MLX，替换模型
        if self.use_mlx_flag:
            self._apply_mlx_optimizations()
    
    def _apply_mlx_optimizations(self):
        """应用 MLX 优化"""
        pass
```

#### Task 4.2: 实现模型替换逻辑

实现 `_apply_mlx_optimizations` 方法

#### Task 4.3: 重写 infer 方法

重写推理方法以使用 MLX 模型

**验证**: 基础加载测试

---

### Day 5: 集成和测试

#### Task 5.1: 添加工厂函数

在 `infer_v2.py` 末尾添加：

```python
def create_tts(model_dir: str, use_mlx: bool = False, **kwargs):
    """
    工厂函数：创建 TTS 实例
    
    Args:
        use_mlx: 是否使用 MLX 优化版本
        **kwargs: 其他参数
    
    Returns:
        IndexTTS2 或 IndexTTS2MLX 实例
    """
    if use_mlx:
        try:
            from indextts.mlx.infer_mlx import IndexTTS2MLX
            return IndexTTS2MLX(model_dir=model_dir, use_mlx=True, **kwargs)
        except ImportError as e:
            print(f">> MLX not available: {e}")
            print(">> Falling back to PyTorch version")
            return IndexTTS2(model_dir=model_dir, **kwargs)
    else:
        return IndexTTS2(model_dir=model_dir, **kwargs)
```

#### Task 5.2: 基础功能测试

```bash
# 测试原版
python -c "from indextts.infer_v2 import IndexTTS2; tts = IndexTTS2('./checkpoints')"

# 测试 MLX 版本
python -c "from indextts.infer_v2 import create_tts; tts = create_tts('./checkpoints', use_mlx=True)"
```

#### Task 5.3: 第一次提交

```bash
git add indextts/mlx/ indextts/gpt/mlx/ indextts/utils/mlx/
git commit -m "🏗️ Week 1: MLX 插件化架构基础"
```

---

## 🗓️ Week 2: 功能迁移（5 天）

### Day 6-7: 完整实现 MLXModelLoader

#### Task 6.1: 完成 GPT 加载逻辑

从 `infer_v2.py` 提取完整的 GPT 加载代码

#### Task 6.2: 完成 S2MEL 加载逻辑

#### Task 6.3: 完成 BigVGAN 加载逻辑

#### Task 6.4: 测试所有加载器

```bash
python benchmark_v1_baseline.py
```

**验证**: 加载成功，内存占用正确

---

### Day 8-9: 完整实现 MemoryOptimizer

#### Task 8.1: 完成 Semantic Model 优化

从 `infer_v2.py` Line 193-200, 572-603 提取逻辑

#### Task 8.2: 完成 Qwen Emotion 优化

从 `infer_v2.py` Line 112-115, 563-569 提取逻辑

#### Task 8.3: 测试内存优化

验证内存释放是否生效

---

### Day 10: 完整实现 IndexTTS2MLX.infer

#### Task 10.1: 实现推理逻辑

从 `infer_v2.py` 的 `infer` 方法提取 MLX 相关逻辑

#### Task 10.2: 功能完整性测试

```bash
python benchmark_v1_baseline.py
```

**目标**: 
- 基准值 6.43s ± 10%
- 音色正确
- 内存优化生效

#### Task 10.3: 第二次提交

```bash
git commit -m "✨ Week 2: MLX 功能完整迁移"
```

---

## 🗓️ Week 3: 清理和集成（5 天）

### Day 11-12: 最小化修改原有文件

#### Task 11.1: 清理 infer_v2.py

**目标**: 仅保留必要的修改

1. 删除所有 MLX 相关的内联代码
2. 只保留原有的 PyTorch 逻辑
3. 在末尾添加工厂函数（~20 行）

#### Task 11.2: 更新 cli.py

```python
# 只添加 MLX 参数和使用工厂函数
parser.add_argument("--mlx", action="store_true")
tts = create_tts(args.model_dir, use_mlx=args.mlx)
```

#### Task 11.3: 更新 webui.py

```python
# 只添加环境变量和使用工厂函数
USE_MLX = os.getenv("USE_MLX", "true").lower() == "true"
tts = create_tts("./checkpoints", use_mlx=USE_MLX)
```

**验证**: `git diff main...HEAD` 确认原有文件改动 <50 行

---

### Day 13: 全面测试

#### Task 13.1: 功能测试

```bash
# 测试 MLX 版本
python -m indextts.cli "今天天气真不错" -v examples/zh_vo_Main_Linaxita_2_4_24_6.wav --mlx

# 测试原版
python -m indextts.cli "今天天气真不错" -v examples/zh_vo_Main_Linaxita_2_4_24_6.wav
```

#### Task 13.2: 性能测试

```bash
python benchmark_v1_baseline.py
```

**目标**: 
- V1 基准: 6.43s ± 10%
- 内存: 1.8GB
- 音色准确

#### Task 13.3: 回退测试

```bash
# 测试禁用 MLX
python -m indextts.cli "测试" -v voice.wav  # 不加 --mlx

# 测试删除 mlx 模块
mv indextts/mlx indextts/mlx.bak
python -m indextts.cli "测试" -v voice.wav  # 应该正常工作
mv indextts/mlx.bak indextts/mlx
```

---

### Day 14: 文档和代码审查

#### Task 14.1: 更新文档

- [ ] 更新 `README_MEMORY_OPTIMIZATION.md`
- [ ] 创建 `MIGRATION_GUIDE.md`
- [ ] 更新 `MLX_OPTIMIZATION_SUMMARY.md`

#### Task 14.2: 代码审查清单

- [ ] 所有新代码有适当的注释
- [ ] 所有公共方法有文档字符串
- [ ] 删除了 debug 代码和注释
- [ ] 代码符合 PEP 8
- [ ] 没有 TODO 或 FIXME

#### Task 14.3: 最终提交

```bash
git add .
git commit -m "🎉 完成无侵入式 MLX 重构

改动统计:
- 原有文件改动: <50 行 (-95%)
- 新增文件: ~1000 行
- 功能: 100% 保留
- 性能: 6.43s (符合预期)
- 内存: 1.8GB (优化生效)

架构:
- IndexTTS2MLX 继承类
- MLXModelLoader 独立加载
- MemoryOptimizer 独立优化
- 完全向后兼容"
```

---

### Day 15: PR 准备

#### Task 15.1: 创建 MIGRATION_GUIDE.md

#### Task 15.2: 准备 PR 描述

#### Task 15.3: 推送到远端

```bash
git push fork refactor/cleanup-unused-files
```

---

## ✅ 验收标准

### 功能性

- [ ] MLX 版本功能完整
- [ ] 原版功能不受影响
- [ ] 可通过参数切换
- [ ] 可完全禁用 MLX

### 性能

- [ ] 基准值: 6.43s ± 10%
- [ ] 内存: 1.8GB
- [ ] 音色准确（人工验证）

### 代码质量

- [ ] 原有文件改动 <50 行
- [ ] 新代码职责单一
- [ ] 有适当的文档
- [ ] 通过代码审查

### 兼容性

- [ ] 向后兼容
- [ ] 易于回退
- [ ] 无破坏性变更

---

## 📝 每日检查清单

每天工作结束前：

- [ ] 代码已提交
- [ ] 测试通过
- [ ] 文档已更新
- [ ] TODO 列表已更新

---

## 🚨 风险和应对

### 风险 1: 继承导致的复杂性

**应对**: 保持子类简单，只做必要的替换

### 风险 2: 测试覆盖不足

**应对**: 每天运行 `benchmark_v1_baseline.py`

### 风险 3: 性能回退

**应对**: 持续监控基准值，及时调整

---

**准备开始**: Week 1, Day 1, Task 1.1

