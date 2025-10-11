# Full MLX 实现计划 - 分步执行

## 🎯 最终目标
实现100% Pure MLX IndexTTS2，完全移除PyTorch依赖（推理阶段）

## 📋 当前状态 (v1.0)
```
✅ MLX Transformer (24 layers) - 工作正常
✅ KV Cache - 工作正常
✅ Sampling - 已修复
❌ MLX Conformer - 使用Linear简化实现（质量差）
❌ MLX Perceiver - 未经充分验证
```

## 🚀 执行计划

---

### Step 1: 研究MLX Conv1d API ⏱️ 2-3小时

**目标**: 理解MLX Conv1d的正确用法

**任务清单**:
- [ ] 1.1 创建测试文件 `experiments/test_mlx_conv1d_basic.py`
- [ ] 1.2 测试简单的Conv1d操作
- [ ] 1.3 对比PyTorch Conv1d输出
- [ ] 1.4 测试不同参数配置

**测试代码模板**:
```python
import mlx.core as mx
import mlx.nn as nn
import torch
import numpy as np

# Test 1: Basic Conv1d
print("Test 1: Basic Conv1d")
# Input: (batch=1, channels=64, seq=100)
x_mlx = mx.random.normal((1, 100, 64))  # MLX格式
x_pt = torch.randn(1, 64, 100)  # PyTorch格式

# MLX Conv1d
conv_mlx = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
out_mlx = conv_mlx(x_mlx)

# PyTorch Conv1d
conv_pt = torch.nn.Conv1d(64, 128, 3)
out_pt = conv_pt(x_pt)

print(f"MLX output shape: {out_mlx.shape}")
print(f"PyTorch output shape: {out_pt.shape}")
```

**成功标准**:
- ✅ 理解输入格式: (batch, seq, channels) 还是 (batch, channels, seq)
- ✅ 理解输出格式
- ✅ 理解权重初始化

---

### Step 2: 实现MLX Depthwise Convolution ⏱️ 4-6小时

**目标**: 实现Conformer需要的depthwise separable convolution

**任务清单**:
- [ ] 2.1 实现pointwise convolution (1x1)
- [ ] 2.2 实现depthwise convolution (groups=channels)
- [ ] 2.3 实现完整的depthwise separable conv
- [ ] 2.4 添加GLU activation
- [ ] 2.5 数值验证

**实现文件**: `indextts/gpt/mlx_conditioning.py` (更新)

**代码结构**:
```python
class MLXConvolutionModule(nn.Module):
    """
    Conformer Convolution Module with proper Conv1d
    """
    def __init__(self, channels: int, kernel_size: int = 31):
        super().__init__()
        
        # Pointwise expansion
        self.pointwise1 = nn.Conv1d(channels, 2*channels, kernel_size=1)
        
        # Depthwise convolution
        self.depthwise = nn.Conv1d(
            2*channels, 2*channels, 
            kernel_size=kernel_size,
            groups=2*channels,  # depthwise
            padding=kernel_size//2
        )
        
        # Batch norm
        self.norm = nn.LayerNorm(2*channels)
        
        # Pointwise projection
        self.pointwise2 = nn.Conv1d(channels, channels, kernel_size=1)
    
    def __call__(self, x):
        # x: (batch, seq, channels)
        
        # GLU: split and multiply
        x = self.pointwise1(x)
        x1, x2 = mx.split(x, 2, axis=-1)
        x = x1 * nn.sigmoid(x2)
        
        # Depthwise conv
        x = self.depthwise(x)
        
        # Swish activation
        x = x * nn.sigmoid(x)
        
        # Project back
        x = self.pointwise2(x)
        
        return x
```

**测试脚本**: `experiments/test_conformer_conv.py`

**成功标准**:
- ✅ 与PyTorch输出误差 < 1e-3
- ✅ Shape正确
- ✅ 支持不同kernel size

---

### Step 3: 验证MLX Conformer ⏱️ 6-8小时

**目标**: 确保MLX Conformer输出质量与PyTorch相当

**任务清单**:
- [ ] 3.1 更新 `MLXConformerBlock` 使用真正的Conv1d
- [ ] 3.2 逐层验证 (attention, conv, feedforward)
- [ ] 3.3 验证完整encoder
- [ ] 3.4 对比conditioning输出

**测试脚本**: `experiments/test_mlx_conformer.py`

**测试代码**:
```python
import torch
import mlx.core as mx
from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.mlx_conditioning import MLXConformerEncoder

# 创建相同输入
batch, seq, dim = 1, 121, 1024
x_pt = torch.randn(batch, seq, dim)
x_mlx = mx.array(x_pt.numpy())

# PyTorch Conformer
conformer_pt = ConformerEncoder(...)
out_pt = conformer_pt(x_pt)

# MLX Conformer
conformer_mlx = MLXConformerEncoder(...)
out_mlx = conformer_mlx(x_mlx)

# 对比
diff = abs(torch.from_numpy(np.array(out_mlx)) - out_pt)
print(f"Max diff: {diff.max()}")
print(f"Mean diff: {diff.mean()}")
```

**成功标准**:
- ✅ 逐层误差 < 1e-3
- ✅ 端到端误差 < 1e-2
- ✅ 输出shape正确

---

### Step 4: 集成纯MLX Conditioning ⏱️ 4-6小时

**目标**: 在inference中启用纯MLX conditioning

**任务清单**:
- [ ] 4.1 在 `UnifiedVoiceMLX` 中设置 `use_mlx_conditioning=True`
- [ ] 4.2 加载PyTorch权重到MLX Conformer
- [ ] 4.3 加载PyTorch权重到MLX Perceiver
- [ ] 4.4 测试权重加载正确性

**修改文件**: `indextts/infer_v2.py`

**修改代码**:
```python
# 在 __init__ 中
self.mlx_transformer = UnifiedVoiceMLX(
    use_mlx_conditioning=True,  # 启用纯MLX
    **self.cfg.gpt
)
```

**权重转换**: `experiments/convert_conformer_weights.py`

**成功标准**:
- ✅ 权重加载无错误
- ✅ 初始输出与hybrid模式接近

---

### Step 5: 端到端测试 ⏱️ 4-6小时

**目标**: 验证纯MLX模式生成质量

**任务清单**:
- [ ] 5.1 运行纯MLX推理
- [ ] 5.2 检查token质量（无重复）
- [ ] 5.3 检查音频质量
- [ ] 5.4 性能benchmark

**测试脚本**: `experiments/test_pure_mlx_quality.py`

**测试内容**:
```python
# 生成多个样本
texts = ["今天", "你好世界", "今天天气很好"]
for text in texts:
    # Pure MLX
    codes_mlx, audio_mlx = infer_pure_mlx(text)
    
    # PyTorch baseline
    codes_pt, audio_pt = infer_pytorch(text)
    
    # 质量指标
    metrics = {
        'token_unique': len(set(codes_mlx)),
        'token_repetition': check_repetition(codes_mlx),
        'audio_std': audio_mlx.std(),
        'duration_diff': abs(len(audio_mlx) - len(audio_pt))
    }
    
    print(f"{text}: {metrics}")
```

**成功标准**:
- ✅ Token unique count > 50
- ✅ Token重复率 < 10%
- ✅ 音频STD: 0.04-0.12
- ✅ 能正常hit stop token

---

### Step 6: 调试和优化 ⏱️ 根据需要

**如果Step 5失败，可能的问题和解决方案**:

#### 问题1: Token质量差
**症状**: 重复token，unique count低
**调试**:
```python
# 检查conditioning质量
cond_mlx = model.get_conditioning_mlx(...)
print(f"Conditioning std: {cond_mlx.std()}")
print(f"Conditioning range: [{cond_mlx.min()}, {cond_mlx.max()}]")

# 对比PyTorch
cond_pt = model_pt.get_conditioning(...)
diff = abs(cond_mlx - cond_pt)
print(f"Conditioning diff: {diff.mean()}")
```

**解决方案**:
- 检查Conv1d实现
- 检查normalization
- 检查activation函数

#### 问题2: 数值不稳定
**症状**: NaN或Inf
**调试**:
```python
# 添加梯度检查
mx.eval(output)  # 强制计算
assert not mx.isnan(output).any()
assert not mx.isinf(output).any()
```

**解决方案**:
- 添加gradient clipping
- 检查除零操作
- 使用更稳定的activation

#### 问题3: 性能不如Hybrid
**调试**:
```python
# Profiling
import time
start = time.time()
output = model(input)
mx.eval(output)  # 等待计算完成
print(f"Time: {time.time() - start}")
```

---

## 📊 Progress Tracker

```
[✓] Step 1: MLX Conv1d API研究        (4/4) ✅
[✓] Step 2: Depthwise Conv实现       (5/5) ✅
[✓] Step 3: Conformer验证           (6/6) ✅
[✓] Step 4: 纯MLX集成               (4/4) ✅
[✓] Step 5: 端到端测试              (5/5) ✅
[ ] Step 6: 权重加载和优化          (0/?) ← 当前

总进度: 83% → 目标: 100% Pure MLX
```

## 🎯 Checkpoints

**Checkpoint 1**: Conv1d工作
- 能运行基本Conv1d
- 输出与PyTorch匹配

**Checkpoint 2**: Conformer工作  
- 完整Conformer通过测试
- Conditioning质量验证

**Checkpoint 3**: Pure MLX推理
- 能生成音频
- Token质量可接受

**Checkpoint 4**: Production Ready
- 音频质量 ≥ Hybrid
- 性能 ≥ Hybrid

---

## 🛠️ 开发环境设置

```bash
# 创建实验目录
mkdir -p experiments

# 创建测试脚本模板
touch experiments/test_mlx_conv1d_basic.py
touch experiments/test_conformer_conv.py
touch experiments/test_mlx_conformer.py
touch experiments/test_pure_mlx_quality.py
touch experiments/convert_conformer_weights.py

# 设置权限
chmod +x experiments/*.py
```

---

## 📝 每步的Git Commit

```
Step 1 完成: "feat: Add MLX Conv1d research and basic tests"
Step 2 完成: "feat: Implement MLX depthwise separable convolution"
Step 3 完成: "feat: Validate MLX Conformer encoder"
Step 4 完成: "feat: Integrate pure MLX conditioning"
Step 5 完成: "test: End-to-end pure MLX validation"
Step 6 完成: "perf: Optimize pure MLX implementation"
最终:      "🎉 Milestone: Pure MLX implementation complete"
```

---

**准备好了吗？让我们从Step 1开始！** 🚀

回复 "开始 Step 1" 来开始第一步。

