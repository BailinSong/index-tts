# MLX Conv1d API 研究结果 - Step 1

## 🎯 关键发现

### 1. Input Format 差异 ✅

| Framework | Input Format | Example Shape |
|-----------|-------------|---------------|
| **PyTorch** | NCL: `(batch, channels, seq)` | `(1, 64, 100)` |
| **MLX** | NLC: `(batch, seq, channels)` | `(1, 100, 64)` |

**结论**: MLX 使用 "自然顺序" (batch, length, features)，类似 NLP 任务的常见格式

### 2. Weight Format 差异 ✅

| Framework | Weight Format | Example Shape |
|-----------|--------------|---------------|
| **PyTorch** | `(out, in, kernel)` | `(128, 64, 3)` |
| **MLX** | `(out, kernel, in)` | `(128, 3, 64)` |

**结论**: MLX 交换了最后两个维度

**权重转换公式**:
```python
# PyTorch → MLX
mlx_weight = pytorch_weight.permute(0, 2, 1)  # (out, in, k) → (out, k, in)

# MLX → PyTorch
pytorch_weight = mlx_weight.permute(0, 2, 1)  # (out, k, in) → (out, in, k)
```

### 3. Padding 行为 ⚠️

- **PyTorch**: 支持 `padding` 参数，默认 0
  ```python
  Conv1d(64, 128, kernel_size=3, padding=1)  # Same shape output
  ```

- **MLX**: Conv1d **无 padding 参数**
  - 输入: `(1, 100, 64)`, kernel=3
  - 输出: `(1, 98, 64)`  ← 少了 2 个位置
  - 输出长度: `seq_out = seq_in - kernel_size + 1`

**解决方案**: 需要手动 padding 或使用其他策略

### 4. Depthwise Convolution 支持 ❓

- **PyTorch**: 使用 `groups` 参数
  ```python
  Conv1d(64, 64, kernel_size=31, groups=64)  # Depthwise
  ```

- **MLX**: Conv1d **无 groups 参数**
  - 测试中运行了，但权重 shape 不符合 depthwise 的预期
  - PyTorch depthwise weight: `(channels, 1, kernel)` 
  - MLX weight: `(64, 31, 64)` ← 不是 depthwise!

**结论**: MLX Conv1d 不支持 depthwise convolution，需要自定义实现

### 5. Pointwise Convolution ✅

- Kernel_size=1 时，Conv1d **等价于** Linear layer
- 可以用 Linear 代替 pointwise conv

```python
# These are equivalent:
conv1x1 = nn.Conv1d(64, 128, kernel_size=1)
linear = nn.Linear(64, 128)
```

---

## 📋 Step 1 完成任务清单

- [x] 1.1 创建测试文件 `experiments/test_mlx_conv1d_basic.py`
- [x] 1.2 测试简单的Conv1d操作
- [x] 1.3 对比PyTorch Conv1d输出
- [x] 1.4 测试不同参数配置

**成功标准**:
- ✅ 理解输入格式: NLC `(batch, seq, channels)`
- ✅ 理解输出格式: NLC `(batch, seq_out, channels_out)`
- ✅ 理解权重初始化: `(out, kernel, in)`

---

## 🚨 关键问题和解决方案

### 问题 1: MLX Conv1d 无 padding 参数

**影响**: 输出长度会缩短

**解决方案**:
```python
# 方案 A: 手动 padding input
def pad_1d(x, padding):
    # x: (batch, seq, channels)
    pad_left = mx.zeros((x.shape[0], padding, x.shape[2]))
    pad_right = mx.zeros((x.shape[0], padding, x.shape[2]))
    return mx.concatenate([pad_left, x, pad_right], axis=1)

x_padded = pad_1d(x, kernel_size // 2)
out = conv(x_padded)
```

```python
# 方案 B: 使用 MLX 的 pad function
import mlx.core as mx
x_padded = mx.pad(x, [(0, 0), (pad, pad), (0, 0)])
out = conv(x_padded)
```

### 问题 2: MLX Conv1d 无 groups 参数（不支持 depthwise）

**影响**: 无法直接实现 Conformer 的 depthwise separable convolution

**解决方案**: 手动实现 depthwise convolution

```python
class MLXDepthwiseConv1d(nn.Module):
    """Depthwise Conv1d: each channel has its own kernel"""
    
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        
        # Weight: (channels, kernel_size, 1)
        # Each channel has its own 1D kernel
        self.weight = mx.random.normal((channels, kernel_size, 1)) * 0.02
    
    def __call__(self, x):
        # x: (batch, seq, channels)
        batch, seq, channels = x.shape
        
        # Reshape for per-channel convolution
        # Option 1: Loop over channels (slow)
        outputs = []
        for c in range(channels):
            x_c = x[:, :, c:c+1]  # (batch, seq, 1)
            # Apply 1D conv for this channel
            # Need to implement...
        
        # Option 2: Use einsum or other tricks
        # TBD...
        
        return output
```

**更好的方案**: 使用 Linear layers 代替（如当前的简化实现）

---

## 📊 测试结果总结

| Test | PyTorch Output | MLX Output | Status |
|------|---------------|-----------|--------|
| Basic Conv1d | `(1, 128, 100)` | `(1, 98, 128)` | ✅ Works (no padding) |
| Depthwise | `(1, 64, 100)` | `(1, 70, 64)` | ⚠️ Not true depthwise |
| Pointwise (1x1) | `(1, 128, 100)` | `(1, 100, 128)` | ✅ Perfect |

---

## 🔄 Next Steps (Step 2)

1. ✅ **理解了 MLX Conv1d API**
2. → **实现 depthwise separable convolution**
   - Option A: 真正的 depthwise conv (需要自定义实现)
   - Option B: 使用 Linear 近似（当前方案）
3. → **测试数值准确性**
4. → **集成到 Conformer**

---

## 💡 建议

### 对于 Conformer ConvolutionModule:

**方案 A: 简化实现（推荐用于快速原型）**
```python
class MLXConvolutionModule(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 31):
        super().__init__()
        # Use Linear layers instead of Conv
        self.pointwise1 = nn.Linear(channels, 2*channels)  # Expand
        self.pointwise2 = nn.Linear(channels, channels)    # Project
        # Skip depthwise for now
```

**优点**: 简单，快速实现
**缺点**: 丢失了局部感受野（convolution 的核心优势）

**方案 B: 真实 Depthwise Conv（推荐用于最终实现）**
```python
class MLXDepthwiseSeparableConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 31):
        super().__init__()
        # Pointwise expansion
        self.expand = nn.Linear(channels, 2*channels)
        
        # Depthwise (custom implementation)
        self.depthwise = MLXDepthwiseConv1d(2*channels, kernel_size)
        
        # Pointwise projection
        self.project = nn.Linear(channels, channels)
```

**优点**: 保留 convolution 的语义和局部特性
**缺点**: 需要自定义 depthwise 实现，可能较慢

---

## ✅ Step 1 完成状态

**Progress: 4/4 tasks completed** ✅

- [x] 理解 MLX Conv1d 输入/输出格式
- [x] 理解权重格式
- [x] 识别 API 差异（padding, groups）
- [x] 提出解决方案

**下一步**: Step 2 - 实现 Depthwise Separable Convolution

