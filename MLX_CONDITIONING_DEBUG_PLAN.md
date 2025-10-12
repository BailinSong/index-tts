# MLX Conditioning 调试计划

**创建日期**: 2025-10-12  
**状态**: 🔴 Blocked - 需要修复 MLX Conformer/Perceiver  
**优先级**: 🔥 高 - 阻塞 Pure MLX 实现

---

## 📊 问题总结

### 当前状态
- ✅ **Hybrid Mode**: 正常工作 (PyTorch Cond + MLX Transformer)
- ❌ **Pure MLX Mode**: 音频质量异常 (需要修复)

### 数值差异
```
MLX vs PyTorch Conditioning:
  Max absolute diff: 89.98
  Mean absolute diff: 0.67
  Correlation: 0.055 (几乎无相关)
  
影响:
  ❌ Speaker 特征完全丢失
  ❌ 男声 → 女声
  ❌ 内容错误 ("今天" → 无意义音节)
```

---

## 🎯 调试目标

找出并修复 MLX Conditioning 中的错误，使其输出与 PyTorch 一致（max_diff < 2.0, correlation > 0.95）

---

## 🔍 Phase 1: 逐层诊断 (预计 2-3 小时)

### 1.1 创建逐层对比脚本

**脚本**: `experiments/debug_mlx_conformer_layers.py`

```python
# 对比 Conformer 每一层的输出
# 输入: Speaker embedding (1, 121, 1024)
# 
# Layer 0: input_proj + pos_encoding
#   PyTorch: ...
#   MLX: ...
#   Diff: ...
#
# Layer 1-6: Conformer blocks
#   每个 block:
#     - ff_macaron (前馈)
#     - self_attn (注意力)
#     - conv_module (卷积)
#     - ff (后馈)
#   对比每个子层的输出
#
# 找出第一个产生大差异的层
```

### 1.2 创建 Perceiver 对比脚本

**脚本**: `experiments/debug_mlx_perceiver_layers.py`

```python
# 假设 Conformer 输出正确
# 对比 Perceiver 每一层的输出
#
# Layer 0: proj_context (512 → 1280)
#   PyTorch: ...
#   MLX: ...
#
# Layer 1-2: Cross-attention layers
#   每个 layer:
#     - Cross-attention (latents attend to context)
#     - Feed-forward (GEGLU)
#   对比每个子层的输出
#
# Final: RMSNorm
```

### 1.3 预期输出

找出问题所在层，例如：
```
✓ Conformer input_proj: diff < 0.1
✓ Conformer block 1-5: diff < 0.5
❌ Conformer block 6: diff = 45.2  ← 问题在这里！
  ❌ conv_module: diff = 45.2
    ✓ pointwise1: diff < 0.1
    ❌ depthwise: diff = 45.2  ← 根本原因
```

---

## 🔧 Phase 2: 修复具体问题 (预计 2-4 小时)

根据 Phase 1 的发现，修复对应的层。

### 可能的问题类型

#### 类型 A: 权重加载错误

**症状**: 某层输出差异巨大  
**原因**: 权重形状、转置、索引错误  
**修复**:
```python
# 检查权重加载
print(f"PyTorch weight: {pt_weight.shape}")
print(f"MLX weight: {mlx_weight.shape}")
print(f"Diff: {np.abs(pt_weight - mlx_weight).max()}")

# 验证转置
if pt_conv_weight.shape == (out, in, kernel):
    mlx_conv_weight = pt_conv_weight.transpose(1, 2, 0)
```

#### 类型 B: 激活函数错误

**症状**: 某层输出符号/范围错误  
**原因**: Swish、SiLU、GEGLU 实现不准确  
**修复**:
```python
# Swish (x * sigmoid(x))
def swish(x):
    return x * mx.sigmoid(x)  # 确保没有额外的系数

# GEGLU
def geglu(x):
    x, gate = mx.split(x, 2, axis=-1)
    return x * nn.gelu(gate)  # 确保使用正确的 GELU
```

#### 类型 C: Conv1d 实现错误

**症状**: Depthwise conv 输出完全错误  
**原因**: MLX Conv1d API 理解错误  
**修复**:
```python
# 检查 depthwise conv 的实现
# 权重形状: (channels, kernel_size) 而非 (channels, 1, kernel_size)
# Padding: 需要手动实现
class MLXDepthwiseConv1d:
    def __call__(self, x):
        # x: (batch, seq, channels)
        if self.padding > 0:
            x = mx.pad(x, [(0,0), (self.padding, self.padding), (0,0)])
        # ... 滑动窗口卷积
```

#### 类型 D: LayerNorm/BatchNorm 错误

**症状**: 数值偏移、方差不对  
**原因**: epsilon、归一化轴错误  
**修复**:
```python
# 检查 LayerNorm
ln = nn.LayerNorm(dims, eps=1e-5)  # 确保 eps 与 PyTorch 一致

# 检查归一化轴
# PyTorch LayerNorm: normalized_shape=[-1]
# MLX LayerNorm: dims (最后一维)
```

#### 类型 E: Attention Mask 错误

**症状**: Attention 输出全零或异常大  
**原因**: Mask 形状、取值错误  
**修复**:
```python
# 检查 mask
# PyTorch: mask = (1 - mask) * -10000
# MLX: mask 应该是 (seq, seq) 的 bool/float

# Relative position encoding
# 确保 pos_emb 的形状和索引正确
```

---

## 🧪 Phase 3: 逐步验证 (预计 1-2 小时)

### 3.1 单层验证

修复一层后，验证该层输出：
```bash
python experiments/verify_conformer_block_X.py
# 预期: diff < 0.5, correlation > 0.95
```

### 3.2 累积验证

修复所有层后，验证整体 Conditioning：
```bash
python experiments/critical_debug_conditioning.py
# 预期: max_diff < 2.0, correlation > 0.95
```

### 3.3 端到端验证

验证生成的音频质量：
```bash
python -m indextts.cli --mlx --text "今天" \
  --ref_audio examples/voice_01.wav \
  --output test_pure_mlx.wav

# 听音验证: 男声, "今天", 清晰
```

---

## 📋 Phase 4: 切换回 Pure MLX (预计 0.5 小时)

### 4.1 修改代码

```python
# indextts/infer_v2.py
self.mlx_transformer = UnifiedVoiceMLX(
    use_mlx_conditioning=True,  # ✅ 恢复 Pure MLX
    **self.cfg.gpt
)
```

### 4.2 完整测试

运行完整的测试套件：
```bash
# 多个测试用例
python experiments/step6_4_final_comparison.py

# 预期结果:
# ✓ MLX 音频与 PyTorch 音频质量相当
# ✓ Conditioning diff < 2.0
# ✓ 音色、内容正确
```

### 4.3 性能对比

```bash
# 测试性能
python experiments/benchmark_pure_mlx.py

# 目标:
# Pure MLX < Hybrid < Pure PyTorch
# 理想: Pure MLX RTF < 5.0x (vs Hybrid 6.5x)
```

---

## 🛠️ 调试工具

### 工具 1: 权重检查器

```python
def check_weights(pt_weight, mlx_weight, name):
    diff = np.abs(pt_weight - mlx_weight)
    print(f"{name}:")
    print(f"  Shape: PT={pt_weight.shape}, MLX={mlx_weight.shape}")
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")
    if diff.max() > 0.01:
        print(f"  ⚠️  WARNING: Large difference!")
```

### 工具 2: 前向传播跟踪

```python
def trace_forward(model, x, layer_names):
    outputs = {}
    for name in layer_names:
        outputs[name] = model.get_layer_output(name, x)
    return outputs

# 对比 PyTorch 和 MLX 的每层输出
pt_outputs = trace_forward(pt_model, x, layers)
mlx_outputs = trace_forward(mlx_model, x, layers)

for name in layers:
    diff = compare_outputs(pt_outputs[name], mlx_outputs[name])
    print(f"{name}: diff={diff:.6f}")
```

### 工具 3: 可视化对比

```python
import matplotlib.pyplot as plt

def visualize_difference(pt_out, mlx_out, title):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    axes[0].imshow(pt_out[0].T, aspect='auto', cmap='viridis')
    axes[0].set_title(f'PyTorch - {title}')
    
    axes[1].imshow(mlx_out[0].T, aspect='auto', cmap='viridis')
    axes[1].set_title(f'MLX - {title}')
    
    diff = np.abs(pt_out - mlx_out)
    axes[2].imshow(diff[0].T, aspect='auto', cmap='hot')
    axes[2].set_title(f'Absolute Difference (max={diff.max():.4f})')
    
    plt.tight_layout()
    plt.savefig(f'debug_{title}.png')
```

---

## 📊 成功标准

### 最低标准 (可接受)
- [ ] Max absolute diff < 5.0
- [ ] Mean absolute diff < 1.0
- [ ] Correlation > 0.90
- [ ] 音频: 音色基本正确，内容清晰

### 理想标准 (目标)
- [ ] Max absolute diff < 2.0
- [ ] Mean absolute diff < 0.5
- [ ] Correlation > 0.95
- [ ] 音频: 与 PyTorch 质量无明显差异

### 卓越标准 (最佳)
- [ ] Max absolute diff < 0.5
- [ ] Mean absolute diff < 0.1
- [ ] Correlation > 0.99
- [ ] 音频: 完全一致

---

## 📅 时间估算

| Phase | 任务 | 预计时间 |
|-------|------|----------|
| 1 | 逐层诊断 | 2-3 小时 |
| 2 | 修复问题 | 2-4 小时 |
| 3 | 验证测试 | 1-2 小时 |
| 4 | 切换和部署 | 0.5 小时 |
| **总计** | | **5.5-9.5 小时** |

---

## 🎯 下一步行动

**立即开始** (如果用户同意):
1. 创建 `experiments/debug_mlx_conformer_layers.py`
2. 运行逐层对比，找出问题层
3. 根据发现制定具体修复方案

**或者暂停** (如果需要休息):
- Hybrid Mode 当前正常工作 ✅
- 可以先使用 Hybrid Mode 进行其他开发
- 稍后再返回调试 Pure MLX Conditioning

---

**当前状态**: ⏸️ 等待用户决定  
**选项 A**: 立即开始调试 MLX Conditioning  
**选项 B**: 先使用 Hybrid Mode，稍后调试  
**选项 C**: 其他优先事项 (如性能优化、S2MEL/BigVGAN MLX 化)

