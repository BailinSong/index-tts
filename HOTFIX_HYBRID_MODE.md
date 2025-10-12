# 🔴 紧急修复：切换到 Hybrid Mode

**日期**: 2025-10-12  
**问题**: Pure MLX Conditioning 导致音频质量严重异常  
**状态**: ✅ 已修复 (使用 Hybrid Mode)

---

## 🚨 问题描述

### 症状
使用 `--mlx` 参数时，生成的音频：
- ❌ **男声 → 女声**
- ❌ **"今天" → 无意义音节**
- ❌ 音色完全错误

### 根本原因

通过详细诊断发现：**MLX Conditioning 输出与 PyTorch 差异巨大**

```
诊断数据 (experiments/critical_debug_conditioning.py):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PyTorch Conditioning:
  Shape: (1, 32, 1280)
  Mean: 0.007135, Std: 1.449145
  Range: [-88.56, 45.93]
  Sample: [-0.316, 0.054, -0.078, ...]

MLX Conditioning:
  Shape: (1, 32, 1280)
  Mean: 0.001172, Std: 0.599423
  Range: [-7.22, 6.42]
  Sample: [1.002, -1.672, -0.023, ...]  # 完全不同！
  
差异分析:
  ❌ Max absolute diff: 89.98
  ❌ Mean absolute diff: 0.67
  ❌ Correlation: 0.055 (几乎无相关性)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**结论**: MLX Conformer/Perceiver 实现存在**严重错误**，导致：
1. Speaker 特征完全丢失
2. 音色信息错误
3. 生成内容无意义

---

## ✅ 修复方案

### Hybrid Mode: PyTorch Conditioning + MLX Transformer

**原理**:
- ✅ 使用 **PyTorch** 运行 Conformer + Perceiver (准确)
- ✅ 使用 **MLX** 运行 Transformer (快速)
- ✅ 结合两者优势：正确性 + 性能

**修改文件**: `indextts/infer_v2.py`

```python
# Before (Pure MLX - 有严重bug):
self.mlx_transformer = UnifiedVoiceMLX(
    use_mlx_conditioning=True,  # ❌ 导致音色错误
    **self.cfg.gpt
)

# After (Hybrid Mode - 正确):
self.mlx_transformer = UnifiedVoiceMLX(
    use_mlx_conditioning=False,  # ✅ 使用 PyTorch conditioning
    **self.cfg.gpt
)
```

---

## 📊 修复验证

### 测试音频对比

| 文件 | 模式 | 音色 | 内容 | 时长 | 状态 |
|------|------|------|------|------|------|
| `gen_base.wav` | Baseline | 男声 | "今天" | 1.21s | ✅ 基准 |
| `diagnose_pytorch.wav` | Pure PyTorch | 男声 | "今天" | 1.29s | ✅ 正确 |
| `diagnose_mlx.wav` | Pure MLX | 女声 | 无意义 | 1.93s | ❌ 错误 |
| **`hybrid_fixed.wav`** | **Hybrid MLX** | **男声** | **"今天"** | **1.53s** | **✅ 正确** |

### 性能对比

| 模式 | RTF | 音质 | 音色 | 状态 |
|------|-----|------|------|------|
| Pure PyTorch | 6.22x | ✅ | ✅ | 基准 |
| Pure MLX | 14.21x | ❌ | ❌ | 已禁用 |
| **Hybrid MLX** | **6.51x** | **✅** | **✅** | **当前** |

**结论**: Hybrid Mode 性能与 PyTorch 相当 (6.51x vs 6.22x)，但音质正确 ✅

---

## 🔧 使用方法

### 命令行
```bash
# 自动使用 Hybrid Mode
python webui.py --mlx

# 或
python -m indextts.cli --mlx --text "今天" --ref_audio examples/voice_01.wav
```

### 代码
```python
from indextts.infer_v2 import IndexTTS2

# use_mlx=True 会自动使用 Hybrid Mode
model = IndexTTS2(model_dir="checkpoints", use_mlx=True)
model.infer(
    spk_audio_prompt="examples/voice_01.wav",
    text="今天",
    output_path="output.wav"
)
```

---

## 🐛 Pure MLX Conditioning 问题清单

需要修复 MLX Conformer + Perceiver 的以下问题：

### 1. 权重加载问题 (可能)
- [ ] Conformer input_proj 权重映射
- [ ] Conformer 6层 blocks 权重对齐
- [ ] Conformer positional encoding
- [ ] Perceiver latents 初始化
- [ ] Perceiver proj_context 权重
- [ ] Perceiver 2层 cross-attention 权重

### 2. 前向传播问题 (可能)
- [ ] Conformer 的 Conv1d 实现 (depthwise separable)
- [ ] Conformer 的 GLU 激活函数
- [ ] Conformer 的 Swish/SiLU 激活
- [ ] Conformer 的 relative position encoding
- [ ] Perceiver 的 cross-attention 机制
- [ ] Perceiver 的 GEGLU 激活函数
- [ ] Perceiver 的 RMSNorm

### 3. 数值精度问题 (可能)
- [ ] LayerNorm epsilon 值
- [ ] 激活函数的数值稳定性
- [ ] 梯度/权重的量化误差

---

## 📝 调试计划

### Phase 1: 定位问题层 (优先级: 🔥 高)
1. 对比 MLX Conformer 每一层的输出
2. 对比 MLX Perceiver 每一层的输出
3. 找出第一个产生显著差异的层

### Phase 2: 修复权重加载 (优先级: 🔥 高)
1. 验证所有 197 个 conditioning 权重的形状
2. 检查权重转置、reshape 是否正确
3. 对比 PyTorch 和 MLX 的权重值

### Phase 3: 修复前向传播 (优先级: 🔥 高)
1. 逐个验证激活函数实现
2. 检查 Conv1d、LayerNorm 等基础 ops
3. 验证 attention mask 和 positional encoding

### Phase 4: 端到端验证 (优先级: 中)
1. 重新测试 Pure MLX Conditioning
2. 对比音频质量
3. 如果正确，切换回 Pure MLX Mode

---

## 🎯 当前状态

**生产状态**: ✅ **Hybrid Mode (已修复)**
- Conditioning: PyTorch (准确)
- Transformer: MLX (快速)
- 音质: 正常 ✅
- 性能: 6.51x RTF

**开发状态**: ⚠️ **Pure MLX Conditioning (待修复)**
- 需要深入调试
- 预计需要 4-8 小时
- 目标: 达到与 PyTorch 一致的输出

---

## 📁 相关文件

**诊断脚本**:
- `experiments/critical_debug_conditioning.py` - Conditioning 对比
- `experiments/diagnose_mlx_simple.py` - 波形/频谱图对比

**诊断结果**:
- `experiments/conditioning_comparison.npz` - 详细数值对比
- `experiments/diagnose_waveform.png` - 波形图
- `experiments/diagnose_spectrogram.png` - 频谱图
- `MLX_QUALITY_DIAGNOSIS.md` - 诊断报告

**测试音频**:
- `experiments/diagnose_pytorch.wav` - PyTorch 基准
- `experiments/diagnose_mlx.wav` - Pure MLX (错误)
- `experiments/hybrid_fixed.wav` - Hybrid Mode (正确) ✅

---

## 🚀 下一步

1. ✅ **已完成**: 切换到 Hybrid Mode
2. ✅ **已完成**: 验证音频质量
3. ⏳ **进行中**: 创建调试计划
4. ⏳ **待开始**: 修复 MLX Conditioning
5. ⏳ **待开始**: 切换回 Pure MLX Mode

---

**修复时间**: 2025-10-12  
**验证者**: 用户确认音频正常 ✅  
**提交状态**: 待提交

