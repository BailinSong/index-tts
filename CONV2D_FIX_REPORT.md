# Conv2d Subsampling 修复报告

## 🎯 目标
实现完整的 Pure MLX Conditioning (Conformer + Perceiver)

## 🔍 发现的问题

### 问题 1: 缺少 Conv2d Subsampling
**症状**:
```
PyTorch After embed: (1, 60, 512)   # 121 → 60 下采样
MLX After embed:     (1, 121, 512)  # 没有下采样
```

**根本原因**:
- PyTorch Conformer 使用 `Conv2dSubsampling2` 进行下采样
- MLX 实现只有简单的 Linear + LayerNorm
- 这是 **架构级别的缺陷**，不是小的权重加载错误

**解决方案**:
1. 实现 `MLXConv2d` (支持 stride=2, kernel=3, padding=0)
2. 实现 `MLXConv2dSubsampling2Fixed` 模块
3. 更新 `MLXConformerEncoder` 使用新的 subsampling 层

**关键代码**:
```python
# PyTorch 的 Conv2d 计算:
# Input: (batch, 1, time=121, freq=1024)
# Conv2d(1→512, kernel=3, stride=2, padding=0)
# Output: (batch, 512, time'=60, freq'=511)
# Flatten: (batch, 60, 512*511=261632)
# Linear: 261632 → 512

# 输出大小公式:
# out = floor((in - kernel) / stride) + 1
# time: (121 - 3) / 2 + 1 = 60 ✅
# freq: (1024 - 3) / 2 + 1 = 511 ✅
```

### 问题 2: 缺少 xscale (Positional Encoding Scaling)
**症状**:
```
After Linear projection:
  PyTorch: mean=0.38, std=17.6
  MLX:     mean=0.38, std=17.6 ✅

After Positional encoding:
  PyTorch: mean=8.5, std=398.8 🔴
  MLX:     mean=0.76, std=17.6  ❌
```

**根本原因**:
```python
# PyTorch 的 positional encoding:
self.xscale = math.sqrt(self.d_model)  # sqrt(512) = 22.627
x = x * self.xscale + pos_emb

# MLX 原实现 (错误):
x = x + pos_emb  # 缺少 * xscale!
```

**解决方案**:
```python
# MLXConformerEncoder.__init__
self.xscale = math.sqrt(output_dim)  # 22.627

# MLXConformerEncoder.__call__
x = x * self.xscale + pos_emb  # ✅
```

**数值验证**:
```
17.6 * 22.627 = 398.2 ≈ 398.8 ✅
```

## 📊 改进效果

### Conformer After Embed (Conv2d + xscale 修复后)
| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| Mean diff | 299.54 | 0.58 |
| Max diff | 1937.11 | 1.01 |
| Correlation | 0.9994 | 0.999999 ✅ |

### Final Conditioning Output
| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| Correlation | 0.055 | 0.534 |
| Max diff | 90.0 | 43.7 |

**改善幅度**: Correlation 提升 **9.7倍** (0.055 → 0.534)

## 🎵 音频生成测试

**测试命令**:
```bash
python -m indextts.cli --mlx -v examples/voice_01.wav -o test_pure_mlx_v2.wav "今天天气很好"
```

**结果**:
- ✅ 生成成功: 1.71秒音频
- ✅ 86 tokens 生成
- ✅ RTF: 8.48x (可接受)
- ⚠️  **音质待验证** (correlation 0.53 仍有改善空间)

## 📝 新增文件

1. **indextts/gpt/mlx_subsampling.py**
   - `MLXConv2d`: 2D 卷积实现 (stride, padding 支持)
   - `MLXConv2dSubsampling2Fixed`: Conv2d + Linear 投影

2. **experiments/**
   - `debug_conformer_layer_by_layer.py`: 逐层诊断
   - `debug_pytorch_embed.py`: PyTorch embed 详细分析
   - `test_mlx_subsampling.py`: Conv2d 形状测试
   - `test_full_conditioning.py`: 完整 conditioning 对比

## 🔧 修改的文件

1. **indextts/gpt/mlx_conditioning.py**
   - 添加 `self.xscale = math.sqrt(output_dim)`
   - 使用 `MLXConv2dSubsampling2Fixed`
   - 修复 forward 中的 positional encoding

2. **indextts/gpt/mlx_model.py**
   - 更新 `_load_conformer_weights` 加载 Conv2d 权重
   - 修复 `after_norm` 属性名

3. **indextts/infer_v2.py**
   - 切换到 Pure MLX Conditioning (use_mlx_conditioning=True)

## 🚧 已知问题

1. **Conditioning correlation 仍然偏低 (0.53)**
   - Conformer blocks 内部可能还有实现差异
   - Perceiver 模块可能需要进一步调试
   
2. **音质待验证**
   - 需要人工听测确认音色、音节是否正确
   - 可能需要进一步调整

## 🎯 下一步建议

**选项 A**: 继续优化 MLX Conditioning
- 逐层对比 Conformer blocks (attention, conv, FF)
- 调试 Perceiver 模块
- 目标: correlation > 0.90

**选项 B**: 接受当前方案
- Correlation 0.53 可能已足够用于生成
- 先验证音质，如果可接受则转向性能优化
- 实现 S2MEL/BigVGAN MLX 化 (性能提升 50%)

**选项 C**: 回退到 Hybrid Mode
- PyTorch Conditioning (准确) + MLX Transformer (快速)
- 已验证音质正常
- RTF: 6.5x

## ✅ 完成的工作

- [x] 发现架构缺陷 (Conv2d + xscale)
- [x] 实现 MLX Conv2d Subsampling
- [x] 验证形状匹配 (121 → 60)
- [x] 加载 Conv2d 权重
- [x] 修复 xscale
- [x] 端到端音频生成测试
- [x] Correlation 提升 9.7倍

## 📈 性能对比

| 模式 | RTF | 音质 | Status |
|------|-----|------|--------|
| PyTorch Full | 15.12x | ✅ 完美 | Baseline |
| Hybrid (PT Cond + MLX Trans) | 6.51x | ✅ 正常 | 已验证 |
| Pure MLX (修复前) | N/A | ❌ 严重异常 | 废弃 |
| Pure MLX (修复后) | 8.48x | ⚠️  待验证 | 当前 |

**结论**: Conv2d + xscale 是 **关键修复**，让 Pure MLX Conditioning 从完全不可用变为基本可用。

