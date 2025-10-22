# MLX Feature Request: ConvTranspose with Groups Parameter

## 📝 请求概述

**功能**: 为 `mlx.nn.ConvTranspose1d/2d/3d` 添加 `groups` 参数支持

**优先级**: 高

**用例**: BigVGAN vocoder和其他需要depthwise transposed convolution的模型

---

## 🎯 背景

### 当前问题

MLX的`ConvTranspose1d`缺少`groups`参数，导致无法实现depthwise transposed convolution，这在许多音频/图像生成模型中是核心特性。

```python
# PyTorch (支持)
torch.nn.ConvTranspose1d(
    in_channels=768, 
    out_channels=768,
    kernel_size=12,
    stride=2,
    groups=768  # ✅ 支持depthwise
)

# MLX (不支持)
mlx.nn.ConvTranspose1d(
    in_channels=768,
    out_channels=768, 
    kernel_size=12,
    stride=2
    # ❌ 无groups参数
)
```

---

## 💡 影响的模型

### 1. BigVGAN (Neural Vocoder)
- **用途**: TTS音频生成
- **依赖**: Anti-aliasing upsampling需要depthwise conv_transpose
- **影响**: 无法实现完整MLX版本

### 2. StyleGAN系列
- **用途**: 图像生成
- **依赖**: Anti-aliased upsampling

### 3. 其他音频模型
- HiFi-GAN variants
- UnivNet
- 各种需要anti-aliasing的生成模型

---

## 📊 预期收益

### 性能提升
- Depthwise操作：**100x+ faster**（vs 手动循环）
- 内存效率：更优的GPU/Metal利用

### 功能完整性
- 与PyTorch特性对等
- 支持更多SOTA模型迁移到MLX

---

## 🔧 建议实现

### API设计（与PyTorch一致）

```python
class ConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,  # ✨ 新增参数
        dilation: int = 1,
        bias: bool = True
    ):
        # Constraint: in_channels % groups == 0
        # Constraint: out_channels % groups == 0
        ...
```

### 行为
- `groups=1`: 标准conv_transpose（当前行为）
- `groups=in_channels`: Depthwise conv_transpose
- `groups=n`: Grouped conv_transpose

---

## 📋 参考

### PyTorch文档
https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html

### 相关Issue
- 类似功能已在`Conv1d/2d/3d`中实现（groups参数）
- `ConvTranspose`应当保持API一致性

---

## 🎯 优先级建议

**高优先级**

理由：
1. 功能gap大（PyTorch有，MLX无）
2. 影响范围广（音频、图像生成模型）
3. 实现相对标准（参考Conv的groups实现）
4. 社区需求强（BigVGAN等流行模型）

---

## 📞 联系方式

如需更多技术细节或用例，请参考：
- BigVGAN: https://github.com/NVIDIA/BigVGAN
- IndexTTS: https://github.com/index-tts/index-tts

---

**提交日期**: 2025-10-22  
**提交者**: IndexTTS开发团队  
**平台**: Apple Silicon M4

