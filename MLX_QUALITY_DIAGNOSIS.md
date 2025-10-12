# MLX 音频质量诊断报告

**日期**: 2025-10-12  
**测试文本**: "今天"  
**参考音频**: voice_01.wav  
**基准**: gen_base.wav

---

## 🔍 诊断结果摘要

### ✅ 好消息：MLX 音频能量正常！

| 指标 | Baseline | PyTorch | MLX |
|------|----------|---------|-----|
| **音频长度** | 1.21s | 1.29s | 1.93s |
| **RMS 能量** | 0.1011 | 0.0308 (30.4%) | 0.0903 (**89.3%**) |
| **峰值** | 0.7215 | 0.2983 | 0.4749 |
| **Mean** | 0.000078 | 0.000018 | -0.000209 |
| **Std** | 0.101091 | 0.030777 | 0.090302 |

**关键发现**:
- ✅ **MLX 能量正常** (89.3% of baseline)
- ⚠️ **MLX 比 PyTorch 能量高 293.4%**
- ⚠️ **MLX 音频更长** (1.93s vs PyTorch 1.29s)

---

## 🎯 问题分析

### 1. 这不是"音量太小"的问题

用户说"声音不对"，但诊断显示 MLX 音频能量接近 baseline，甚至比 PyTorch 更大。因此，问题可能是：

#### 可能原因 A: **音色/音质差异**
- MLX Conditioning (Conformer + Perceiver) 可能输出与 PyTorch 不同
- 导致生成的音色、音调、清晰度等有差异
- **不是音量问题，而是质量问题**

#### 可能原因 B: **生成长度异常**
- MLX 生成了 **1.93s** 音频 (PyTorch 只有 1.29s)
- 对于"今天"(2个汉字)，1.93s 可能过长
- 可能生成了额外的停顿、噪音或重复

#### 可能原因 C: **Conditioning 差异**
- 需要对比 MLX 和 PyTorch 的 conditioning latents
- 如果差异很大 (>2.0)，会影响生成质量

---

## 🔬 下一步诊断

### 方案 1: 对比 Conditioning Latents (推荐)

创建测试脚本，直接对比 PyTorch 和 MLX 的 conditioning 输出：

```python
# 提取相同的 speaker embedding
# 对比 PyTorch conditioning vs MLX conditioning
# 如果差异 > 2.0，说明 Conformer/Perceiver 有问题
```

### 方案 2: 听音对比

请手动听听以下音频：
- `gen_base.wav` - 基线
- `experiments/diagnose_pytorch.wav` - PyTorch 生成
- `experiments/diagnose_mlx.wav` - MLX 生成

回答问题：
1. MLX 音频是否**完全无声**？(诊断显示应该不是)
2. MLX 音频**音色是否异常**？(如机械音、失真、噪音)
3. MLX 音频**时长是否过长**？(1.93s vs 1.29s)
4. MLX 音频**内容是否正确**？(是否说了"今天")

### 方案 3: 查看波形和频谱图

已生成的可视化文件：
- `experiments/diagnose_waveform.png` - 波形对比
- `experiments/diagnose_spectrogram.png` - 频谱图对比

检查：
- MLX 波形是否有明显异常模式
- MLX 频谱是否缺少高频或低频成分

---

## 🔧 可能的修复方向

### 如果是 Conditioning 差异导致

**问题**: MLX Conformer/Perceiver 输出与 PyTorch 不一致

**修复步骤**:
1. 验证所有 197 个 conditioning 权重是否正确加载
2. 对比 Conformer 每一层的中间输出
3. 对比 Perceiver 每一层的中间输出
4. 检查激活函数 (Swish, SiLU, GEGLU) 是否正确实现
5. 检查 LayerNorm/BatchNorm 是否正确

### 如果是生成长度问题

**问题**: MLX 生成了过多 tokens

**可能原因**:
- Stop token 检测不准确
- Temperature/sampling 参数不同
- MLX transformer 的 generation 逻辑有差异

**修复步骤**:
1. 对比 PyTorch 和 MLX 生成的 token 序列
2. 检查 stop token (8193) 是否正确检测
3. 验证 sampling 策略 (categorical vs argmax)

---

## 📊 测试环境

- **环境**: conda indextts2
- **设备**: Apple Silicon M4 (MPS)
- **PyTorch模式**: CPU/MPS inference
- **MLX模式**: Pure MLX Conditioning + Transformer

---

## 🎯 结论

**诊断状态**: ⚠️ **需要进一步分析**

**确定的事实**:
- ✅ MLX 音频能量正常 (89% of baseline)
- ✅ MLX 成功生成音频 (不是无声)
- ⚠️ MLX 音频长度异常 (1.93s vs 1.29s)
- ❓ MLX 音质是否异常 (需要听音确认)

**最可能的问题**: 
1. **Conditioning 差异** → 音色不同
2. **生成长度异常** → 多余的停顿/噪音

**推荐行动**:
1. **听音对比** - 确定具体是哪种"不对"
2. **对比 Conditioning** - 量化差异程度
3. **对比 Token 序列** - 检查生成是否正确

---

## 📁 诊断文件

已生成的诊断文件:
- `experiments/diagnose_pytorch.wav` - PyTorch 音频
- `experiments/diagnose_mlx.wav` - MLX 音频
- `experiments/diagnose_waveform.png` - 波形对比图
- `experiments/diagnose_spectrogram.png` - 频谱图对比

---

**下一步**: 需要用户提供听音反馈，以确定具体问题类型。

