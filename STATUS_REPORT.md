# IndexTTS MLX 实现 - 状态报告

**日期**: 2025-10-12  
**分支**: full_mlx  
**最新提交**: 8b723ce (HOTFIX: Hybrid Mode)

---

## 📊 总体状态

### ✅ 已完成的工作

#### 1. Hybrid MLX Implementation (可用于生产) ✅
- **状态**: ✅ **正常工作**
- **架构**: PyTorch Conditioning + MLX Transformer
- **音质**: 正常 (用户已验证)
- **性能**: RTF 6.51x (与 PyTorch 6.22x 相当)
- **使用方法**: `python webui.py --mlx`

#### 2. MLX Transformer (核心引擎) ✅
- **状态**: ✅ **正常工作**
- **功能**:
  - 24 层 GPT2-style Transformer
  - Multi-head attention with KV caching
  - Causal mask for autoregressive generation
  - Position embeddings (text + mel)
  - Categorical sampling (正确)
- **性能**: 生成速度正常，stop token 检测正确

#### 3. MLX 缓存系统 ✅
- **状态**: ✅ **正常工作**
- **功能**:
  - 自动转换 PyTorch → MLX (.npz)
  - 缓存 GPT, S2MEL, BigVGAN 权重
  - 快速加载 (3.3GB GPT 瞬间加载)

### ⚠️ 已知问题

#### 1. Pure MLX Conditioning ❌
- **状态**: ❌ **有严重bug**
- **问题**: MLX Conformer/Perceiver 输出与 PyTorch 差异巨大
- **影响**: 音色错误，内容错误
- **数据**:
  - Max diff: 89.98
  - Correlation: 0.055
  - Speaker 特征完全丢失
- **修复计划**: `MLX_CONDITIONING_DEBUG_PLAN.md`
- **预计时间**: 5.5-9.5 小时

---

## 🎯 当前可用功能

### ✅ 可以使用的功能

| 功能 | 状态 | 命令 |
|------|------|------|
| **Hybrid MLX 推理** | ✅ 正常 | `python webui.py --mlx` |
| 文本转语音 | ✅ 正常 | 支持中文 |
| 说话人控制 | ✅ 正常 | 通过参考音频 |
| 情感控制 | ✅ 正常 | 通过 emo_text |
| 流式输出 | ✅ 正常 | stream_return=True |
| 音频质量 | ✅ 正常 | 与 PyTorch 一致 |

### ⏳ 部分可用的功能

| 功能 | 状态 | 说明 |
|------|------|------|
| Pure MLX Mode | ❌ 暂不可用 | Conditioning 有bug |
| 性能加速 | ⚠️ 有限 | Hybrid 与 PyTorch 性能相当 |
| S2MEL MLX | ⏸️ 未实现 | 仍使用 PyTorch |
| BigVGAN MLX | ⏸️ 未实现 | 仍使用 PyTorch |

---

## 📈 性能数据

### 当前性能 (Hybrid Mode)

**测试**: "今天" (voice_01.wav)

| 指标 | 数值 |
|------|------|
| 生成 tokens | 77 |
| 音频长度 | 1.53s |
| GPT 时间 | 2.92s |
| S2MEL 时间 | 3.36s |
| BigVGAN 时间 | 0.38s |
| **总时间** | **9.97s** |
| **RTF** | **6.51x** |

### 性能对比

| 模式 | RTF | 音质 | 可用性 |
|------|-----|------|--------|
| Pure PyTorch | 6.22x | ✅ | ✅ 基准 |
| Hybrid MLX | 6.51x | ✅ | ✅ **当前** |
| Pure MLX (broken) | 14.21x | ❌ | ❌ 已禁用 |

---

## 🔧 技术细节

### Hybrid Mode 架构

```
输入文本 → TextTokenizer
参考音频 → SeamlessM4T → Speaker Embedding (1, 121, 1024)
                                    ↓
                        PyTorch Conformer (6 layers, 512D)
                                    ↓
                        PyTorch Perceiver (2 layers, 1280D)
                                    ↓
                          Conditioning Latents (1, 32, 1280)
                                    ↓
                    [转换到 MLX] mx.array()
                                    ↓
                        MLX Transformer (24 layers) ✅
                                    ↓
                          Mel Tokens (77 tokens)
                                    ↓
                    [转换回 PyTorch] torch.from_numpy()
                                    ↓
                        S2MEL (PyTorch) → Mel Spectrogram
                                    ↓
                        BigVGAN (PyTorch) → Waveform
                                    ↓
                          Output Audio (1.53s)
```

**关键点**:
- ✅ Conditioning 使用 PyTorch (准确)
- ✅ Transformer 使用 MLX (M4 优化)
- ⚠️ 数据在 PyTorch ↔ MLX 之间转换 (有开销)

---

## 📋 待办事项

### 🔥 高优先级

1. **修复 MLX Conditioning** (预计 5.5-9.5 小时)
   - [ ] Phase 1: 逐层诊断 Conformer
   - [ ] Phase 2: 修复权重/前向传播
   - [ ] Phase 3: 验证质量
   - [ ] Phase 4: 切换回 Pure MLX Mode

2. **S2MEL MLX 化** (预计 4-6 小时)
   - [ ] 实现 Flow Matching in MLX
   - [ ] 转换 S2MEL 权重
   - [ ] 验证输出一致性
   - [ ] 预期提升: ~3.36s → ~1.5s

3. **BigVGAN MLX 化** (预计 2-4 小时)
   - [ ] 实现 Vocoder in MLX
   - [ ] 转换 BigVGAN 权重
   - [ ] 验证音频质量
   - [ ] 预期提升: ~0.38s → ~0.2s

### 🔸 中优先级

4. **性能优化**
   - [ ] 减少 PyTorch ↔ MLX 转换开销
   - [ ] 优化 MLX Transformer KV cache
   - [ ] 并行化部分计算

5. **测试和验证**
   - [ ] 创建自动化测试套件
   - [ ] 多样本质量测试
   - [ ] 长文本压力测试

### 🔹 低优先级

6. **文档和示例**
   - [ ] API 文档
   - [ ] 使用示例
   - [ ] 性能调优指南

---

## 🎯 路线图

### Phase 1: 稳定 Hybrid Mode (已完成) ✅
- [x] 实现 Hybrid Mode
- [x] 修复音频质量问题
- [x] 用户验证
- [x] 提交 hotfix

### Phase 2: 修复 Pure MLX Conditioning (进行中) ⏳
- [ ] 诊断 Conformer/Perceiver 问题
- [ ] 逐层修复
- [ ] 验证质量
- [ ] 切换回 Pure MLX Mode

### Phase 3: Full MLX Pipeline (未开始) ⏸️
- [ ] S2MEL MLX 化
- [ ] BigVGAN MLX 化
- [ ] 端到端 Pure MLX
- [ ] 性能优化

### Phase 4: 生产部署 (未开始) ⏸️
- [ ] 完整测试
- [ ] 文档完善
- [ ] 合并到 main 分支
- [ ] 发布

---

## 📊 预期最终性能

假设所有模块都转为 MLX：

| 模块 | 当前 (Hybrid) | 目标 (Pure MLX) | 预期提升 |
|------|---------------|-----------------|----------|
| GPT Cond | 0.05s (PT) | 0.05s (PT) | - |
| GPT Gen | 2.92s (MLX) | 2.50s (MLX) | -14% |
| S2MEL | 3.36s (PT) | 1.50s (MLX) | -55% |
| BigVGAN | 0.38s (PT) | 0.20s (MLX) | -47% |
| **总计** | **9.97s** | **~5.0s** | **-50%** |
| **RTF** | **6.51x** | **~3.3x** | **-49%** |

**目标**: 实时因子 (RTF) < 3.5x

---

## 🚀 如何使用当前版本

### 命令行

```bash
# 启动 WebUI (推荐)
python webui.py --mlx

# CLI 推理
python -m indextts.cli \
  --mlx \
  --text "今天天气很好" \
  --ref_audio examples/voice_01.wav \
  --output output.wav
```

### Python API

```python
from indextts.infer_v2 import IndexTTS2

# 初始化 (use_mlx=True 自动使用 Hybrid Mode)
model = IndexTTS2(
    model_dir="/Users/bailin/index-tts/checkpoints",
    use_mlx=True
)

# 推理
model.infer(
    spk_audio_prompt="examples/voice_01.wav",
    text="今天天气很好",
    output_path="output.wav",
    emo_audio_prompt="examples/voice_01.wav",
    use_emo_text=True,
    emo_text="今天天气很好"
)
```

---

## 📁 关键文件

### 配置和检查点
- `checkpoints/config.yaml` - 模型配置
- `checkpoints/gpt.pth` - GPT 权重 (PyTorch)
- `checkpoints/mlx/gpt.npz` - GPT 权重 (MLX 缓存)
- `checkpoints/mlx/s2mel.npz` - S2MEL 权重 (MLX 缓存)

### 核心代码
- `indextts/infer_v2.py` - 主推理接口 ⭐
- `indextts/gpt/mlx_model.py` - MLX Transformer 实现 ⭐
- `indextts/gpt/mlx_conditioning.py` - MLX Conditioning (有bug)
- `indextts/utils/mlx_cache.py` - MLX 缓存管理

### 诊断和测试
- `experiments/critical_debug_conditioning.py` - Conditioning 诊断 ⭐
- `experiments/diagnose_mlx_simple.py` - 音频质量诊断
- `experiments/step6_4_final_comparison.py` - 完整对比测试

### 文档
- `HOTFIX_HYBRID_MODE.md` - Hotfix 详情 ⭐
- `MLX_CONDITIONING_DEBUG_PLAN.md` - 调试计划
- `MLX_QUALITY_DIAGNOSIS.md` - 质量诊断报告
- `STATUS_REPORT.md` (本文件)

---

## 🎓 经验教训

### ✅ 成功经验

1. **逐步验证**: 分层测试 (Transformer, Conditioning 分别验证)
2. **Hybrid 策略**: 在 Pure MLX 有问题时，Hybrid 是很好的中间方案
3. **详细诊断**: 创建诊断脚本帮助快速定位问题
4. **用户验证**: 实际听音测试比数值对比更重要

### ⚠️ 需要改进

1. **前期测试不足**: Pure MLX Conditioning 应该更早进行端到端测试
2. **权重验证**: 加载权重后应立即验证输出，而非等到生成音频
3. **文档先行**: 应该先创建架构文档，再实现代码

---

## 📞 下一步决策

**请选择下一步行动**:

### 选项 A: 立即修复 MLX Conditioning 🔥
- 时间: 5.5-9.5 小时
- 收益: 恢复 Pure MLX Mode
- 优先级: 高

### 选项 B: 先实现 S2MEL/BigVGAN MLX 化 ⚡
- 时间: 6-10 小时
- 收益: 显著性能提升 (50%)
- 优先级: 高

### 选项 C: 使用当前 Hybrid Mode，进行其他开发 ✅
- Hybrid Mode 已经可用
- 可以先完成其他功能
- 稍后再优化 Pure MLX

### 选项 D: 完善测试和文档 📚
- 创建测试套件
- 完善 API 文档
- 准备生产部署

---

**当前状态**: ⏸️ 等待决策  
**推荐**: 选项 B (S2MEL/BigVGAN MLX 化) - 性能提升最明显  
**备选**: 选项 A (修复 Conditioning) - 完成 Pure MLX 目标

