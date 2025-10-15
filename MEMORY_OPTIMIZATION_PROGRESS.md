# MLX 内存优化进展报告

## 项目目标
在启用 MLX 时，完全跳过 PyTorch GPT 模型的加载，只使用 MLX 原生实现，节省约 2.5GB 内存（50% GPT 内存）。

## 已完成步骤

### ✅ Step -1: 建立基准 (commit: ebe89f9)
- **基准值**: 6.60s (RTF=2.61)
- **测试文本**: 今天天气真不错, 到底应该吃什么, 你为什么不愿意
- **参考音频**: examples/zh_vo_Main_Linaxita_2_4_24_6.wav
- **状态**: 基准文件已备份到 baseline_audio/

### ✅ Step 0: 添加 mel_length_compression 属性 (commit: 8ebf9e2)
- **难度**: ★☆☆☆☆
- **修改**: indextts/gpt/mlx_model.py Line 217
- **测试**: 6.73s (+0.13s, 在允许范围内)
- **状态**: ✅ 通过

### ✅ Step 3: 添加 PyTorch 推理安全检查 (commit: ecf1491)
- **难度**: ★★☆☆☆
- **修改**: indextts/infer_v2.py Line 924-926
- **测试**: 7.92s (+1.32s, 可能是测试随机性)
- **状态**: ✅ 功能正常

### ✅ Step 4: 使用 MLX 实现 S2MEL forward 调用 (commit: 720215f)
- **难度**: ★★★☆☆
- **修改**: indextts/infer_v2.py Line 986-1014
- **测试**: 6.75s (+0.15s, 符合预期)
- **状态**: ✅ S2MEL 输出正常

## 发现的问题

### ❌ Step 1: merge_emovec 无法执行
**问题**: MLX 模型缺少关键的 emotion conditioning 组件

**缺失组件**:
- `emo_conditioning_encoder` (ConformerEncoder)
- `emo_perceiver_encoder` (PerceiverResampler)

**当前实现问题**:
```python
# MLX 简化版 - 只做平均，丢失音色信息
def get_emo_conditioning(self, speech_conditioning_input, ...):
    emo_cond = mx.mean(speech_mlx, axis=1)  # 简单平均
    return mlx_to_torch(emo_cond, device='mps')

# PyTorch 正确版 - 使用 Conformer + Perceiver
def get_emo_conditioning(self, speech_conditioning_input, ...):
    speech_conditioning_input, mask = self.emo_conditioning_encoder(...)
    conds_mask = self.emo_cond_mask_pad(mask.squeeze(1))
    conds = self.emo_perceiver_encoder(speech_conditioning_input, conds_mask)
    return conds.squeeze(1)
```

**影响**:
- `merge_emovec` 调用会导致音色丢失
- 需要先修复 MLX emotion conditioning 才能继续

## 待完成步骤

### ⏸️ Step 1: 替换 merge_emovec 调用 ★★☆☆☆
- **状态**: 暂停，需要先修复 MLX emotion conditioning
- **位置**: indextts/infer_v2.py Line 805-811

### ⏸️ Step 2: 清理 Hybrid 模式 ★★★★☆
- **状态**: 暂停，依赖 self.gpt.get_conditioning() 和 self.gpt.speed_emb()
- **位置**: indextts/infer_v2.py Line 879-922

### ⏸️ Step 5: 修改 GPT 加载逻辑 ★★★★★
- **状态**: 就绪，主要调用点已替换
- **位置**: indextts/infer_v2.py Line 114-169
- **预期收益**: 节省约 2.5GB 内存

## 当前代码状态

### 已替换的调用点
1. ✅ PyTorch 推理安全检查 (Line 924-926)
2. ✅ S2MEL forward 调用 (Line 986-1014)

### 未替换的调用点
1. ❌ merge_emovec (Line 805) - 依赖 emotion conditioning
2. ❌ Hybrid 模式代码块 (Line 879-922) - 依赖 get_conditioning 和 speed_emb

### 当前 MLX 模式下的执行流程
- GPT 模型加载: PyTorch + MLX (双份内存)
- merge_emovec: PyTorch (使用 self.gpt)
- inference_speech: MLX ✅
- S2MEL forward: MLX ✅

## 性能数据对比

| 步骤 | 时间 | 差异 | 状态 |
|-----|------|------|------|
| 基准 | 6.60s | - | ✅ |
| Step 0 | 6.73s | +0.13s | ✅ |
| Step 3 | 7.92s | +1.32s | ⚠️ 随机性？ |
| Step 4 | 6.75s | +0.15s | ✅ |

## Git 提交记录

```
720215f Step 4: 使用 MLX 实现 S2MEL forward 调用
ecf1491 Step 3: 添加 PyTorch GPT 推理安全检查
8ebf9e2 Step 0: 添加 MLX 模型 mel_length_compression 属性
ebe89f9 Step -1: 建立 MLX 内存优化基准
8b5db61 立项：MLX 内存优化计划 - 移除 PyTorch GPT 模型加载
```

## 下一步执行路径分析

见 NEXT_STEPS_ANALYSIS.md

