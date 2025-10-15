# MLX 内存优化计划 - 按难度递增执行

## 目标
在启用 MLX 时，完全跳过 PyTorch GPT 模型的加载，只使用 MLX 原生实现，从而减少约 50% 的 GPT 模型内存占用。

## 难度评估

| 步骤 | 位置 | 难度 | 原因 |
|-----|------|------|------|
| Step 0 | 添加属性 | ★☆☆☆☆ | 只添加属性，零风险 |
| Step 1 | merge_emovec | ★★☆☆☆ | 简单条件判断，MLX 已实现 |
| Step 3 | PyTorch 推理检查 | ★★☆☆☆ | 只添加安全检查，不改逻辑 |
| Step 4 | S2MEL forward | ★★★☆☆ | 涉及音频生成，需仔细验证 |
| Step 2 | 清理 Hybrid | ★★★★☆ | 删除代码块，影响范围大 |
| Step 5 | 加载逻辑 | ★★★★★ | 核心修改，影响整个初始化 |

## 执行策略

**原则**:
1. 按难度从低到高执行
2. 每步完成后立即测试
3. 与基准对比，误差/差异率极低才提交 git
4. 发现问题立即分析并修复

## 测试配置

- **测试脚本**: `benchmark_v1_baseline.py`
- **测试环境**: `conda run -n indextts2 python benchmark_v1_baseline.py`
- **参考音频**: `examples/zh_vo_Main_Linaxita_2_4_24_6.wav`
- **基准文件**: `baseline_audio/gen_v1_*.wav`
- **验证标准**: 
  - 性能差异 < ±0.5s
  - 音频质量目测/听测无明显差异
  - 无报错或异常

---

## Step -1: 建立基准 ★☆☆☆☆

**目的**: 生成基准音频文件和性能数据

**操作**:
```bash
cd /Users/bailin/index-tts
conda run -n indextts2 python benchmark_v1_baseline.py
```

**生成文件**:
- `gen_v1_1.wav` - 预热（"今天天气真不错"）
- `gen_v1_2.wav` - Run 1（"到底应该吃什么"）
- `gen_v1_3.wav` - Run 2（"你为什么不愿意"）
- `gen_v1_4.wav` - Run 3（"今天天气真不错"）
- `BASELINE_V1_OPTIMIZED.md` - 性能数据

**备份**:
```bash
mkdir -p baseline_audio
cp gen_v1_*.wav baseline_audio/
cp BASELINE_V1_OPTIMIZED.md baseline_audio/
```

**提交**: 
```bash
git add baseline_audio/
git commit -m "建立 MLX 内存优化基准"
```

---

## Step 0: 添加 mel_length_compression 属性 ★☆☆☆☆

**难度**: 最简单，零风险

**文件**: `indextts/gpt/mlx_model.py` Line ~217

**修改**:
```python
# 在 UnifiedVoiceMLX.__init__() 中添加
self.mel_length_compression = kwargs.get('mel_length_compression', 1024)
```

**测试**:
```bash
conda run -n indextts2 python benchmark_v1_baseline.py
```

**验证**:
- 性能: 与基准差异 < ±0.5s
- 音频: `gen_v1_*.wav` 与 `baseline_audio/gen_v1_*.wav` 对比

**提交条件**: 
- ✅ 测试通过
- ✅ 性能无变化
- ✅ 音频无差异

**提交**:
```bash
git add indextts/gpt/mlx_model.py
git commit -m "添加 MLX 模型 mel_length_compression 属性"
```

---

## Step 1: 替换 merge_emovec 调用 ★★☆☆☆

**难度**: 简单，MLX 实现已验证

**文件**: `indextts/infer_v2.py` Line 805-811

**修改**:
```python
# 当前
emovec = self.gpt.merge_emovec(...)

# 修改为
if self.gpt_is_mlx:
    emovec = self.mlx_transformer.merge_emovec(
        spk_cond_emb, emo_cond_emb,
        torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
        torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
        alpha=emo_alpha
    )
else:
    emovec = self.gpt.merge_emovec(
        spk_cond_emb, emo_cond_emb,
        torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
        torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
        alpha=emo_alpha
    )
```

**测试**: 同 Step 0

**重点验证**: 情感表达是否正常

**提交条件**: 
- ✅ 测试通过
- ✅ 情感表达无变化
- ✅ 音频质量一致

**提交**:
```bash
git add indextts/infer_v2.py
git commit -m "使用 MLX 实现 merge_emovec"
```

---

## Step 3: 添加 PyTorch 推理安全检查 ★★☆☆☆

**难度**: 简单，只添加检查

**文件**: `indextts/infer_v2.py` Line 923-945

**修改**:
```python
else:
    # PyTorch inference
    if self.gpt is None:
        raise RuntimeError("PyTorch GPT not loaded. Cannot use PyTorch inference.")
    
    # 🔧 For debugging: force greedy decoding when num_beams=1
    use_sampling_mode = False if num_beams == 1 else do_sample
    
    codes, speech_conditioning_latent = self.gpt.inference_speech(...)
```

**测试**: 同 Step 0

**重点验证**: MLX 模式下不会进入此分支，功能正常

**提交条件**: 
- ✅ 测试通过
- ✅ 无性能/功能影响

**提交**:
```bash
git add indextts/infer_v2.py
git commit -m "添加 PyTorch GPT 推理安全检查"
```

---

## Step 4: 替换 S2MEL forward 调用 ★★★☆☆

**难度**: 中等，涉及音频生成核心

**文件**: `indextts/infer_v2.py` Line 982-993

**修改**:
```python
# 当前
latent = self.gpt(...)

# 修改为
if self.gpt_is_mlx:
    latent = self.mlx_transformer(...)
else:
    if self.gpt is None:
        raise RuntimeError("PyTorch GPT not loaded.")
    latent = self.gpt(...)
```

**测试**: 同 Step 0

**重点验证**: 
- S2MEL 生成的音频质量
- 仔细对比音频细节

**提交条件**: 
- ✅ 测试通过
- ✅ 音频质量完全一致
- ✅ 无异常噪音或失真

**提交**:
```bash
git add indextts/infer_v2.py
git commit -m "使用 MLX 实现 S2MEL forward 调用"
```

**如果出现问题**: 
- 检查 MLX `__call__` 实现
- 对比 latent shape 和数值
- 必要时回滚此步

---

## Step 2: 清理 Hybrid 模式 ★★★★☆

**难度**: 较难，删除代码块

**文件**: `indextts/infer_v2.py` Line 879-922

**操作**: 删除整个 hybrid 模式代码块

**原因**: Pure MLX 已包含所有功能

**测试**: 同 Step 0

**重点验证**: 
- 推理流程完整
- 无遗漏功能

**提交条件**: 
- ✅ 测试通过
- ✅ 功能完整
- ✅ 音频无变化

**提交**:
```bash
git add indextts/infer_v2.py
git commit -m "清理 Hybrid MLX 模式代码"
```

**如果出现问题**: 
- 检查是否有遗漏的逻辑
- 考虑保留 hybrid 模式
- 回滚此步

---

## Step 5: 修改 GPT 加载逻辑 ★★★★★

**难度**: 最难，核心优化步骤

**文件**: `indextts/infer_v2.py` Line 114-169

**修改**: MLX 模式下设置 `self.gpt = None`，不加载 PyTorch 模型

**测试**:
```bash
conda run -n indextts2 python benchmark_v1_baseline.py
```

**额外监控内存**:
```bash
# 另一个终端运行
watch -n 1 'ps aux | grep python | grep benchmark'
# 或使用 Activity Monitor
```

**重点验证**: 
1. 功能完整性
2. 音频质量一致
3. 内存减少约 2.5GB
4. 加载速度提升

**提交条件**: 
- ✅ 测试完全通过
- ✅ 音频与基准一致
- ✅ 内存确实减少
- ✅ 无性能倒退

**提交**:
```bash
git add indextts/infer_v2.py
git commit -m "优化 GPT 加载逻辑，MLX 模式下跳过 PyTorch 模型 (-2.5GB)"
```

**如果出现问题**: 
- 仔细检查错误信息
- 验证所有调用点都已替换
- 检查降级机制
- 必要时回滚

---

## Step 6: 最终完整验证 ✓

**功能测试**:
```bash
conda run -n indextts2 python benchmark_v1_baseline.py
```

**对比验证**:
1. 音频质量: 逐一对比 `gen_v1_*.wav` 与基准
2. 性能数据: 检查 `BASELINE_V1_OPTIMIZED.md`
3. 内存占用: 确认减少约 2.5GB

**降级测试**:
临时修改代码模拟 MLX 加载失败，验证降级到 PyTorch 正常

**文档记录**:
创建 `MEMORY_OPTIMIZATION_REPORT.md` 记录优化成果

**最终提交**:
```bash
git add MEMORY_OPTIMIZATION_REPORT.md
git commit -m "完成 MLX 内存优化，添加验证报告"
git push
```

---

## 预期收益

- **内存**: 节省约 2.5GB (50% GPT 内存)
- **加载速度**: 节省 3-5 秒
- **音频质量**: 完全一致（通过逐步验证）
- **代码质量**: 更清晰，每个调用点都有明确的模式选择

## 风险控制

- ✅ 按难度递增执行
- ✅ 每步独立测试和验证
- ✅ 基准对比，误差极低才提交
- ✅ 发现问题立即分析修复
- ✅ 保留降级机制

## 执行检查清单

- [ ] Step -1: 建立基准 ★☆☆☆☆ (运行测试 + 备份 + 提交)
- [ ] Step 0: 添加属性 ★☆☆☆☆ (修改 + 测试对比 + 提交)
- [ ] Step 1: merge_emovec ★★☆☆☆ (修改 + 测试对比 + 提交)
- [ ] Step 3: PyTorch检查 ★★☆☆☆ (修改 + 测试对比 + 提交)
- [ ] Step 4: S2MEL forward ★★★☆☆ (修改 + 测试对比 + 提交)
- [ ] Step 2: 清理Hybrid ★★★★☆ (删除 + 测试对比 + 提交)
- [ ] Step 5: 加载逻辑 ★★★★★ (核心修改 + 内存监控 + 提交)
- [ ] Step 6: 最终验证 ✓ (全面测试 + 文档 + 提交)

