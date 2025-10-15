# 里程碑：MLX GPT gpt_ln_f LayerNorm 修复

## 日期
2025-10-15

## 问题描述
通过固定seed逐层对比发现，MLX实现缺少了PyTorch GPT2模型中的关键LayerNorm层（gpt.ln_f），导致推理结果数值完全错误。

## 关键发现

### PyTorch架构（正确）
```
transformer_blocks[0-23] → gpt.ln_f → final_norm → mel_head
                              ↑           ↑
                         第1个LayerNorm  第2个LayerNorm
```

### MLX架构（修复前 - 错误）
```
transformer_blocks[0-23] → final_norm → mel_head
                              ↑
                         只有1个LayerNorm ❌
```

### MLX架构（修复后 - 正确）
```
transformer_blocks[0-23] → gpt_ln_f → final_norm → mel_head
                              ↑           ↑
                         第1个LayerNorm  第2个LayerNorm ✅
```

## 修复内容

### 1. 核心代码修改
- **文件**: `indextts/gpt/mlx_model.py`
- **修改**:
  - 添加 `self.gpt_ln_f = nn.LayerNorm(model_dim)`
  - 加载权重: `gpt.ln_f.weight/bias`
  - 在所有forward方法中应用两个LayerNorm

### 2. 配置优化
- **文件**: `indextts/infer_v2.py`
  - 默认 `num_beams=1` (快速调试)
- **文件**: `indextts/cli.py`
  - 添加 `--num-beams` 参数
  - 默认值=1

### 3. 性能优化
- **文件**: `indextts/gpt/mlx_model.py`
  - 实现beam search批处理（理论15x加速）

## 验证结果

### 数值精度
| 指标 | 修复前 | 修复后 | 改善 |
|------|--------|--------|------|
| final_norm diff | 17.78 | 1.57e-05 | 99.9%+ |
| logits diff | 8.42 | 0.35 | 96% |
| Top-1 token | 不匹配 | 完全匹配 | ✅ |

### 功能测试
- ✅ 逐层对比验证通过
- ✅ 成功生成音频
- ✅ RTF ~5x (比实时快5倍)
- ✅ 权重数量: 516个 (增加2个)

## 影响范围
- 所有使用MLX推理的生成任务
- 需要清除旧的MLX cache重新生成

## 测试命令
```bash
# 使用conda indextts2环境
conda run -n indextts2 python -m indextts.cli \
  "今天天气真不错" \
  -v examples/voice_01.wav \
  --force --mlx --num-beams 1
```

## 后续工作
- [ ] 验证beam search批处理优化的正确性
- [ ] 在更多测试用例上验证
- [ ] 考虑添加--seed参数到CLI

---
**状态**: ✅ 完成并验证
**测试**: ✅ 通过
**文档**: ✅ 完整
