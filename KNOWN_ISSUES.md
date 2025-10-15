# Known Issues

## ⚠️ MLX Conditioning在某些Voice上数值不稳定

### 症状
MLX在不同voice上的生成长度差异很大：

| Voice | PyTorch | MLX | 差异 | 状态 |
|---|---|---|---|---|
| voice_01.wav | 2.35s | 3.07s | +31% | ✅ 可接受 |
| zh_vo_Main_Linaxita_2_4_24_6.wav | 2.61s | 5.99s | +130% | ❌ 问题 |

### 测试方法
```bash
# Voice 01 (正常)
python -m indextts.cli "今天天气真不错" -v examples/voice_01.wav --mlx

# Linaxita (异常)
python -m indextts.cli "今天天气真不错" -v examples/zh_vo_Main_Linaxita_2_4_24_6.wav --mlx
```

### 已验证的一致性
使用**fake input**（all 1s）测试时，PyTorch和MLX完全一致：
- ✅ 所有24层Transformer: max_diff < 0.00012
- ✅ Final Norm: max_diff < 0.000005
- ✅ Mel Head Logits: 完全相同
- ✅ Top-10 Tokens: 完全相同

### 可能的原因
1. **Conformer数值稳定性**
   - 某些audio特征范围导致计算差异
   - Relative Position Attention在极端值下的行为

2. **Perceiver Resampler**
   - Cross-attention计算可能有细微差异
   - Query/Key/Value的数值范围问题

3. **数据类型和精度**
   - MLX和PyTorch的float32精度差异
   - 累积误差在某些case下被放大

### Workaround
当前建议：
- ✅ 优先使用voice_01.wav等表现良好的voice
- ⚠️ 对于问题voice，考虑使用PyTorch模式
- 🔧 等待MLX conditioning数值稳定性修复

### 调试进展
- [x] 验证了fake input下完全一致
- [x] 发现真实audio conditioning有差异
- [ ] 定位具体是Conformer还是Perceiver的问题
- [ ] 修复数值稳定性问题

### 相关文件
- `indextts/gpt/mlx_conditioning.py` - Conformer + Perceiver实现
- `compare_voices.py` - Voice对比测试脚本
- `FINAL_FIX_SUMMARY.md` - 已修复的问题总结

---

## ✅ 已修复的问题

### 1. Conformer Position Encoding (已修复)
- Bug: `x = x * xscale + pos_emb`
- Fix: `x = x * xscale`
- Status: ✅ 完全一致

### 2. Convolution Module顺序 (已修复)
- Bug: Norm在开头
- Fix: Norm在Depthwise之后
- Status: ✅ 完全一致

### 3. Causal Mask实现 (已修复)
- Bug: 使用加法mask
- Fix: 使用`mx.where()`
- Status: ✅ 完全一致

### 4. Final Norm权重加载 (已修复) ⭐ 最重要
- Bug: 加载了错误的`final_norm.weight`
- Fix: 使用`gpt.ln_f.weight`
- Status: ✅ 完全一致
- Impact: 解决了fake input下的所有问题

---

更新时间: 2025-01-14




