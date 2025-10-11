# MLX IndexTTS2 快速开始

## ✅ 状态：成功运行！

MLX Hybrid模式已成功实现，性能提升18.7%，音频质量良好。

## 🚀 使用方法

### MLX Hybrid模式（推荐 - 更快）
```bash
python -m indextts.cli "你的文本" -v examples/voice_01.wav --mlx --force
```

### 纯PyTorch模式（兼容性最好）
```bash
python -m indextts.cli "你的文本" -v examples/voice_01.wav --force
```

## 📊 性能对比

| 指标 | PyTorch | MLX Hybrid | 提升 |
|------|---------|------------|------|
| 推理时间 | 8.11s | 6.59s | **18.7%** ⚡ |
| RTF | 7.59 | 5.51 | **27.4%** ⚡ |
| 音频质量 | ✅ | ✅ | 相当 |

## 🔧 技术细节

**MLX Hybrid = PyTorch Conditioning + MLX Transformer**

- ✅ 利用PyTorch的成熟Conformer/Perceiver
- ✅ 利用MLX的高效Transformer + KV cache
- ✅ 在M4上性能最优

## 🐛 已解决的问题

1. **Token重复循环** - 修复sampling实现
2. **后处理hang** - 修复MPS dtype兼容性
3. **音频无声** - 修复tensor转换

## 📝 详细文档

- `MLX_SUCCESS_REPORT.md` - 完整技术报告
- `MLX_DIAGNOSIS.md` - 问题诊断记录
- `MLX_STATUS_FINAL.md` - 开发过程记录

## 🎯 下一步

- [ ] 进一步性能优化
- [ ] 实现纯MLX Conformer（Conv1d修复）
- [ ] 端到端纯MLX推理

---

**版本**: v1.0  
**日期**: 2025-10-12  
**硬件**: Apple Silicon M4  
**状态**: ✅ Production Ready
