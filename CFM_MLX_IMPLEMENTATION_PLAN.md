# CFM MLX化实施计划

## 📋 项目概述

**目标**: 完整实现S2MEL CFM的MLX版本，包括GPT-fast Transformer和WaveNet final layer

**当前状态**: 架构代码已完成，需要权重加载和测试验证

**预估工作量**: 2-3天

---

## ✅ 已完成组件 (Day 1)

### 核心架构实现

1. **MLX RoPE** ✅
   - `precompute_freqs_cis_mlx`: 预计算旋转频率
   - `apply_rotary_emb_mlx`: 应用旋转embedding
   - 测试: 数学逻辑正确

2. **MLX AdaptiveLayerNorm** ✅
   - 条件LayerNorm
   - 支持timestep conditioning
   - 代码: `indextts/s2mel/modules/mlx_gpt_fast.py`

3. **MLX Attention (GPT-fast风格)** ✅
   - RoPE支持
   - Grouped Query Attention
   - Self/Cross attention
   - Mask处理

4. **MLX FeedForward (SwiGLU)** ✅
   - w2(silu(w1(x)) * w3(x))
   - 标准GPT-fast实现

5. **MLX TransformerBlock** ✅
   - AdaLN + Attention + FFN
   - U-ViT skip connections
   - time_as_token支持

6. **MLX Transformer** ✅
   - 13层完整实现
   - U-ViT架构
   - Freqs预计算

7. **MLX WaveNet** ✅
   - Dilated convolutions
   - Gated activation (tanh * sigmoid)
   - Residual/skip connections

8. **MLX TimestepEmbedder** ✅
   - 测试通过: diff < 0.0000001 ⭐
   - 完全一致

9. **MLX DiT** ✅
   - 完整forward实现
   - WaveNet final layer支持
   - CFG masking支持

10. **MLX CFM** ✅
    - Euler solver
    - CFG (Classifier-free guidance)
    - PyTorch兼容接口

---

## 🔜 待完成任务 (Day 2-3)

### Day 2: 权重加载

#### Task 1: Transformer权重映射
- [ ] 映射GPT-fast Transformer到MLX Transformer
- [ ] 处理wqkv权重分割
- [ ] AdaLN权重加载
- [ ] RMSNorm权重加载
- [ ] 预估时间: 4-6小时

#### Task 2: WaveNet权重映射
- [ ] Conv1d权重格式转换
- [ ] Conditioning layer权重
- [ ] Residual/skip layer权重
- [ ] 预估时间: 2-3小时

#### Task 3: DiT其他组件
- [ ] x_embedder, cond_projection权重
- [ ] TimestepEmbedder权重（已测试）
- [ ] Merge linear权重
- [ ] Final layer权重
- [ ] 预估时间: 2-3小时

### Day 3: 测试与集成

#### Task 4: 单元测试
- [ ] 测试单个Transformer layer
- [ ] 测试完整Transformer
- [ ] 测试DiT forward
- [ ] 测试CFM单步
- [ ] 预估时间: 4-5小时

#### Task 5: 端到端测试
- [ ] 测试完整CFM inference (5步)
- [ ] 对比PyTorch vs MLX输出
- [ ] 调试数值差异
- [ ] 预估时间: 3-4小时

#### Task 6: 性能优化
- [ ] JIT预热
- [ ] 内存优化
- [ ] Batch优化
- [ ] 预估时间: 2-3小时

#### Task 7: 集成到infer_v2.py
- [ ] 添加MLX CFM选项
- [ ] 缓存机制
- [ ] 错误处理
- [ ] 预估时间: 1-2小时

---

## 🎯 关键挑战

### 1. 权重映射复杂度 ⭐⭐⭐⭐⭐

GPT-fast使用独特的权重组织：
```python
# PyTorch: wqkv.weight = [wq, wk, wv] concatenated
# Need to split and map to MLX's separate wq, wk, wv

# Example mapping:
transformer.layers.0.attention.wqkv.weight
  → mlx_transformer.layers[0].attention.wqkv.weight
```

需要处理：
- 13层 × (attention + ffn + norms) × (weight + bias)
- 约 13 × 10 = 130+ 权重张量

### 2. WaveNet结构适配 ⭐⭐⭐⭐

WaveNet有复杂的dilated convolutions：
```python
# dilation = 2^i for i in range(n_layers)
# 需要正确处理padding和dilation
```

### 3. 数值一致性验证 ⭐⭐⭐⭐

Diffusion模型对数值精度敏感：
- 20-25步累积误差
- 每步需要高精度
- 需要仔细验证

---

## 📊 预期成果

### 性能提升（预估）

| 组件 | PyTorch | MLX (预估) | 提升 |
|------|---------|-----------|------|
| DiT forward (单步) | 0.04s | 0.03s | 1.3x |
| CFM total (25步) | 1.0s | 0.75s | 1.3x |
| S2MEL total | 1.5s | 0.8s | **1.9x** |

### 一致性目标

| 指标 | 目标 | 可接受 |
|------|------|--------|
| Max diff | < 0.01 | < 0.1 |
| Correlation | > 0.99 | > 0.95 |
| 音频质量 | 无差异 | 微小差异 |

---

## 🔧 实施策略

### 渐进式验证

1. **Layer by layer**
   - 先测试单个Transformer block
   - 再测试完整Transformer
   - 最后测试DiT

2. **Step by step**
   - 先测试1步diffusion
   - 再测试5步
   - 最后测试25步

3. **Component by component**
   - TimestepEmbedder ✅ (已测试)
   - Transformer (待测试)
   - WaveNet (待测试)
   - DiT (待测试)
   - CFM (待测试)

---

## 📁 文件结构

### 实现文件
- `indextts/s2mel/modules/mlx_gpt_fast.py` - GPT-fast Transformer ✅
- `indextts/s2mel/modules/mlx_wavenet.py` - WaveNet ✅
- `indextts/s2mel/modules/mlx_cfm.py` - DiT + CFM ✅

### 测试文件
- `test_mlx_timestep_embedder.py` - TimestepEmbedder测试 ✅
- `test_mlx_transformer.py` - Transformer测试 (待创建)
- `test_mlx_dit.py` - DiT测试 (待创建)
- `test_mlx_cfm.py` - CFM测试 (待创建)

---

## 🚦 里程碑

### Milestone 1: 基础组件 ✅ (Day 1完成)
- [x] RoPE实现
- [x] AdaLN实现
- [x] Attention实现
- [x] Transformer实现
- [x] WaveNet实现
- [x] DiT架构
- [x] CFM Euler solver

### Milestone 2: 权重加载 🔜 (Day 2)
- [ ] Transformer权重映射
- [ ] WaveNet权重映射
- [ ] DiT其他组件权重
- [ ] 完整权重加载测试

### Milestone 3: 验证测试 🔜 (Day 3)
- [ ] 单层测试
- [ ] 端到端测试
- [ ] 性能benchmark
- [ ] 音频质量验证

### Milestone 4: 集成部署 🔜 (Day 3)
- [ ] 集成到infer_v2.py
- [ ] 缓存优化
- [ ] 文档完善
- [ ] 提交推送

---

## 💡 成功标准

### 必须达到
1. ✅ 代码编译运行无错误
2. ✅ 权重正确加载
3. ✅ 生成的mel-spectrogram correlation > 0.95
4. ✅ 音频质量可接受

### 期望达到
1. ⭐ Correlation > 0.99
2. ⭐ 性能提升 > 20%
3. ⭐ 音频质量完全一致

---

**当前进度**: 架构完成 (Day 1) ✅  
**下一步**: 权重加载 (Day 2)  
**预计完成**: 2-3天  
**更新时间**: 2025-10-22

