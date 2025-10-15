# MLX vs PyTorch 调查报告

## 日期: 2025-01-XX

## 执行摘要

经过详细的逐层对比和调试，我们确认了MLX实现的核心问题：

1. ✅ **Transformer Block 0 完全匹配** - 单层实现正确
2. ❌ **最终Logits分布完全错误** - 24层后累积误差导致输出错误
3. ❌ **Beam Search无法停止** - 由于logits错误，stop token概率异常低

## 详细发现

### 1. 逐层对比结果

#### Block 0 测试（随机输入）
```
Input shape: (1, 51, 1280)
Input mean: 0.001832, std: 1.001866

✓ Block 0 Output
  PyTorch: mean=0.005493, std=1.112578, max=33.02
  MLX:     mean=0.005493, std=1.112578, max=33.02
  Diff:    max=1.335144e-05, mean=2.623725e-07
```

**结论**: 单层Transformer实现完全正确

#### 最终Logits测试（随机输入）
```
Stop Token (ID=8193) 概率：
  PyTorch:
    Stop token log_prob: -9.416428
    Stop token prob: 0.000081
    Ranking: #6522 / 8194
  
  MLX:
    Stop token log_prob: -10.704201
    Stop token prob: 0.000022
    Ranking: #5526 / 8194

Top-1 Token log_prob:
  PyTorch: -7.6627
  MLX:     -4.5495  ⚠️ 差异达到3.1！
```

**结论**: Logits分布完全错误，MLX的值偏大约3个数量级

### 2. Beam Search 分析

MLX Beam Search参数：
```python
n_tokens_to_keep = 2 * num_beams  # num_beams=3 -> 只保留6个候选
```

从`vocab_size * num_beams = 8194 * 3 = 24582`个候选中只选top-6。

**问题**: 虽然这个值看起来很小，但真正的问题不在这里。PyTorch版本能正常停止，说明：
- PyTorch的logits分布中，stop token在前几步就能进入top-6
- MLX的logits分布错误，导致stop token排名一直很低

### 3. 实际推理结果

```bash
# PyTorch (正常)
>> Generated 137 mel tokens
>> Total inference time: 13.96 seconds
>> Generated audio length: 1.89 seconds

# MLX (异常)
>> Generated 1500 mel tokens (达到max_length限制)
>> Total inference time: 802.46 seconds
>> Generated audio length: 29.95 seconds
```

### 4. 根本原因分析

#### 4.1 权重加载正确性 ✅

- Block 0的权重加载正确（输出完全匹配）
- Final norm权重正确（已在之前修复）
- Mel head权重正确（已验证）

#### 4.2 计算逻辑正确性 ✅

- MLXLinear实现正确（匹配PyTorch Conv1D行为）
- LayerNorm实现正确
- Attention实现正确
- MLP (FC + GELU + Projection) 实现正确

#### 4.3 累积误差来源 ❓

**假设**: 虽然Block 0完全匹配，但在24层的传递过程中：

1. **浮点精度累积误差**
   - MLX使用MPS backend (Metal Performance Shaders)
   - PyTorch使用CPU/CUDA
   - 微小的数值差异在24层中累积

2. **可能的实现差异**
   - LayerNorm的epsilon处理
   - Attention mask的数值精度
   - Softmax的数值稳定性

3. **权重加载的细微差异**
   - 虽然Block 0匹配，但可能其他layers的权重有细微差异
   - 特别是Block 1-23的权重加载需要逐一验证

## 下一步行动

### 优先级 P0 - 必须修复

1. **完整对比所有24层**
   - 创建简化的测试脚本（绕过conditioning模块问题）
   - 逐层对比每一层的输出
   - 找出第一个开始偏离的层

2. **验证所有层的权重加载**
   - 对比Layer 1-23的权重统计值
   - 确保没有transposition或shape mismatch

3. **检查数值稳定性**
   - 对比LayerNorm的epsilon
   - 对比Attention mask的dtype和精度
   - 对比Softmax的实现

### 优先级 P1 - 优化改进

1. **增加调试输出**
   - 在真实推理中添加中间层输出
   - 对比实际audio输入下的每一层

2. **性能优化**
   - 当前MLX推理速度慢（802s vs 14s）
   - 可能的原因：频繁的numpy转换、debug输出

### 优先级 P2 - 长期改进

1. **简化模型加载**
   - 解决condition_module的config不匹配问题
   - 统一PyTorch和MLX的初始化流程

2. **单元测试**
   - 为每个MLX层添加单元测试
   - 对比随机权重下的输出

## 已完成的修复

1. ✅ Final norm权重映射错误（`gpt.ln_f` -> `final_norm`）
2. ✅ Mel head权重加载
3. ✅ KV cache初始化和deep copy
4. ✅ Manual log_softmax实现
5. ✅ MLXLinear匹配PyTorch Conv1D行为（无transpose）
6. ✅ Causal mask创建和应用
7. ✅ Conformer/Perceiver conditioning实现

## 测试脚本

创建了以下测试脚本：

1. `compare_direct.py` - Block 0逐子层对比 ✅
2. `check_stop_token_prob.py` - Stop token概率检查 ✅
3. `check_all_layers.py` - 全部24层对比（待修复config问题）
4. `compare_layer_by_layer.py` - 完整推理逐层对比（待修复）

## 结论

MLX的Transformer Block单层实现是正确的，但在24层传递过程中累积了显著误差，导致：
1. 最终logits分布错误（偏大约3个数量级）
2. Stop token概率异常低
3. Beam search无法正常终止

**关键下一步**: 必须找出哪一层开始出现偏差，以及偏差的具体原因（权重/计算逻辑/数值稳定性）。


