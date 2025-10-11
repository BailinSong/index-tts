# Step 6.1: Weight Mapping Strategy

## 🎯 发现的权重结构

### PyTorch Architecture (从checkpoint分析)

```
conditioning_encoder (Conformer)
  ├─ input: (batch, 1024, seq) - speaker embedding
  ├─ output_dim: 512
  └─ 6 encoder layers

perceiver_encoder
  ├─ input: (batch, seq, 512) - from Conformer
  ├─ proj_context: 512 → 1280
  ├─ latents: (32, 1280)
  └─ output: (batch, 32, 1280)
```

### 关键发现

**维度差异** ⚠️:
- **PyTorch Conformer**: 输出 512 维
- **MLX Conformer**: 当前实现输出 1280 维
- **不匹配！需要调整**

**解决方案**:
1. 修改 MLX Conformer 输出为 512 维
2. 在 Perceiver 前添加 proj_context (512 → 1280)
3. 保持其他架构不变

---

## 📋 权重映射表

### 1. Conformer Encoder (conditioning_encoder)

PyTorch: `conditioning_encoder.*` → MLX: `conditioning_module.conformer.*`

#### Input Projection
```
PT: conditioning_encoder.embed.conv.0.weight (512, 1, 3, 3)
    → MLX: 需要处理2D conv，可能跳过或flatten

PT: conditioning_encoder.embed.out.0.weight (512, 261632)
    → MLX: conformer.input_proj.0.weight (512, 1024)
    注意：维度不匹配，需要切片或处理
```

#### Position Encoding
```
PT: conditioning_encoder.embed.pos_enc.pe (1, 5000, 512)
    → MLX: conformer.pos_encoding (5000, 512)
    需要 squeeze 第一维
```

#### Conformer Blocks (6 layers)
对于每一层 i ∈ [0, 5]:

**Self-Attention**:
```
PT: conditioning_encoder.encoders.{i}.self_attn.linear_q.weight (512, 512)
    → MLX: conformer.blocks[{i}].attn.q_proj.weight (512, 512)
    
PT: conditioning_encoder.encoders.{i}.self_attn.linear_k.weight (512, 512)
    → MLX: conformer.blocks[{i}].attn.k_proj.weight (512, 512)
    
PT: conditioning_encoder.encoders.{i}.self_attn.linear_v.weight (512, 512)
    → MLX: conformer.blocks[{i}].attn.v_proj.weight (512, 512)
    
PT: conditioning_encoder.encoders.{i}.self_attn.linear_out.weight (512, 512)
    → MLX: conformer.blocks[{i}].attn.out_proj.weight (512, 512)

PT: conditioning_encoder.encoders.{i}.self_attn.linear_pos.weight (512, 512)
    → MLX: conformer.blocks[{i}].attn.pos_proj.weight (512, 512)

PT: conditioning_encoder.encoders.{i}.self_attn.pos_bias_u (8, 64)
    → MLX: 相对位置编码的 bias，需要存储
PT: conditioning_encoder.encoders.{i}.self_attn.pos_bias_v (8, 64)
    → MLX: 同上
```

**Feed-Forward**:
```
PT: conditioning_encoder.encoders.{i}.feed_forward.w_1.weight (2048, 512)
    → MLX: conformer.blocks[{i}].ff_macaron.1.weight (2048, 512)
    或 conformer.blocks[{i}].ff.1.weight (取决于是第一个还是第二个FF)

PT: conditioning_encoder.encoders.{i}.feed_forward.w_2.weight (512, 2048)
    → MLX: conformer.blocks[{i}].ff_macaron.3.weight 或 ff.3.weight
```

**Convolution Module**:
```
PT: conditioning_encoder.encoders.{i}.conv_module.pointwise_conv1.weight (1024, 512, 1)
    → MLX: conformer.blocks[{i}].conv.pointwise1.weight
    需要 squeeze 最后一维: (1024, 512)

PT: conditioning_encoder.encoders.{i}.conv_module.depthwise_conv.weight (512, 1, 15)
    → MLX: conformer.blocks[{i}].conv.depthwise.weight
    需要 squeeze 中间维: (512, 15)

PT: conditioning_encoder.encoders.{i}.conv_module.pointwise_conv2.weight (512, 512, 1)
    → MLX: conformer.blocks[{i}].conv.pointwise2.weight
    需要 squeeze: (512, 512)
```

**Layer Norms**:
```
PT: conditioning_encoder.encoders.{i}.norm_mha.weight (512,)
    → MLX: conformer.blocks[{i}].norm_attn.weight

PT: conditioning_encoder.encoders.{i}.norm_conv.weight (512,)
    → MLX: conformer.blocks[{i}].norm_conv.weight

PT: conditioning_encoder.encoders.{i}.norm_ff.weight (512,)
    → MLX: conformer.blocks[{i}].ff.0.weight (如果是Sequential)

PT: conditioning_encoder.encoders.{i}.norm_final.weight (512,)
    → MLX: conformer.blocks[{i}].norm_final.weight
```

#### Final Norm
```
PT: conditioning_encoder.after_norm.weight (512,)
    → MLX: conformer.norm.weight
```

---

### 2. Perceiver Resampler (perceiver_encoder)

PyTorch: `perceiver_encoder.*` → MLX: `conditioning_module.perceiver.*`

#### Latents
```
PT: perceiver_encoder.latents (32, 1280)
    → MLX: perceiver.latents (32, 1280)
    直接复制
```

#### Context Projection (重要！)
```
PT: perceiver_encoder.proj_context.weight (1280, 512)
    → MLX: perceiver.proj_context.weight (1280, 512)
    
PT: perceiver_encoder.proj_context.bias (1280,)
    → MLX: perceiver.proj_context.bias (1280,)
```

#### Perceiver Layers (2 layers)
对于每一层 i ∈ [0, 1]:

**Attention**:
```
PT: perceiver_encoder.layers.{i}.0.to_q.weight (512, 1280)
    → MLX: perceiver.layers[{i}][0].to_q.weight (512, 1280)

PT: perceiver_encoder.layers.{i}.0.to_kv.weight (1024, 1280)
    → MLX: perceiver.layers[{i}][0].to_kv.weight (1024, 1280)

PT: perceiver_encoder.layers.{i}.0.to_out.weight (1280, 512)
    → MLX: perceiver.layers[{i}][0].to_out.weight (1280, 512)
```

**Feed-Forward (GEGLU)**:
```
PT: perceiver_encoder.layers.{i}.1.0.weight (3412, 1280)
    → MLX: perceiver.layers[{i}][1].net[0].weight (3412, 1280)
    注意：3412 = 1280 * 2 * 1.333... (mult=4/3)

PT: perceiver_encoder.layers.{i}.1.2.weight (1280, 1706)
    → MLX: perceiver.layers[{i}][1].net[2].weight (1280, 1706)
    注意：1706 = 3412 / 2 (GEGLU splits)
```

#### Final Norm
```
PT: perceiver_encoder.norm.gamma (1280,)
    → MLX: perceiver.norm.scale (1280,)
    RMSNorm 使用 gamma 而不是 weight
```

---

### 3. Emotion Perceiver (emo_perceiver_encoder)

类似 perceiver_encoder，但维度不同：
- input_dim: 512
- model_dim: 1024
- latents: (1, 1024) - 只有1个latent而不是32个

映射策略类似，但映射到 MLX 的 emo_conditioning_module (如果实现)

---

## 🔧 实现步骤

### Step 6.2: 调整 MLX 架构

**需要修改的文件**: `indextts/gpt/mlx_conditioning.py`

1. **修改 MLXConformerEncoder**:
   ```python
   def __init__(
       self,
       input_dim: int = 1024,
       output_dim: int = 512,  # 改为512！
       num_layers: int = 6,
       ...
   )
   ```

2. **确保 MLXPerceiverResampler 有 proj_context**:
   ```python
   if dim_context != dim:
       self.proj_context = nn.Linear(dim_context, dim)
   ```

3. **更新 MLXConditioningModule**:
   ```python
   self.conformer = MLXConformerEncoder(
       input_dim=1024,
       output_dim=512,  # 匹配 PyTorch
       num_layers=6
   )
   
   self.perceiver = MLXPerceiverResampler(
       dim=1280,
       dim_context=512,  # 来自 Conformer
       num_latents=32
   )
   ```

### Step 6.3: 实现权重加载

**需要修改的文件**: `indextts/gpt/mlx_model.py`

扩展 `load_weights_from_dict` 方法：

```python
def load_conditioning_weights(self, weights):
    """Load Conformer and Perceiver weights from PyTorch checkpoint"""
    
    if not self.use_mlx_conditioning:
        return 0
    
    loaded = 0
    
    # Load Conformer weights
    loaded += self._load_conformer_weights(weights)
    
    # Load Perceiver weights
    loaded += self._load_perceiver_weights(weights)
    
    return loaded
```

---

## 📊 预期结果

完成后:
- ✅ MLX Conformer 维度: 1024 → 512
- ✅ MLX Perceiver 维度: 512 → 1280 
- ✅ 匹配 PyTorch 架构
- ✅ 可以加载 PyTorch 权重
- ✅ 数值输出应该与 PyTorch 接近

---

## ⚠️ 注意事项

### 1. 权重转置
- PyTorch Linear: `weight.shape = (out, in)`
- MLX Linear: `weight.shape = (out, in)` - 相同！
- 但是需要检查每个层的实际使用

### 2. Conv1d 权重处理
PyTorch conv weights 有额外的维度 (kernel_size 维度)，需要 squeeze：
```python
# PyTorch: (out, in, kernel)
# MLX: (out, kernel) for depthwise
weight_mlx = weight_pt.squeeze(1)  # Remove middle dim
```

### 3. Pointwise Conv (kernel=1)
```python
# PyTorch: (out, in, 1)
# MLX: (out, in) - Linear
weight_mlx = weight_pt.squeeze(-1)  # Remove last dim
```

### 4. RMSNorm vs LayerNorm
- PyTorch 使用 gamma (可能)
- MLX RMSNorm 使用 scale
- 需要映射 `gamma → scale`

---

## 🎯 下一步

1. ✅ **Step 6.1 完成**: 权重结构分析完成
2. → **Step 6.2**: 调整 MLX 架构匹配 PyTorch
3. → **Step 6.3**: 实现权重加载
4. → **Step 6.4**: 验证数值匹配
5. → **Step 6.5**: 端到端测试

---

*生成时间: 2025-10-11*  
*文件: experiments/step6_1_weight_mapping.md*

