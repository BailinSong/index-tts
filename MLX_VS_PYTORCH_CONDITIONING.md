# MLX vs PyTorch Conditioning 实现逻辑对比

## 📊 总体流程对比

### PyTorch 流程

```
音频输入 (16kHz)
  ↓
W2V-BERT特征提取 (get_emb)
  ↓
Semantic Codec Quantize
  ↓
spk_cond_emb (语义特征) [B, 1024, T]
  ↓
直接传入 GPT.inference_speech() 
  ↓
GPT内部调用 conditioning_encoder + perceiver_encoder
  ↓
speech_conditioning_latent [B, 32, 1280]
```

### MLX 流程

```
音频输入 (16kHz)
  ↓
W2V-BERT特征提取 (PyTorch get_emb - 共享)
  ↓
Semantic Codec Quantize (PyTorch - 共享)
  ↓
spk_cond_emb (语义特征) [B, 1024, T]
  ↓
转换为MLX array: torch_to_mlx()
  ↓
Transpose: (B, 1024, T) -> (B, T, 1024)
  ↓
MLX Conformer Encoder
  ↓
MLX Perceiver Encoder
  ↓
speech_conditioning_latent_mlx [B, 32, 1280]
```

---

## 🔍 关键差异点

### 1. 输入格式差异

#### PyTorch
```python
# indextts/gpt/model_v2.py
# Conformer expects: (B, T, D)
semantic_features_permuted = spk_cond_emb.transpose(1, 2)  # (B, 1024, T) -> (B, T, 1024)
conformer_out, _ = self.conditioning_encoder(semantic_features_permuted, lengths)
```

#### MLX
```python
# indextts/gpt/mlx_model.py
# 需要手动transpose
speech_condition_mlx = torch_to_mlx(speech_condition.cpu())
if speech_condition_mlx.shape[1] == 1024:
    speech_condition_mlx = speech_condition_mlx.transpose(0, 2, 1)  # (B, 1024, T) -> (B, T, 1024)

conformer_out_mlx, _ = self.conditioning_encoder(speech_condition_mlx, lengths_mlx)
```

**风险**: Transpose操作可能有误，或者维度判断逻辑不对

---

### 2. Conformer实现差异

#### PyTorch Conformer
```python
# 使用 ESPnet 的 ConformerEncoder
class ConformerEncoder(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size=512,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        ...
    ):
        # 完整的PyTorch实现
        # - MultiHeadedAttention
        - Convolution module
        - Feed-forward module
        - Layer normalization
```

#### MLX Conformer
```python
# indextts/gpt/conformer/*.py
# 需要手动实现MLX版本的:
# - mx.nn.MultiHeadAttention
# - Conv1D (使用mx.nn.Conv1d)
# - Layer normalization (mx.nn.LayerNorm)
# - Swish activation (mx.nn.SiLU)
```

**风险**: 
1. MLX的attention实现可能与PyTorch不完全一致
2. Convolution的padding处理可能不同
3. Layer norm的epsilon可能不同
4. Dropout在inference时的行为

---

### 3. Perceiver实现差异

#### PyTorch Perceiver
```python
# PerceiverResampler
class PerceiverResampler(nn.Module):
    def __init__(
        self,
        dim,
        dim_context=None,
        num_latents=64,
        heads=8,
        ...
    ):
        # Learnable latent queries
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        
        # Cross-attention layers
        self.layers = nn.ModuleList([
            PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads, ...),
            FeedForward(dim=dim, mult=ff_mult),
        ])
```

#### MLX Perceiver
```python
# MLX版本需要实现:
# - mx.nn.Parameter for latents
# - Cross-attention (query from latents, key/value from conformer output)
# - Feed-forward network
```

**风险**:
1. Latent的初始化值可能不同
2. Cross-attention的实现细节可能不同
3. Masking处理可能不同

---

### 4. 数据流差异

#### PyTorch
```python
# indextts/infer_v2.py (Line 775)
codes, speech_conditioning_latent = self.gpt.inference_speech(
    speech_conditioning_latent=None,  # PyTorch内部会计算
    text_inputs=text_tokens,
    ...
)

# 在gpt内部:
# 1. conditioning_encoder处理输入
# 2. perceiver_encoder生成latents
# 3. 组合: [latents] + [text] + [mel]
# 4. Transformer处理
```

#### MLX
```python
# indextts/gpt/mlx_model.py (Line 1660)
speech_conditioning_latent_mlx = self.get_conditioning_mlx(
    speech_condition_mlx, 
    cond_lengths_mlx
)

# 手动组合:
speech_cond_with_emo = speech_conditioning_latent_mlx + emo_vec_mlx
conds_mlx = mx.concatenate([
    speech_cond_with_emo,
    duration_emb_half_mlx,
    duration_emb_mlx
], axis=1)

# 然后:
# context = mx.concatenate([conds_mlx, text_emb], axis=1)
# sequence = mx.concatenate([context, start_mel_emb], axis=1)
```

**风险**: 组合顺序或维度可能不一致

---

## 🎯 重点检查项

### 检查点1: Transpose是否正确

```python
# PyTorch输入: (B, 1024, T)
# Conformer expects: (B, T, D=1024)

# PyTorch做法:
semantic_features.transpose(1, 2)  # (B, 1024, T) -> (B, T, 1024) ✅

# MLX做法:
speech_condition_mlx.transpose(0, 2, 1)  # (B, 1024, T) -> (B, T, 1024) ✅
```

**结论**: Transpose逻辑一致

---

### 检查点2: Conformer权重加载

```python
# 需要验证的权重:
conditioning_encoder.encoder.*.self_attn.linear_q.weight
conditioning_encoder.encoder.*.self_attn.linear_k.weight
conditioning_encoder.encoder.*.self_attn.linear_v.weight
conditioning_encoder.encoder.*.self_attn.linear_out.weight
conditioning_encoder.encoder.*.conv_module.pointwise_conv1.weight
conditioning_encoder.encoder.*.conv_module.depthwise_conv.weight
conditioning_encoder.encoder.*.feed_forward.w_1.weight
conditioning_encoder.encoder.*.feed_forward.w_2.weight
conditioning_encoder.encoder.*.norm_*.weight
conditioning_encoder.encoder.*.norm_*.bias
```

**检查方法**:
```python
# 对比权重
torch_weight = torch_model.conditioning_encoder.encoder[0].self_attn.linear_q.weight
mlx_weight = mlx_model.conditioning_encoder.encoder[0].self_attn.linear_q.weight

print(f"Shape: {torch_weight.shape} vs {mlx_weight.shape}")
print(f"Max diff: {np.abs(torch_weight.numpy() - mlx_weight.numpy()).max()}")
```

---

### 检查点3: Perceiver权重加载

```python
# 需要验证的权重:
perceiver_encoder.latents                    # (32, 1280)
perceiver_encoder.layers.*.0.to_q.weight    # Cross-attention Q
perceiver_encoder.layers.*.0.to_kv.weight   # Cross-attention KV
perceiver_encoder.layers.*.0.to_out.weight  # Cross-attention output
perceiver_encoder.layers.*.1.*.weight       # Feed-forward
```

**检查方法**:
```python
# 对比latents初始化
torch_latents = torch_model.perceiver_encoder.latents
mlx_latents = mlx_model.perceiver_encoder.latents

print(f"Latents shape: {torch_latents.shape} vs {mlx_latents.shape}")
print(f"Max diff: {np.abs(torch_latents.numpy() - mlx_latents.numpy()).max()}")
```

---

### 检查点4: Forward实现细节

#### Conformer Attention

**PyTorch** (使用ESPnet MultiHeadedAttention):
```python
# espnet/nets/pytorch_backend/transformer/attention.py
q = self.linear_q(query)
k = self.linear_k(key)
v = self.linear_v(value)

scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)
attn = torch.softmax(scores, dim=-1)
attn = self.dropout(attn)
output = torch.matmul(attn, v)
```

**MLX** (需要实现相同逻辑):
```python
q = self.linear_q(query)
k = self.linear_k(key)
v = self.linear_v(value)

scores = mx.matmul(q, k.transpose(0, 2, 1)) / math.sqrt(d_k)
if mask is not None:
    scores = mx.where(mask == 0, -1e9, scores)  # 注意: mx.where的参数顺序!
attn = mx.softmax(scores, axis=-1)
# Dropout在inference时应该关闭
output = mx.matmul(attn, v)
```

**风险**: `mx.where` vs `torch.masked_fill` 的行为可能不同

---

## 🐛 可能的Bug

### Bug 1: Mask处理不一致

```python
# PyTorch
scores.masked_fill(mask == 0, -1e9)  # 将mask为0的位置填充为-1e9

# MLX (错误写法)
mx.where(mask == 0, scores, -1e9)  # 参数顺序错误!

# MLX (正确写法)
mx.where(mask == 0, -1e9, scores)  # condition, true_value, false_value
```

---

### Bug 2: Layer Norm的epsilon

```python
# PyTorch默认epsilon=1e-5
nn.LayerNorm(dim, eps=1e-5)

# MLX默认epsilon可能不同
mx.nn.LayerNorm(dim, eps=???)  # 需要明确设置为1e-5
```

---

### Bug 3: Dropout在inference时未关闭

```python
# PyTorch
model.eval()  # 自动关闭dropout

# MLX
# 需要确保在inference时不应用dropout
# 或者dropout rate设置为0.0
```

---

### Bug 4: 维度顺序混淆

```python
# PyTorch: (B, T, D)
# MLX可能误用: (B, D, T)

# 关键检查点:
# 1. Conformer输入/输出
# 2. Perceiver输入/输出
# 3. 组合conditioning时的维度
```

---

## 📋 调试步骤

### Step 1: 验证权重加载

```python
python compare_weights_conditioning.py
# 输出每一层的权重差异
```

### Step 2: 验证Conformer输出

```python
python compare_conformer_output.py
# 用相同输入对比Conformer的输出
```

### Step 3: 验证Perceiver输出

```python
python compare_perceiver_output.py
# 用Conformer输出对比Perceiver的输出
```

### Step 4: 验证最终conditioning

```python
python compare_final_conditioning.py
# 对比最终的32个latents
```

---

## ✅ 修复建议

### 优先级1: 检查Conformer权重

```bash
# 创建权重对比脚本
cd /Users/bailin/index-tts
python -c "
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.gpt.mlx_model import UnifiedVoiceMLX
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.mlx_cache import MLXModelCache
from omegaconf import OmegaConf
import numpy as np

cfg = OmegaConf.load('checkpoints/config.yaml')

# Load PyTorch
torch_model = UnifiedVoice(**cfg.gpt)
load_checkpoint(torch_model, 'checkpoints/gpt.pth')

# Load MLX
mlx_cache = MLXModelCache(cache_dir='./cache/mlx')
mlx_weights = mlx_cache.get_or_convert('gpt', 'checkpoints/gpt.pth')
mlx_model = UnifiedVoiceMLX(**cfg.gpt)
mlx_model.load_weights_from_dict(mlx_weights)

# 对比第一层的Q权重
torch_q = torch_model.conditioning_encoder.encoders[0].self_attn.linear_q.weight.detach().cpu().numpy()
mlx_q = np.array(mlx_model.conditioning_encoder.encoders[0].self_attn.linear_q.weight)

print(f'Shape: {torch_q.shape} vs {mlx_q.shape}')
print(f'Max diff: {np.abs(torch_q - mlx_q).max()}')
"
```

### 优先级2: 检查维度处理

在MLX的`get_conditioning_mlx`中添加详细的shape打印:

```python
def get_conditioning_mlx(self, speech_features, lengths=None):
    print(f"[DEBUG] Input shape: {speech_features.shape}")
    
    # Transpose检查
    if speech_features.shape[-1] != 1024:
        speech_features = speech_features.transpose(0, 2, 1)
        print(f"[DEBUG] After transpose: {speech_features.shape}")
    
    # Conformer
    conformer_out, _ = self.conditioning_encoder(speech_features, lengths)
    print(f"[DEBUG] Conformer output: {conformer_out.shape}")
    
    # Perceiver
    latents = self.perceiver_encoder(conformer_out)
    print(f"[DEBUG] Perceiver output: {latents.shape}")
    
    return latents
```

### 优先级3: 数值精度检查

使用float64进行测试，排除精度问题:

```python
# 在MLX中临时使用更高精度
speech_features_mlx = torch_to_mlx(speech_features.cpu()).astype(mx.float64)
```

---

## 🎯 总结

**核心差异**:
1. ✅ 输入格式处理 - 已验证一致
2. ❓ Conformer权重加载 - **需要验证**
3. ❓ Perceiver权重加载 - **需要验证**
4. ❓ Forward实现细节 - **需要验证**
5. ❓ Mask处理 - **可能有差异**

**修复路径**:
```
检查权重加载 → 检查forward实现 → 检查数值精度 → 完成修复
```

**预期结果**:
修复后，MLX和PyTorch在相同输入下应该产生几乎一致的conditioning latents (误差 < 0.01)。






