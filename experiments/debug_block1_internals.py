#!/usr/bin/env python3
"""
深入检查 Conformer Block 1 的内部组件
找出导致 correlation 下降的具体子模块
"""

import sys
sys.path.insert(0, '/Users/bailin/index-tts')

import warnings
warnings.filterwarnings('ignore')

import torch
import mlx.core as mx
import numpy as np
import torchaudio

print("\n" + "="*70)
print("🔬 Block 1 内部组件详细诊断")
print("="*70)

# Load models
print("\n加载模型...")
from indextts.infer_v2 import IndexTTS2
from indextts.gpt.mlx_model import UnifiedVoiceMLX
from indextts.utils.mlx_cache import MLXModelCache
from omegaconf import OmegaConf

pt_model = IndexTTS2(model_dir="/Users/bailin/index-tts/checkpoints", use_mlx=False)

cfg = OmegaConf.load("/Users/bailin/index-tts/checkpoints/config.yaml")
mlx_cache = MLXModelCache(cache_dir="/Users/bailin/index-tts/checkpoints/mlx")
mlx_gpt_weights = mlx_cache.get_or_convert("gpt", "/Users/bailin/index-tts/checkpoints/gpt.pth")

mlx_transformer = UnifiedVoiceMLX(
    use_mlx_conditioning=True,
    **cfg.gpt
)
mlx_transformer.load_weights_from_dict(mlx_gpt_weights)
print("✓ Models loaded")

# Helper functions
def make_pad_mask(lengths, maxlen=None):
    batch_size = lengths.size(0)
    if maxlen is None:
        maxlen = lengths.max()
    seq_range = torch.arange(0, maxlen, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, maxlen)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask

def compare(pt_tensor, mlx_array, name):
    pt_np = pt_tensor.cpu().numpy()
    mlx_np = np.array(mlx_array)
    diff = np.abs(pt_np - mlx_np)
    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    print(f"{name:30s}: corr={corr:.6f}, max_diff={diff.max():.6f}, "
          f"PT_std={pt_np.std():.4f}, MLX_std={mlx_np.std():.4f}")
    return corr

# Get speaker embedding and prepare inputs
print("\n准备输入...")
ref_audio = "/Users/bailin/index-tts/examples/voice_01.wav"
audio, sr = torchaudio.load(ref_audio)
if sr != 16000:
    audio = torchaudio.functional.resample(audio, sr, 16000)
if audio.shape[0] > 1:
    audio = audio.mean(0, keepdim=True)

with torch.no_grad():
    inputs = pt_model.extract_features(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(pt_model.device)
    attention_mask = inputs["attention_mask"].to(pt_model.device)
    speaker_emb = pt_model.get_emb(input_features, attention_mask)

# ============================================================================
# PyTorch: Get input to Block 1
# ============================================================================

pt_conformer = pt_model.gpt.conditioning_encoder
xs = speaker_emb
cond_lengths = torch.tensor([xs.shape[1]], device=pt_model.device)

with torch.no_grad():
    masks = (~make_pad_mask(cond_lengths, xs.size(1))).to(xs.device).unsqueeze(1)
    xs, pos_emb, masks = pt_conformer.embed(xs, masks)
    mask_pad = masks
    
    # Block 0 forward
    xs, masks, _, _ = pt_conformer.encoders[0](xs, masks, pos_emb, mask_pad)
    
    # Now xs is the input to Block 1
    pt_block1_input = xs.clone()
    print(f"\n✓ PyTorch Block 1 输入: {pt_block1_input.shape}")

# ============================================================================
# MLX: Get input to Block 1
# ============================================================================

mlx_conformer = mlx_transformer.conditioning_module.conformer
mlx_emb = mx.array(speaker_emb.cpu().numpy())

x = mlx_emb
x = mlx_conformer.subsampling(x)
seq_len = x.shape[1]
pos_emb_mlx = mlx_conformer.pos_encoding[:seq_len]
pos_emb_mlx = mx.broadcast_to(
    pos_emb_mlx.reshape(1, seq_len, mlx_conformer.output_dim),
    (x.shape[0], seq_len, mlx_conformer.output_dim)
)
x = x * mlx_conformer.xscale + pos_emb_mlx

# Block 0 forward
x = mlx_conformer.blocks[0](x, pos_emb_mlx, mask=None, mask_pad=None)

mlx_block1_input = x
print(f"✓ MLX Block 1 输入: {mlx_block1_input.shape}")

# Compare inputs
print("\n" + "="*70)
print("对比 Block 1 输入")
print("="*70)
input_corr = compare(pt_block1_input, mlx_block1_input, "Block 1 Input")

# ============================================================================
# PyTorch Block 1: Forward with full block
# ============================================================================

print("\n" + "="*70)
print("PyTorch Block 1 前向传播 (完整)")
print("="*70)

pt_block = pt_conformer.encoders[1]
xs = pt_block1_input.clone()

with torch.no_grad():
    # Use full block forward
    xs, masks, _, _ = pt_block(xs, masks, pos_emb, mask_pad)
    pt_block1_output = xs.clone()

print(f"✓ PyTorch Block 1 完成: {pt_block1_output.shape}")

# We can't get intermediate outputs easily from PyTorch block
# So we'll just compare final outputs
pt_after_ff1 = None
pt_after_attn = None  
pt_after_conv = None
pt_after_ff2 = None

# ============================================================================
# MLX Block 1: Detailed forward
# ============================================================================

print("\n" + "="*70)
print("MLX Block 1 详细前向传播")
print("="*70)

mlx_block = mlx_conformer.blocks[1]
x = mlx_block1_input

# MLX Conformer Block structure:
# 1. ff_macaron (first half)
# 2. attn (self-attention)
# 3. conv (convolution module)
# 4. ff (second half)
# 5. norm_final

# 1. FF macaron (first half)
residual = x
x = residual + 0.5 * mlx_block.ff_macaron(x)
mlx_after_ff1 = x

# 2. Self-attention
residual = x
x_norm = mlx_block.norm_attn(x)
x_att = mlx_block.attn(x_norm, pos_emb_mlx)
x = residual + x_att
mlx_after_attn = x

# 3. Convolution
residual = x
x_norm = mlx_block.norm_conv(x)
x_conv = mlx_block.conv(x_norm, mask_pad=None)
x = residual + x_conv
mlx_after_conv = x

# 4. FF (second half)
residual = x
x = residual + 0.5 * mlx_block.ff(x)
mlx_after_ff2 = x

# 5. Final norm
x = mlx_block.norm_final(x)
mlx_block1_output = x

print(f"✓ MLX Block 1 各步骤完成")

# ============================================================================
# Comparison
# ============================================================================

print("\n" + "="*70)
print("📊 对比分析")
print("="*70)

print("\nMLX Block 1 各步骤统计:")
print(f"  Input:         mean={float(mlx_block1_input.mean()):.6f}, std={float(mlx_block1_input.std()):.6f}")
print(f"  After FF1:     mean={float(mlx_after_ff1.mean()):.6f}, std={float(mlx_after_ff1.std()):.6f}")
print(f"  After Attn:    mean={float(mlx_after_attn.mean()):.6f}, std={float(mlx_after_attn.std()):.6f}")
print(f"  After Conv:    mean={float(mlx_after_conv.mean()):.6f}, std={float(mlx_after_conv.std()):.6f}")
print(f"  After FF2:     mean={float(mlx_after_ff2.mean()):.6f}, std={float(mlx_after_ff2.std()):.6f}")
print(f"  Output:        mean={float(mlx_block1_output.mean()):.6f}, std={float(mlx_block1_output.std()):.6f}")

print("\n最终输出对比:")
corr_output = compare(pt_block1_output, mlx_block1_output, "Block 1 Output")

# ============================================================================
# Analysis
# ============================================================================

print("\n" + "="*70)
print("🎯 分析结论")
print("="*70)

print(f"\nCorrelation 变化:")
print(f"  Input:  {input_corr:.6f}")
print(f"  Output: {corr_output:.6f}")
print(f"  Delta:  {corr_output - input_corr:+.6f}")

if corr_output < 0.85:
    print(f"\n❌ Block 1 存在显著问题 (correlation 从 {input_corr:.4f} → {corr_output:.4f})")
    print("   可能的原因:")
    print("   1. Self-Attention 的 relative positional encoding 实现错误")
    print("   2. Convolution Module 的 GLU 或 Swish 激活函数错误")
    print("   3. Feed-Forward 的权重或激活函数错误")
    print("   4. LayerNorm 的实现差异")
else:
    print(f"\n✓ Block 1 基本正常")

print("\n" + "="*70 + "\n")

