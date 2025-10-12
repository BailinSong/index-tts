#!/usr/bin/env python3
"""
紧急诊断：对比 PyTorch 和 MLX 的 conditioning 输出
找出为什么 MLX 生成了完全错误的音色和内容
"""

import sys
sys.path.insert(0, '/Users/bailin/index-tts')

import warnings
warnings.filterwarnings('ignore')

import torch
import mlx.core as mx
import numpy as np

print("\n" + "="*70)
print("🚨 紧急诊断：Conditioning 对比")
print("="*70)
print("\n问题: MLX 生成了女声+无意义音节 (应该是男声说'今天')")
print("目标: 找出 MLX Conditioning 哪里出错了\n")

test_text = "今天"
ref_audio = "/Users/bailin/index-tts/examples/voice_01.wav"

# ============================================================================
# Step 1: 加载模型
# ============================================================================

print("Step 1: 加载模型...")
from indextts.infer_v2 import IndexTTS2

print("  加载 PyTorch 模型...")
pt_model = IndexTTS2(model_dir="/Users/bailin/index-tts/checkpoints", use_mlx=False)

print("  加载 MLX 模型...")
mlx_model = IndexTTS2(model_dir="/Users/bailin/index-tts/checkpoints", use_mlx=True)

print("✓ 模型加载完成\n")

# ============================================================================
# Step 2: 提取 speaker embedding (相同输入)
# ============================================================================

print("="*70)
print("Step 2: 提取 speaker embedding (使用相同的参考音频)")
print("="*70)

import torchaudio

# Load reference audio
audio, sr = torchaudio.load(ref_audio)
if sr != 16000:
    audio = torchaudio.functional.resample(audio, sr, 16000)
if audio.shape[0] > 1:
    audio = audio.mean(0, keepdim=True)

print(f"\n参考音频: {ref_audio}")
print(f"  Shape: {audio.shape}, SR: 16000Hz")

# Extract features using PyTorch pipeline
with torch.no_grad():
    inputs = pt_model.extract_features(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(pt_model.device)
    attention_mask = inputs["attention_mask"].to(pt_model.device)
    
    # Get embedding
    speaker_emb = pt_model.get_emb(input_features, attention_mask)

print(f"✓ Speaker embedding 提取完成")
print(f"  Shape: {speaker_emb.shape}")
print(f"  Stats: mean={speaker_emb.mean().item():.6f}, std={speaker_emb.std().item():.6f}")
print(f"  Range: [{speaker_emb.min().item():.6f}, {speaker_emb.max().item():.6f}]")

# ============================================================================
# Step 3: PyTorch Conditioning
# ============================================================================

print("\n" + "="*70)
print("Step 3: PyTorch Conditioning (基线)")
print("="*70)

with torch.no_grad():
    # Prepare input
    cond_input = speaker_emb.transpose(1, 2)  # (B, 1024, T)
    cond_lengths = torch.tensor([cond_input.shape[-1]], device=pt_model.device)
    
    print(f"\nConditioning 输入:")
    print(f"  Shape: {cond_input.shape}")
    print(f"  Lengths: {cond_lengths}")
    
    # Get conditioning
    pt_cond = pt_model.gpt.get_conditioning(cond_input, cond_lengths)

print(f"\n✓ PyTorch Conditioning 输出:")
print(f"  Shape: {pt_cond.shape}")
print(f"  Mean: {pt_cond.mean().item():.6f}")
print(f"  Std: {pt_cond.std().item():.6f}")
print(f"  Range: [{pt_cond.min().item():.6f}, {pt_cond.max().item():.6f}]")

# Sample values
pt_cond_np = pt_cond.cpu().numpy()
print(f"\n  前 5 个 latent 的前 5 个维度:")
for i in range(min(5, pt_cond_np.shape[1])):
    print(f"    Latent {i}: [{pt_cond_np[0,i,:5]}]")

# ============================================================================
# Step 4: MLX Conditioning
# ============================================================================

print("\n" + "="*70)
print("Step 4: MLX Conditioning (待诊断)")
print("="*70)

# Convert to MLX
mlx_emb = mx.array(speaker_emb.cpu().numpy())
mlx_cond_lengths = mx.array([mlx_emb.shape[-1]])

print(f"\nMLX Conditioning 输入:")
print(f"  Shape: {mlx_emb.shape}")
print(f"  Lengths: {mlx_cond_lengths}")

# Get MLX conditioning
mlx_cond = mlx_model.mlx_transformer.get_conditioning_mlx(mlx_emb, mlx_cond_lengths)

print(f"\n✓ MLX Conditioning 输出:")
print(f"  Shape: {mlx_cond.shape}")
print(f"  Mean: {float(mlx_cond.mean()):.6f}")
print(f"  Std: {float(mlx_cond.std()):.6f}")
print(f"  Range: [{float(mlx_cond.min()):.6f}, {float(mlx_cond.max()):.6f}]")

# Sample values
mlx_cond_np = np.array(mlx_cond)
print(f"\n  前 5 个 latent 的前 5 个维度:")
for i in range(min(5, mlx_cond_np.shape[1])):
    print(f"    Latent {i}: [{mlx_cond_np[0,i,:5]}]")

# ============================================================================
# Step 5: 详细对比分析
# ============================================================================

print("\n" + "="*70)
print("Step 5: 详细对比分析")
print("="*70)

# Compute differences
abs_diff = np.abs(pt_cond_np - mlx_cond_np)
rel_diff = abs_diff / (np.abs(pt_cond_np) + 1e-8)

print(f"\n📊 数值差异:")
print(f"  Max absolute diff:  {abs_diff.max():.6f}")
print(f"  Mean absolute diff: {abs_diff.mean():.6f}")
print(f"  Median abs diff:    {np.median(abs_diff):.6f}")
print(f"  Max relative diff:  {rel_diff.max():.6f}")
print(f"  Mean relative diff: {rel_diff.mean():.6f}")

# Check for structural issues
print(f"\n🔍 结构检查:")
print(f"  PyTorch 零值比例: {(pt_cond_np == 0).sum() / pt_cond_np.size * 100:.2f}%")
print(f"  MLX 零值比例:     {(mlx_cond_np == 0).sum() / mlx_cond_np.size * 100:.2f}%")
print(f"  PyTorch NaN/Inf:  {np.isnan(pt_cond_np).any() or np.isinf(pt_cond_np).any()}")
print(f"  MLX NaN/Inf:      {np.isnan(mlx_cond_np).any() or np.isinf(mlx_cond_np).any()}")

# Correlation
flat_pt = pt_cond_np.flatten()
flat_mlx = mlx_cond_np.flatten()
correlation = np.corrcoef(flat_pt, flat_mlx)[0, 1]
print(f"  相关系数:          {correlation:.6f}")

# ============================================================================
# Step 6: 诊断结论
# ============================================================================

print("\n" + "="*70)
print("🎯 诊断结论")
print("="*70)

print(f"\n📈 差异等级:")
if abs_diff.max() < 0.1:
    level = "✅ 极小 (< 0.1)"
    severity = "EXCELLENT"
elif abs_diff.max() < 0.5:
    level = "✓ 很小 (< 0.5)"
    severity = "GOOD"
elif abs_diff.max() < 2.0:
    level = "⚠️  中等 (< 2.0)"
    severity = "MODERATE"
elif abs_diff.max() < 10.0:
    level = "❌ 较大 (< 10.0)"
    severity = "LARGE"
else:
    level = "❌ 巨大 (>= 10.0)"
    severity = "CRITICAL"

print(f"  {level}")
print(f"  Max diff: {abs_diff.max():.6f}")
print(f"  Mean diff: {abs_diff.mean():.6f}")
print(f"  Correlation: {correlation:.6f}")

print(f"\n💡 问题分析:")
if severity in ["CRITICAL", "LARGE"]:
    print(f"  ❌ MLX Conditioning 输出与 PyTorch 差异巨大!")
    print(f"  这解释了为什么生成的音色和内容完全错误。")
    print(f"\n  可能原因:")
    print(f"  1. MLX Conformer 权重加载错误")
    print(f"  2. MLX Perceiver 权重加载错误")
    print(f"  3. MLX 前向传播逻辑有严重bug")
    print(f"  4. 激活函数、归一化层实现错误")
    
elif severity == "MODERATE":
    print(f"  ⚠️  MLX Conditioning 输出与 PyTorch 有中等差异")
    print(f"  这可能导致音色和内容偏差。")
    print(f"\n  建议检查:")
    print(f"  - 权重加载的精度")
    print(f"  - 激活函数是否完全一致")
    print(f"  - LayerNorm/BatchNorm 的实现")
    
else:
    print(f"  ✓ MLX Conditioning 输出与 PyTorch 基本一致")
    print(f"  问题可能不在 Conditioning，而在:")
    print(f"  - MLX Transformer 生成逻辑")
    print(f"  - Token 采样策略")
    print(f"  - S2MEL/BigVGAN 的处理")

# Specific value comparison
print(f"\n📊 样本对比 (Latent 0, 前 10 维):")
print(f"  PyTorch: {pt_cond_np[0, 0, :10]}")
print(f"  MLX:     {mlx_cond_np[0, 0, :10]}")
print(f"  Diff:    {abs_diff[0, 0, :10]}")

print(f"\n🔧 建议修复步骤:")
if severity in ["CRITICAL", "LARGE"]:
    print(f"  1. 🔴 紧急: 使用 Hybrid Mode (PyTorch Conditioning + MLX Transformer)")
    print(f"  2. 逐层检查 MLX Conformer 的权重和前向传播")
    print(f"  3. 逐层检查 MLX Perceiver 的权重和前向传播")
    print(f"  4. 对比每一层的激活函数输出")
else:
    print(f"  1. 继续调试 MLX Transformer 生成逻辑")
    print(f"  2. 检查 token 采样和 stop token 检测")

# Save detailed comparison
np.savez('experiments/conditioning_comparison.npz',
         pytorch=pt_cond_np,
         mlx=mlx_cond_np,
         diff=abs_diff)
print(f"\n💾 详细数据已保存: experiments/conditioning_comparison.npz")

print("\n" + "="*70 + "\n")

