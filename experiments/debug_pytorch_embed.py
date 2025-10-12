#!/usr/bin/env python3
"""
详细检查 PyTorch Conformer embed 的每一步
"""

import sys
sys.path.insert(0, '/Users/bailin/index-tts')

import torch
import torchaudio

print("加载模型...")
from indextts.infer_v2 import IndexTTS2
pt_model = IndexTTS2(model_dir="/Users/bailin/index-tts/checkpoints", use_mlx=False)

print("\n提取 speaker embedding...")
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

print(f"✓ Speaker embedding: {speaker_emb.shape}")

# Test PyTorch embed step by step
conformer = pt_model.gpt.conditioning_encoder
xs = speaker_emb  # (1, 121, 1024) - already in correct format!

print(f"\nInput to embed (time, freq): {xs.shape}")
print(f"  mean={xs.mean():.6f}, std={xs.std():.6f}")

# Go into embed
with torch.no_grad():
    # Step 1: unsqueeze
    x = xs.unsqueeze(1)  # (b, c=1, t, f)
    print(f"\nAfter unsqueeze: {x.shape}")
    print(f"  mean={x.mean():.6f}, std={x.std():.6f}")
    
    # Step 2: Conv2d
    x = conformer.embed.conv(x)
    print(f"\nAfter Conv2d: {x.shape}")
    print(f"  mean={x.mean():.6f}, std={x.std():.6f}")
    print(f"  range=[{x.min():.6f}, {x.max():.6f}]")
    
    # Step 3: Flatten and Linear
    b, c, t, f = x.size()
    x = conformer.embed.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
    print(f"\nAfter Linear projection: {x.shape}")
    print(f"  mean={x.mean():.6f}, std={x.std():.6f}")
    print(f"  range=[{x.min():.6f}, {x.max():.6f}]")
    
    # Step 4: Positional encoding
    x, pos_emb = conformer.embed.pos_enc(x, offset=0)
    print(f"\nAfter Positional encoding: {x.shape}")
    print(f"  mean={x.mean():.6f}, std={x.std():.6f}")
    print(f"  range=[{x.min():.6f}, {x.max():.6f}]")
    
    print(f"\nPositional embedding (returned separately): {pos_emb.shape}")
    print(f"  mean={pos_emb.mean():.6f}, std={pos_emb.std():.6f}")
    print(f"  range=[{pos_emb.min():.6f}, {pos_emb.max():.6f}]")

