#!/usr/bin/env python3
"""
Check PyTorch model weights to understand Conformer and Perceiver structure.
"""

import sys
sys.path.insert(0, '/Users/bailin/index-tts')

import torch
from omegaconf import OmegaConf

# Load config
cfg = OmegaConf.load('/Users/bailin/index-tts/checkpoints/config.yaml')

# Load PyTorch model
print("Loading PyTorch GPT model...")
from indextts.gpt.model_v2 import UnifiedVoice

gpt = UnifiedVoice(**cfg.gpt)
checkpoint = torch.load('/Users/bailin/index-tts/checkpoints/indextts2_gpt.pt', map_location='cpu')

if 'model' in checkpoint:
    gpt.load_state_dict(checkpoint['model'], strict=False)
else:
    gpt.load_state_dict(checkpoint, strict=False)

print("\n" + "="*70)
print("PyTorch Model Structure")
print("="*70)

# Get all parameter names
all_keys = list(gpt.state_dict().keys())

print(f"\nTotal parameters: {len(all_keys)}")

# Filter for conditioning-related weights
print("\n" + "="*70)
print("Conformer Encoder Weights")
print("="*70)
conformer_keys = [k for k in all_keys if 'conformer' in k.lower() or 'speech_conditioning_encoder' in k]
for key in sorted(conformer_keys[:20]):  # Show first 20
    print(f"  {key}: {gpt.state_dict()[key].shape}")
if len(conformer_keys) > 20:
    print(f"  ... and {len(conformer_keys) - 20} more conformer keys")
print(f"\nTotal Conformer keys: {len(conformer_keys)}")

print("\n" + "="*70)
print("Perceiver Resampler Weights")
print("="*70)
perceiver_keys = [k for k in all_keys if 'perceiver' in k.lower() or 'resampler' in k.lower()]
for key in sorted(perceiver_keys[:20]):
    print(f"  {key}: {gpt.state_dict()[key].shape}")
if len(perceiver_keys) > 20:
    print(f"  ... and {len(perceiver_keys) - 20} more perceiver keys")
print(f"\nTotal Perceiver keys: {len(perceiver_keys)}")

# Check for emo conditioning
print("\n" + "="*70)
print("Emotion Conditioning Weights")
print("="*70)
emo_keys = [k for k in all_keys if 'emo' in k.lower() and 'conformer' not in k.lower()]
for key in sorted(emo_keys):
    print(f"  {key}: {gpt.state_dict()[key].shape}")
print(f"\nTotal Emo keys: {len(emo_keys)}")

# Check for transformer weights
print("\n" + "="*70)
print("Transformer Weights (sample)")
print("="*70)
transformer_keys = [k for k in all_keys if 'gpt.h.' in k]
print(f"Total transformer keys: {len(transformer_keys)}")
print("\nLayer 0 sample:")
layer0_keys = [k for k in transformer_keys if 'gpt.h.0.' in k]
for key in sorted(layer0_keys):
    print(f"  {key}: {gpt.state_dict()[key].shape}")

# Save structure to file
print("\n" + "="*70)
print("Saving full structure to pytorch_weights_structure.txt")
print("="*70)

with open('experiments/pytorch_weights_structure.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("PyTorch GPT Model Weight Structure\n")
    f.write("="*70 + "\n\n")
    
    for key in sorted(all_keys):
        f.write(f"{key}: {gpt.state_dict()[key].shape}\n")

print("✓ Done!")

