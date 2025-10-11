#!/usr/bin/env python3
"""
Step 6.1: Inspect PyTorch checkpoint directly
Goal: Understand weight structure without needing MLX cache
"""

import torch
import os

print("\n" + "="*70)
print("Step 6.1: Inspecting PyTorch Checkpoint Directly")
print("="*70)

# Load PyTorch checkpoint
checkpoint_path = "/Users/bailin/index-tts/checkpoints/gpt.pth"

if not os.path.exists(checkpoint_path):
    print(f"\n❌ Checkpoint not found: {checkpoint_path}")
    exit(1)

print(f"\n✓ Found checkpoint: {checkpoint_path}")
print(f"  Size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.1f} MB")

print("\nLoading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Extract state_dict
if 'model' in checkpoint:
    state_dict = checkpoint['model']
    print("✓ Found 'model' key in checkpoint")
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    print("✓ Found 'state_dict' key in checkpoint")
else:
    state_dict = checkpoint
    print("✓ Using checkpoint directly as state_dict")

all_keys = list(state_dict.keys())
print(f"Total keys: {len(all_keys)}")

# Categorize keys
conformer_keys = []
perceiver_keys = []
emo_keys = []
transformer_keys = []
conditioning_keys = []
other_keys = []

for key in all_keys:
    key_lower = key.lower()
    if 'conformer' in key_lower:
        conformer_keys.append(key)
    elif 'perceiver' in key_lower or 'resampler' in key_lower:
        perceiver_keys.append(key)
    elif 'conditioning' in key_lower:
        conditioning_keys.append(key)
    elif 'emo' in key_lower and 'conformer' not in key_lower:
        emo_keys.append(key)
    elif 'gpt.h.' in key or '.attn.' in key or '.mlp.' in key:
        transformer_keys.append(key)
    else:
        other_keys.append(key)

# Report
print("\n" + "="*70)
print("Weight Categories")
print("="*70)
print(f"Conformer keys:       {len(conformer_keys)}")
print(f"Perceiver keys:       {len(perceiver_keys)}")
print(f"Conditioning keys:    {len(conditioning_keys)}")
print(f"Emotion keys:         {len(emo_keys)}")
print(f"Transformer keys:     {len(transformer_keys)}")
print(f"Other keys:           {len(other_keys)}")

# Detailed inspection
if conformer_keys:
    print("\n" + "="*70)
    print("Conformer Weights")
    print("="*70)
    for key in sorted(conformer_keys)[:40]:
        shape = tuple(state_dict[key].shape)
        dtype = state_dict[key].dtype
        print(f"  {key}")
        print(f"    Shape: {shape}, dtype: {dtype}")
    if len(conformer_keys) > 40:
        print(f"  ... and {len(conformer_keys) - 40} more")

if perceiver_keys:
    print("\n" + "="*70)
    print("Perceiver Weights")
    print("="*70)
    for key in sorted(perceiver_keys)[:40]:
        shape = tuple(state_dict[key].shape)
        dtype = state_dict[key].dtype
        print(f"  {key}")
        print(f"    Shape: {shape}, dtype: {dtype}")
    if len(perceiver_keys) > 40:
        print(f"  ... and {len(perceiver_keys) - 40} more")

if conditioning_keys:
    print("\n" + "="*70)
    print("Conditioning Weights")
    print("="*70)
    for key in sorted(conditioning_keys)[:40]:
        shape = tuple(state_dict[key].shape)
        dtype = state_dict[key].dtype
        print(f"  {key}")
        print(f"    Shape: {shape}, dtype: {dtype}")
    if len(conditioning_keys) > 40:
        print(f"  ... and {len(conditioning_keys) - 40} more")

# Check for alternative patterns
print("\n" + "="*70)
print("Searching for Alternative Patterns")
print("="*70)

patterns = {
    'speech_conditioning': [],
    'speech_encoder': [],
    'audio_encoder': [],
    'encoder': [],
}

for key in all_keys:
    for pattern in patterns.keys():
        if pattern in key.lower() and key not in conformer_keys + perceiver_keys + conditioning_keys:
            patterns[pattern].append(key)

for pattern, keys in patterns.items():
    if keys:
        print(f"\nPattern '{pattern}': {len(keys)} keys")
        for key in sorted(keys)[:10]:
            print(f"  {key}: {tuple(state_dict[key].shape)}")
        if len(keys) > 10:
            print(f"  ... and {len(keys) - 10} more")

# Emotion conditioning
if emo_keys:
    print("\n" + "="*70)
    print("Emotion Conditioning Weights")
    print("="*70)
    for key in sorted(emo_keys):
        shape = tuple(state_dict[key].shape)
        print(f"  {key}: {shape}")

# Sample other important keys
print("\n" + "="*70)
print("Other Important Keys (Sample)")
print("="*70)

embedding_keys = [k for k in other_keys if 'embedding' in k.lower()]
norm_keys = [k for k in other_keys if ('norm' in k.lower() or 'ln_' in k) and 'gpt.h.' not in k]
head_keys = [k for k in other_keys if 'head' in k.lower()]

print(f"\nEmbedding keys ({len(embedding_keys)}):")
for key in sorted(embedding_keys)[:10]:
    print(f"  {key}: {tuple(state_dict[key].shape)}")

print(f"\nNorm keys (non-transformer, {len(norm_keys)}):")
for key in sorted(norm_keys)[:10]:
    print(f"  {key}: {tuple(state_dict[key].shape)}")

print(f"\nHead keys ({len(head_keys)}):")
for key in sorted(head_keys):
    print(f"  {key}: {tuple(state_dict[key].shape)}")

# Save full structure
output_file = "experiments/step6_1_pytorch_structure.txt"
print(f"\n" + "="*70)
print(f"Saving full structure to: {output_file}")
print("="*70)

with open(output_file, 'w') as f:
    f.write("="*70 + "\n")
    f.write("PyTorch GPT Checkpoint Structure\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Checkpoint: {checkpoint_path}\n")
    f.write(f"Total keys: {len(all_keys)}\n\n")
    
    f.write("Categories:\n")
    f.write(f"  Conformer:       {len(conformer_keys)}\n")
    f.write(f"  Perceiver:       {len(perceiver_keys)}\n")
    f.write(f"  Conditioning:    {len(conditioning_keys)}\n")
    f.write(f"  Emotion:         {len(emo_keys)}\n")
    f.write(f"  Transformer:     {len(transformer_keys)}\n")
    f.write(f"  Other:           {len(other_keys)}\n\n")
    
    f.write("="*70 + "\n")
    f.write("All Keys (Sorted by Category)\n")
    f.write("="*70 + "\n\n")
    
    if conformer_keys:
        f.write("CONFORMER:\n")
        for key in sorted(conformer_keys):
            f.write(f"  {key}: {tuple(state_dict[key].shape)}\n")
        f.write("\n")
    
    if perceiver_keys:
        f.write("PERCEIVER:\n")
        for key in sorted(perceiver_keys):
            f.write(f"  {key}: {tuple(state_dict[key].shape)}\n")
        f.write("\n")
    
    if conditioning_keys:
        f.write("CONDITIONING:\n")
        for key in sorted(conditioning_keys):
            f.write(f"  {key}: {tuple(state_dict[key].shape)}\n")
        f.write("\n")
    
    if emo_keys:
        f.write("EMOTION:\n")
        for key in sorted(emo_keys):
            f.write(f"  {key}: {tuple(state_dict[key].shape)}\n")
        f.write("\n")
    
    f.write("TRANSFORMER:\n")
    for key in sorted(transformer_keys)[:20]:
        f.write(f"  {key}: {tuple(state_dict[key].shape)}\n")
    f.write(f"  ... and {len(transformer_keys) - 20} more transformer keys\n\n")
    
    f.write("OTHER:\n")
    for key in sorted(other_keys):
        f.write(f"  {key}: {tuple(state_dict[key].shape)}\n")

print(f"✓ Saved to {output_file}")

# Summary and recommendations
print("\n" + "="*70)
print("Summary")
print("="*70)

total_conditioning = len(conformer_keys) + len(perceiver_keys) + len(conditioning_keys)

if total_conditioning == 0:
    print("\n⚠️  WARNING: No explicit Conformer or Perceiver weights found!")
    print("\nPossible scenarios:")
    print("  1. PyTorch model doesn't use Conformer/Perceiver architecture")
    print("  2. Conditioning is simple (Linear + pooling) - already in transformer weights")
    print("  3. Model uses different conditioning approach")
    print("\n→ MLX implementation uses complex Conformer + Perceiver")
    print("→ Weights will remain randomly initialized (as in Steps 1-5)")
    print("→ Can still proceed with Step 6.2-6.5 for testing and optimization")
else:
    print(f"\n✓ Found {total_conditioning} conditioning weights")
    print(f"  Conformer: {len(conformer_keys)}")
    print(f"  Perceiver: {len(perceiver_keys)}")
    print(f"  Other conditioning: {len(conditioning_keys)}")
    print("\n→ Ready to create weight mapping in Step 6.2")

print("="*70 + "\n")

