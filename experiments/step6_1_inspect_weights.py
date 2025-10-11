#!/usr/bin/env python3
"""
Step 6.1: Inspect PyTorch weights for Conformer and Perceiver
Goal: Understand weight structure and create mapping
"""

import mlx.core as mx
import os

print("\n" + "="*70)
print("Step 6.1: Inspecting PyTorch Weights in MLX Cache")
print("="*70)

# Check MLX cache
cache_dir = "/Users/bailin/index-tts/.mlx_cache"
gpt_cache = os.path.join(cache_dir, "gpt.npz")

if not os.path.exists(gpt_cache):
    print(f"\n❌ MLX cache not found: {gpt_cache}")
    print("   Need to run inference once to create cache")
    exit(1)

print(f"\n✓ Found MLX cache: {gpt_cache}")
print(f"  Size: {os.path.getsize(gpt_cache) / 1024 / 1024:.1f} MB")

# Load weights
print("\nLoading weights...")
weights = mx.load(gpt_cache)
all_keys = list(weights.keys())

print(f"Total keys: {len(all_keys)}")

# Categorize keys
conformer_keys = []
perceiver_keys = []
emo_keys = []
transformer_keys = []
other_keys = []

for key in all_keys:
    key_lower = key.lower()
    if 'conformer' in key_lower or 'speech_conditioning_encoder' in key_lower:
        conformer_keys.append(key)
    elif 'perceiver' in key_lower or 'resampler' in key_lower:
        perceiver_keys.append(key)
    elif 'emo' in key_lower and 'conformer' not in key_lower:
        emo_keys.append(key)
    elif 'gpt.h.' in key:
        transformer_keys.append(key)
    else:
        other_keys.append(key)

# Report
print("\n" + "="*70)
print("Weight Categories")
print("="*70)
print(f"Conformer keys:   {len(conformer_keys)}")
print(f"Perceiver keys:   {len(perceiver_keys)}")
print(f"Emotion keys:     {len(emo_keys)}")
print(f"Transformer keys: {len(transformer_keys)}")
print(f"Other keys:       {len(other_keys)}")

# Detailed inspection
if conformer_keys:
    print("\n" + "="*70)
    print("Conformer Weights (First 30)")
    print("="*70)
    for key in sorted(conformer_keys)[:30]:
        shape = weights[key].shape
        print(f"  {key}")
        print(f"    Shape: {shape}")
    if len(conformer_keys) > 30:
        print(f"  ... and {len(conformer_keys) - 30} more")
else:
    print("\n⚠️  No Conformer weights found!")
    print("   This means PyTorch model doesn't have Conformer,")
    print("   or it's named differently.")

if perceiver_keys:
    print("\n" + "="*70)
    print("Perceiver Weights (First 30)")
    print("="*70)
    for key in sorted(perceiver_keys)[:30]:
        shape = weights[key].shape
        print(f"  {key}")
        print(f"    Shape: {shape}")
    if len(perceiver_keys) > 30:
        print(f"  ... and {len(perceiver_keys) - 30} more")
else:
    print("\n⚠️  No Perceiver weights found!")

# Check for alternative conditioning names
print("\n" + "="*70)
print("Searching for Conditioning-Related Keys")
print("="*70)

conditioning_patterns = [
    'speech_conditioning',
    'conditioning_encoder',
    'conditioning_perceiver',
    'speech_encoder',
    'audio_encoder',
    'conditioning_latent',
]

found_patterns = {}
for pattern in conditioning_patterns:
    matching = [k for k in all_keys if pattern in k.lower()]
    if matching:
        found_patterns[pattern] = matching

if found_patterns:
    for pattern, keys in found_patterns.items():
        print(f"\nPattern '{pattern}': {len(keys)} keys")
        for key in sorted(keys)[:10]:
            print(f"  {key}: {weights[key].shape}")
        if len(keys) > 10:
            print(f"  ... and {len(keys) - 10} more")
else:
    print("\n⚠️  No conditioning-related patterns found")
    print("   PyTorch model might use different architecture")

# Check emotion conditioning
if emo_keys:
    print("\n" + "="*70)
    print("Emotion Conditioning Weights")
    print("="*70)
    for key in sorted(emo_keys):
        shape = weights[key].shape
        print(f"  {key}: {shape}")

# Sample other keys to understand structure
print("\n" + "="*70)
print("Other Important Keys (Sample)")
print("="*70)

embedding_keys = [k for k in other_keys if 'embedding' in k.lower()]
norm_keys = [k for k in other_keys if 'norm' in k.lower() or 'ln' in k.lower()]
head_keys = [k for k in other_keys if 'head' in k.lower()]

print(f"\nEmbedding keys: {len(embedding_keys)}")
for key in sorted(embedding_keys):
    print(f"  {key}: {weights[key].shape}")

print(f"\nHead keys: {len(head_keys)}")
for key in sorted(head_keys):
    print(f"  {key}: {weights[key].shape}")

# Save full structure
output_file = "experiments/step6_1_weight_structure.txt"
print(f"\n" + "="*70)
print(f"Saving full structure to: {output_file}")
print("="*70)

with open(output_file, 'w') as f:
    f.write("="*70 + "\n")
    f.write("PyTorch GPT Weights Structure (from MLX Cache)\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Total keys: {len(all_keys)}\n\n")
    
    f.write("Categories:\n")
    f.write(f"  Conformer:   {len(conformer_keys)}\n")
    f.write(f"  Perceiver:   {len(perceiver_keys)}\n")
    f.write(f"  Emotion:     {len(emo_keys)}\n")
    f.write(f"  Transformer: {len(transformer_keys)}\n")
    f.write(f"  Other:       {len(other_keys)}\n\n")
    
    f.write("="*70 + "\n")
    f.write("All Keys (Sorted)\n")
    f.write("="*70 + "\n\n")
    
    for key in sorted(all_keys):
        f.write(f"{key}: {weights[key].shape}\n")

print(f"✓ Saved to {output_file}")

# Summary
print("\n" + "="*70)
print("Summary")
print("="*70)

if not conformer_keys and not perceiver_keys:
    print("\n⚠️  WARNING: No Conformer or Perceiver weights found!")
    print("\nPossible reasons:")
    print("  1. PyTorch model uses different conditioning architecture")
    print("  2. Conditioning is part of main transformer (not separate)")
    print("  3. Weight names are different from expected")
    print("\n→ Need to check PyTorch model architecture directly")
else:
    print("\n✓ Found conditioning weights")
    print(f"  Conformer: {len(conformer_keys)} weights")
    print(f"  Perceiver: {len(perceiver_keys)} weights")
    print("\n→ Ready to create weight mapping")

print("="*70 + "\n")

