#!/usr/bin/env python3
"""
Inspect cached MLX weights to see what's available.
"""

import os
import mlx.core as mx

cache_dir = "/Users/bailin/index-tts/.mlx_cache"

print("="*70)
print("Inspecting MLX Cached Weights")
print("="*70)

gpt_cache = os.path.join(cache_dir, "gpt.npz")

if os.path.exists(gpt_cache):
    print(f"\nLoading: {gpt_cache}")
    weights = mx.load(gpt_cache)
    
    all_keys = list(weights.keys())
    print(f"\nTotal keys: {len(all_keys)}")
    
    # Group by category
    conformer_keys = [k for k in all_keys if 'conformer' in k.lower() or 'speech_conditioning_encoder' in k]
    perceiver_keys = [k for k in all_keys if 'perceiver' in k.lower() or 'resampler' in k.lower()]
    emo_keys = [k for k in all_keys if 'emo' in k.lower() and 'conformer' not in k.lower()]
    transformer_keys = [k for k in all_keys if 'gpt.h.' in k]
    
    print(f"\n" + "="*70)
    print(f"Conformer keys: {len(conformer_keys)}")
    print("="*70)
    if conformer_keys:
        for key in sorted(conformer_keys)[:30]:
            print(f"  {key}: {weights[key].shape}")
        if len(conformer_keys) > 30:
            print(f"  ... and {len(conformer_keys) - 30} more")
    else:
        print("  (None found)")
    
    print(f"\n" + "="*70)
    print(f"Perceiver keys: {len(perceiver_keys)}")
    print("="*70)
    if perceiver_keys:
        for key in sorted(perceiver_keys)[:30]:
            print(f"  {key}: {weights[key].shape}")
        if len(perceiver_keys) > 30:
            print(f"  ... and {len(perceiver_keys) - 30} more")
    else:
        print("  (None found)")
    
    print(f"\n" + "="*70)
    print(f"Emotion keys: {len(emo_keys)}")
    print("="*70)
    if emo_keys:
        for key in sorted(emo_keys):
            print(f"  {key}: {weights[key].shape}")
    else:
        print("  (None found)")
    
    print(f"\n" + "="*70)
    print(f"Transformer keys: {len(transformer_keys)}")
    print("="*70)
    print(f"  Layer 0 sample:")
    layer0_keys = [k for k in transformer_keys if 'gpt.h.0.' in k]
    for key in sorted(layer0_keys)[:10]:
        print(f"    {key}: {weights[key].shape}")
    
    # Save full list
    print(f"\n" + "="*70)
    print("Saving full list to mlx_weights_structure.txt")
    print("="*70)
    
    with open('experiments/mlx_weights_structure.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("MLX Cached GPT Weights\n")
        f.write("="*70 + "\n\n")
        
        for key in sorted(all_keys):
            f.write(f"{key}: {weights[key].shape}\n")
    
    print("✓ Done!")
    
else:
    print(f"\n❌ Cache file not found: {gpt_cache}")

