"""
Convert S2MEL CFM weights from PyTorch to MLX format and cache.
This script should be run once to create the MLX cache.
"""

import os
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'

import torch
from omegaconf import OmegaConf
from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
from indextts.s2mel.modules.mlx_cfm import MLXCFM
from indextts.utils.mlx_cache import MLXModelCache
import mlx.core as mx


def convert_and_cache_cfm():
    """
    Convert S2MEL CFM from PyTorch to MLX and cache it.
    """
    print("="*70)
    print("Converting S2MEL CFM to MLX format")
    print("="*70)
    
    # Load config
    cfg = OmegaConf.load('checkpoints/config.yaml')
    
    # Load PyTorch S2MEL
    print("\n>> Step 1: Loading PyTorch S2MEL...")
    s2mel_pt = MyModel(cfg.s2mel, use_gpt_latent=True)
    s2mel_pt, _, _, _ = load_checkpoint2(
        s2mel_pt, None, 'checkpoints/s2mel.pth',
        load_only_params=True, ignore_modules=[], is_distributed=False
    )
    print(">> PyTorch S2MEL loaded")
    
    # Get state dict
    print("\n>> Step 2: Extracting PyTorch weights...")
    state_dict = s2mel_pt.state_dict()
    state_dict_np = {k: v.cpu().numpy() for k, v in state_dict.items()}
    print(f">> Extracted {len(state_dict_np)} weights")
    
    # Create MLX CFM
    print("\n>> Step 3: Creating MLX CFM...")
    mlx_cfm = MLXCFM(cfg.s2mel)
    print(">> MLX CFM created")
    
    # Load weights
    print("\n>> Step 4: Loading weights into MLX CFM...")
    try:
        loaded = mlx_cfm.load_weights_from_pytorch(state_dict_np, prefix="models.cfm.")
        print(f">> Successfully loaded {loaded} weights")
    except Exception as e:
        print(f">> Error loading weights: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Cache the weights
    print("\n>> Step 5: Caching MLX weights...")
    cache_dir = 'checkpoints/mlx'
    mlx_cache = MLXModelCache(cache_dir=cache_dir)
    
    # Collect all weights from MLX CFM
    # This is a simplified approach - in practice, we'd use MLXModelCache's convert_and_cache
    # But for now, let's just save the state dict
    
    try:
        # Get all MLX arrays from the model
        mlx_weights = extract_mlx_weights(mlx_cfm.estimator)
        
        # Save using MLXModelCache
        cache_path = os.path.join(cache_dir, "s2mel_cfm.npz")
        mx.savez(cache_path, **mlx_weights)
        print(f">> Cached {len(mlx_weights)} weights to {cache_path}")
        
        # Get cache size
        size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print(f">> Cache size: {size_mb:.2f} MB")
        
    except Exception as e:
        print(f">> Error caching weights: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("✅ S2MEL CFM successfully converted and cached!")
    print("="*70)
    print(f"Cache location: {cache_path}")
    print(f"Next run will load from cache (much faster!)")
    
    return True


def extract_mlx_weights(model, prefix=""):
    """
    Recursively extract all MLX arrays from a model.
    
    Returns:
        dict of {name: mx.array}
    """
    weights = {}
    
    # Get all attributes
    for name in dir(model):
        if name.startswith('_'):
            continue
        
        attr = getattr(model, name)
        full_name = f"{prefix}.{name}" if prefix else name
        
        if isinstance(attr, mx.array):
            weights[full_name] = attr
        elif isinstance(attr, list):
            # Handle lists of layers
            for i, item in enumerate(attr):
                if hasattr(item, '__dict__'):
                    sub_weights = extract_mlx_weights(item, f"{full_name}.{i}")
                    weights.update(sub_weights)
        elif hasattr(attr, '__dict__') and not callable(attr):
            # Recursively extract from sub-modules
            try:
                sub_weights = extract_mlx_weights(attr, full_name)
                weights.update(sub_weights)
            except:
                pass
    
    return weights


if __name__ == "__main__":
    success = convert_and_cache_cfm()
    
    if success:
        print("\n✅ Conversion successful!")
        print("\nYou can now use:")
        print("  tts = IndexTTS2(..., use_mlx=True)")
        print("  # CFM will load from cache automatically")
    else:
        print("\n❌ Conversion failed. Check errors above.")

