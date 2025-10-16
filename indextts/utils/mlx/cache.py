"""
MLX Model Cache Management for Apple Silicon M4

Handles conversion, saving, and loading of MLX models with intelligent caching.
Converts PyTorch models to MLX format once, then reuses cached versions.
"""

import os
import time
import torch
from pathlib import Path


class MLXModelCache:
    """
    Manages MLX model caching for fast loading.
    
    Strategy:
    1. First run: Convert PyTorch → MLX → Save to cache
    2. Subsequent runs: Load directly from cache (fast!)
    """
    
    def __init__(self, cache_dir="checkpoints/mlx"):
        """
        Initialize MLX cache manager.
        
        Args:
            cache_dir: Directory to store MLX model caches
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        print(f">> MLX Cache Directory: {cache_dir}")
    
    def get_cache_path(self, model_name):
        """Get cache file path for a model."""
        return os.path.join(self.cache_dir, f"{model_name}.npz")
    
    def is_cached(self, model_name):
        """Check if model is already cached."""
        cache_path = self.get_cache_path(model_name)
        return os.path.exists(cache_path)
    
    def convert_and_cache(self, model_name, pytorch_checkpoint_path=None, state_dict=None):
        """
        Convert PyTorch checkpoint to MLX format and cache it.
        
        Args:
            model_name: Name of the model (for cache key)
            pytorch_checkpoint_path: Path to PyTorch .pth file (optional if state_dict provided)
            state_dict: Direct state dict (optional if pytorch_checkpoint_path provided)
            
        Returns:
            Path to cached MLX model, or None if conversion failed
        """
        try:
            import mlx.core as mx
            
            print(f"\n{'='*70}")
            print(f"Converting {model_name.upper()} to MLX Format")
            print(f"{'='*70}")
            
            start_time = time.time()
            
            # Get state dict
            if state_dict is None:
                if pytorch_checkpoint_path is None:
                    raise ValueError("Either pytorch_checkpoint_path or state_dict must be provided")
                
                print(f"Source: {pytorch_checkpoint_path}")
                print(">> Loading PyTorch checkpoint...")
                checkpoint = torch.load(pytorch_checkpoint_path, map_location='cpu')
                
                # Extract state dict (handle different formats)
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                    print(">> Format: Standard (model key)")
                elif 'net' in checkpoint:
                    # S2MEL format - flatten nested structure
                    print(">> Format: S2MEL (net key)")
                    state_dict = {}
                    for key in checkpoint['net']:
                        for param_name, param_value in checkpoint['net'][key].items():
                            state_dict[f"{key}.{param_name}"] = param_value
                else:
                    state_dict = checkpoint
                    print(">> Format: Direct state dict")
            else:
                print("Source: Provided state_dict")
            
            # Statistics
            num_params = len(state_dict)
            total_size_mb = sum(
                v.numel() * v.element_size() 
                for v in state_dict.values() 
                if isinstance(v, torch.Tensor)
            ) / (1024 * 1024)
            
            print(f">> Parameters: {num_params}")
            print(f">> Size: {total_size_mb:.2f} MB")
            
            # Convert to MLX format
            print(">> Converting tensors to MLX arrays...")
            mlx_state_dict = {}
            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor):
                    numpy_array = value.cpu().numpy()
                    mlx_state_dict[key] = mx.array(numpy_array)
                else:
                    mlx_state_dict[key] = value
            
            # Save to cache
            cache_path = self.get_cache_path(model_name)
            print(f">> Saving to cache: {cache_path}")
            mx.savez(cache_path, **mlx_state_dict)
            
            elapsed = time.time() - start_time
            print(f">> ✓ Conversion completed in {elapsed:.2f}s")
            print(f">> ✓ Cached for future use")
            print(f"{'='*70}\n")
            
            return cache_path
            
        except Exception as e:
            print(f">> ✗ Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_from_cache(self, model_name):
        """
        Load MLX model from cache.
        
        Args:
            model_name: Name of the model
            
        Returns:
            MLX state dict, or None if not cached
        """
        try:
            import mlx.core as mx
            
            cache_path = self.get_cache_path(model_name)
            if not os.path.exists(cache_path):
                return None
            
            print(f">> Loading {model_name.upper()} from MLX cache...")
            print(f"   Cache: {cache_path}")
            
            state_dict = mx.load(cache_path)
            
            # Get size
            cache_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
            print(f"   Size: {cache_size_mb:.2f} MB")
            print(f">> ✓ Loaded from cache (fast!)")
            
            return state_dict
            
        except Exception as e:
            print(f">> ✗ Cache loading failed: {e}")
            return None
    
    def get_or_convert(self, model_name, pytorch_checkpoint_path):
        """
        Get MLX model from cache, or convert if not cached.
        
        This is the main entry point for getting MLX models.
        
        Args:
            model_name: Name of the model
            pytorch_checkpoint_path: Path to PyTorch checkpoint
            
        Returns:
            MLX state dict
        """
        # Try to load from cache first
        mlx_state = self.load_from_cache(model_name)
        
        if mlx_state is not None:
            return mlx_state
        
        # Not cached - convert and cache
        print(f">> No cache found for {model_name}, converting...")
        cache_path = self.convert_and_cache(model_name, pytorch_checkpoint_path)
        
        if cache_path is None:
            raise RuntimeError(f"Failed to convert {model_name} to MLX")
        
        # Load the newly cached model
        return self.load_from_cache(model_name)

