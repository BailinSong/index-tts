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
    
    def convert_and_cache(self, model_name, pytorch_checkpoint_path=None, state_dict=None, cfm_fix=False):
        """
        Convert PyTorch checkpoint to MLX format and cache it.

        Args:
            model_name: Name of the model (for cache key)
            pytorch_checkpoint_path: Path to PyTorch .pth file (optional if state_dict provided)
            state_dict: Direct state dict (optional if pytorch_checkpoint_path provided)
            cfm_fix: Apply CFM weight mapping fix (for s2mel_cfm)

        Returns:
            Path to cached MLX model, or None if conversion failed
        """
        try:
            import mlx.core as mx

            print(f"\n{'='*70}")
            print(f"Converting {model_name.upper()} to MLX Format")
            if cfm_fix:
                print("🔧 CFM Weight Mapping Fix: ENABLED")
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

            # Apply CFM fix if requested
            if cfm_fix and model_name == "s2mel_cfm":
                state_dict = self._apply_cfm_weight_fix(state_dict)

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
                    numpy_array = value.cpu().numpy().astype('float32')  # 确保使用float32
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
            if cfm_fix:
                print(">> ✓ CFM weight mapping fix applied")
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

    def _apply_cfm_weight_fix(self, state_dict):
        """
        Apply CFM weight mapping fix.

        The issue was that MLX CFM was loading weights with incorrect prefixes.
        This method ensures the correct weight mapping for s2mel_cfm.

        Args:
            state_dict: Original state dict from PyTorch checkpoint

        Returns:
            Fixed state dict with correct CFM weight mapping
        """
        print(f"🔧 Applying CFM weight mapping fix...")

        # Extract only CFM-related weights with correct prefix
        cfm_weights = {}
        cfm_prefix = "cfm."

        # Count weights before fix
        original_cfm_count = sum(1 for k in state_dict.keys() if k.startswith(cfm_prefix))
        print(f"   Original CFM weights found: {original_cfm_count}")

        # Extract CFM weights with proper mapping
        for key, value in state_dict.items():
            if key.startswith(cfm_prefix):
                # Keep CFM weights as-is (they have correct prefix)
                cfm_weights[key] = value

        print(f"   Fixed CFM weights extracted: {len(cfm_weights)}")

        # Validate we have the expected DiT components
        expected_components = [
            "cfm.estimator.x_embedder",
            "cfm.estimator.t_embedder",
            "cfm.estimator.transformer",
            "cfm.estimator.final_layer"
        ]

        found_components = []
        for component in expected_components:
            if any(k.startswith(component) for k in cfm_weights.keys()):
                found_components.append(component)

        print(f"   Found DiT components: {found_components}")

        if len(cfm_weights) < 200:  # 期望235个左右的权重
            print(f"   ⚠️  Warning: Only {len(cfm_weights)} CFM weights found, expected ~235")
            print(f"   Available keys sample:")
            sample_keys = list(state_dict.keys())[:10]
            for k in sample_keys:
                print(f"      {k}")
        else:
            print(f"   ✅ CFM weights count looks correct: {len(cfm_weights)}")

        return cfm_weights

    def get_or_convert(self, model_name, pytorch_checkpoint_path, force_cfm_fix=False):
        """
        Get MLX model from cache, or convert if not cached.

        This is the main entry point for getting MLX models.

        Args:
            model_name: Name of the model
            pytorch_checkpoint_path: Path to PyTorch checkpoint
            force_cfm_fix: Force CFM weight mapping fix (for s2mel_cfm)

        Returns:
            MLX state dict
        """
        # For s2mel_cfm, always check if we need to apply the fix
        need_cfm_fix = (model_name == "s2mel_cfm" and force_cfm_fix)

        # If CFM fix is needed, delete old cache to force regeneration
        if need_cfm_fix and self.is_cached(model_name):
            cache_path = self.get_cache_path(model_name)
            print(f"🔧 Deleting old s2mel_cfm cache to apply fix: {cache_path}")
            os.remove(cache_path)

        # Try to load from cache first (only if no fix needed)
        if not need_cfm_fix:
            mlx_state = self.load_from_cache(model_name)
            if mlx_state is not None:
                return mlx_state

        # Not cached or fix needed - convert and cache
        if need_cfm_fix:
            print(f">> Converting {model_name} with CFM weight mapping fix...")
        else:
            print(f">> No cache found for {model_name}, converting...")

        cache_path = self.convert_and_cache(
            model_name,
            pytorch_checkpoint_path,
            cfm_fix=need_cfm_fix
        )

        if cache_path is None:
            raise RuntimeError(f"Failed to convert {model_name} to MLX")

        # Load the newly cached model
        return self.load_from_cache(model_name)

