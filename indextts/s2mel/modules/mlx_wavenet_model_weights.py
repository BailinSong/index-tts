"""
Weight loading utilities for MLX WaveNet Model
Handles PyTorch to MLX weight conversion for WaveNet components
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def load_wavenet_weights(mlx_model, pytorch_state_dict, prefix=""):
    """
    Load WaveNet weights from PyTorch to MLX.
    
    Args:
        mlx_model: MLX WaveNet model
        pytorch_state_dict: PyTorch state dict (with numpy arrays)
        prefix: Key prefix
    
    Returns:
        loaded: Number of weights loaded
    """
    loaded = 0
    
    print(f">> Loading WaveNet weights from PyTorch...")
    
    # 1. Load input projection
    print("   [1/3] Loading input projection...")
    
    if f"{prefix}input_projection.weight" in pytorch_state_dict:
        mlx_model.input_projection.weight = mx.array(pytorch_state_dict[f"{prefix}input_projection.weight"])
        loaded += 1
        print(f"   ✅ Loaded input_projection.weight: {pytorch_state_dict[f'{prefix}input_projection.weight'].shape}")
    
    if f"{prefix}input_projection.bias" in pytorch_state_dict:
        mlx_model.input_projection.bias = mx.array(pytorch_state_dict[f"{prefix}input_projection.bias"])
        loaded += 1
        print(f"   ✅ Loaded input_projection.bias: {pytorch_state_dict[f'{prefix}input_projection.bias'].shape}")
    
    # 2. Load dilated convolutions
    print("   [2/3] Loading dilated convolutions...")
    
    for i, layer in enumerate(mlx_model.layers):
        layer_prefix = f"{prefix}layers.{i}."
        
        # Dilated convolution weights
        if f"{layer_prefix}dilated_conv.weight" in pytorch_state_dict:
            layer.dilated_conv.weight = mx.array(pytorch_state_dict[f"{layer_prefix}dilated_conv.weight"])
            loaded += 1
        
        if f"{layer_prefix}dilated_conv.bias" in pytorch_state_dict:
            layer.dilated_conv.bias = mx.array(pytorch_state_dict[f"{layer_prefix}dilated_conv.bias"])
            loaded += 1
        
        # Gated activation weights
        if f"{layer_prefix}gated_conv.weight" in pytorch_state_dict:
            layer.gated_conv.weight = mx.array(pytorch_state_dict[f"{layer_prefix}gated_conv.weight"])
            loaded += 1
        
        if f"{layer_prefix}gated_conv.bias" in pytorch_state_dict:
            layer.gated_conv.bias = mx.array(pytorch_state_dict[f"{layer_prefix}gated_conv.bias"])
            loaded += 1
        
        # Skip connection weights
        if f"{layer_prefix}skip.weight" in pytorch_state_dict:
            layer.skip.weight = mx.array(pytorch_state_dict[f"{layer_prefix}skip.weight"])
            loaded += 1
        
        if f"{layer_prefix}skip.bias" in pytorch_state_dict:
            layer.skip.bias = mx.array(pytorch_state_dict[f"{layer_prefix}skip.bias"])
            loaded += 1
        
        # Residual connection weights
        if f"{layer_prefix}residual.weight" in pytorch_state_dict:
            layer.residual.weight = mx.array(pytorch_state_dict[f"{layer_prefix}residual.weight"])
            loaded += 1
        
        if f"{layer_prefix}residual.bias" in pytorch_state_dict:
            layer.residual.bias = mx.array(pytorch_state_dict[f"{layer_prefix}residual.bias"])
            loaded += 1
    
    # 3. Load output layers
    print("   [3/3] Loading output layers...")
    
    # Skip projection
    if f"{prefix}skip_projection.weight" in pytorch_state_dict:
        mlx_model.skip_projection.weight = mx.array(pytorch_state_dict[f"{prefix}skip_projection.weight"])
        loaded += 1
        print(f"   ✅ Loaded skip_projection.weight: {pytorch_state_dict[f'{prefix}skip_projection.weight'].shape}")
    
    if f"{prefix}skip_projection.bias" in pytorch_state_dict:
        mlx_model.skip_projection.bias = mx.array(pytorch_state_dict[f"{prefix}skip_projection.bias"])
        loaded += 1
        print(f"   ✅ Loaded skip_projection.bias: {pytorch_state_dict[f'{prefix}skip_projection.bias'].shape}")
    
    # Output projection
    if f"{prefix}output_projection.weight" in pytorch_state_dict:
        mlx_model.output_projection.weight = mx.array(pytorch_state_dict[f"{prefix}output_projection.weight"])
        loaded += 1
        print(f"   ✅ Loaded output_projection.weight: {pytorch_state_dict[f'{prefix}output_projection.weight'].shape}")
    
    if f"{prefix}output_projection.bias" in pytorch_state_dict:
        mlx_model.output_projection.bias = mx.array(pytorch_state_dict[f"{prefix}output_projection.bias"])
        loaded += 1
        print(f"   ✅ Loaded output_projection.bias: {pytorch_state_dict[f'{prefix}output_projection.bias'].shape}")
    
    print(f">> MLX WaveNet loaded {loaded} weights total")
    return loaded


def load_wavenet_from_cache(mlx_model, cache_dict, prefix=""):
    """
    Load WaveNet weights from MLX cache.
    
    Args:
        mlx_model: MLX WaveNet model
        cache_dict: dict from mx.load() with cached weights
        prefix: Key prefix
    
    Returns:
        loaded: Number of weights loaded
    """
    loaded = 0
    
    print(f">> Loading WaveNet weights from cache...")
    
    # Load input projection from cache
    input_keys = [
        f"{prefix}input_projection.weight",
        f"{prefix}input_projection.bias"
    ]
    
    for key in input_keys:
        if key in cache_dict:
            attr_name = key.split('.')[-1]
            if hasattr(mlx_model, 'input_projection'):
                setattr(mlx_model.input_projection, attr_name, cache_dict[key])
                loaded += 1
                print(f"   ✅ Loaded {key}")
    
    # Load layers from cache
    for i in range(len(mlx_model.layers)):
        layer_prefix = f"{prefix}layers.{i}."
        layer = mlx_model.layers[i]
        
        # Load layer weights
        layer_keys = [
            f"{layer_prefix}dilated_conv.weight",
            f"{layer_prefix}dilated_conv.bias",
            f"{layer_prefix}gated_conv.weight",
            f"{layer_prefix}gated_conv.bias",
            f"{layer_prefix}skip.weight",
            f"{layer_prefix}skip.bias",
            f"{layer_prefix}residual.weight",
            f"{layer_prefix}residual.bias"
        ]
        
        for key in layer_keys:
            if key in cache_dict:
                attr_path = key.replace(f"{layer_prefix}", "").split('.')
                obj = layer
                for attr in attr_path[:-1]:
                    obj = getattr(obj, attr)
                setattr(obj, attr_path[-1], cache_dict[key])
                loaded += 1
    
    # Load output layers from cache
    output_keys = [
        f"{prefix}skip_projection.weight",
        f"{prefix}skip_projection.bias",
        f"{prefix}output_projection.weight",
        f"{prefix}output_projection.bias"
    ]
    
    for key in output_keys:
        if key in cache_dict:
            attr_name = key.split('.')[-1]
            layer_name = key.split('.')[-2]
            if hasattr(mlx_model, layer_name):
                setattr(getattr(mlx_model, layer_name), attr_name, cache_dict[key])
                loaded += 1
                print(f"   ✅ Loaded {key}")
    
    print(f">> MLX WaveNet loaded {loaded} weights from cache")
    return loaded


def load_wavenet_improved_weights(mlx_model, pytorch_state_dict, prefix=""):
    """
    Load WaveNet Improved weights from PyTorch to MLX.
    
    Args:
        mlx_model: MLX WaveNet Improved model
        pytorch_state_dict: PyTorch state dict (with numpy arrays)
        prefix: Key prefix
    
    Returns:
        loaded: Number of weights loaded
    """
    loaded = 0
    
    print(f">> Loading WaveNet Improved weights from PyTorch...")
    
    # Similar structure to regular WaveNet but with additional improvements
    # Load basic WaveNet weights first
    loaded += load_wavenet_weights(mlx_model, pytorch_state_dict, prefix)
    
    # Load additional improved components
    print("   [4/4] Loading improved components...")
    
    # Additional layers specific to improved version
    if f"{prefix}improved_layer.weight" in pytorch_state_dict:
        mlx_model.improved_layer.weight = mx.array(pytorch_state_dict[f"{prefix}improved_layer.weight"])
        loaded += 1
        print(f"   ✅ Loaded improved_layer.weight: {pytorch_state_dict[f'{prefix}improved_layer.weight'].shape}")
    
    print(f">> MLX WaveNet Improved loaded {loaded} weights total")
    return loaded
