"""
Weight loading utilities for MLX Commons Components
Handles PyTorch to MLX weight conversion for common components
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def load_commons_weights(mlx_model, pytorch_state_dict, prefix=""):
    """
    Load Commons weights from PyTorch to MLX.
    
    Args:
        mlx_model: MLX Commons model (MLXGPTLayer, MLXLengthRegulator, etc.)
        pytorch_state_dict: PyTorch state dict (with numpy arrays)
        prefix: Key prefix
    
    Returns:
        loaded: Number of weights loaded
    """
    loaded = 0
    
    print(f">> Loading Commons weights from PyTorch...")
    
    # 1. Load MLXGPTLayer weights
    if hasattr(mlx_model, 'layer0') and hasattr(mlx_model, 'layer1') and hasattr(mlx_model, 'layer2'):
        print("   [1/3] Loading MLXGPTLayer...")
        
        if f"{prefix}layer0.weight" in pytorch_state_dict:
            mlx_model.layer0.weight = mx.array(pytorch_state_dict[f"{prefix}layer0.weight"])
            loaded += 1
            print(f"   ✅ Loaded layer0.weight: {pytorch_state_dict[f'{prefix}layer0.weight'].shape}")
        
        if f"{prefix}layer0.bias" in pytorch_state_dict:
            mlx_model.layer0.bias = mx.array(pytorch_state_dict[f"{prefix}layer0.bias"])
            loaded += 1
            print(f"   ✅ Loaded layer0.bias: {pytorch_state_dict[f'{prefix}layer0.bias'].shape}")
        
        if f"{prefix}layer1.weight" in pytorch_state_dict:
            mlx_model.layer1.weight = mx.array(pytorch_state_dict[f"{prefix}layer1.weight"])
            loaded += 1
            print(f"   ✅ Loaded layer1.weight: {pytorch_state_dict[f'{prefix}layer1.weight'].shape}")
        
        if f"{prefix}layer1.bias" in pytorch_state_dict:
            mlx_model.layer1.bias = mx.array(pytorch_state_dict[f"{prefix}layer1.bias"])
            loaded += 1
            print(f"   ✅ Loaded layer1.bias: {pytorch_state_dict[f'{prefix}layer1.bias'].shape}")
        
        if f"{prefix}layer2.weight" in pytorch_state_dict:
            mlx_model.layer2.weight = mx.array(pytorch_state_dict[f"{prefix}layer2.weight"])
            loaded += 1
            print(f"   ✅ Loaded layer2.weight: {pytorch_state_dict[f'{prefix}layer2.weight'].shape}")
        
        if f"{prefix}layer2.bias" in pytorch_state_dict:
            mlx_model.layer2.bias = mx.array(pytorch_state_dict[f"{prefix}layer2.bias"])
            loaded += 1
            print(f"   ✅ Loaded layer2.bias: {pytorch_state_dict[f'{prefix}layer2.bias'].shape}")
    
    # 2. Load MLXLengthRegulator weights
    if hasattr(mlx_model, 'upsampling_layers'):
        print("   [2/3] Loading MLXLengthRegulator...")
        
        for i, layer in enumerate(mlx_model.upsampling_layers):
            layer_prefix = f"{prefix}upsampling_layers.{i}."
            
            if f"{layer_prefix}conv.weight" in pytorch_state_dict:
                layer.conv.weight = mx.array(pytorch_state_dict[f"{layer_prefix}conv.weight"])
                loaded += 1
            
            if f"{layer_prefix}conv.bias" in pytorch_state_dict:
                layer.conv.bias = mx.array(pytorch_state_dict[f"{layer_prefix}conv.bias"])
                loaded += 1
            
            if f"{layer_prefix}norm.weight" in pytorch_state_dict:
                layer.norm.weight = mx.array(pytorch_state_dict[f"{layer_prefix}norm.weight"])
                loaded += 1
            
            if f"{layer_prefix}norm.bias" in pytorch_state_dict:
                layer.norm.bias = mx.array(pytorch_state_dict[f"{layer_prefix}norm.bias"])
                loaded += 1
    
    # 3. Load other common components
    print("   [3/3] Loading other components...")
    
    # Check for other common layer patterns
    common_patterns = [
        'linear', 'conv', 'norm', 'embedding', 'projection'
    ]
    
    for pattern in common_patterns:
        for key, value in pytorch_state_dict.items():
            if key.startswith(prefix) and pattern in key:
                # Extract attribute path
                attr_path = key.replace(prefix, "").split('.')
                if len(attr_path) >= 2:
                    try:
                        obj = mlx_model
                        for attr in attr_path[:-1]:
                            obj = getattr(obj, attr)
                        setattr(obj, attr_path[-1], mx.array(value))
                        loaded += 1
                        print(f"   ✅ Loaded {key}")
                    except AttributeError:
                        # Skip if attribute doesn't exist
                        continue
    
    print(f">> MLX Commons loaded {loaded} weights total")
    return loaded


def load_commons_from_cache(mlx_model, cache_dict, prefix=""):
    """
    Load Commons weights from MLX cache.
    
    Args:
        mlx_model: MLX Commons model
        cache_dict: dict from mx.load() with cached weights
        prefix: Key prefix
    
    Returns:
        loaded: Number of weights loaded
    """
    loaded = 0
    
    print(f">> Loading Commons weights from cache...")
    
    # Load MLXGPTLayer from cache
    if hasattr(mlx_model, 'layer0'):
        gpt_keys = [
            f"{prefix}layer0.weight",
            f"{prefix}layer0.bias",
            f"{prefix}layer1.weight",
            f"{prefix}layer1.bias",
            f"{prefix}layer2.weight",
            f"{prefix}layer2.bias"
        ]
        
        for key in gpt_keys:
            if key in cache_dict:
                attr_name = key.split('.')[-1]
                layer_name = key.split('.')[-2]
                if hasattr(mlx_model, layer_name):
                    setattr(getattr(mlx_model, layer_name), attr_name, cache_dict[key])
                    loaded += 1
                    print(f"   ✅ Loaded {key}")
    
    # Load MLXLengthRegulator from cache
    if hasattr(mlx_model, 'upsampling_layers'):
        for i in range(len(mlx_model.upsampling_layers)):
            layer_prefix = f"{prefix}upsampling_layers.{i}."
            layer = mlx_model.upsampling_layers[i]
            
            layer_keys = [
                f"{layer_prefix}conv.weight",
                f"{layer_prefix}conv.bias",
                f"{layer_prefix}norm.weight",
                f"{layer_prefix}norm.bias"
            ]
            
            for key in layer_keys:
                if key in cache_dict:
                    attr_path = key.replace(f"{layer_prefix}", "").split('.')
                    obj = layer
                    for attr in attr_path[:-1]:
                        obj = getattr(obj, attr)
                    setattr(obj, attr_path[-1], cache_dict[key])
                    loaded += 1
    
    # Load other components from cache
    for key, value in cache_dict.items():
        if key.startswith(prefix):
            attr_path = key.replace(prefix, "").split('.')
            if len(attr_path) >= 2:
                try:
                    obj = mlx_model
                    for attr in attr_path[:-1]:
                        obj = getattr(obj, attr)
                    setattr(obj, attr_path[-1], value)
                    loaded += 1
                    print(f"   ✅ Loaded {key}")
                except AttributeError:
                    # Skip if attribute doesn't exist
                    continue
    
    print(f">> MLX Commons loaded {loaded} weights from cache")
    return loaded


def load_gpt_layer_weights(mlx_gpt_layer, pytorch_state_dict, prefix=""):
    """
    Specifically load MLXGPTLayer weights.
    
    Args:
        mlx_gpt_layer: MLXGPTLayer instance
        pytorch_state_dict: PyTorch state dict (with numpy arrays)
        prefix: Key prefix
    
    Returns:
        loaded: Number of weights loaded
    """
    loaded = 0
    
    print(f">> Loading MLXGPTLayer weights...")
    
    layers = ['layer0', 'layer1', 'layer2']
    
    for layer_name in layers:
        if hasattr(mlx_gpt_layer, layer_name):
            layer = getattr(mlx_gpt_layer, layer_name)
            
            if f"{prefix}{layer_name}.weight" in pytorch_state_dict:
                layer.weight = mx.array(pytorch_state_dict[f"{prefix}{layer_name}.weight"])
                loaded += 1
                print(f"   ✅ Loaded {layer_name}.weight: {pytorch_state_dict[f'{prefix}{layer_name}.weight'].shape}")
            
            if f"{prefix}{layer_name}.bias" in pytorch_state_dict:
                layer.bias = mx.array(pytorch_state_dict[f"{prefix}{layer_name}.bias"])
                loaded += 1
                print(f"   ✅ Loaded {layer_name}.bias: {pytorch_state_dict[f'{prefix}{layer_name}.bias'].shape}")
    
    print(f">> MLXGPTLayer loaded {loaded} weights")
    return loaded


def load_length_regulator_weights(mlx_length_regulator, pytorch_state_dict, prefix=""):
    """
    Specifically load MLXLengthRegulator weights.
    
    Args:
        mlx_length_regulator: MLXLengthRegulator instance
        pytorch_state_dict: PyTorch state dict (with numpy arrays)
        prefix: Key prefix
    
    Returns:
        loaded: Number of weights loaded
    """
    loaded = 0
    
    print(f">> Loading MLXLengthRegulator weights...")
    
    if hasattr(mlx_length_regulator, 'upsampling_layers'):
        for i, layer in enumerate(mlx_length_regulator.upsampling_layers):
            layer_prefix = f"{prefix}upsampling_layers.{i}."
            
            # Load convolution weights
            if f"{layer_prefix}conv.weight" in pytorch_state_dict:
                layer.conv.weight = mx.array(pytorch_state_dict[f"{layer_prefix}conv.weight"])
                loaded += 1
            
            if f"{layer_prefix}conv.bias" in pytorch_state_dict:
                layer.conv.bias = mx.array(pytorch_state_dict[f"{layer_prefix}conv.bias"])
                loaded += 1
            
            # Load normalization weights
            if f"{layer_prefix}norm.weight" in pytorch_state_dict:
                layer.norm.weight = mx.array(pytorch_state_dict[f"{layer_prefix}norm.weight"])
                loaded += 1
            
            if f"{layer_prefix}norm.bias" in pytorch_state_dict:
                layer.norm.bias = mx.array(pytorch_state_dict[f"{layer_prefix}norm.bias"])
                loaded += 1
    
    print(f">> MLXLengthRegulator loaded {loaded} weights")
    return loaded
