"""
Weight loading utilities for MLX BigVGAN Model
Handles PyTorch to MLX weight conversion for BigVGAN components
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def load_bigvgan_weights(mlx_model, pytorch_state_dict, prefix=""):
    """
    Load BigVGAN weights from PyTorch to MLX.
    
    Args:
        mlx_model: MLX BigVGAN model
        pytorch_state_dict: PyTorch state dict (with numpy arrays)
        prefix: Key prefix
    
    Returns:
        loaded: Number of weights loaded
    """
    loaded = 0
    
    print(f">> Loading BigVGAN weights from PyTorch...")
    
    # 1. Load input projection
    print("   [1/4] Loading input projection...")
    
    if f"{prefix}input_projection.weight" in pytorch_state_dict:
        mlx_model.input_projection.weight = mx.array(pytorch_state_dict[f"{prefix}input_projection.weight"])
        loaded += 1
        print(f"   ✅ Loaded input_projection.weight: {pytorch_state_dict[f'{prefix}input_projection.weight'].shape}")
    
    if f"{prefix}input_projection.bias" in pytorch_state_dict:
        mlx_model.input_projection.bias = mx.array(pytorch_state_dict[f"{prefix}input_projection.bias"])
        loaded += 1
        print(f"   ✅ Loaded input_projection.bias: {pytorch_state_dict[f'{prefix}input_projection.bias'].shape}")
    
    # 2. Load upsampling layers
    print("   [2/4] Loading upsampling layers...")
    
    for i, layer in enumerate(mlx_model.upsampling_layers):
        layer_prefix = f"{prefix}upsampling_layers.{i}."
        
        # Upsampling convolution weights
        if f"{layer_prefix}conv.weight" in pytorch_state_dict:
            layer.conv.weight = mx.array(pytorch_state_dict[f"{layer_prefix}conv.weight"])
            loaded += 1
        
        if f"{layer_prefix}conv.bias" in pytorch_state_dict:
            layer.conv.bias = mx.array(pytorch_state_dict[f"{layer_prefix}conv.bias"])
            loaded += 1
        
        # Batch normalization weights
        if f"{layer_prefix}bn.weight" in pytorch_state_dict:
            layer.bn.weight = mx.array(pytorch_state_dict[f"{layer_prefix}bn.weight"])
            loaded += 1
        
        if f"{layer_prefix}bn.bias" in pytorch_state_dict:
            layer.bn.bias = mx.array(pytorch_state_dict[f"{layer_prefix}bn.bias"])
            loaded += 1
        
        if f"{layer_prefix}bn.running_mean" in pytorch_state_dict:
            layer.bn.running_mean = mx.array(pytorch_state_dict[f"{layer_prefix}bn.running_mean"])
            loaded += 1
        
        if f"{layer_prefix}bn.running_var" in pytorch_state_dict:
            layer.bn.running_var = mx.array(pytorch_state_dict[f"{layer_prefix}bn.running_var"])
            loaded += 1
    
    # 3. Load residual blocks
    print("   [3/4] Loading residual blocks...")
    
    for i, block in enumerate(mlx_model.residual_blocks):
        block_prefix = f"{prefix}residual_blocks.{i}."
        
        # First convolution
        if f"{block_prefix}conv1.weight" in pytorch_state_dict:
            block.conv1.weight = mx.array(pytorch_state_dict[f"{block_prefix}conv1.weight"])
            loaded += 1
        
        if f"{block_prefix}conv1.bias" in pytorch_state_dict:
            block.conv1.bias = mx.array(pytorch_state_dict[f"{block_prefix}conv1.bias"])
            loaded += 1
        
        # Second convolution
        if f"{block_prefix}conv2.weight" in pytorch_state_dict:
            block.conv2.weight = mx.array(pytorch_state_dict[f"{block_prefix}conv2.weight"])
            loaded += 1
        
        if f"{block_prefix}conv2.bias" in pytorch_state_dict:
            block.conv2.bias = mx.array(pytorch_state_dict[f"{block_prefix}conv2.bias"])
            loaded += 1
        
        # Batch normalization
        if f"{block_prefix}bn1.weight" in pytorch_state_dict:
            block.bn1.weight = mx.array(pytorch_state_dict[f"{block_prefix}bn1.weight"])
            loaded += 1
        
        if f"{block_prefix}bn1.bias" in pytorch_state_dict:
            block.bn1.bias = mx.array(pytorch_state_dict[f"{block_prefix}bn1.bias"])
            loaded += 1
        
        if f"{block_prefix}bn2.weight" in pytorch_state_dict:
            block.bn2.weight = mx.array(pytorch_state_dict[f"{block_prefix}bn2.weight"])
            loaded += 1
        
        if f"{block_prefix}bn2.bias" in pytorch_state_dict:
            block.bn2.bias = mx.array(pytorch_state_dict[f"{block_prefix}bn2.bias"])
            loaded += 1
        
        # Skip connection
        if f"{block_prefix}skip.weight" in pytorch_state_dict:
            block.skip.weight = mx.array(pytorch_state_dict[f"{block_prefix}skip.weight"])
            loaded += 1
        
        if f"{block_prefix}skip.bias" in pytorch_state_dict:
            block.skip.bias = mx.array(pytorch_state_dict[f"{block_prefix}skip.bias"])
            loaded += 1
    
    # 4. Load output layers
    print("   [4/4] Loading output layers...")
    
    # Final convolution
    if f"{prefix}final_conv.weight" in pytorch_state_dict:
        mlx_model.final_conv.weight = mx.array(pytorch_state_dict[f"{prefix}final_conv.weight"])
        loaded += 1
        print(f"   ✅ Loaded final_conv.weight: {pytorch_state_dict[f'{prefix}final_conv.weight'].shape}")
    
    if f"{prefix}final_conv.bias" in pytorch_state_dict:
        mlx_model.final_conv.bias = mx.array(pytorch_state_dict[f"{prefix}final_conv.bias"])
        loaded += 1
        print(f"   ✅ Loaded final_conv.bias: {pytorch_state_dict[f'{prefix}final_conv.bias'].shape}")
    
    print(f">> MLX BigVGAN loaded {loaded} weights total")
    return loaded


def load_bigvgan_from_cache(mlx_model, cache_dict, prefix=""):
    """
    Load BigVGAN weights from MLX cache.
    
    Args:
        mlx_model: MLX BigVGAN model
        cache_dict: dict from mx.load() with cached weights
        prefix: Key prefix
    
    Returns:
        loaded: Number of weights loaded
    """
    loaded = 0
    
    print(f">> Loading BigVGAN weights from cache...")
    
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
    
    # Load upsampling layers from cache
    for i in range(len(mlx_model.upsampling_layers)):
        layer_prefix = f"{prefix}upsampling_layers.{i}."
        layer = mlx_model.upsampling_layers[i]
        
        layer_keys = [
            f"{layer_prefix}conv.weight",
            f"{layer_prefix}conv.bias",
            f"{layer_prefix}bn.weight",
            f"{layer_prefix}bn.bias",
            f"{layer_prefix}bn.running_mean",
            f"{layer_prefix}bn.running_var"
        ]
        
        for key in layer_keys:
            if key in cache_dict:
                attr_path = key.replace(f"{layer_prefix}", "").split('.')
                obj = layer
                for attr in attr_path[:-1]:
                    obj = getattr(obj, attr)
                setattr(obj, attr_path[-1], cache_dict[key])
                loaded += 1
    
    # Load residual blocks from cache
    for i in range(len(mlx_model.residual_blocks)):
        block_prefix = f"{prefix}residual_blocks.{i}."
        block = mlx_model.residual_blocks[i]
        
        block_keys = [
            f"{block_prefix}conv1.weight",
            f"{block_prefix}conv1.bias",
            f"{block_prefix}conv2.weight",
            f"{block_prefix}conv2.bias",
            f"{block_prefix}bn1.weight",
            f"{block_prefix}bn1.bias",
            f"{block_prefix}bn2.weight",
            f"{block_prefix}bn2.bias",
            f"{block_prefix}skip.weight",
            f"{block_prefix}skip.bias"
        ]
        
        for key in block_keys:
            if key in cache_dict:
                attr_path = key.replace(f"{block_prefix}", "").split('.')
                obj = block
                for attr in attr_path[:-1]:
                    obj = getattr(obj, attr)
                setattr(obj, attr_path[-1], cache_dict[key])
                loaded += 1
    
    # Load output layers from cache
    output_keys = [
        f"{prefix}final_conv.weight",
        f"{prefix}final_conv.bias"
    ]
    
    for key in output_keys:
        if key in cache_dict:
            attr_name = key.split('.')[-1]
            if hasattr(mlx_model, 'final_conv'):
                setattr(mlx_model.final_conv, attr_name, cache_dict[key])
                loaded += 1
                print(f"   ✅ Loaded {key}")
    
    print(f">> MLX BigVGAN loaded {loaded} weights from cache")
    return loaded


def load_bigvgan_complete_weights(mlx_model, pytorch_state_dict, prefix=""):
    """
    Load BigVGAN Complete weights from PyTorch to MLX.
    
    Args:
        mlx_model: MLX BigVGAN Complete model
        pytorch_state_dict: PyTorch state dict (with numpy arrays)
        prefix: Key prefix
    
    Returns:
        loaded: Number of weights loaded
    """
    loaded = 0
    
    print(f">> Loading BigVGAN Complete weights from PyTorch...")
    
    # Load basic BigVGAN weights first
    loaded += load_bigvgan_weights(mlx_model, pytorch_state_dict, prefix)
    
    # Load additional complete components
    print("   [5/5] Loading complete components...")
    
    # Additional layers specific to complete version
    if f"{prefix}complete_layer.weight" in pytorch_state_dict:
        mlx_model.complete_layer.weight = mx.array(pytorch_state_dict[f"{prefix}complete_layer.weight"])
        loaded += 1
        print(f"   ✅ Loaded complete_layer.weight: {pytorch_state_dict[f'{prefix}complete_layer.weight'].shape}")
    
    print(f">> MLX BigVGAN Complete loaded {loaded} weights total")
    return loaded
