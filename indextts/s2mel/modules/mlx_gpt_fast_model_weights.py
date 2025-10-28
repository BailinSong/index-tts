"""
Weight loading utilities for MLX GPT Fast Model
Handles PyTorch to MLX weight conversion for GPT components
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def load_gpt_fast_weights(mlx_model, pytorch_state_dict, prefix=""):
    """
    Load GPT Fast weights from PyTorch to MLX.
    
    Args:
        mlx_model: MLX GPT Fast model
        pytorch_state_dict: PyTorch state dict (with numpy arrays)
        prefix: Key prefix
    
    Returns:
        loaded: Number of weights loaded
    """
    loaded = 0
    
    print(f">> Loading GPT Fast weights from PyTorch...")
    
    # 1. Load embedding weights
    print("   [1/4] Loading embeddings...")
    
    # token_embedding
    if f"{prefix}token_embedding.weight" in pytorch_state_dict:
        mlx_model.token_embedding.weight = mx.array(pytorch_state_dict[f"{prefix}token_embedding.weight"])
        loaded += 1
        print(f"   ✅ Loaded token_embedding.weight: {pytorch_state_dict[f'{prefix}token_embedding.weight'].shape}")
    
    # position_embedding
    if f"{prefix}position_embedding.weight" in pytorch_state_dict:
        mlx_model.position_embedding.weight = mx.array(pytorch_state_dict[f"{prefix}position_embedding.weight"])
        loaded += 1
        print(f"   ✅ Loaded position_embedding.weight: {pytorch_state_dict[f'{prefix}position_embedding.weight'].shape}")
    
    # 2. Load transformer blocks
    print("   [2/4] Loading transformer blocks...")
    
    for i, block in enumerate(mlx_model.transformer_blocks):
        block_prefix = f"{prefix}transformer_blocks.{i}."
        
        # Attention weights
        if f"{block_prefix}attention.w_q.weight" in pytorch_state_dict:
            block.attention.w_q.weight = mx.array(pytorch_state_dict[f"{block_prefix}attention.w_q.weight"])
            loaded += 1
        
        if f"{block_prefix}attention.w_k.weight" in pytorch_state_dict:
            block.attention.w_k.weight = mx.array(pytorch_state_dict[f"{block_prefix}attention.w_k.weight"])
            loaded += 1
        
        if f"{block_prefix}attention.w_v.weight" in pytorch_state_dict:
            block.attention.w_v.weight = mx.array(pytorch_state_dict[f"{block_prefix}attention.w_v.weight"])
            loaded += 1
        
        if f"{block_prefix}attention.w_o.weight" in pytorch_state_dict:
            block.attention.w_o.weight = mx.array(pytorch_state_dict[f"{block_prefix}attention.w_o.weight"])
            loaded += 1
        
        # FeedForward weights
        if f"{block_prefix}feedforward.w1.weight" in pytorch_state_dict:
            block.feedforward.w1.weight = mx.array(pytorch_state_dict[f"{block_prefix}feedforward.w1.weight"])
            loaded += 1
        
        if f"{block_prefix}feedforward.w2.weight" in pytorch_state_dict:
            block.feedforward.w2.weight = mx.array(pytorch_state_dict[f"{block_prefix}feedforward.w2.weight"])
            loaded += 1
        
        if f"{block_prefix}feedforward.w3.weight" in pytorch_state_dict:
            block.feedforward.w3.weight = mx.array(pytorch_state_dict[f"{block_prefix}feedforward.w3.weight"])
            loaded += 1
        
        # Layer norms
        if f"{block_prefix}attention_norm.weight" in pytorch_state_dict:
            block.attention_norm.weight = mx.array(pytorch_state_dict[f"{block_prefix}attention_norm.weight"])
            loaded += 1
        
        if f"{block_prefix}ffn_norm.weight" in pytorch_state_dict:
            block.ffn_norm.weight = mx.array(pytorch_state_dict[f"{block_prefix}ffn_norm.weight"])
            loaded += 1
        
        # AdaLN weights
        if f"{block_prefix}ada_ln.project_layer.weight" in pytorch_state_dict:
            block.ada_ln.project_layer.weight = mx.array(pytorch_state_dict[f"{block_prefix}ada_ln.project_layer.weight"])
            loaded += 1
        
        if f"{block_prefix}ada_ln.project_layer.bias" in pytorch_state_dict:
            block.ada_ln.project_layer.bias = mx.array(pytorch_state_dict[f"{block_prefix}ada_ln.project_layer.bias"])
            loaded += 1
    
    # 3. Load final layer norm
    print("   [3/4] Loading final layer norm...")
    
    if f"{prefix}final_norm.weight" in pytorch_state_dict:
        mlx_model.final_norm.weight = mx.array(pytorch_state_dict[f"{prefix}final_norm.weight"])
        loaded += 1
        print(f"   ✅ Loaded final_norm.weight: {pytorch_state_dict[f'{prefix}final_norm.weight'].shape}")
    
    # 4. Load output head
    print("   [4/4] Loading output head...")
    
    if f"{prefix}output_head.weight" in pytorch_state_dict:
        mlx_model.output_head.weight = mx.array(pytorch_state_dict[f"{prefix}output_head.weight"])
        loaded += 1
        print(f"   ✅ Loaded output_head.weight: {pytorch_state_dict[f'{prefix}output_head.weight'].shape}")
    
    print(f">> MLX GPT Fast loaded {loaded} weights total")
    return loaded


def load_gpt_fast_from_cache(mlx_model, cache_dict, prefix=""):
    """
    Load GPT Fast weights from MLX cache.
    
    Args:
        mlx_model: MLX GPT Fast model
        cache_dict: dict from mx.load() with cached weights
        prefix: Key prefix
    
    Returns:
        loaded: Number of weights loaded
    """
    loaded = 0
    
    print(f">> Loading GPT Fast weights from cache...")
    
    # Load weights directly from cache
    weight_keys = [
        f"{prefix}token_embedding.weight",
        f"{prefix}position_embedding.weight",
        f"{prefix}final_norm.weight",
        f"{prefix}output_head.weight"
    ]
    
    for key in weight_keys:
        if key in cache_dict:
            # Extract the attribute name from the key
            attr_name = key.split('.')[-2] if '.' in key else key.split('.')[-1]
            if hasattr(mlx_model, attr_name):
                setattr(mlx_model, attr_name, cache_dict[key])
                loaded += 1
                print(f"   ✅ Loaded {key}")
    
    # Load transformer blocks from cache
    for i in range(len(mlx_model.transformer_blocks)):
        block_prefix = f"{prefix}transformer_blocks.{i}."
        block = mlx_model.transformer_blocks[i]
        
        # Load attention weights
        attention_keys = [
            f"{block_prefix}attention.w_q.weight",
            f"{block_prefix}attention.w_k.weight", 
            f"{block_prefix}attention.w_v.weight",
            f"{block_prefix}attention.w_o.weight"
        ]
        
        for key in attention_keys:
            if key in cache_dict:
                attr_path = key.replace(f"{block_prefix}", "").split('.')
                obj = block
                for attr in attr_path[:-1]:
                    obj = getattr(obj, attr)
                setattr(obj, attr_path[-1], cache_dict[key])
                loaded += 1
        
        # Load feedforward weights
        ff_keys = [
            f"{block_prefix}feedforward.w1.weight",
            f"{block_prefix}feedforward.w2.weight",
            f"{block_prefix}feedforward.w3.weight"
        ]
        
        for key in ff_keys:
            if key in cache_dict:
                attr_path = key.replace(f"{block_prefix}", "").split('.')
                obj = block
                for attr in attr_path[:-1]:
                    obj = getattr(obj, attr)
                setattr(obj, attr_path[-1], cache_dict[key])
                loaded += 1
        
        # Load layer norms
        norm_keys = [
            f"{block_prefix}attention_norm.weight",
            f"{block_prefix}ffn_norm.weight"
        ]
        
        for key in norm_keys:
            if key in cache_dict:
                attr_path = key.replace(f"{block_prefix}", "").split('.')
                obj = block
                for attr in attr_path[:-1]:
                    obj = getattr(obj, attr)
                setattr(obj, attr_path[-1], cache_dict[key])
                loaded += 1
    
    print(f">> MLX GPT Fast loaded {loaded} weights from cache")
    return loaded
