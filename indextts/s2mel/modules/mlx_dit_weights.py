"""
Weight loading utilities for MLX DiT
Handles PyTorch to MLX weight conversion
"""

import mlx.core as mx
import numpy as np


def load_weight_norm(state_dict, prefix):
    """
    Load weight from weight_g and weight_v (weight normalization).
    
    Args:
        state_dict: PyTorch state dict (numpy arrays)
        prefix: Key prefix (e.g., "x_embedder")
    
    Returns:
        weight: Reconstructed weight
    """
    g_key = f"{prefix}.weight_g"
    v_key = f"{prefix}.weight_v"
    
    if g_key in state_dict and v_key in state_dict:
        g = state_dict[g_key]
        v = state_dict[v_key]
        
        # Compute weight: w = g * v / ||v||
        # Norm over all dims except first (output dim)
        axes = tuple(range(1, len(v.shape)))
        norm_v = np.linalg.norm(v, axis=axes, keepdims=True)
        w = g * v / (norm_v + 1e-8)
        
        return w
    elif f"{prefix}.weight" in state_dict:
        # Already removed weight_norm
        return state_dict[f"{prefix}.weight"]
    else:
        return None


def convert_conv1d_weight(w):
    """PyTorch Conv1d (O, I, K) -> MLX (O, K, I)"""
    if len(w.shape) == 3:
        return w.transpose(0, 2, 1)
    return w


def load_dit_weights(mlx_dit, pytorch_state_dict, prefix="models.cfm.estimator."):
    """
    Load DiT weights from PyTorch to MLX.
    
    Args:
        mlx_dit: MLX DiT model
        pytorch_state_dict: PyTorch state dict (with numpy arrays)
        prefix: Key prefix
    
    Returns:
        loaded: Number of weights loaded
    """
    loaded = 0
    
    print(f">> Loading DiT weights from PyTorch...")
    
    # 1. Load embedders
    print("   [1/5] Loading embedders...")
    
    # x_embedder (with weight_norm)
    w = load_weight_norm(pytorch_state_dict, f"{prefix}x_embedder")
    if w is not None:
        mlx_dit.x_embedder.weight = mx.array(w)
        loaded += 1
    if f"{prefix}x_embedder.bias" in pytorch_state_dict:
        mlx_dit.x_embedder.bias = mx.array(pytorch_state_dict[f"{prefix}x_embedder.bias"])
        loaded += 1
    
    # cond_projection
    if f"{prefix}cond_projection.weight" in pytorch_state_dict:
        mlx_dit.cond_projection.weight = mx.array(pytorch_state_dict[f"{prefix}cond_projection.weight"])
        loaded += 1
    if f"{prefix}cond_projection.bias" in pytorch_state_dict:
        mlx_dit.cond_projection.bias = mx.array(pytorch_state_dict[f"{prefix}cond_projection.bias"])
        loaded += 1
    
    # t_embedder
    if f"{prefix}t_embedder.freqs" in pytorch_state_dict:
        mlx_dit.t_embedder.freqs = mx.array(pytorch_state_dict[f"{prefix}t_embedder.freqs"])
        loaded += 1
    if f"{prefix}t_embedder.mlp.0.weight" in pytorch_state_dict:
        mlx_dit.t_embedder.mlp_0.weight = mx.array(pytorch_state_dict[f"{prefix}t_embedder.mlp.0.weight"])
        loaded += 1
    if f"{prefix}t_embedder.mlp.0.bias" in pytorch_state_dict:
        mlx_dit.t_embedder.mlp_0.bias = mx.array(pytorch_state_dict[f"{prefix}t_embedder.mlp.0.bias"])
        loaded += 1
    if f"{prefix}t_embedder.mlp.2.weight" in pytorch_state_dict:
        mlx_dit.t_embedder.mlp_2.weight = mx.array(pytorch_state_dict[f"{prefix}t_embedder.mlp.2.weight"])
        loaded += 1
    if f"{prefix}t_embedder.mlp.2.bias" in pytorch_state_dict:
        mlx_dit.t_embedder.mlp_2.bias = mx.array(pytorch_state_dict[f"{prefix}t_embedder.mlp.2.bias"])
        loaded += 1
    
    # cond_x_merge_linear
    if f"{prefix}cond_x_merge_linear.weight" in pytorch_state_dict:
        mlx_dit.cond_x_merge_linear.weight = mx.array(pytorch_state_dict[f"{prefix}cond_x_merge_linear.weight"])
        loaded += 1
    if f"{prefix}cond_x_merge_linear.bias" in pytorch_state_dict:
        mlx_dit.cond_x_merge_linear.bias = mx.array(pytorch_state_dict[f"{prefix}cond_x_merge_linear.bias"])
        loaded += 1
    
    # skip_linear
    if mlx_dit.long_skip_connection:
        if f"{prefix}skip_linear.weight" in pytorch_state_dict:
            mlx_dit.skip_linear.weight = mx.array(pytorch_state_dict[f"{prefix}skip_linear.weight"])
            loaded += 1
        if f"{prefix}skip_linear.bias" in pytorch_state_dict:
            mlx_dit.skip_linear.bias = mx.array(pytorch_state_dict[f"{prefix}skip_linear.bias"])
            loaded += 1
    
    # 2. Load Transformer layers
    print("   [2/5] Loading Transformer layers...")
    
    n_layers = mlx_dit.depth
    for layer_idx in range(n_layers):
        layer = mlx_dit.transformer.layers[layer_idx]
        pt_prefix = f"{prefix}transformer.layers.{layer_idx}"
        
        # Attention: wqkv (combined Q, K, V)
        wqkv_key = f"{pt_prefix}.attention.wqkv.weight"
        if wqkv_key in pytorch_state_dict:
            wqkv = pytorch_state_dict[wqkv_key]  # (total_head_dim, dim)
            # Split into q, k, v
            kv_size = layer.attention.n_local_heads * layer.attention.head_dim
            q_w = wqkv[:kv_size, :]
            k_w = wqkv[kv_size:kv_size*2, :]
            v_w = wqkv[kv_size*2:, :]
            
            layer.attention.wqkv.weight = mx.array(wqkv)
            loaded += 1
        
        # Attention output
        wo_key = f"{pt_prefix}.attention.wo.weight"
        if wo_key in pytorch_state_dict:
            layer.attention.wo.weight = mx.array(pytorch_state_dict[wo_key])
            loaded += 1
        
        # FeedForward
        for weight_name in ['w1', 'w2', 'w3']:
            w_key = f"{pt_prefix}.feed_forward.{weight_name}.weight"
            if w_key in pytorch_state_dict:
                setattr(
                    layer.feed_forward,
                    weight_name,
                    nn.Linear(1, 1, bias=False)  # Placeholder
                )
                getattr(layer.feed_forward, weight_name).weight = mx.array(pytorch_state_dict[w_key])
                loaded += 1
        
        # AdaptiveLayerNorm for attention
        aln_prefix = f"{pt_prefix}.attention_norm"
        if f"{aln_prefix}.project_layer.weight" in pytorch_state_dict:
            layer.attention_norm.project_layer.weight = mx.array(pytorch_state_dict[f"{aln_prefix}.project_layer.weight"])
            loaded += 1
        if f"{aln_prefix}.project_layer.bias" in pytorch_state_dict:
            layer.attention_norm.project_layer.bias = mx.array(pytorch_state_dict[f"{aln_prefix}.project_layer.bias"])
            loaded += 1
        if f"{aln_prefix}.norm.weight" in pytorch_state_dict:
            layer.attention_norm.norm.weight = mx.array(pytorch_state_dict[f"{aln_prefix}.norm.weight"])
            loaded += 1
        
        # AdaptiveLayerNorm for FFN
        ffn_prefix = f"{pt_prefix}.ffn_norm"
        if f"{ffn_prefix}.project_layer.weight" in pytorch_state_dict:
            layer.ffn_norm.project_layer.weight = mx.array(pytorch_state_dict[f"{ffn_prefix}.project_layer.weight"])
            loaded += 1
        if f"{ffn_prefix}.project_layer.bias" in pytorch_state_dict:
            layer.ffn_norm.project_layer.bias = mx.array(pytorch_state_dict[f"{ffn_prefix}.project_layer.bias"])
            loaded += 1
        if f"{ffn_prefix}.norm.weight" in pytorch_state_dict:
            layer.ffn_norm.norm.weight = mx.array(pytorch_state_dict[f"{ffn_prefix}.norm.weight"])
            loaded += 1
        
        # U-ViT skip connection
        if mlx_dit.transformer.uvit_skip_connection and hasattr(layer, 'skip_in_linear'):
            skip_key = f"{pt_prefix}.skip_in_linear.weight"
            if skip_key in pytorch_state_dict:
                layer.skip_in_linear.weight = mx.array(pytorch_state_dict[skip_key])
                loaded += 1
            skip_bias_key = f"{pt_prefix}.skip_in_linear.bias"
            if skip_bias_key in pytorch_state_dict:
                layer.skip_in_linear.bias = mx.array(pytorch_state_dict[skip_bias_key])
                loaded += 1
    
    # 3. Load final transformer norm
    print("   [3/5] Loading final norm...")
    
    norm_prefix = f"{prefix}transformer.norm"
    if f"{norm_prefix}.project_layer.weight" in pytorch_state_dict:
        mlx_dit.transformer.norm.project_layer.weight = mx.array(pytorch_state_dict[f"{norm_prefix}.project_layer.weight"])
        loaded += 1
    if f"{norm_prefix}.project_layer.bias" in pytorch_state_dict:
        mlx_dit.transformer.norm.project_layer.bias = mx.array(pytorch_state_dict[f"{norm_prefix}.project_layer.bias"])
        loaded += 1
    if f"{norm_prefix}.norm.weight" in pytorch_state_dict:
        mlx_dit.transformer.norm.norm.weight = mx.array(pytorch_state_dict[f"{norm_prefix}.norm.weight"])
        loaded += 1
    
    # 4. Load final layer (WaveNet or MLP)
    print("   [4/5] Loading final layer...")
    
    if mlx_dit.final_layer_type == 'wavenet':
        # Load WaveNet weights
        loaded += load_wavenet_weights(mlx_dit, pytorch_state_dict, prefix)
    else:
        # Load MLP final layer
        if f"{prefix}final_mlp.0.weight" in pytorch_state_dict:
            mlx_dit.final_mlp_0.weight = mx.array(pytorch_state_dict[f"{prefix}final_mlp.0.weight"])
            loaded += 1
        if f"{prefix}final_mlp.0.bias" in pytorch_state_dict:
            mlx_dit.final_mlp_0.bias = mx.array(pytorch_state_dict[f"{prefix}final_mlp.0.bias"])
            loaded += 1
        if f"{prefix}final_mlp.2.weight" in pytorch_state_dict:
            mlx_dit.final_mlp_2.weight = mx.array(pytorch_state_dict[f"{prefix}final_mlp.2.weight"])
            loaded += 1
        if f"{prefix}final_mlp.2.bias" in pytorch_state_dict:
            mlx_dit.final_mlp_2.bias = mx.array(pytorch_state_dict[f"{prefix}final_mlp.2.bias"])
            loaded += 1
    
    # 5. Load content mask embedder
    print("   [5/5] Loading misc...")
    
    if f"{prefix}content_mask_embedder.weight" in pytorch_state_dict:
        mlx_dit.content_mask_embedder.weight = mx.array(pytorch_state_dict[f"{prefix}content_mask_embedder.weight"])
        loaded += 1
    
    print(f">> Loaded {loaded} DiT weights total")
    return loaded


def load_wavenet_weights(mlx_dit, pytorch_state_dict, prefix):
    """Load WaveNet-specific weights"""
    loaded = 0
    wn_prefix = f"{prefix}"
    
    # t_embedder2
    if f"{wn_prefix}t_embedder2.mlp.0.weight" in pytorch_state_dict:
        mlx_dit.t_embedder2.mlp_0.weight = mx.array(pytorch_state_dict[f"{wn_prefix}t_embedder2.mlp.0.weight"])
        loaded += 1
    if f"{wn_prefix}t_embedder2.mlp.0.bias" in pytorch_state_dict:
        mlx_dit.t_embedder2.mlp_0.bias = mx.array(pytorch_state_dict[f"{wn_prefix}t_embedder2.mlp.0.bias"])
        loaded += 1
    if f"{wn_prefix}t_embedder2.mlp.2.weight" in pytorch_state_dict:
        mlx_dit.t_embedder2.mlp_2.weight = mx.array(pytorch_state_dict[f"{wn_prefix}t_embedder2.mlp.2.weight"])
        loaded += 1
    if f"{wn_prefix}t_embedder2.mlp.2.bias" in pytorch_state_dict:
        mlx_dit.t_embedder2.mlp_2.bias = mx.array(pytorch_state_dict[f"{wn_prefix}t_embedder2.mlp.2.bias"])
        loaded += 1
    if f"{wn_prefix}t_embedder2.freqs" in pytorch_state_dict:
        mlx_dit.t_embedder2.freqs = mx.array(pytorch_state_dict[f"{wn_prefix}t_embedder2.freqs"])
        loaded += 1
    
    # conv1
    if f"{wn_prefix}conv1.weight" in pytorch_state_dict:
        mlx_dit.conv1.weight = mx.array(pytorch_state_dict[f"{wn_prefix}conv1.weight"])
        loaded += 1
    if f"{wn_prefix}conv1.bias" in pytorch_state_dict:
        mlx_dit.conv1.bias = mx.array(pytorch_state_dict[f"{wn_prefix}conv1.bias"])
        loaded += 1
    
    # conv2 (Conv1d)
    w2 = load_weight_norm(pytorch_state_dict, f"{wn_prefix}conv2")
    if w2 is not None:
        mlx_dit.conv2.weight = mx.array(convert_conv1d_weight(w2))
        loaded += 1
    if f"{wn_prefix}conv2.bias" in pytorch_state_dict:
        mlx_dit.conv2.bias = mx.array(pytorch_state_dict[f"{wn_prefix}conv2.bias"])
        loaded += 1
    
    # res_projection
    if f"{wn_prefix}res_projection.weight" in pytorch_state_dict:
        mlx_dit.res_projection.weight = mx.array(pytorch_state_dict[f"{wn_prefix}res_projection.weight"])
        loaded += 1
    if f"{wn_prefix}res_projection.bias" in pytorch_state_dict:
        mlx_dit.res_projection.bias = mx.array(pytorch_state_dict[f"{wn_prefix}res_projection.bias"])
        loaded += 1
    
    # WaveNet layers
    n_layers = mlx_dit.wavenet.n_layers
    for i in range(n_layers):
        layer_prefix = f"{wn_prefix}wavenet.in_layers.{i}.conv.conv"
        
        # Load with weight_norm
        w = load_weight_norm(pytorch_state_dict, layer_prefix)
        if w is not None:
            mlx_dit.wavenet.in_layers[i].weight = mx.array(convert_conv1d_weight(w))
            loaded += 1
        if f"{layer_prefix}.bias" in pytorch_state_dict:
            mlx_dit.wavenet.in_layers[i].bias = mx.array(pytorch_state_dict[f"{layer_prefix}.bias"])
            loaded += 1
        
        # res_skip layers
        res_prefix = f"{wn_prefix}wavenet.res_skip_layers.{i}.conv.conv"
        w_res = load_weight_norm(pytorch_state_dict, res_prefix)
        if w_res is not None:
            mlx_dit.wavenet.res_skip_layers[i].weight = mx.array(convert_conv1d_weight(w_res))
            loaded += 1
        if f"{res_prefix}.bias" in pytorch_state_dict:
            mlx_dit.wavenet.res_skip_layers[i].bias = mx.array(pytorch_state_dict[f"{res_prefix}.bias"])
            loaded += 1
    
    # WaveNet cond_layer (if exists)
    if mlx_dit.wavenet.cond_layer is not None:
        cond_prefix = f"{wn_prefix}wavenet.cond_layer.conv.conv"
        w_cond = load_weight_norm(pytorch_state_dict, cond_prefix)
        if w_cond is not None:
            mlx_dit.wavenet.cond_layer.weight = mx.array(convert_conv1d_weight(w_cond))
            loaded += 1
        if f"{cond_prefix}.bias" in pytorch_state_dict:
            mlx_dit.wavenet.cond_layer.bias = mx.array(pytorch_state_dict[f"{cond_prefix}.bias"])
            loaded += 1
    
    # final_layer (FinalLayer with AdaLN)
    final_prefix = f"{wn_prefix}final_layer"
    
    # linear (with weight_norm)
    w_final = load_weight_norm(pytorch_state_dict, f"{final_prefix}.linear")
    if w_final is not None:
        mlx_dit.final_layer.linear.weight = mx.array(w_final)
        loaded += 1
    if f"{final_prefix}.linear.bias" in pytorch_state_dict:
        mlx_dit.final_layer.linear.bias = mx.array(pytorch_state_dict[f"{final_prefix}.linear.bias"])
        loaded += 1
    
    # adaLN_modulation
    if f"{final_prefix}.adaLN_modulation.1.weight" in pytorch_state_dict:
        mlx_dit.final_layer.adaLN_0.weight = mx.array(pytorch_state_dict[f"{final_prefix}.adaLN_modulation.1.weight"])
        loaded += 1
    if f"{final_prefix}.adaLN_modulation.1.bias" in pytorch_state_dict:
        mlx_dit.final_layer.adaLN_0.bias = mx.array(pytorch_state_dict[f"{final_prefix}.adaLN_modulation.1.bias"])
        loaded += 1
    
    return loaded

