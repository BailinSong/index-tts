"""
MLX Implementation of GPT-fast Transformer for DiT
Based on Meta's gpt-fast implementation
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
import math


def precompute_freqs_cis_mlx(seq_len: int, n_elem: int, base: int = 10000):
    """
    Precompute RoPE frequencies.
    
    Returns:
        freqs_cis: (seq_len, n_elem//2, 2) containing [cos, sin] values
    """
    freqs = 1.0 / (base ** (mx.arange(0, n_elem, 2, dtype=mx.float32)[: (n_elem // 2)] / n_elem))
    t = mx.arange(seq_len, dtype=mx.float32)
    # Outer product: (seq_len,) x (n_elem//2,) -> (seq_len, n_elem//2)
    freqs_outer = t.reshape(-1, 1) * freqs.reshape(1, -1)
    
    # Compute cos and sin
    freqs_cos = mx.cos(freqs_outer)
    freqs_sin = mx.sin(freqs_outer)
    
    # Stack as (seq_len, n_elem//2, 2) where [..., 0] is cos, [..., 1] is sin
    freqs_cis = mx.stack([freqs_cos, freqs_sin], axis=-1)
    
    return freqs_cis


def apply_rotary_emb_mlx(x, freqs_cis):
    """
    Apply rotary position embedding.
    
    Args:
        x: (batch, seq_len, n_heads, head_dim)
        freqs_cis: (seq_len, head_dim//2, 2) containing [cos, sin]
    
    Returns:
        x_rotated: same shape as x
    """
    # Reshape x to separate real/imag parts: (batch, seq_len, n_heads, head_dim//2, 2)
    xshaped = x.reshape(*x.shape[:-1], -1, 2)
    
    # Reshape freqs_cis to broadcast: (1, seq_len, 1, head_dim//2, 2)
    freqs_cis = freqs_cis.reshape(1, freqs_cis.shape[0], 1, freqs_cis.shape[1], 2)
    
    # Apply rotation
    # Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    x_out_real = xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1]
    x_out_imag = xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1]
    
    x_out = mx.stack([x_out_real, x_out_imag], axis=-1)
    x_out = x_out.reshape(*x.shape)
    
    return x_out


class MLXRMSNorm(nn.Module):
    """MLX implementation of RMSNorm"""
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))
    
    def __call__(self, x):
        """
        x: (..., dim)
        """
        # RMS: sqrt(mean(x^2))
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        x_normed = x / rms
        return x_normed * self.weight


class MLXAdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization conditioned on timestep embedding.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.project_layer = nn.Linear(d_model, 2 * d_model, bias=True)
        self.norm = MLXRMSNorm(d_model, eps=eps)
        self.eps = eps
    
    def __call__(self, x, embedding=None):
        """
        x: (batch, seq_len, d_model)
        embedding: (batch, d_model) - timestep embedding
        
        If embedding is None, just apply regular normalization.
        """
        if embedding is None:
            return self.norm(x)
        
        # Project embedding to get weight and bias
        projected = self.project_layer(embedding)  # (batch, ..., 2*d_model)
        
        # Split along last dimension
        weight = projected[..., :self.d_model]
        bias = projected[..., self.d_model:]
        
        # Ensure correct shape for broadcasting with x (batch, seq_len, d_model)
        # If weight is (batch, d_model), reshape to (batch, 1, d_model)
        # If weight is (batch, 1, d_model), keep as is
        if weight.ndim == 2:
            weight = weight[:, None, :]  # (batch, d_model) -> (batch, 1, d_model)
            bias = bias[:, None, :]
        
        return weight * self.norm(x) + bias


class MLXAttentionGPTFast(nn.Module):
    """
    MLX implementation of GPT-fast Attention with RoPE.
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        n_local_heads: int = None,
        is_cross_attention: bool = False,
        context_dim: int = 0
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_local_heads = n_local_heads or n_heads
        self.is_cross_attention = is_cross_attention
        
        if is_cross_attention:
            # Cross-attention: Q from x, KV from context
            self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
            self.wkv = nn.Linear(context_dim, 2 * self.n_local_heads * head_dim, bias=False)
        else:
            # Self-attention: QKV from x
            total_head_dim = (n_heads + 2 * self.n_local_heads) * head_dim
            self.wqkv = nn.Linear(dim, total_head_dim, bias=False)
        
        self.wo = nn.Linear(head_dim * n_heads, dim, bias=False)
        
        self.kv_cache = None  # Will be set if using cache
    
    def __call__(
        self,
        x,
        freqs_cis,
        mask=None,
        input_pos=None,
        context=None,
        context_freqs_cis=None
    ):
        """
        x: (batch, seq_len, dim)
        freqs_cis: (seq_len, head_dim//2, 2)
        mask: (batch, 1, seq_len, seq_len) or None
        context: (batch, context_len, context_dim) for cross-attention
        """
        batch, seq_len, _ = x.shape
        
        kv_size = self.n_local_heads * self.head_dim
        
        if self.is_cross_attention and context is not None:
            # Cross-attention
            q = self.wq(x)  # (batch, seq_len, n_heads * head_dim)
            kv = self.wkv(context)  # (batch, context_len, 2 * n_local_heads * head_dim)
            k, v = mx.split(kv, 2, axis=-1)  # Each: (batch, context_len, kv_size)
            context_seq_len = context.shape[1]
        else:
            # Self-attention
            qkv = self.wqkv(x)  # (batch, seq_len, total_head_dim)
            q, k, v = mx.split(qkv, [kv_size, kv_size * 2], axis=-1)
            # q: (batch, seq_len, kv_size)
            # k: (batch, seq_len, kv_size)
            # v: (batch, seq_len, kv_size)
            context_seq_len = seq_len
        
        # Reshape for multi-head
        q = q.reshape(batch, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch, context_seq_len, self.n_local_heads, self.head_dim)
        v = v.reshape(batch, context_seq_len, self.n_local_heads, self.head_dim)
        
        # Apply RoPE
        freqs_to_use = context_freqs_cis if context_freqs_cis is not None else freqs_cis
        q = apply_rotary_emb_mlx(q, freqs_cis)
        k = apply_rotary_emb_mlx(k, freqs_to_use)
        
        # Transpose to (batch, n_heads, seq_len, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Repeat k,v if n_local_heads < n_heads (Grouped Query Attention)
        if self.n_local_heads != self.n_heads:
            repeat_factor = self.n_heads // self.n_local_heads
            # Repeat along head dimension
            k = mx.repeat(k, repeat_factor, axis=1)
            v = mx.repeat(v, repeat_factor, axis=1)
        
        # Scaled dot-product attention
        # scores = Q @ K^T / sqrt(head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            # mask: True = allowed, False = masked
            # scores: apply -inf to masked positions
            scores = mx.where(mask, scores, float('-inf'))
        
        # Softmax
        attn_weights = mx.softmax(scores, axis=-1)
        
        # Apply to values
        output = attn_weights @ v  # (batch, n_heads, seq_len, head_dim)
        
        # Reshape back
        output = output.transpose(0, 2, 1, 3)  # (batch, seq_len, n_heads, head_dim)
        output = output.reshape(batch, seq_len, self.n_heads * self.head_dim)
        
        # Output projection
        output = self.wo(output)
        
        return output


class MLXFeedForward(nn.Module):
    """
    MLX implementation of SwiGLU feedforward (GPT-fast style).
    """
    
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_size, bias=False)
        self.w3 = nn.Linear(dim, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, dim, bias=False)
    
    def __call__(self, x):
        """
        SwiGLU: w2(silu(w1(x)) * w3(x))
        """
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class MLXTransformerBlock(nn.Module):
    """
    MLX implementation of GPT-fast TransformerBlock with AdaLN.
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        intermediate_size: int,
        n_local_heads: int = None,
        eps: float = 1e-5,
        has_cross_attention: bool = False,
        context_dim: int = 0,
        uvit_skip_connection: bool = False,
        time_as_token: bool = False
    ):
        super().__init__()
        
        self.dim = dim
        self.has_cross_attention = has_cross_attention
        self.uvit_skip_connection = uvit_skip_connection
        self.time_as_token = time_as_token
        
        # Self-attention
        self.attention = MLXAttentionGPTFast(
            dim, n_heads, head_dim, n_local_heads or n_heads,
            is_cross_attention=False
        )
        self.attention_norm = MLXAdaptiveLayerNorm(dim, eps=eps)
        
        # Feed-forward
        self.feed_forward = MLXFeedForward(dim, intermediate_size)
        self.ffn_norm = MLXAdaptiveLayerNorm(dim, eps=eps)
        
        # Cross-attention (if needed)
        if has_cross_attention:
            self.cross_attention = MLXAttentionGPTFast(
                dim, n_heads, head_dim, n_local_heads or n_heads,
                is_cross_attention=True,
                context_dim=context_dim
            )
            self.cross_attention_norm = MLXAdaptiveLayerNorm(dim, eps=eps)
        
        # U-ViT skip connection
        if uvit_skip_connection:
            self.skip_in_linear = nn.Linear(dim * 2, dim, bias=True)
    
    def __call__(
        self,
        x,
        c,
        input_pos,
        freqs_cis,
        mask,
        context=None,
        context_freqs_cis=None,
        cross_attention_mask=None,
        skip_in_x=None
    ):
        """
        x: (batch, seq_len, dim)
        c: (batch, dim) - conditioning (timestep embedding), None if time_as_token
        freqs_cis: (seq_len, head_dim//2, 2)
        mask: (batch, 1, seq_len, seq_len)
        skip_in_x: (batch, seq_len, dim) for U-ViT
        """
        # Time as token mode: don't use c for AdaLN
        c_for_norm = None if self.time_as_token else c
        
        # U-ViT skip connection input
        if self.uvit_skip_connection and skip_in_x is not None:
            x = self.skip_in_linear(mx.concatenate([x, skip_in_x], axis=-1))
        
        # Self-attention with residual
        h = x + self.attention(
            self.attention_norm(x, c_for_norm),
            freqs_cis,
            mask,
            input_pos
        )
        
        # Cross-attention (if enabled)
        if self.has_cross_attention and context is not None:
            h = h + self.cross_attention(
                self.cross_attention_norm(h, c_for_norm),
                freqs_cis,
                cross_attention_mask,
                input_pos,
                context,
                context_freqs_cis
            )
        
        # Feed-forward with residual
        out = h + self.feed_forward(self.ffn_norm(h, c_for_norm))
        
        return out


class MLXTransformerGPTFast(nn.Module):
    """
    MLX implementation of GPT-fast Transformer.
    """
    
    def __init__(
        self,
        dim: int = 512,
        n_layer: int = 13,
        n_head: int = 8,
        head_dim: int = 64,
        intermediate_size: int = 2048,
        n_local_heads: int = None,
        block_size: int = 16384,
        rope_base: int = 10000,
        norm_eps: float = 1e-5,
        has_cross_attention: bool = False,
        context_dim: int = 0,
        uvit_skip_connection: bool = False,
        time_as_token: bool = False
    ):
        super().__init__()
        
        self.dim = dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.head_dim = head_dim
        self.uvit_skip_connection = uvit_skip_connection
        self.time_as_token = time_as_token
        
        n_local_heads = n_local_heads or n_head
        
        # Transformer blocks
        self.layers = [
            MLXTransformerBlock(
                dim, n_head, head_dim, intermediate_size,
                n_local_heads, norm_eps,
                has_cross_attention, context_dim,
                uvit_skip_connection, time_as_token
            )
            for _ in range(n_layer)
        ]
        
        # Final norm (AdaptiveLayerNorm)
        self.norm = MLXAdaptiveLayerNorm(dim, eps=norm_eps)
        
        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis_mlx(block_size, head_dim, rope_base)
        
        # U-ViT skip connection indices
        if uvit_skip_connection:
            self.layers_emit_skip = [i for i in range(n_layer) if i < n_layer // 2]
            self.layers_receive_skip = [i for i in range(n_layer) if i > n_layer // 2]
        else:
            self.layers_emit_skip = []
            self.layers_receive_skip = []
        
        print(f">> MLX TransformerGPTFast initialized:")
        print(f"   Layers: {n_layer}, Heads: {n_head}, Dim: {dim}")
        print(f"   Head dim: {head_dim}, Intermediate: {intermediate_size}")
        print(f"   U-ViT skip: {uvit_skip_connection}")
    
    def __call__(
        self,
        x,
        c,
        input_pos=None,
        mask=None,
        context=None,
        context_input_pos=None,
        cross_attention_mask=None
    ):
        """
        x: (batch, seq_len, dim)
        c: (batch, dim) - conditioning (timestep embedding)
        input_pos: (seq_len,) - position indices
        mask: (batch, 1, seq_len, seq_len)
        """
        # Get position indices if not provided
        if input_pos is None:
            input_pos = mx.arange(x.shape[1], dtype=mx.int32)
        
        # Get freqs for current positions
        freqs_cis = self.freqs_cis[input_pos]  # (seq_len, head_dim//2, 2)
        
        if context is not None and context_input_pos is not None:
            context_freqs_cis = self.freqs_cis[context_input_pos]
        else:
            context_freqs_cis = None
        
        # U-ViT skip connections
        skip_in_x_list = []
        
        for i, layer in enumerate(self.layers):
            # Get skip input if needed
            if self.uvit_skip_connection and i in self.layers_receive_skip:
                skip_in_x = skip_in_x_list.pop(-1) if skip_in_x_list else None
            else:
                skip_in_x = None
            
            # Forward through layer
            x = layer(
                x, c, input_pos, freqs_cis, mask,
                context, context_freqs_cis, cross_attention_mask,
                skip_in_x
            )
            
            # Save skip output if needed
            if self.uvit_skip_connection and i in self.layers_emit_skip:
                skip_in_x_list.append(x)
        
        # Final normalization
        x = self.norm(x, c)
        
        return x


def create_mlx_transformer_from_config(config):
    """
    Create MLX Transformer from DiT config.
    """
    dit_cfg = config.DiT
    
    # Calculate intermediate size (SwiGLU uses 2/3 * 4 * dim)
    hidden_dim = 4 * dit_cfg.hidden_dim
    n_hidden = int(2 * hidden_dim / 3)
    # Round to multiple of 256 for efficiency
    intermediate_size = ((n_hidden + 255) // 256) * 256
    
    return MLXTransformerGPTFast(
        dim=dit_cfg.hidden_dim,
        n_layer=dit_cfg.depth,
        n_head=dit_cfg.num_heads,
        head_dim=dit_cfg.hidden_dim // dit_cfg.num_heads,
        intermediate_size=intermediate_size,
        n_local_heads=dit_cfg.num_heads,  # Same as n_head for DiT
        block_size=16384,
        rope_base=10000,
        norm_eps=1e-5,
        has_cross_attention=False,
        uvit_skip_connection=dit_cfg.get('uvit_skip_connection', False),
        time_as_token=dit_cfg.get('time_as_token', False)
    )

