#!/usr/bin/env python3
"""
MLX implementation of Conformer Encoder and Perceiver Resampler for conditioning.

This module provides native MLX implementations of the complex conditioning pipeline
used in IndexTTS2, replacing PyTorch's Conformer + PerceiverResampler.
"""

import mlx.core as mx
import mlx.nn as nn
import math


# ============================================================================
# Helper Functions
# ============================================================================

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# ============================================================================
# RMSNorm (used in Perceiver)
# ============================================================================

class MLXRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = mx.ones((dim,))
    
    def __call__(self, x):
        # x: (batch, seq, dim)
        norm = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x / norm * self.scale


# ============================================================================
# Attention (used in Perceiver)
# ============================================================================

class MLXPerceiverAttention(nn.Module):
    """
    Cross-attention module for Perceiver Resampler.
    
    Args:
        dim: Model dimension
        dim_head: Dimension per attention head
        heads: Number of attention heads
        cross_attn_include_queries: Whether to include queries in context
    """
    
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        cross_attn_include_queries: bool = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.cross_attn_include_queries = cross_attn_include_queries
        
        dim_inner = dim_head * heads
        
        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = nn.Linear(dim, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)
    
    def __call__(self, x, context=None, mask=None):
        """
        Args:
            x: Query tensor (batch, n_latents, dim)
            context: Key/Value context (batch, seq_len, dim)
            mask: Optional attention mask
        
        Returns:
            Attended output (batch, n_latents, dim)
        """
        h = self.heads
        has_context = exists(context)
        
        context = default(context, x)
        
        # Include queries in context if specified (Perceiver cross-attention style)
        if has_context and self.cross_attn_include_queries:
            context = mx.concatenate([x, context], axis=1)
        
        # Project to Q, K, V
        q = self.to_q(x)  # (batch, n_latents, dim_inner)
        kv = self.to_kv(context)  # (batch, context_len, dim_inner * 2)
        k, v = mx.split(kv, 2, axis=-1)
        
        # Reshape for multi-head: (batch, heads, seq, dim_per_head)
        batch, n_q, _ = q.shape
        _, n_kv, _ = k.shape
        dim_per_head = q.shape[-1] // h
        
        q = q.reshape(batch, n_q, h, dim_per_head).transpose(0, 2, 1, 3)
        k = k.reshape(batch, n_kv, h, dim_per_head).transpose(0, 2, 1, 3)
        v = v.reshape(batch, n_kv, h, dim_per_head).transpose(0, 2, 1, 3)
        
        # Attention: (batch, heads, n_q, n_kv)
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = mx.where(mask, scores, -10000.0)
        
        attn = mx.softmax(scores, axis=-1)
        
        # Apply to values
        out = attn @ v  # (batch, heads, n_q, dim_per_head)
        
        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(batch, n_q, -1)
        
        return self.to_out(out)


# ============================================================================
# Feed-Forward (used in Perceiver)
# ============================================================================

class MLXGEGLU(nn.Module):
    """Gated GLU activation."""
    
    def __call__(self, x):
        x, gate = mx.split(x, 2, axis=-1)
        return x * nn.gelu(gate)


class MLXFeedForward(nn.Module):
    """Feed-forward network with GEGLU activation."""
    
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        dim_inner = int(dim * mult)
        
        self.net = [
            nn.Linear(dim, dim_inner * 2),
            MLXGEGLU(),
            nn.Linear(dim_inner, dim)
        ]
    
    def __call__(self, x):
        for layer in self.net:
            x = layer(x)
        return x


# ============================================================================
# Perceiver Resampler
# ============================================================================

class MLXPerceiverResampler(nn.Module):
    """
    Perceiver Resampler for compressing variable-length features to fixed latents.
    
    This is a key component of the conditioning pipeline, taking the output from
    the Conformer encoder and compressing it to a fixed number of latent vectors.
    
    Args:
        dim: Model dimension (1280 for IndexTTS2)
        depth: Number of cross-attention + FF layers (default 2)
        dim_context: Input context dimension (if different from dim)
        num_latents: Number of output latent vectors (32 for IndexTTS2)
        dim_head: Dimension per attention head
        heads: Number of attention heads
        ff_mult: Feed-forward expansion multiplier
    """
    
    def __init__(
        self,
        dim: int,
        depth: int = 2,
        dim_context: int = None,
        num_latents: int = 32,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4
    ):
        super().__init__()
        dim_context = default(dim_context, dim)
        
        # Project context to model dim if needed
        if dim_context != dim:
            self.proj_context = nn.Linear(dim_context, dim)
        else:
            self.proj_context = None
        
        # Learnable latent queries
        self.latents = mx.random.normal((num_latents, dim)) * 0.02
        
        # Cross-attention + FF layers
        self.layers = []
        for _ in range(depth):
            self.layers.append([
                MLXPerceiverAttention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    cross_attn_include_queries=True
                ),
                MLXFeedForward(dim=dim, mult=ff_mult)
            ])
        
        self.norm = MLXRMSNorm(dim)
    
    def __call__(self, x, mask=None):
        """
        Args:
            x: Input features (batch, seq_len, dim_context)
            mask: Optional mask for padding
        
        Returns:
            Compressed latents (batch, num_latents, dim)
        """
        batch = x.shape[0]
        
        # Project context if needed
        if self.proj_context is not None:
            x = self.proj_context(x)
        
        # Expand latents for batch
        latents = mx.broadcast_to(
            self.latents.reshape(1, *self.latents.shape),
            (batch, *self.latents.shape)
        )
        
        # Apply cross-attention + FF layers
        for attn, ff in self.layers:
            latents = attn(latents, x, mask=mask) + latents
            latents = ff(latents) + latents
        
        return self.norm(latents)


# ============================================================================
# Simplified Conformer Components
# ============================================================================

class MLXRelativeMultiHeadAttention(nn.Module):
    """
    Relative position multi-head attention for Conformer.
    Implements Transformer-XL style attention with learnable position biases.
    Paper: https://arxiv.org/abs/1901.02860
    """
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Positional encoding projection (no bias)
        self.pos_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # ✅ NEW: Learnable position biases (content and position)
        # These are used in matrix C and matrix D as described in Transformer-XL paper
        self.pos_bias_u = mx.random.normal((self.num_heads, self.head_dim)) * 0.02
        self.pos_bias_v = mx.random.normal((self.num_heads, self.head_dim)) * 0.02
    
    def __call__(self, x, pos_emb, mask=None):
        """
        Args:
            x: Input (batch, seq, embed_dim)
            pos_emb: Positional embeddings (batch, seq, embed_dim) or (seq, embed_dim)
            mask: Optional attention mask
        
        Returns:
            Attended output (batch, seq, embed_dim)
        """
        batch, seq_len, _ = x.shape
        
        # Handle pos_emb broadcast if needed
        if len(pos_emb.shape) == 2:
            # pos_emb is (seq, embed_dim), broadcast to (batch, seq, embed_dim)
            pos_emb = mx.broadcast_to(
                pos_emb.reshape(1, seq_len, self.embed_dim),
                (batch, seq_len, self.embed_dim)
            )
        
        # Project Q, K, V
        # Shape: (batch, num_heads, seq_len, head_dim)
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Project positional embeddings
        # Shape: (batch, num_heads, seq_len, head_dim)
        p = self.pos_proj(pos_emb).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # ✅ RELATIVE POSITIONAL ATTENTION (Transformer-XL style)
        # Compute q + pos_bias_u and q + pos_bias_v
        # pos_bias_u/v: (num_heads, head_dim) -> need to broadcast to (batch, num_heads, seq_len, head_dim)
        pos_bias_u_broadcast = self.pos_bias_u.reshape(1, self.num_heads, 1, self.head_dim)
        pos_bias_v_broadcast = self.pos_bias_v.reshape(1, self.num_heads, 1, self.head_dim)
        
        q_with_bias_u = q + pos_bias_u_broadcast  # (batch, num_heads, seq_len, head_dim)
        q_with_bias_v = q + pos_bias_v_broadcast  # (batch, num_heads, seq_len, head_dim)
        
        # Matrix AC: content-based attention
        # (batch, num_heads, seq_len, seq_len)
        matrix_ac = q_with_bias_u @ k.transpose(0, 1, 3, 2)
        
        # Matrix BD: position-based attention
        # (batch, num_heads, seq_len, seq_len)
        matrix_bd = q_with_bias_v @ p.transpose(0, 1, 3, 2)
        
        # NOTE: PyTorch Conformer removed rel_shift() for speech recognition
        # "Remove rel_shift since it is useless in speech recognition"
        # So we don't apply rel_shift here
        
        # Combine and scale
        scores = (matrix_ac + matrix_bd) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = mx.where(mask, scores, -10000.0)
        
        attn = mx.softmax(scores, axis=-1)
        
        # Apply to values
        out = attn @ v
        
        # Reshape and project
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.embed_dim)
        return self.out_proj(out)


class MLXDepthwiseConv1d(nn.Module):
    """
    Depthwise 1D Convolution using MLX's native Conv1d with groups.
    
    This matches PyTorch's depthwise convolution: Conv1d with groups=channels.
    Each input channel is convolved with its own kernel.
    """
    
    def __init__(self, channels: int, kernel_size: int, padding: int = 0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        # ✅ Use MLX's native Conv1d with groups=channels for depthwise convolution
        # This is much more efficient and numerically accurate than manual implementation
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=channels,  # This makes it depthwise!
            bias=True
        )
    
    @property
    def weight(self):
        """Expose weight for loading from checkpoint"""
        return self.conv.weight
    
    @weight.setter
    def weight(self, value):
        """Set weight, handling format conversion if needed"""
        self.conv.weight = value
    
    @property
    def bias(self):
        return self.conv.bias
    
    @bias.setter
    def bias(self, value):
        self.conv.bias = value
    
    def __call__(self, x):
        """
        Args:
            x: (batch, seq, channels) - MLX format
        Returns:
            out: (batch, seq, channels)
        """
        # MLX Conv1d expects (batch, seq, in_channels) which is what we have
        return self.conv(x)


class MLXConvolutionModule(nn.Module):
    """
    Conformer Convolution Module with proper depthwise separable convolution.
    
    Architecture:
        1. LayerNorm
        2. Pointwise expansion with GLU
        3. Depthwise convolution
        4. BatchNorm (LayerNorm in MLX)
        5. Swish activation
        6. Pointwise projection
    """
    
    def __init__(self, channels: int, kernel_size: int = 31):
        super().__init__()
        
        # Pointwise expansion (for GLU: 2x channels)
        self.pointwise1 = nn.Linear(channels, 2 * channels)
        
        # Depthwise convolution
        padding = kernel_size // 2
        self.depthwise = MLXDepthwiseConv1d(channels, kernel_size, padding)
        
        # Norm after depthwise (matches PyTorch)
        self.bn = nn.LayerNorm(channels, eps=1e-05)  # eps匹配PyTorch
        
        # Pointwise projection
        self.pointwise2 = nn.Linear(channels, channels)
    
    def __call__(self, x, mask_pad=None):
        """
        Args:
            x: Input (batch, seq, channels)
            mask_pad: Optional padding mask
        
        Returns:
            Output (batch, seq, channels)
        """
        # ✅ FIX: 按照PyTorch顺序
        # 1. Pointwise expansion (NOT norm first!)
        x = self.pointwise1(x)  # (batch, seq, 2*channels)
        
        # 2. GLU: split and gate
        x1, x2 = mx.split(x, 2, axis=-1)
        x = x1 * nn.sigmoid(x2)  # (batch, seq, channels)
        
        # 3. Depthwise convolution
        x = self.depthwise(x)
        
        # 4. Norm (after depthwise)
        x = self.bn(x)
        
        # 5. Swish activation
        x = x * nn.sigmoid(x)
        
        # 6. Pointwise projection
        x = self.pointwise2(x)
        
        # Apply mask if provided
        if mask_pad is not None:
            # mask_pad could be (batch, 1, 1, seq) or (batch, 1, seq) depending on source
            # Squeeze all size-1 dimensions to get (batch, seq)
            mask_squeezed = mask_pad
            while len(mask_squeezed.shape) > 2:
                # Find and squeeze size-1 dimensions
                for i in range(1, len(mask_squeezed.shape)):
                    if mask_squeezed.shape[i] == 1:
                        mask_squeezed = mask_squeezed.squeeze(i)
                        break
                else:
                    break  # No more size-1 dims
            
            # mask_squeezed should now be (batch, seq_len)
            mask_expanded = mask_squeezed.reshape(mask_squeezed.shape[0], mask_squeezed.shape[1], 1)
            x = x * mask_expanded
        
        return x


class MLXConformerBlock(nn.Module):
    """
    Conformer encoder block.
    
    Architecture (WITHOUT macaron style, matching PyTorch):
        1. Multi-head self-attention with relative positional encoding
        2. Convolution module
        3. Feed-forward
        4. Final norm
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ff_mult: int = 4,
        conv_kernel_size: int = 15,
        dropout: float = 0.1
    ):
        super().__init__()
        
        ff_dim = dim * ff_mult
        
        # Multi-head attention
        self.norm_attn = nn.LayerNorm(dim, eps=1e-05)  # norm_mha in PyTorch, eps匹配
        self.attn = MLXRelativeMultiHeadAttention(dim, num_heads)
        
        # Convolution module
        self.norm_conv = nn.LayerNorm(dim, eps=1e-05)  # eps匹配PyTorch
        self.conv = MLXConvolutionModule(dim, conv_kernel_size)
        
        # Feed-forward (只有一个！)
        self.norm_ff = nn.LayerNorm(dim, eps=1e-05)  # eps匹配PyTorch
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.SiLU(),
            nn.Linear(ff_dim, dim)
        )
        
        self.norm_final = nn.LayerNorm(dim, eps=1e-05)  # eps匹配PyTorch
    
    def __call__(self, x, pos_emb, mask=None, mask_pad=None):
        """
        Args:
            x: Input (batch, seq, dim)
            pos_emb: Positional embeddings
            mask: Attention mask
            mask_pad: Padding mask
        
        Returns:
            Output (batch, seq, dim)
        """
        # ✅ FIXED: 匹配 PyTorch 的非 macaron 结构
        # PyTorch (without macaron): Attention → Conv → FF → Final Norm
        
        # 1. Multi-head self-attention
        residual = x
        x = self.norm_attn(x)
        x = self.attn(x, pos_emb, mask)
        x = residual + x
        
        # 2. Convolution module
        residual = x
        x = self.norm_conv(x)
        x = self.conv(x, mask_pad)
        x = residual + x
        
        # 3. Feed-forward
        residual = x
        x = self.norm_ff(x)
        x = self.ff(x)
        x = residual + x  # ← ff_scale = 1.0 (no macaron)
        
        # 4. Final normalization
        x = self.norm_final(x)
        
        return x


# ============================================================================
# Conformer Encoder
# ============================================================================

class MLXConformerEncoder(nn.Module):
    """
    Conformer encoder for speech feature extraction.
    
    Args:
        input_dim: Input feature dimension (e.g., 1024 for speaker embeddings)
        output_dim: Output dimension (512 for IndexTTS2)
        num_layers: Number of Conformer blocks (default 6)
        num_heads: Number of attention heads
        ff_mult: Feed-forward expansion multiplier
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_mult: int = 4,
        conv_kernel_size: int = 15
    ):
        super().__init__()
        
        self.output_dim = output_dim
        
        # ✅ FIX: Use Conv2d Subsampling to match PyTorch (121 → 60)
        from indextts.gpt.mlx_subsampling import MLXConv2dSubsampling2Fixed
        self.subsampling = MLXConv2dSubsampling2Fixed(input_dim, output_dim)
        
        # ✅ FIX: xscale for positional encoding (sqrt(d_model))
        import math
        self.xscale = math.sqrt(output_dim)  # sqrt(512) = 22.627
        
        # Positional encoding (will be loaded from checkpoint)
        max_len = 1000
        self.pos_encoding = mx.random.normal((max_len, output_dim)) * 0.02
        
        # Conformer blocks
        self.blocks = []
        for _ in range(num_layers):
            self.blocks.append(
                MLXConformerBlock(
                    dim=output_dim,
                    num_heads=num_heads,
                    ff_mult=ff_mult,
                    conv_kernel_size=conv_kernel_size
                )
            )
        
        # Final norm (matches PyTorch after_norm)
        self.after_norm = nn.LayerNorm(output_dim, eps=1e-05)  # eps匹配PyTorch
    
    def __call__(self, x, lengths=None):
        """
        Args:
            x: Input features (batch, seq_len, input_dim)
            lengths: Sequence lengths for masking (batch,)
        
        Returns:
            Encoded features (batch, seq_len', output_dim) where seq_len' = seq_len // 2
            Mask (batch, seq_len')
        """
        # ✅ FIX: Apply Conv2d subsampling (seq_len → seq_len // 2)
        x = self.subsampling(x)  # (batch, seq_len//2, output_dim)
        
        # ✅ FIX: Apply xscale ONLY (matches PyTorch)
        # PyTorch: x = x * xscale (pos_emb is separate!)
        x = x * self.xscale  # ✅ CRITICAL: xscale = sqrt(512) = 22.627
        
        # ✅ FIX: Get positional encoding (separate from x)
        seq_len = x.shape[1]
        pos_emb = self.pos_encoding[:seq_len]
        pos_emb = mx.broadcast_to(
            pos_emb.reshape(1, seq_len, self.output_dim),
            (x.shape[0], seq_len, self.output_dim)
        )
        
        # Create mask if lengths provided (adjust for subsampling)
        mask = None
        mask_pad = None
        if lengths is not None:
            # Lengths are halved due to subsampling
            lengths_subsampled = lengths // 2
            batch_size = x.shape[0]
            positions = mx.arange(seq_len).reshape(1, -1)
            lengths_expanded = lengths_subsampled.reshape(-1, 1)
            mask_pad = positions < lengths_expanded  # (batch, seq)
            mask_pad = mask_pad.reshape(batch_size, 1, 1, seq_len)  # For attention
        
        # Apply Conformer blocks
        for block in self.blocks:
            x = block(x, pos_emb, mask, mask_pad)
        
        # Final normalization (matches PyTorch after_norm)
        x = self.after_norm(x)
        
        # Return encoded features and mask
        # For speaker conditioning, mask_pad remains None (not used)
        # For emotion conditioning, we need to return the actual mask
        # PyTorch ConformerEncoder returns (encoded, mask) where mask shape is (batch, 1, seq_len)
        if mask_pad is not None:
            # Reshape mask_pad from (batch, 1, 1, seq_len) to (batch, 1, seq_len) for compatibility
            return_mask = mask_pad.squeeze(2)  # (batch, 1, seq_len)
        else:
            # For speaker conditioning (32 latents), no mask is typically returned
            return_mask = None
        
        return x, return_mask


# ============================================================================
# Complete Conditioning Module
# ============================================================================

class MLXConditioningModule(nn.Module):
    """
    Complete conditioning module: Conformer Encoder + Perceiver Resampler.
    
    This replaces the PyTorch conditioning pipeline with a pure MLX implementation.
    
    Architecture matches PyTorch:
        - Conformer: input_dim (1024) → conformer_dim (512)
        - Perceiver: conformer_dim (512) → model_dim (1280) via proj_context
        - Output: (batch, num_latents=32, model_dim=1280)
    
    Args:
        input_dim: Input dimension (1024 for speaker embeddings)
        conformer_dim: Conformer output dimension (512 to match PyTorch)
        model_dim: Final model dimension (1280 for IndexTTS2)
        num_latents: Number of output latents (32 for IndexTTS2)
        conformer_layers: Number of Conformer layers (6)
        perceiver_depth: Depth of Perceiver Resampler (2)
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        conformer_dim: int = 512,  # ← Changed to match PyTorch!
        model_dim: int = 1280,
        num_latents: int = 32,
        conformer_layers: int = 6,
        conformer_heads: int = 8,      # ← NEW: attention heads
        conformer_ff_mult: int = 4,    # ← NEW: feed-forward multiplier
        perceiver_depth: int = 2,
        perceiver_heads: int = 8,      # ← NEW: perceiver attention heads
        perceiver_ff_mult: int = 4     # ← NEW: perceiver feed-forward multiplier
    ):
        super().__init__()
        
        # Store num_latents for mask processing
        self.num_latents = num_latents
        
        # Conformer: 1024 → 512 (matches PyTorch)
        self.conformer = MLXConformerEncoder(
            input_dim=input_dim,
            output_dim=conformer_dim,  # 512
            num_layers=conformer_layers,
            num_heads=conformer_heads,  # Pass from config
            ff_mult=conformer_ff_mult   # Pass from config
        )
        
        # Perceiver: 512 → 1280 (with proj_context)
        self.perceiver = MLXPerceiverResampler(
            dim=model_dim,  # 1280
            depth=perceiver_depth,
            dim_context=conformer_dim,  # 512 from Conformer
            num_latents=num_latents,
            heads=perceiver_heads,      # Pass from config
            ff_mult=perceiver_ff_mult   # Pass from config
        )
    
    def __call__(self, x, lengths=None):
        """
        Args:
            x: Input features (batch, seq_len, input_dim)
            lengths: Sequence lengths (batch,)
        
        Returns:
            Conditioning latents (batch, num_latents, model_dim)
        """
        import mlx.core as mx
        
        # Conformer encoding
        x, mask = self.conformer(x, lengths)
        
        # Process mask for perceiver (matches PyTorch cond_mask_pad)
        # PyTorch: conds_mask = self.cond_mask_pad(mask.squeeze(1))
        # cond_mask_pad = nn.ConstantPad1d((num_latents, 0), True)
        if mask is not None:
            mask_squeezed = mask.squeeze(1) if len(mask.shape) > 2 and mask.shape[1] == 1 else mask
            # Add num_latents True values at the beginning
            padding = mx.ones((mask_squeezed.shape[0], self.num_latents), dtype=mx.bool_)
            conds_mask = mx.concatenate([padding, mask_squeezed], axis=1)
        else:
            # If no mask, create one for all positions
            batch_size, seq_len, _ = x.shape
            conds_mask = mx.ones((batch_size, self.num_latents + seq_len), dtype=mx.bool_)
        
        # Perceiver resampling with processed mask
        latents = self.perceiver(x, conds_mask)
        
        return latents

