"""
Native MLX Implementation of GPT Model for Apple Silicon M4

Full Transformer implementation with proper attention and feed-forward layers.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
import math


class MLXLinear(nn.Module):
    """MLX Linear layer matching PyTorch behavior."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        scale = (1.0 / in_features) ** 0.5
        self.weight = mx.random.uniform(-scale, scale, (out_features, in_features))
        self.bias = mx.zeros(out_features)
    
    def __call__(self, x):
        return x @ self.weight.T + self.bias


class MLXEmbedding(nn.Module):
    """MLX Embedding layer."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = mx.random.normal((num_embeddings, embedding_dim)) * 0.02
    
    def __call__(self, indices):
        return self.weight[indices]


class MLXMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention for MLX.
    Implements scaled dot-product attention with multiple heads.
    """
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections
        self.q_proj = MLXLinear(embed_dim, embed_dim)
        self.k_proj = MLXLinear(embed_dim, embed_dim)
        self.v_proj = MLXLinear(embed_dim, embed_dim)
        self.out_proj = MLXLinear(embed_dim, embed_dim)
    
    def __call__(self, x, causal_mask=None, past_kv=None, use_cache=False):
        """
        Multi-head attention with KV caching support.
        
        Args:
            x: Input (batch, seq_len, embed_dim)
            causal_mask: Optional causal attention bias (bool array, True=allowed, False=masked)
            past_kv: Tuple of (past_key, past_value) cache
            use_cache: If True, return updated (key, value) cache
        
        Returns:
            output: Attention output
            cache: Updated (key, value) if use_cache=True, else None
        """
        batch, seq_len, _ = x.shape
        
        # Project and reshape for multi-head
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Use cached K, V if available (for autoregressive generation)
        kv_seq_len = k.shape[2]
        if past_kv is not None:
            past_k, past_v = past_kv
            k = mx.concatenate([past_k, k], axis=2)  # Concat along seq dimension
            v = mx.concatenate([past_v, v], axis=2)
            kv_seq_len = k.shape[2]
        
        # Attention scores
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        
        # Apply causal mask (GPT2 style: True=allowed, False=masked)
        if causal_mask is not None:
            # When using KV cache (seq_len=1), we're at position kv_seq_len-1
            # and can attend to all previous positions
            if past_kv is not None and seq_len == 1:
                # Current position is kv_seq_len - 1
                current_pos = kv_seq_len - 1
                # Extract mask for current position: can attend to positions [0, kv_seq_len)
                mask_slice = causal_mask[:, :, current_pos:current_pos+1, :kv_seq_len]
            else:
                # No cache or full sequence: extract mask for all query positions
                mask_slice = causal_mask[:, :, :seq_len, :kv_seq_len]
            
            # Convert bool mask to attention mask: True -> 0.0, False -> -10000.0
            attn_mask = mx.where(mask_slice, 0.0, -10000.0)
            scores = scores + attn_mask
        
        # Softmax and apply to values
        attn_weights = mx.softmax(scores, axis=-1)
        attn_output = attn_weights @ v
        
        # Reshape and project
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        
        if use_cache:
            return output, (k, v)
        return output


class MLXTransformerBlock(nn.Module):
    """
    Transformer block with self-attention and feed-forward network.
    """
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = MLXMultiHeadAttention(embed_dim, num_heads)
        self.ln_2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.mlp_fc = MLXLinear(embed_dim, embed_dim * 4)
        self.mlp_proj = MLXLinear(embed_dim * 4, embed_dim)
    
    def __call__(self, x, causal_mask=None, past_kv=None, use_cache=False):
        """
        Transformer block forward with KV caching.
        
        Args:
            x: Input tensor
            causal_mask: Optional causal attention mask
            past_kv: Cached (key, value) from previous step
            use_cache: Whether to return updated cache
            
        Returns:
            x: Output tensor
            cache: Updated (key, value) if use_cache=True
        """
        # Self-attention with residual
        attn_out = self.attn(self.ln_1(x), causal_mask=causal_mask, past_kv=past_kv, use_cache=use_cache)
        
        if use_cache:
            attn_out, new_kv = attn_out
            x = x + attn_out
        else:
            x = x + attn_out
        
        # Feed-forward with residual and NewGELU activation
        h = self.mlp_fc(self.ln_2(x))
        # NewGELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # NOTE: Can produce small negative values, don't clip to 0!
        h = 0.5 * h * (1 + mx.tanh(math.sqrt(2 / math.pi) * (h + 0.044715 * h ** 3)))
        x = x + self.mlp_proj(h)
        
        if use_cache:
            return x, new_kv
        return x


class UnifiedVoiceMLX(nn.Module):
    """
    Full MLX GPT model with complete Transformer architecture.
    
    Includes all transformer layers for proper weight loading and inference.
    Can optionally use pure MLX conditioning (Conformer + Perceiver) or accept
    pre-computed conditioning from PyTorch.
    """
    
    def __init__(
        self,
        layers=24,
        model_dim=1280,
        heads=20,
        number_text_tokens=12000,
        number_mel_codes=8194,
        start_mel_token=8192,
        stop_mel_token=8193,
        use_mlx_conditioning=True,  # NEW: Enable pure MLX conditioning
        **kwargs  # Accept extra args for compatibility
    ):
        """
        Initialize full MLX GPT model.
        
        Args:
            layers: Number of transformer layers (default 24)
            model_dim: Model dimension (default 1280)
            heads: Number of attention heads (default 20)
            number_text_tokens: Text vocabulary size
            number_mel_codes: Mel code vocabulary size
            start_mel_token: Start token for mel
            stop_mel_token: Stop token for mel
            use_mlx_conditioning: If True, use pure MLX conditioning pipeline
        """
        super().__init__()
        
        self.layers = layers
        self.model_dim = model_dim
        self.heads = heads
        self.number_mel_codes = number_mel_codes
        self.start_mel_token = start_mel_token
        self.stop_mel_token = stop_mel_token
        self.use_mlx_conditioning = use_mlx_conditioning
        
        # Pure MLX conditioning (Conformer + Perceiver)
        if use_mlx_conditioning:
            from indextts.gpt.mlx_conditioning import MLXConditioningModule
            self.conditioning_module = MLXConditioningModule(
                input_dim=1024,         # Speaker embedding dimension
                conformer_dim=512,      # Conformer output (matches PyTorch)
                model_dim=model_dim,    # 1280
                num_latents=32,         # Output 32 conditioning latents
                conformer_layers=6,     # 6 layers (matches PyTorch)
                perceiver_depth=2       # 2 layers (matches PyTorch)
            )
            print(">> MLX: Using pure MLX conditioning (Conformer 512D + Perceiver 1280D)")
        else:
            self.conditioning_module = None
        
        # Speed embeddings (for duration control)
        self.speed_emb = MLXEmbedding(2, model_dim)
        
        # Emotion layer (for emotion vector processing)
        self.emo_layer = nn.Linear(model_dim, model_dim)
        
        # Core embeddings
        self.text_embedding = MLXEmbedding(number_text_tokens + 1, model_dim)
        self.mel_embedding = MLXEmbedding(number_mel_codes, model_dim)
        
        # Transformer layers (GPT2 style)
        self.transformer_blocks = [
            MLXTransformerBlock(model_dim, heads) for _ in range(layers)
        ]
        
        # Positional embeddings (learned)
        max_mel_tokens = kwargs.get('max_mel_tokens', 1815)
        max_text_tokens = kwargs.get('max_text_tokens', 600)
        self.mel_pos_embedding = MLXEmbedding(max_mel_tokens, model_dim)
        self.text_pos_embedding = MLXEmbedding(max_text_tokens + 2, model_dim)
        
        # Conditioning parameters
        self.cond_num = kwargs.get('condition_num_latent', 32)
        
        # Speed/duration embeddings
        self.speed_emb = MLXEmbedding(2, model_dim)
        
        # Emotion and speaker conditioning layers
        # Simplified: use linear projections instead of full ConformerEncoder
        self.emo_layer = MLXLinear(model_dim, model_dim)
        self.emovec_layer = MLXLinear(1024, model_dim)
        self.cond_projection = MLXLinear(1024, model_dim)  # Project semantic features
        
        # Output heads
        self.mel_head = MLXLinear(model_dim, number_mel_codes)
        self.text_head = MLXLinear(model_dim, number_text_tokens + 1)
        
        # Normalization
        self.final_norm = nn.LayerNorm(model_dim)
        
        # Create causal attention mask (GPT2 style)
        # Maximum sequence length for mask (must be >= max context length in training)
        max_positions = 2420  # Match PyTorch GPT2 default
        # Create lower triangular matrix: True for allowed positions, False for masked
        causal_mask = mx.tril(mx.ones((max_positions, max_positions), dtype=mx.bool_))
        # Add batch and head dimensions: (1, 1, max_pos, max_pos)
        self.causal_mask = causal_mask.reshape(1, 1, max_positions, max_positions)
        
        print(f">> Initialized UnifiedVoiceMLX (full transformer, layers={layers}, dim={model_dim}, heads={heads})")
        print(f"   Conditioning: {self.cond_num} latents, Positional: mel={max_mel_tokens}, text={max_text_tokens}")
        print(f"   Causal mask: {max_positions}x{max_positions} (GPT2 style)")
    
    def load_weights_from_dict(self, mlx_weights):
        """
        Load weights from MLX npz cache including all transformer layers.
        
        Args:
            mlx_weights: Dictionary of MLX arrays from cache
        """
        loaded = 0
        
        # Load embeddings and heads
        # Note: Position embeddings use .emb.weight in PyTorch checkpoint
        simple_mappings = {
            'text_embedding.weight': ('text_embedding', 'weight'),
            'mel_embedding.weight': ('mel_embedding', 'weight'),
            'mel_pos_embedding.emb.weight': ('mel_pos_embedding', 'weight'),  # FIXED: added .emb
            'text_pos_embedding.emb.weight': ('text_pos_embedding', 'weight'),  # FIXED: added .emb
            'mel_head.weight': ('mel_head', 'weight'),
            'mel_head.bias': ('mel_head', 'bias'),
            'text_head.weight': ('text_head', 'weight'),
            'text_head.bias': ('text_head', 'bias'),
            'final_norm.weight': ('final_norm', 'weight'),
            'final_norm.bias': ('final_norm', 'bias'),
            'speed_emb.weight': ('speed_emb', 'weight'),
            'emo_layer.weight': ('emo_layer', 'weight'),
            'emo_layer.bias': ('emo_layer', 'bias'),
            'emovec_layer.weight': ('emovec_layer', 'weight'),
            'emovec_layer.bias': ('emovec_layer', 'bias'),
        }
        
        for key, (module_name, attr_name) in simple_mappings.items():
            if key in mlx_weights:
                module = getattr(self, module_name)
                setattr(module, attr_name, mlx_weights[key])
                loaded += 1
        
        # Load transformer layers
        # PyTorch GPT2 format: gpt.h.{layer_idx}.{component}.{param}
        for layer_idx in range(self.layers):
            block = self.transformer_blocks[layer_idx]
            prefix = f"gpt.h.{layer_idx}"
            
            # Attention weights (c_attn combines q,k,v projections in PyTorch)
            c_attn_weight = f"{prefix}.attn.c_attn.weight"
            c_attn_bias = f"{prefix}.attn.c_attn.bias"
            
            if c_attn_weight in mlx_weights:
                # Split combined qkv weight into separate q, k, v
                # PyTorch GPT2 c_attn.weight shape: (in_features, out_features) = (D, 3*D)
                # This is already in the correct format from checkpoint
                combined = mlx_weights[c_attn_weight]  # (D, 3*D)
                
                # Verify shape
                expected_shape = (self.model_dim, 3 * self.model_dim)
                if combined.shape != expected_shape:
                    print(f">> WARNING: c_attn weight shape: {combined.shape} vs expected {expected_shape}")
                
                # Split along axis=1 (columns): (D, 3*D) -> 3 x (D, D)
                split_size = self.model_dim
                q_w = combined[:, :split_size]  # First D columns -> (D, D)
                k_w = combined[:, split_size:2*split_size]  # Second D columns -> (D, D)
                v_w = combined[:, 2*split_size:]  # Last D columns -> (D, D)
                
                # MLX Linear expects (out_features, in_features), so transpose
                block.attn.q_proj.weight = q_w.T  # (D, D) -> (D, D)
                block.attn.k_proj.weight = k_w.T
                block.attn.v_proj.weight = v_w.T
                
                if layer_idx == 0:
                    # Debug: print first layer weight shapes
                    print(f"   Layer 0 attn weights after load: q={block.attn.q_proj.weight.shape}, "
                          f"k={block.attn.k_proj.weight.shape}, v={block.attn.v_proj.weight.shape}")
                
                loaded += 1
            
            if c_attn_bias in mlx_weights:
                # PyTorch GPT2 c_attn.bias shape: (3*model_dim,)
                combined = mlx_weights[c_attn_bias]
                
                # Manual split by slicing
                split_size = self.model_dim
                q_b = combined[:split_size]  # (D,)
                k_b = combined[split_size:2*split_size]  # (D,)
                v_b = combined[2*split_size:]  # (D,)
                
                block.attn.q_proj.bias = q_b
                block.attn.k_proj.bias = k_b
                block.attn.v_proj.bias = v_b
                loaded += 1
            
            # Attention output projection  
            # IMPORTANT: PyTorch GPT2 uses Conv1D which stores weight as (in, out)
            # MLX Linear expects (out, in), so we MUST transpose
            c_proj_weight = f"{prefix}.attn.c_proj.weight"
            c_proj_bias = f"{prefix}.attn.c_proj.bias"
            if c_proj_weight in mlx_weights:
                w = mlx_weights[c_proj_weight]  # Conv1D format: (in, out)
                block.attn.out_proj.weight = w.T  # Transpose to (out, in) for MLX Linear
                loaded += 1
            if c_proj_bias in mlx_weights:
                block.attn.out_proj.bias = mlx_weights[c_proj_bias]
                loaded += 1
            
            # Layer norms
            ln1_weight = f"{prefix}.ln_1.weight"
            ln1_bias = f"{prefix}.ln_1.bias"
            if ln1_weight in mlx_weights:
                block.ln_1.weight = mlx_weights[ln1_weight]
                loaded += 1
            if ln1_bias in mlx_weights:
                block.ln_1.bias = mlx_weights[ln1_bias]
                loaded += 1
            
            ln2_weight = f"{prefix}.ln_2.weight"
            ln2_bias = f"{prefix}.ln_2.bias"
            if ln2_weight in mlx_weights:
                block.ln_2.weight = mlx_weights[ln2_weight]
                loaded += 1
            if ln2_bias in mlx_weights:
                block.ln_2.bias = mlx_weights[ln2_bias]
                loaded += 1
            
            # MLP (feed-forward)
            # IMPORTANT: PyTorch GPT2 uses Conv1D, so weight is (in, out)
            # MLX Linear expects (out, in), must transpose
            mlp_fc_weight = f"{prefix}.mlp.c_fc.weight"
            mlp_fc_bias = f"{prefix}.mlp.c_fc.bias"
            if mlp_fc_weight in mlx_weights:
                w = mlx_weights[mlp_fc_weight]  # Conv1D: (D, 4*D)
                block.mlp_fc.weight = w.T  # Transpose to (4*D, D)
                loaded += 1
            if mlp_fc_bias in mlx_weights:
                block.mlp_fc.bias = mlx_weights[mlp_fc_bias]
                loaded += 1
            
            mlp_proj_weight = f"{prefix}.mlp.c_proj.weight"
            mlp_proj_bias = f"{prefix}.mlp.c_proj.bias"
            if mlp_proj_weight in mlx_weights:
                w = mlx_weights[mlp_proj_weight]  # Conv1D: (4*D, D)
                block.mlp_proj.weight = w.T  # Transpose to (D, 4*D)
                loaded += 1
            if mlp_proj_bias in mlx_weights:
                block.mlp_proj.bias = mlx_weights[mlp_proj_bias]
                loaded += 1
        
        # Load conditioning weights if using MLX conditioning
        if self.use_mlx_conditioning and self.conditioning_module is not None:
            print("\n>> Loading MLX Conditioning weights...")
            cond_loaded = self._load_conditioning_weights(mlx_weights)
            loaded += cond_loaded
            print(f">> Loaded {cond_loaded} conditioning weights")
        
        print(f">> Loaded {loaded} weight tensors total from MLX cache")
        return loaded
    
    def _load_conditioning_weights(self, weights):
        """Load Conformer and Perceiver weights from PyTorch checkpoint"""
        loaded = 0
        
        # Load Perceiver weights (simpler, do first)
        loaded += self._load_perceiver_weights(weights)
        
        # Load Conformer weights (more complex)
        loaded += self._load_conformer_weights(weights)
        
        return loaded
    
    def _load_perceiver_weights(self, weights):
        """Load Perceiver Resampler weights"""
        loaded = 0
        perceiver = self.conditioning_module.perceiver
        
        # Latents
        if 'perceiver_encoder.latents' in weights:
            perceiver.latents = weights['perceiver_encoder.latents']
            loaded += 1
        
        # proj_context (512 → 1280)
        if 'perceiver_encoder.proj_context.weight' in weights:
            perceiver.proj_context.weight = weights['perceiver_encoder.proj_context.weight']
            loaded += 1
        if 'perceiver_encoder.proj_context.bias' in weights:
            perceiver.proj_context.bias = weights['perceiver_encoder.proj_context.bias']
            loaded += 1
        
        # Perceiver layers (2 layers)
        for layer_idx in range(2):
            layer = perceiver.layers[layer_idx]
            attn, ff = layer  # (attention, feed-forward)
            
            prefix = f"perceiver_encoder.layers.{layer_idx}"
            
            # Attention
            if f"{prefix}.0.to_q.weight" in weights:
                attn.to_q.weight = weights[f"{prefix}.0.to_q.weight"]
                loaded += 1
            if f"{prefix}.0.to_kv.weight" in weights:
                attn.to_kv.weight = weights[f"{prefix}.0.to_kv.weight"]
                loaded += 1
            if f"{prefix}.0.to_out.weight" in weights:
                attn.to_out.weight = weights[f"{prefix}.0.to_out.weight"]
                loaded += 1
            
            # Feed-forward (GEGLU)
            if f"{prefix}.1.0.weight" in weights:
                ff.net[0].weight = weights[f"{prefix}.1.0.weight"]
                loaded += 1
            if f"{prefix}.1.0.bias" in weights:
                ff.net[0].bias = weights[f"{prefix}.1.0.bias"]
                loaded += 1
            if f"{prefix}.1.2.weight" in weights:
                ff.net[2].weight = weights[f"{prefix}.1.2.weight"]
                loaded += 1
            if f"{prefix}.1.2.bias" in weights:
                ff.net[2].bias = weights[f"{prefix}.1.2.bias"]
                loaded += 1
        
        # Final norm (RMSNorm uses 'scale' instead of 'weight')
        if 'perceiver_encoder.norm.gamma' in weights:
            perceiver.norm.scale = weights['perceiver_encoder.norm.gamma']
            loaded += 1
        
        return loaded
    
    def _load_conformer_weights(self, weights):
        """Load Conformer Encoder weights"""
        loaded = 0
        conformer = self.conditioning_module.conformer
        
        # ✅ FIX: Load Conv2d Subsampling weights
        # Conv2d: conditioning_encoder.embed.conv.0.weight (512, 1, 3, 3)
        if 'conditioning_encoder.embed.conv.0.weight' in weights:
            # PyTorch: (out_channels=512, in_channels=1, H=3, W=3)
            # MLX: same shape
            conformer.subsampling.conv.weight = weights['conditioning_encoder.embed.conv.0.weight']
            loaded += 1
        
        if 'conditioning_encoder.embed.conv.0.bias' in weights:
            # PyTorch: (512,)
            conformer.subsampling.conv.bias = weights['conditioning_encoder.embed.conv.0.bias']
            loaded += 1
        
        # Linear projection after Conv2d: conditioning_encoder.embed.out.0
        if 'conditioning_encoder.embed.out.0.weight' in weights:
            # PyTorch: (512, 261632) where 261632 = 512 * 511
            # MLX: same
            conformer.subsampling.out.weight = weights['conditioning_encoder.embed.out.0.weight']
            loaded += 1
        
        if 'conditioning_encoder.embed.out.0.bias' in weights:
            conformer.subsampling.out.bias = weights['conditioning_encoder.embed.out.0.bias']
            loaded += 1
        
        # Position encoding
        if 'conditioning_encoder.embed.pos_enc.pe' in weights:
            pe = weights['conditioning_encoder.embed.pos_enc.pe']  # (1, 5000, 512)
            conformer.pos_encoding = pe.squeeze(0)[:1000]  # Take first 1000, remove batch dim
            loaded += 1
        
        # Conformer blocks (6 layers)
        for layer_idx in range(6):
            block = conformer.blocks[layer_idx]
            prefix = f"conditioning_encoder.encoders.{layer_idx}"
            
            # Self-attention
            if f"{prefix}.self_attn.linear_q.weight" in weights:
                block.attn.q_proj.weight = weights[f"{prefix}.self_attn.linear_q.weight"]
                loaded += 1
            if f"{prefix}.self_attn.linear_q.bias" in weights:
                block.attn.q_proj.bias = weights[f"{prefix}.self_attn.linear_q.bias"]
                loaded += 1
            if f"{prefix}.self_attn.linear_k.weight" in weights:
                block.attn.k_proj.weight = weights[f"{prefix}.self_attn.linear_k.weight"]
                loaded += 1
            if f"{prefix}.self_attn.linear_k.bias" in weights:
                block.attn.k_proj.bias = weights[f"{prefix}.self_attn.linear_k.bias"]
                loaded += 1
            if f"{prefix}.self_attn.linear_v.weight" in weights:
                block.attn.v_proj.weight = weights[f"{prefix}.self_attn.linear_v.weight"]
                loaded += 1
            if f"{prefix}.self_attn.linear_v.bias" in weights:
                block.attn.v_proj.bias = weights[f"{prefix}.self_attn.linear_v.bias"]
                loaded += 1
            if f"{prefix}.self_attn.linear_out.weight" in weights:
                block.attn.out_proj.weight = weights[f"{prefix}.self_attn.linear_out.weight"]
                loaded += 1
            if f"{prefix}.self_attn.linear_out.bias" in weights:
                block.attn.out_proj.bias = weights[f"{prefix}.self_attn.linear_out.bias"]
                loaded += 1
            if f"{prefix}.self_attn.linear_pos.weight" in weights:
                block.attn.pos_proj.weight = weights[f"{prefix}.self_attn.linear_pos.weight"]
                loaded += 1
            
            # ✅ NEW: Load Relative Positional Attention biases (Transformer-XL style)
            if f"{prefix}.self_attn.pos_bias_u" in weights:
                block.attn.pos_bias_u = weights[f"{prefix}.self_attn.pos_bias_u"]
                loaded += 1
            if f"{prefix}.self_attn.pos_bias_v" in weights:
                block.attn.pos_bias_v = weights[f"{prefix}.self_attn.pos_bias_v"]
                loaded += 1
            
            # ✅ FIXED: Feed-forward (只有一个 FF，不是 macaron style)
            # PyTorch 只有 feed_forward，没有 feed_forward_macaron
            if f"{prefix}.feed_forward.w_1.weight" in weights:
                block.ff.layers[0].weight = weights[f"{prefix}.feed_forward.w_1.weight"]
                loaded += 1
            if f"{prefix}.feed_forward.w_1.bias" in weights:
                block.ff.layers[0].bias = weights[f"{prefix}.feed_forward.w_1.bias"]
                loaded += 1
            if f"{prefix}.feed_forward.w_2.weight" in weights:
                block.ff.layers[2].weight = weights[f"{prefix}.feed_forward.w_2.weight"]
                loaded += 1
            if f"{prefix}.feed_forward.w_2.bias" in weights:
                block.ff.layers[2].bias = weights[f"{prefix}.feed_forward.w_2.bias"]
                loaded += 1
            
            # Convolution module
            if f"{prefix}.conv_module.pointwise_conv1.weight" in weights:
                w = weights[f"{prefix}.conv_module.pointwise_conv1.weight"]  # (1024, 512, 1)
                block.conv.pointwise1.weight = w.squeeze(-1)  # Remove kernel dim: (1024, 512)
                loaded += 1
            if f"{prefix}.conv_module.pointwise_conv1.bias" in weights:
                block.conv.pointwise1.bias = weights[f"{prefix}.conv_module.pointwise_conv1.bias"]
                loaded += 1
            
            if f"{prefix}.conv_module.depthwise_conv.weight" in weights:
                w = weights[f"{prefix}.conv_module.depthwise_conv.weight"]  # (512, 1, 15)
                block.conv.depthwise.weight = w.squeeze(1)  # Remove middle dim: (512, 15)
                loaded += 1
            if f"{prefix}.conv_module.depthwise_conv.bias" in weights:
                block.conv.depthwise.bias = weights[f"{prefix}.conv_module.depthwise_conv.bias"]
                loaded += 1
            
            if f"{prefix}.conv_module.pointwise_conv2.weight" in weights:
                w = weights[f"{prefix}.conv_module.pointwise_conv2.weight"]  # (512, 512, 1)
                block.conv.pointwise2.weight = w.squeeze(-1)  # (512, 512)
                loaded += 1
            if f"{prefix}.conv_module.pointwise_conv2.bias" in weights:
                block.conv.pointwise2.bias = weights[f"{prefix}.conv_module.pointwise_conv2.bias"]
                loaded += 1
            
            if f"{prefix}.conv_module.norm.weight" in weights:
                block.conv.bn.weight = weights[f"{prefix}.conv_module.norm.weight"]
                loaded += 1
            if f"{prefix}.conv_module.norm.bias" in weights:
                block.conv.bn.bias = weights[f"{prefix}.conv_module.norm.bias"]
                loaded += 1
            
            # ✅ FIXED: Layer norms (PyTorch 没有 norm_ff_macaron)
            # norm_mha (for multi-head attention)
            if f"{prefix}.norm_mha.weight" in weights:
                block.norm_attn.weight = weights[f"{prefix}.norm_mha.weight"]
                loaded += 1
            if f"{prefix}.norm_mha.bias" in weights:
                block.norm_attn.bias = weights[f"{prefix}.norm_mha.bias"]
                loaded += 1
            
            # norm_conv (for convolution module)
            if f"{prefix}.norm_conv.weight" in weights:
                block.norm_conv.weight = weights[f"{prefix}.norm_conv.weight"]
                loaded += 1
            if f"{prefix}.norm_conv.bias" in weights:
                block.norm_conv.bias = weights[f"{prefix}.norm_conv.bias"]
                loaded += 1
            
            # norm_ff (for second FF)
            if f"{prefix}.norm_ff.weight" in weights:
                block.norm_ff.weight = weights[f"{prefix}.norm_ff.weight"]
                loaded += 1
            if f"{prefix}.norm_ff.bias" in weights:
                block.norm_ff.bias = weights[f"{prefix}.norm_ff.bias"]
                loaded += 1
            
            # norm_final (after all sub-modules)
            if f"{prefix}.norm_final.weight" in weights:
                block.norm_final.weight = weights[f"{prefix}.norm_final.weight"]
                loaded += 1
            if f"{prefix}.norm_final.bias" in weights:
                block.norm_final.bias = weights[f"{prefix}.norm_final.bias"]
                loaded += 1
        
        # Final norm (after all conformer blocks)
        if 'conditioning_encoder.after_norm.weight' in weights:
            conformer.after_norm.weight = weights['conditioning_encoder.after_norm.weight']
            loaded += 1
        if 'conditioning_encoder.after_norm.bias' in weights:
            conformer.after_norm.bias = weights['conditioning_encoder.after_norm.bias']
            loaded += 1
        
        return loaded
    
    def simple_forward(self, text_tokens, conditioning=None, max_length=1500, **kwargs):
        """
        Full forward pass with transformer layers for generation.
        
        Args:
            text_tokens: Text token IDs (batch, text_len) - MLX array
            conditioning: Conditioning latent (batch, cond_len, model_dim) - MLX array
            max_length: Maximum mel tokens to generate
            **kwargs: Extra arguments (ignored for compatibility)
            
        Returns:
            Generated mel codes (batch, mel_len) - MLX array
        """
        # 🔧 修复：重置随机状态以避免推理间状态累积
        # 每次推理使用新的随机种子（基于时间），确保推理独立性
        import time
        import os
        
        seed = kwargs.get('seed', None)
        if seed is None:
            # 检查是否有固定种子的环境变量（用于调试）
            fixed_seed_str = os.environ.get('MLX_FIXED_SEED', None)
            if fixed_seed_str:
                seed = int(fixed_seed_str)
            else:
                # 🔧 修复首单词吞音：使用固定种子确保稳定性
                # 而不是时间戳（会导致每次输出不同）
                # 如果需要随机性，可以通过 seed 参数或 MLX_RANDOM_SEED=1 环境变量启用
                use_random_seed = os.environ.get('MLX_RANDOM_SEED', '0') == '1'
                if use_random_seed:
                    # 随机模式：使用时间戳
                    seed = int(time.time() * 1000000) % (2**32)
                else:
                    # 默认：固定种子，确保稳定输出
                    seed = 42
        
        # Debug: 打印种子（可选）
        debug_generation = kwargs.get('debug_generation', False)
        if debug_generation:
            print(f"[DEBUG] Random seed: {seed}")
        
        mx.random.seed(seed)
        
        # Create default conditioning if not provided
        if conditioning is None:
            batch_size = text_tokens.shape[0]
            cond_len = 32
            conditioning = mx.zeros((batch_size, cond_len, self.model_dim))
        
        # Get text embeddings
        text_emb = self.text_embedding(text_tokens)  # (B, T, D)
        
        # Add text positional embeddings (like PyTorch LearnedPositionEmbeddings)
        # PyTorch returns (seq_len, dim), we need to expand for batch
        text_seq_len = text_tokens.shape[1]
        max_text_pos = self.text_pos_embedding.weight.shape[0]
        
        # 检查是否越界
        if text_seq_len > max_text_pos:
            print(f"⚠️  WARNING: text_seq_len ({text_seq_len}) > max_text_pos ({max_text_pos})")
            text_seq_len = max_text_pos
        
        text_pos_emb = mx.stack([self.text_pos_embedding.weight[i] for i in range(text_seq_len)], axis=0)  # (T, D)
        text_emb = text_emb[:, :text_seq_len] + text_pos_emb  # Broadcasting: (B, T, D) + (T, D) -> (B, T, D)
        
        # Combine conditioning + text
        context = mx.concatenate([conditioning, text_emb], axis=1)  # (B, C+T, D)
        
        # CRITICAL: Add start_mel_token embedding (like PyTorch version)
        batch_size = text_tokens.shape[0]
        start_token_ids = mx.full((batch_size, 1), self.start_mel_token, dtype=mx.int32)
        start_token_emb = self.mel_embedding(start_token_ids)  # (B, 1, D)
        
        # CRITICAL: Track absolute position in sequence for position encoding
        context_len = context.shape[1]  # Length of conditioning + text
        
        # Add mel position encoding for start_mel_token at absolute position context_len
        # PyTorch uses mel_pos_embedding for mel tokens, returns (1, dim) without batch
        start_pos = context_len
        start_token_emb = start_token_emb + self.mel_pos_embedding.weight[start_pos:start_pos+1]  # (B, 1, D) + (1, D)
        
        # Full initial sequence: [context] + [start_mel_token]
        sequence = mx.concatenate([context, start_token_emb], axis=1)
        
        # Autoregressive generation with KV caching for efficiency
        generated = []
        
        debug_generation = kwargs.get('debug_generation', False)
        
        if debug_generation:
            print(f"\n[DEBUG] Generation Setup:")
            print(f"  Text tokens shape: {text_tokens.shape}")
            print(f"  Text tokens (first 10): {text_tokens[0, :min(10, text_tokens.shape[1])].tolist()}")
            print(f"  Conditioning shape: {conditioning.shape}")
            print(f"  Context length: {context.shape[1]}")
            print(f"  Start token ID: {self.start_mel_token}")
            print(f"  Start token position: {start_pos}")
        
        print(f">> [MLX] Starting autoregressive loop with KV cache (max_length={max_length})...")
        print(f"   Initial sequence: {sequence.shape[1]} tokens (context={context.shape[1]} + start_token=1)")
        print(f"   Start token position: {start_pos}")
        
        # First pass: process full context (including start_mel_token) and initialize KV cache
        hidden = sequence
        past_kvs = []
        for block in self.transformer_blocks:
            hidden, kv = block(hidden, causal_mask=self.causal_mask, past_kv=None, use_cache=True)
            past_kvs.append(kv)
        hidden = self.final_norm(hidden)
        
        # Get first token
        logits = self.mel_head(hidden[:, -1:, :])
        
        temperature = kwargs.get('temperature', 0.8)
        use_sampling = kwargs.get('use_sampling', True)  # 可配置
        
        if use_sampling and temperature > 0:
            # 随机采样模式
            logits = logits / temperature
            probs = mx.softmax(logits[0, 0], axis=-1)
            next_token_id = mx.random.categorical(mx.log(probs + 1e-10))
            next_token = mx.array([[next_token_id]])
        else:
            # 确定性模式（argmax）- 可能更稳定
            next_token_id = mx.argmax(logits[0, 0])
            next_token = mx.array([[next_token_id]])
        
        token_val = int(next_token[0, 0])
        print(f">> [MLX] First token: {token_val}")
        
        if debug_generation:
            print(f"[DEBUG] First token logits top 10:")
            top_k = 10
            logits_1d = logits[0, 0]
            top_indices = mx.argsort(logits_1d)[-top_k:][::-1]
            top_values = logits_1d[top_indices]
            for i in range(top_k):
                idx = int(top_indices[i])
                val = float(top_values[i])
                print(f"  #{i+1}: token {idx}, logit {val:.4f}")
        
        if token_val == self.stop_mel_token:
            print(f">> [MLX] Hit stop token at step 0")
            codes = mx.zeros((text_tokens.shape[0], 0), dtype=mx.int32)
            return codes
        
        generated.append(next_token)
        
        # Subsequent passes: use KV cache (only process new token)
        for step in range(1, max_length):
            # Progress update every 50 steps
            if step % 50 == 0:
                print(f">> [MLX] Step {step}/{max_length}, {len(generated)} tokens generated", end='\r')
            
            # Embed only the new token
            next_emb = self.mel_embedding(next_token)  # (B, 1, D)
            
            # CRITICAL: Add mel position encoding using ABSOLUTE position in sequence
            # Current absolute position = context_len + step (step=0 was start_token, step=1 is first generated, etc.)
            absolute_pos = context_len + step
            if absolute_pos < self.mel_pos_embedding.weight.shape[0]:
                mel_pos_enc = self.mel_pos_embedding.weight[absolute_pos:absolute_pos+1]  # (1, D)
                next_emb = next_emb + mel_pos_enc  # (B, 1, D) + (1, D)
            
            # Apply transformer with KV cache (much faster!)
            hidden = next_emb
            new_past_kvs = []
            for i, block in enumerate(self.transformer_blocks):
                hidden, kv = block(hidden, causal_mask=self.causal_mask, past_kv=past_kvs[i], use_cache=True)
                new_past_kvs.append(kv)
            
            # DEBUG: Check KV cache size
            if new_past_kvs:
                kv_cache_size = new_past_kvs[0][0].shape[2]  # K's sequence dimension
            
            past_kvs = new_past_kvs
            
            hidden = self.final_norm(hidden)
            
            # Get next token logits
            logits = self.mel_head(hidden)  # (B, 1, vocab)
            
            if use_sampling and temperature > 0:
                # 随机采样模式
                logits = logits / temperature
                probs = mx.softmax(logits[0, 0], axis=-1)  # (vocab,)
                next_token_id = mx.random.categorical(mx.log(probs + 1e-10))  # Sample
                next_token = mx.array([[next_token_id]])  # (1, 1)
            else:
                # 确定性模式（argmax）
                next_token_id = mx.argmax(logits[0, 0])
                next_token = mx.array([[next_token_id]])
            
            # Check stop token
            token_val = int(next_token[0, 0])
            
            
            if token_val == self.stop_mel_token:
                print(f"\n>> [MLX] Hit stop token at step {step}")
                break
            
            generated.append(next_token)
        
        print(f"\n>> [MLX] Generation complete: {len(generated)} tokens (with KV cache)")
        
        # Stack results
        if generated:
            codes = mx.concatenate(generated, axis=1)
        else:
            codes = mx.zeros((text_tokens.shape[0], 0), dtype=mx.int32)
        
        # Debug: print first 10 tokens
        if debug_generation and codes.shape[1] > 0:
            first_n = min(15, codes.shape[1])
            tokens_list = [int(codes[0, i]) for i in range(first_n)]
            print(f"[DEBUG] First {first_n} generated tokens: {tokens_list}")
        
        # 🔧 修复：显式清理 KV cache 和中间变量
        try:
            del past_kvs, new_past_kvs, hidden, logits, probs
            del context, sequence, text_emb, conditioning
        except:
            pass
        
        return codes
    
    def __call__(self, *args, **kwargs):
        """
        Make model callable for compatibility with PyTorch code.
        
        Handles various calling patterns from inference code.
        """
        from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch
        
        # If called with inputs_embeds (typical PyTorch GPT usage)
        if 'inputs_embeds' in kwargs:
            # Return dummy output for compatibility
            inputs_embeds = kwargs['inputs_embeds']
            
            # Simple pass-through for now
            inputs_mlx = torch_to_mlx(inputs_embeds)
            hidden = self.final_norm(inputs_mlx)
            
            # Convert back
            hidden_torch = mlx_to_torch(hidden, device='mps')
            
            # Return format compatible with PyTorch GPT
            class DummyOutput:
                def __init__(self, hidden):
                    self.last_hidden_state = hidden
            
            return DummyOutput(hidden_torch)
        
        # If called with positional args for latent extraction
        # PyTorch version: gpt(speech_latent, text_tokens, text_lens, codes, code_lens, ...)
        # Returns latent that should match codes length
        if len(args) >= 4:
            # args[3] should be codes
            codes = args[3]
            
            # Get batch size and sequence length from codes
            batch_size = codes.shape[0]
            seq_len = codes.shape[1]  # Match codes length!
            
            # Return latent with matching shape
            latent = mx.zeros((batch_size, seq_len, self.model_dim))
            latent_torch = mlx_to_torch(latent, device='mps')
            
            print(f">> [MLX Native] Returning latent with shape {latent_torch.shape} to match codes")
            return latent_torch
        
        # Fallback: ignore all and return dummy
        batch_size = 1
        seq_len = 1
        latent = mx.zeros((batch_size, seq_len, self.model_dim))
        return mlx_to_torch(latent, device='mps')
    
    def get_conditioning(self, speech_conditioning_input, cond_mel_lengths=None):
        """
        Get conditioning latents from semantic features.
        
        Matches model_v2.py: conformer_perceiver mode.
        Input: (b, time, 1024) semantic features already transposed
        
        Args:
            speech_conditioning_input: Semantic features (b, time, 1024) PyTorch tensor
            cond_mel_lengths: Lengths tensor
            
        Returns:
            Conditioning latents (b, cond_num, model_dim) PyTorch tensor
        """
        from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch
        
        # Ensure correct shape (b, time, 1024)
        if speech_conditioning_input.shape[-1] != 1024:
            # Input is (b, 1024, time), transpose it
            speech_conditioning_input = speech_conditioning_input.transpose(1, 2)
        
        # Convert to MLX
        speech_mlx = torch_to_mlx(speech_conditioning_input)  # (b, time, 1024)
        
        # Project from 1024 to model_dim
        projected = self.cond_projection(speech_mlx)  # (b, time, model_dim)
        
        # Simplified PerceiverResampler: use pooling to downsample
        batch_size, seq_len, _ = projected.shape
        num_tokens = self.cond_num
        
        if seq_len >= num_tokens:
            # Average pooling to fixed number of tokens
            pool_size = seq_len / num_tokens
            conds_list = []
            for i in range(num_tokens):
                start = int(i * pool_size)
                end = int((i + 1) * pool_size)
                if start < seq_len:
                    pooled = mx.mean(projected[:, start:end, :], axis=1, keepdims=True)
                    conds_list.append(pooled)
            conds = mx.concatenate(conds_list, axis=1)
        else:
            # Pad if too short
            conds = projected
        
        # Return as PyTorch
        return mlx_to_torch(conds, device='mps')
    
    def get_emo_conditioning(self, speech_conditioning_input, cond_mel_lengths=None):
        """
        Get emotion conditioning vector from semantic features.
        
        Matches model_v2.py logic - returns raw 1024-dim features, not projected.
        
        Args:
            speech_conditioning_input: Semantic features (b, time, 1024) PyTorch tensor
            cond_mel_lengths: Lengths tensor
            
        Returns:
            Emotion vector (b, 1024) PyTorch tensor (raw features, NOT projected)
        """
        from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch
        
        # Ensure correct shape (b, time, 1024)
        if speech_conditioning_input.shape[-1] != 1024:
            # Input is (b, 1024, time), transpose it
            speech_conditioning_input = speech_conditioning_input.transpose(1, 2)
        
        # Convert to MLX
        speech_mlx = torch_to_mlx(speech_conditioning_input)  # (b, time, 1024)
        
        # Simplified: just average over time (no projection yet)
        # The projection will be done by emovec_layer and emo_layer in get_emovec
        emo_cond = mx.mean(speech_mlx, axis=1)  # (b, 1024)
        
        # Return as PyTorch
        return mlx_to_torch(emo_cond, device='mps')
    
    def get_emovec(self, emo_speech_conditioning_latent, emo_cond_lengths):
        """
        Get emotion vector (matches model_v2.py).
        
        Args:
            emo_speech_conditioning_latent: Semantic features (b, 1024, time)
            emo_cond_lengths: Lengths
            
        Returns:
            Emotion vector (PyTorch tensor) (b, model_dim)
        """
        # Get emotion conditioning (function handles transpose internally)
        emo_vec_syn_ori = self.get_emo_conditioning(
            emo_speech_conditioning_latent,  # No transpose, function handles it
            emo_cond_lengths
        )
        
        # Apply emotion layers (convert to MLX, apply, convert back)
        from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch
        
        emo_mlx = torch_to_mlx(emo_vec_syn_ori)
        emo_vec_syn = self.emovec_layer(emo_mlx)
        emo_vec = self.emo_layer(emo_vec_syn)
        
        return mlx_to_torch(emo_vec, device='mps')
    
    def merge_emovec(self, speech_latent, emo_latent, cond_len, emo_len, alpha=1.0):
        """
        Merge emotion vectors (matches model_v2.py).
        
        Returns PyTorch tensor for compatibility.
        """
        # Use get_emovec for both
        emo_vec = self.get_emovec(emo_latent, emo_len)
        base_vec = self.get_emovec(speech_latent, cond_len)
        
        # Merge with alpha blending
        out = base_vec + alpha * (emo_vec - base_vec)
        
        print(f">> [MLX Native] merge_emovec: {out.shape}")
        return out
    
    def get_conditioning_mlx(self, speech_features, lengths=None):
        """
        Get conditioning latents using pure MLX conditioning pipeline.
        
        Args:
            speech_features: Speaker features (batch, 1024, time) or MLX array (batch, time, 1024)
            lengths: Sequence lengths (batch,) or MLX array
        
        Returns:
            Conditioning latents (batch, 32, model_dim) as MLX array
        """
        if not self.use_mlx_conditioning:
            raise RuntimeError("MLX conditioning not enabled. Set use_mlx_conditioning=True in __init__")
        
        # Ensure correct format: (batch, time, features)
        if isinstance(speech_features, mx.array):
            if speech_features.shape[-1] != 1024:
                # Assume it's (batch, 1024, time), transpose to (batch, time, 1024)
                speech_features = speech_features.transpose(0, 2, 1)
        
        # Run through Conformer + Perceiver
        latents = self.conditioning_module(speech_features, lengths)
        
        return latents  # (batch, 32, model_dim)
    
    def inference_speech(
        self,
        speech_condition,
        text_inputs,
        emo_speech_condition=None,
        cond_lengths=None,
        emo_cond_lengths=None,
        emo_vec=None,
        use_speed=False,
        **kwargs
    ):
        """
        Main inference method with pure MLX conditioning.
        
        Args:
            speech_condition: Semantic features (b, 1024, time) PyTorch or (b, time, 1024) MLX
            text_inputs: Text tokens (b, L) PyTorch or MLX
            emo_speech_condition: Emotion semantic features
            cond_lengths, emo_cond_lengths: Lengths tensors
            emo_vec: Pre-computed emotion vector
            use_speed: Speed conditioning flag
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple of (codes, conditioning_latent) as PyTorch tensors
        """
        from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch
        import torch
        import mlx.core as mx
        
        print(">> [MLX Native] Running pure MLX inference with Conformer + Perceiver")
        
        # Handle dimensions
        if speech_condition.ndim == 2:
            speech_condition = speech_condition.unsqueeze(0)
        if emo_speech_condition is None:
            emo_speech_condition = speech_condition
        if cond_lengths is None:
            cond_lengths = torch.tensor([speech_condition.shape[-1]], device=speech_condition.device)
        if emo_cond_lengths is None:
            emo_cond_lengths = torch.tensor([emo_speech_condition.shape[-1]], device=speech_condition.device)
        
        # Convert to MLX for pure MLX conditioning
        speech_condition_mlx = torch_to_mlx(speech_condition.cpu())
        if speech_condition_mlx.ndim == 3 and speech_condition_mlx.shape[1] == 1024:
            speech_condition_mlx = speech_condition_mlx.transpose(0, 2, 1)  # (b, 1024, t) -> (b, t, 1024)
        
        emo_speech_condition_mlx = torch_to_mlx(emo_speech_condition.cpu())
        if emo_speech_condition_mlx.ndim == 3 and emo_speech_condition_mlx.shape[1] == 1024:
            emo_speech_condition_mlx = emo_speech_condition_mlx.transpose(0, 2, 1)
        
        cond_lengths_mlx = torch_to_mlx(cond_lengths.cpu()) if cond_lengths is not None else None
        
        # Get conditioning latents using PURE MLX (Conformer + Perceiver)
        print('>> [MLX] Running Conformer + Perceiver for speech conditioning...')
        speech_conditioning_latent_mlx = self.get_conditioning_mlx(speech_condition_mlx, cond_lengths_mlx)
        
        # Get emotion vector
        if emo_vec is None:
            print('>> [MLX] Running Conformer + Perceiver for emotion conditioning...')
            emo_conditioning_latent_mlx = self.get_conditioning_mlx(emo_speech_condition_mlx, cond_lengths_mlx)
            # Average pool to get emotion vector
            emo_vec_mlx = mx.mean(emo_conditioning_latent_mlx, axis=1)  # (b, model_dim)
            emo_vec_mlx = self.emo_layer(emo_vec_mlx)
        else:
            emo_vec_mlx = torch_to_mlx(emo_vec.cpu())
            print('>> [MLX] Using specified emotion vector')
        
        # Prepare conditioning with speed embeddings (pure MLX)
        batch_size = speech_conditioning_latent_mlx.shape[0]
        duration_emb_mlx = self.speed_emb(mx.zeros((batch_size,), dtype=mx.int32))
        duration_emb_half_mlx = self.speed_emb(mx.ones((batch_size,), dtype=mx.int32))
        
        # Combine conditioning (pure MLX)
        speech_cond_with_emo = speech_conditioning_latent_mlx + emo_vec_mlx.reshape(batch_size, 1, -1)
        conds_mlx = mx.concatenate([
            speech_cond_with_emo,
            duration_emb_half_mlx.reshape(batch_size, 1, -1),
            duration_emb_mlx.reshape(batch_size, 1, -1)
        ], axis=1)  # (b, 34, model_dim)
        
        print(f">> [MLX] Pure MLX conditioning shape: {conds_mlx.shape}")
        
        # Convert text to MLX
        text_mlx = torch_to_mlx(text_inputs)
        
        # Generate codes using full transformer
        print(f">> [MLX] Starting generation (max_length={kwargs.get('max_generate_length', 1500)})")
        codes_mlx = self.simple_forward(
            text_mlx,
            conds_mlx,
            max_length=kwargs.get('max_generate_length', 500),  # Reduce default for testing
            temperature=kwargs.get('temperature', 0.8),
            use_sampling=kwargs.get('use_sampling', True),  # 传递采样模式
            debug_generation=kwargs.get('debug_generation', False),  # 传递 debug 模式
        )
        print(f">> [MLX] Generation complete")
        
        # Convert to PyTorch with correct dtype
        codes = mlx_to_torch(codes_mlx, device='cpu').long().to(speech_condition.device)
        speech_conditioning_latent_torch = mlx_to_torch(speech_conditioning_latent_mlx, device=speech_condition.device)
        
        print(f">> [MLX Native] Generated {codes.shape[1]} mel tokens with pure MLX conditioning")
        
        # 🔧 修复内存泄漏和状态累积：清理 MLX 中间结果
        try:
            # 删除大的中间 MLX 数组
            del speech_condition_mlx, emo_speech_condition_mlx, cond_lengths_mlx
            del speech_conditioning_latent_mlx, emo_vec_mlx, conds_mlx, text_mlx, codes_mlx
            # 清理 MLX 缓存和Metal资源
            mx.metal.clear_cache()
            # 强制垃圾回收
            import gc
            gc.collect()
        except Exception as e:
            # 静默失败，不影响返回结果
            pass
        
        return codes, speech_conditioning_latent_torch


def create_mlx_gpt_from_cache(mlx_cache_dict, config):
    """
    Create MLX GPT model and load weights from cache.
    
    This is the main entry point for loading cached MLX models.
    
    Args:
        mlx_cache_dict: MLX weights dictionary from npz cache
        config: Model configuration
        
    Returns:
        Initialized MLX model with loaded weights
    """
    model = UnifiedVoiceMLX(**config)
    model.load_weights_from_dict(mlx_cache_dict)
    return model

