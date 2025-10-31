"""
Native MLX Implementation of GPT Model for Apple Silicon M4

Full Transformer implementation with proper attention and feed-forward layers.
Uses MLX built-in layers with PyTorch-compatible initialization.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
import math
import numpy as np


def init_linear_pytorch_compatible(linear_layer: nn.Linear, std: float = 0.02):
    """
    Initialize MLX Linear layer to match PyTorch initialization.
    
    Args:
        linear_layer: MLX Linear layer
        std: Standard deviation for normal initialization (default: 0.02)
    """
    # PyTorch uses normal_(mean=0.0, std=std) for Linear layers
    linear_layer.weight = mx.random.normal(linear_layer.weight.shape) * std
    if linear_layer.bias is not None:
        linear_layer.bias = mx.zeros_like(linear_layer.bias)


def init_embedding_pytorch_compatible(embedding_layer: nn.Embedding, std: float = 0.02):
    """
    Initialize MLX Embedding layer to match PyTorch initialization.
    
    Args:
        embedding_layer: MLX Embedding layer
        std: Standard deviation for normal initialization (default: 0.02)
    """
    # PyTorch uses normal_(mean=0.0, std=std) for Embedding layers
    embedding_layer.weight = mx.random.normal(embedding_layer.weight.shape) * std


class MLXLearnedPositionEmbeddings(nn.Module):
    """
    Learned Position Embeddings for MLX.
    Matches PyTorch LearnedPositionEmbeddings behavior.
    """
    
    def __init__(self, seq_len: int, model_dim: int, init=0.02):
        super().__init__()
        # Create embedding weight
        self.emb = mx.random.normal((seq_len, model_dim)) * init
        self.seq_len = seq_len
        self.model_dim = model_dim
    
    def __call__(self, x):
        """
        Args:
            x: Input tensor (for getting sequence length from shape)
        
        Returns:
            Position embeddings for the sequence
        """
        import torch
        from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch
        
        # Get sequence length
        if isinstance(x, torch.Tensor):
            sl = x.shape[1]
            device = x.device
            # Generate position indices like PyTorch: arange(0, sl)
            positions = mx.arange(sl)
            pos_emb = self.emb[positions]
            # Return as PyTorch tensor
            return mlx_to_torch(pos_emb, device=device)
        else:
            # MLX array
            sl = x.shape[1]
            positions = mx.arange(sl)
            return self.emb[positions]
    
    def get_fixed_embedding(self, ind, dev):
        """Get embedding for a specific index."""
        import torch
        from indextts.utils.mlx_utils import mlx_to_torch
        return mlx_to_torch(self.emb[ind:ind+1], device=dev).unsqueeze(0)


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
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Initialize with PyTorch-compatible weights
        init_linear_pytorch_compatible(self.q_proj)
        init_linear_pytorch_compatible(self.k_proj)
        init_linear_pytorch_compatible(self.v_proj)
        init_linear_pytorch_compatible(self.out_proj)
    
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
        
        # Use MLX fast SDPA implementation
        # Prepare mask slice aligned to current kv length and query length
        attn_mask = None
        if causal_mask is not None:
            if past_kv is not None and seq_len == 1:
                current_pos = kv_seq_len - 1
                attn_mask = causal_mask[:, :, current_pos:current_pos+1, :kv_seq_len]
            else:
                attn_mask = causal_mask[:, :, :seq_len, :kv_seq_len]
        
        attn_output = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=attn_mask)
        
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
        self.mlp_fc = nn.Linear(embed_dim, embed_dim * 4)
        self.mlp_proj = nn.Linear(embed_dim * 4, embed_dim)
        
        # Initialize with PyTorch-compatible weights
        init_linear_pytorch_compatible(self.mlp_fc)
        init_linear_pytorch_compatible(self.mlp_proj)
    
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
        self.mel_length_compression = kwargs.get('mel_length_compression', 1024)
        
        # Pure MLX conditioning (Conformer + Perceiver)
        if use_mlx_conditioning:
            from indextts.gpt.mlx_conformer_encoder import MLXConditioningModule
            
            # Speaker conditioning (32 latents)
            # Get speaker conformer config from kwargs
            condition_module = kwargs.get('condition_module', None)
            if condition_module is not None:
                spk_cfg = condition_module
                self.conditioning_module = MLXConditioningModule(
                    input_dim=1024,                      # Speaker embedding dimension
                    conformer_dim=spk_cfg['output_size'],  # 512
                    model_dim=model_dim,                 # 1280
                    num_latents=32,                      # Output 32 conditioning latents
                    conformer_layers=spk_cfg['num_blocks'],  # From config (6)
                    conformer_heads=spk_cfg['attention_heads'],  # From config (8)
                    conformer_ff_mult=spk_cfg['linear_units'] // spk_cfg['output_size'],  # 2048/512 = 4
                    perceiver_depth=2,                   # 2 layers
                    perceiver_heads=spk_cfg['attention_heads'],  # From config (8)
                    perceiver_ff_mult=spk_cfg['perceiver_mult']  # From config (2)
                )
            else:
                # Fallback to defaults
                self.conditioning_module = MLXConditioningModule(
                    input_dim=1024,
                    conformer_dim=512,
                    model_dim=model_dim,
                    num_latents=32,
                    conformer_layers=6,
                    conformer_heads=8,
                    conformer_ff_mult=4,
                    perceiver_depth=2,
                    perceiver_heads=8,
                    perceiver_ff_mult=2
                )
            
            # Emotion conditioning (1 latent for emotion vector)
            # Get emotion conformer config from kwargs
            emo_condition_module = kwargs.get('emo_condition_module', None)
            if emo_condition_module is not None:
                emo_cfg = emo_condition_module
                self.emo_conditioning_module = MLXConditioningModule(
                    input_dim=1024,                          # Emotion embedding dimension
                    conformer_dim=emo_cfg['output_size'],   # 512 (conformer output)
                    model_dim=1024,                          # Keep 1024 for emovec_layer input
                    num_latents=1,                           # Output 1 latent for emotion
                    conformer_layers=emo_cfg['num_blocks'], # From config (4)
                    conformer_heads=emo_cfg['attention_heads'],  # From config (4)
                    conformer_ff_mult=emo_cfg['linear_units'] // emo_cfg['output_size'],  # 1024/512 = 2
                    perceiver_depth=2,                       # 2 layers (matches PyTorch)
                    perceiver_heads=emo_cfg['attention_heads'],  # From config (4)
                    perceiver_ff_mult=emo_cfg['perceiver_mult']  # From config (2)
                )
            else:
                # Fallback to defaults
                self.emo_conditioning_module = MLXConditioningModule(
                    input_dim=1024,
                    conformer_dim=512,
                    model_dim=1024,
                    num_latents=1,
                    conformer_layers=4,      # Default
                    conformer_heads=4,       # Default
                    conformer_ff_mult=2,     # Default
                    perceiver_depth=2
                )
            
            # Emotion projection layers (matches PyTorch)
            self.emovec_layer = nn.Linear(1024, model_dim)  # 1024 → 1280
            self.emo_layer = nn.Linear(model_dim, model_dim)  # 1280 → 1280
            
            print(">> MLX: Using pure MLX conditioning (Conformer 512D + Perceiver 1280D)")
            print(">> MLX: Emotion conditioning with dedicated Conformer + Perceiver")
        else:
            self.conditioning_module = None
            self.emo_conditioning_module = None
            self.emovec_layer = None
            self.emo_layer = None
        
        # Speed embeddings (for duration control)
        self.speed_emb = nn.Embedding(2, model_dim)
        init_embedding_pytorch_compatible(self.speed_emb)
        
        # Core embeddings
        self.text_embedding = nn.Embedding(number_text_tokens + 1, model_dim)
        self.mel_embedding = nn.Embedding(number_mel_codes, model_dim)
        
        # Initialize embeddings with PyTorch-compatible weights
        init_embedding_pytorch_compatible(self.text_embedding)
        init_embedding_pytorch_compatible(self.mel_embedding)
        
        # Transformer layers (GPT2 style)
        self.transformer_blocks = [
            MLXTransformerBlock(model_dim, heads) for _ in range(layers)
        ]
        
        # CRITICAL: GPT2Model has a final LayerNorm (gpt.ln_f) after all transformer blocks
        # This is separate from the final_norm used before mel_head
        # PyTorch uses BOTH: transformer_blocks → gpt_ln_f → final_norm → mel_head
        self.gpt_ln_f = nn.LayerNorm(model_dim)  # GPT2Model's final LayerNorm
        
        # Positional embeddings (learned)
        max_mel_tokens = kwargs.get('max_mel_tokens', 1815)
        max_text_tokens = kwargs.get('max_text_tokens', 600)
        self.mel_pos_embedding = nn.Embedding(max_mel_tokens, model_dim)
        init_embedding_pytorch_compatible(self.mel_pos_embedding)
        # Use MLXLearnedPositionEmbeddings to match PyTorch LearnedPositionEmbeddings behavior
        self.text_pos_embedding = MLXLearnedPositionEmbeddings(max_text_tokens + 2, model_dim, init=0.02)
        
        # Conditioning parameters
        self.cond_num = kwargs.get('condition_num_latent', 32)
        
        # Emotion and speaker conditioning layers (for non-MLX-conditioning mode)
        # Only create if not using pure MLX conditioning
        if not use_mlx_conditioning:
            self.emo_layer = nn.Linear(model_dim, model_dim)
            self.emovec_layer = nn.Linear(1024, model_dim)
            init_linear_pytorch_compatible(self.emo_layer)
            init_linear_pytorch_compatible(self.emovec_layer)
        
        # Conditioning projection (always needed for compatibility)
        self.cond_projection = nn.Linear(1024, model_dim)  # Project semantic features
        init_linear_pytorch_compatible(self.cond_projection)
        
        # Output heads
        self.mel_head = nn.Linear(model_dim, number_mel_codes)
        self.text_head = nn.Linear(model_dim, number_text_tokens + 1)
        init_linear_pytorch_compatible(self.mel_head)
        init_linear_pytorch_compatible(self.text_head)
        
        # Normalization (used in lm_head, after gpt_ln_f)
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
        
        # 🔥 P0优化: JIT预热标志（延迟到load_weights之后）
        self._jit_warmed_up = False
    
    def load_weights_from_dict(self, mlx_weights):
        """
        Load weights from MLX npz cache including all transformer layers.
        
        Args:
            mlx_weights: Dictionary of MLX arrays from cache
        """
        loaded = 0
        
        # Load embeddings and heads
        # Note: Position embeddings use .emb.weight in PyTorch checkpoint
        # CRITICAL: PyTorch has TWO LayerNorms that are BOTH used in inference:
        #   1. gpt.ln_f: inside GPT2Model, after all transformer blocks
        #   2. final_norm: in lm_head, before mel_head
        # Both are applied sequentially: transformer_blocks → gpt.ln_f → final_norm → mel_head
        simple_mappings = {
            'text_embedding.weight': ('text_embedding', 'weight'),
            'mel_embedding.weight': ('mel_embedding', 'weight'),
            'mel_pos_embedding.emb.weight': ('mel_pos_embedding', 'weight'),
            'text_pos_embedding.emb.weight': ('text_pos_embedding', 'emb'),  # Use 'emb' for MLXLearnedPositionEmbeddings
            'mel_head.weight': ('mel_head', 'weight'),
            'mel_head.bias': ('mel_head', 'bias'),
            'text_head.weight': ('text_head', 'weight'),
            'text_head.bias': ('text_head', 'bias'),
            'gpt.ln_f.weight': ('gpt_ln_f', 'weight'),      # GPT2Model's final LayerNorm
            'gpt.ln_f.bias': ('gpt_ln_f', 'bias'),
            'final_norm.weight': ('final_norm', 'weight'),   # lm_head's LayerNorm
            'final_norm.bias': ('final_norm', 'bias'),
            'speed_emb.weight': ('speed_emb', 'weight'),
            'emo_layer.weight': ('emo_layer', 'weight'),
            'emo_layer.bias': ('emo_layer', 'bias'),
            'emovec_layer.weight': ('emovec_layer', 'weight'),
            'emovec_layer.bias': ('emovec_layer', 'bias'),
        }
        
        for key, (module_name, attr_name) in simple_mappings.items():
            # Try original key first, then fixed key
            fixed_key = self._fix_key_name(key)
            weight = None
            
            if key in mlx_weights:
                weight = mlx_weights[key]
            elif fixed_key != key and fixed_key in mlx_weights:
                weight = mlx_weights[fixed_key]
                print(f"   🔧 Used fixed key: '{key}' -> '{fixed_key}'")
            
            if weight is not None:
                module = getattr(self, module_name)
                setattr(module, attr_name, weight)
                loaded += 1
        
        # Load transformer layers
        # PyTorch GPT2 format: gpt.h.{layer_idx}.{component}.{param}
        for layer_idx in range(self.layers):
            block = self.transformer_blocks[layer_idx]
            prefix = f"gpt.h.{layer_idx}"
            
            # Attention weights (c_attn combines q,k,v projections in PyTorch)
            c_attn_weight = f"{prefix}.attn.c_attn.weight"
            c_attn_bias = f"{prefix}.attn.c_attn.bias"
            
            # Try original key first, then fixed key
            fixed_c_attn_weight = self._fix_key_name(c_attn_weight)
            weight = None
            
            if c_attn_weight in mlx_weights:
                weight = mlx_weights[c_attn_weight]
            elif fixed_c_attn_weight != c_attn_weight and fixed_c_attn_weight in mlx_weights:
                weight = mlx_weights[fixed_c_attn_weight]
                print(f"   🔧 Used fixed key: '{c_attn_weight}' -> '{fixed_c_attn_weight}'")
            
            if weight is not None:
                # Split combined qkv weight into separate q, k, v
                # PyTorch GPT2 c_attn.weight shape: (in_features, out_features) = (D, 3*D)
                # This is already in the correct format from checkpoint
                combined = weight  # (D, 3*D)
                
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
            
            # Try original key first, then fixed key for bias
            fixed_c_attn_bias = self._fix_key_name(c_attn_bias)
            bias = None
            
            if c_attn_bias in mlx_weights:
                bias = mlx_weights[c_attn_bias]
            elif fixed_c_attn_bias != c_attn_bias and fixed_c_attn_bias in mlx_weights:
                bias = mlx_weights[fixed_c_attn_bias]
                print(f"   🔧 Used fixed key: '{c_attn_bias}' -> '{fixed_c_attn_bias}'")
            
            if bias is not None:
                # PyTorch GPT2 c_attn.bias shape: (3*model_dim,)
                combined = bias
                
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
            print(f">> Loaded {cond_loaded} speaker conditioning weights")
            
            # Load emotion conditioning weights
            if self.emo_conditioning_module is not None:
                print(">> Loading MLX Emotion Conditioning weights...")
                emo_loaded = self._load_emo_conditioning_weights(mlx_weights)
                loaded += emo_loaded
                print(f">> Loaded {emo_loaded} emotion conditioning weights")
        
        print(f">> Loaded {loaded} weight tensors total from MLX cache")
        
        # 🔥 P0优化: 加载权重后立即JIT预热
        if not self._jit_warmed_up:
            self._warmup_jit()
            self._jit_warmed_up = True
        
        return loaded
    
    def _warmup_jit(self):
        """JIT预热：预先编译Metal kernels，避免首次推理时的编译开销"""
        print(">> 🔥 JIT Warmup: Pre-compiling Metal kernels...")
        import time
        t0 = time.time()
        
        # 创建dummy输入
        dummy_seq = mx.zeros((1, 10, self.model_dim))
        
        # 预热transformer blocks
        for block in self.transformer_blocks:
            dummy_seq, _ = block(dummy_seq, causal_mask=self.causal_mask, use_cache=True)
        
        # 预热LayerNorm
        dummy_seq = self.gpt_ln_f(dummy_seq)
        dummy_seq = self.final_norm(dummy_seq)
        
        # 预热mel_head
        dummy_logits = self.mel_head(dummy_seq)
        
        # 强制执行，触发编译
        mx.eval(dummy_logits)
        
        print(f">> JIT Warmup completed in {time.time()-t0:.2f}s")
    
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
        
        # ✅ FIX: Load Conv2d Subsampling weights with format conversion
        # Conv2d: conditioning_encoder.embed.conv.0.weight
        if 'conditioning_encoder.embed.conv.0.weight' in weights:
            # PyTorch format: (out_channels=512, in_channels=1, H=3, W=3)
            # MLX format: (out_channels=512, H=3, W=3, in_channels=1)
            # Need to permute: (O, I, H, W) → (O, H, W, I)
            pytorch_weight = weights['conditioning_encoder.embed.conv.0.weight']
            # Transpose from (512, 1, 3, 3) to (512, 3, 3, 1)
            mlx_weight = pytorch_weight.transpose(0, 2, 3, 1)
            conformer.subsampling.conv.conv.weight = mlx_weight
            loaded += 1
        
        if 'conditioning_encoder.embed.conv.0.bias' in weights:
            # Bias is the same format: (512,)
            conformer.subsampling.conv.conv.bias = weights['conditioning_encoder.embed.conv.0.bias']
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
        
        # Conformer blocks (dynamic number of layers)
        for layer_idx in range(len(conformer.blocks)):
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
                # PyTorch format: (out_channels, in_channels//groups, kernel_size) = (512, 1, 15)
                # MLX Conv1d format: (out_channels, kernel_size, in_channels//groups) = (512, 15, 1)
                w = weights[f"{prefix}.conv_module.depthwise_conv.weight"]  # (512, 1, 15)
                block.conv.depthwise.conv.weight = w.transpose(0, 2, 1)  # (512, 15, 1)
                loaded += 1
            if f"{prefix}.conv_module.depthwise_conv.bias" in weights:
                block.conv.depthwise.conv.bias = weights[f"{prefix}.conv_module.depthwise_conv.bias"]
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
    
    def _load_emo_conditioning_weights(self, weights):
        """Load Emotion Conditioning (Conformer + Perceiver) weights from PyTorch checkpoint"""
        loaded = 0
        
        # Load emotion Perceiver weights
        loaded += self._load_emo_perceiver_weights(weights)
        
        # Load emotion Conformer weights
        loaded += self._load_emo_conformer_weights(weights)
        
        return loaded
    
    def _load_emo_perceiver_weights(self, weights):
        """Load Emotion Perceiver Resampler weights"""
        loaded = 0
        perceiver = self.emo_conditioning_module.perceiver
        
        # Latents
        if 'emo_perceiver_encoder.latents' in weights:
            perceiver.latents = weights['emo_perceiver_encoder.latents']
            loaded += 1
        
        # proj_context (512 → 1024, note: different from speaker conditioning)
        if 'emo_perceiver_encoder.proj_context.weight' in weights:
            perceiver.proj_context.weight = weights['emo_perceiver_encoder.proj_context.weight']
            loaded += 1
        if 'emo_perceiver_encoder.proj_context.bias' in weights:
            perceiver.proj_context.bias = weights['emo_perceiver_encoder.proj_context.bias']
            loaded += 1
        
        # Perceiver layers (2 layers)
        for layer_idx in range(2):
            layer = perceiver.layers[layer_idx]
            attn, ff = layer
            
            prefix = f"emo_perceiver_encoder.layers.{layer_idx}"
            
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
            
            # Feed-forward
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
        
        # Final norm
        if 'emo_perceiver_encoder.norm.gamma' in weights:
            perceiver.norm.scale = weights['emo_perceiver_encoder.norm.gamma']
            loaded += 1
        
        return loaded
    
    def _load_emo_conformer_weights(self, weights):
        """Load Emotion Conformer Encoder weights"""
        loaded = 0
        conformer = self.emo_conditioning_module.conformer
        
        # Conv2d Subsampling
        if 'emo_conditioning_encoder.embed.conv.0.weight' in weights:
            # PyTorch format: (out_channels, in_channels, H, W) = (512, 1, 3, 3)
            # MLX format: (out_channels, H, W, in_channels) = (512, 3, 3, 1)
            pytorch_weight = weights['emo_conditioning_encoder.embed.conv.0.weight']
            mlx_weight = pytorch_weight.transpose(0, 2, 3, 1)
            conformer.subsampling.conv.conv.weight = mlx_weight
            loaded += 1
        if 'emo_conditioning_encoder.embed.conv.0.bias' in weights:
            conformer.subsampling.conv.conv.bias = weights['emo_conditioning_encoder.embed.conv.0.bias']
            loaded += 1
        
        # Linear projection
        if 'emo_conditioning_encoder.embed.out.0.weight' in weights:
            conformer.subsampling.out.weight = weights['emo_conditioning_encoder.embed.out.0.weight']
            loaded += 1
        if 'emo_conditioning_encoder.embed.out.0.bias' in weights:
            conformer.subsampling.out.bias = weights['emo_conditioning_encoder.embed.out.0.bias']
            loaded += 1
        
        # Position encoding
        if 'emo_conditioning_encoder.embed.pos_enc.pe' in weights:
            pe = weights['emo_conditioning_encoder.embed.pos_enc.pe']
            conformer.pos_encoding = pe.squeeze(0)[:1000]
            loaded += 1
        
        # Conformer blocks (dynamic number of layers)
        for layer_idx in range(len(conformer.blocks)):
            block = conformer.blocks[layer_idx]
            prefix = f"emo_conditioning_encoder.encoders.{layer_idx}"
            
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
            
            # Positional biases
            if f"{prefix}.self_attn.pos_bias_u" in weights:
                block.attn.pos_bias_u = weights[f"{prefix}.self_attn.pos_bias_u"]
                loaded += 1
            if f"{prefix}.self_attn.pos_bias_v" in weights:
                block.attn.pos_bias_v = weights[f"{prefix}.self_attn.pos_bias_v"]
                loaded += 1
            
            # Feed-forward
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
                block.conv.pointwise1.weight = weights[f"{prefix}.conv_module.pointwise_conv1.weight"]
                loaded += 1
            if f"{prefix}.conv_module.pointwise_conv1.bias" in weights:
                block.conv.pointwise1.bias = weights[f"{prefix}.conv_module.pointwise_conv1.bias"]
                loaded += 1
            if f"{prefix}.conv_module.depthwise_conv.weight" in weights:
                # PyTorch format: (out_channels, in_channels//groups, kernel_size) = (512, 1, 15)
                # MLX Conv1d format: (out_channels, kernel_size, in_channels//groups) = (512, 15, 1)
                w = weights[f"{prefix}.conv_module.depthwise_conv.weight"]
                block.conv.depthwise.conv.weight = w.transpose(0, 2, 1)
                loaded += 1
            if f"{prefix}.conv_module.depthwise_conv.bias" in weights:
                block.conv.depthwise.conv.bias = weights[f"{prefix}.conv_module.depthwise_conv.bias"]
                loaded += 1
            if f"{prefix}.conv_module.pointwise_conv2.weight" in weights:
                block.conv.pointwise2.weight = weights[f"{prefix}.conv_module.pointwise_conv2.weight"]
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
            
            # Layer norms
            if f"{prefix}.norm_mha.weight" in weights:
                block.norm_attn.weight = weights[f"{prefix}.norm_mha.weight"]
                loaded += 1
            if f"{prefix}.norm_mha.bias" in weights:
                block.norm_attn.bias = weights[f"{prefix}.norm_mha.bias"]
                loaded += 1
            if f"{prefix}.norm_conv.weight" in weights:
                block.norm_conv.weight = weights[f"{prefix}.norm_conv.weight"]
                loaded += 1
            if f"{prefix}.norm_conv.bias" in weights:
                block.norm_conv.bias = weights[f"{prefix}.norm_conv.bias"]
                loaded += 1
            if f"{prefix}.norm_ff.weight" in weights:
                block.norm_ff.weight = weights[f"{prefix}.norm_ff.weight"]
                loaded += 1
            if f"{prefix}.norm_ff.bias" in weights:
                block.norm_ff.bias = weights[f"{prefix}.norm_ff.bias"]
                loaded += 1
            if f"{prefix}.norm_final.weight" in weights:
                block.norm_final.weight = weights[f"{prefix}.norm_final.weight"]
                loaded += 1
            if f"{prefix}.norm_final.bias" in weights:
                block.norm_final.bias = weights[f"{prefix}.norm_final.bias"]
                loaded += 1
        
        # Final norm
        if 'emo_conditioning_encoder.after_norm.weight' in weights:
            conformer.after_norm.weight = weights['emo_conditioning_encoder.after_norm.weight']
            loaded += 1
        if 'emo_conditioning_encoder.after_norm.bias' in weights:
            conformer.after_norm.bias = weights['emo_conditioning_encoder.after_norm.bias']
            loaded += 1
        
        return loaded
    
    def beam_search_forward(self, text_tokens, conditioning=None, max_length=1500, num_beams=3, length_penalty=1.0, **kwargs):
        """
        True beam search implementation (matches PyTorch transformers).
        
        At each step:
        1. Expand each beam with top-k next tokens
        2. Score all candidates (num_beams * vocab_size)
        3. Keep top num_beams candidates for next step
        4. Apply repetition penalty and length penalty
        
        Args:
            text_tokens: Text token IDs (batch, text_len) - MLX array
            conditioning: Conditioning latent (batch, cond_len, model_dim) - MLX array
            max_length: Maximum mel tokens to generate
            num_beams: Number of beams to maintain
            length_penalty: Length penalty (>1.0 encourages longer sequences)
            
        Returns:
            Best sequence from beam search (batch, mel_len) - MLX array
        """
        import time
        import os
        
        print(f">> [True Beam Search] num_beams={num_beams}, length_penalty={length_penalty}, save_all_beams={kwargs.get('save_all_beams', False)}")
        
        # Set random seed for reproducibility
        base_seed = kwargs.get('seed', None)
        if base_seed is None:
            fixed_seed_str = os.environ.get('MLX_FIXED_SEED', None)
            if fixed_seed_str:
                base_seed = int(fixed_seed_str)
            else:
                base_seed = int(time.time() * 1000000) % (2**32)
        mx.random.seed(base_seed)
        import numpy as np
        np.random.seed(base_seed)  # Also set numpy seed for multinomial sampling
        print(f">> [Beam Search] Using seed: {base_seed}")
        print(f">> [Beam Search] Starting initialization...")
        
        # Create default conditioning if needed
        if conditioning is None:
            batch_size = text_tokens.shape[0]
            cond_len = 32
            conditioning = mx.zeros((batch_size, cond_len, self.model_dim))
        
        batch_size = text_tokens.shape[0]
        assert batch_size == 1, "Beam search currently only supports batch_size=1"
        
        # Process text tokens (add boundary tokens)
        text_input = text_tokens[0]
        text_input_list = text_input.tolist()
        text_input_filtered_list = [t for t in text_input_list if t != 0 and t != 1]
        text_input_processed_list = [0] + text_input_filtered_list + [1]
        text_input_processed = mx.array(text_input_processed_list, dtype=mx.int32)
        
        
        text_seq_len = text_input_processed.shape[0]
        max_text_pos = self.text_pos_embedding.emb.shape[0]
        if text_seq_len > max_text_pos:
            text_seq_len = max_text_pos
            text_input_processed = text_input_processed[:text_seq_len]
        
        # Get text embeddings
        text_emb = self.text_embedding(text_input_processed.reshape(1, -1))
        text_pos_emb = mx.stack([self.text_pos_embedding.emb[j] for j in range(text_seq_len)], axis=0)
        text_emb = text_emb[0] + text_pos_emb
        text_emb = text_emb.reshape(1, -1, self.model_dim)
        
        # Create initial context
        context = mx.concatenate([conditioning, text_emb], axis=1)
        context_len = context.shape[1]
        
        # Add start token
        start_token_ids = mx.full((1, 1), self.start_mel_token, dtype=mx.int32)
        start_token_emb = self.mel_embedding(start_token_ids)
        # 🔧 FIX: Use relative position 0 for start_mel (matches PyTorch)
        start_token_emb = start_token_emb + self.mel_pos_embedding.weight[0:1]
        
        # 🔧 CRITICAL FIX: PyTorch-style beam initialization
        # All beams start with the SAME start_token, but with different scores:
        # beam_scores = [0.0, -1e9, -1e9, ...] 
        # This ensures only the first beam expands in step 1, creating diversity
        
        # First forward pass to get initial KV cache (shared by all beams initially)
        initial_sequence = mx.concatenate([context, start_token_emb], axis=1)  # (1, context+1, D)
        hidden = initial_sequence
        initial_past_kvs = []
        for block in self.transformer_blocks:
            hidden, kv = block(hidden, causal_mask=self.causal_mask, past_kv=None, use_cache=True)
            initial_past_kvs.append(kv)
        hidden = self.gpt_ln_f(hidden)  # GPT2Model's final LayerNorm
        hidden = self.final_norm(hidden)  # lm_head's LayerNorm
        
        # Initialize ALL beams with the SAME start token
        beam_codes = []
        beam_scores = []
        beam_past_kvs = []
        beam_finished = []
        # 🔧 Track full input_ids like PyTorch (for repetition_penalty)
        # PyTorch: input_ids = [fake_inputs..., start_mel_token, generated_tokens...]
        # We simulate this with a list of token IDs
        beam_input_ids = []
        # Fake inputs are all 1s in PyTorch (placeholder tokens)
        fake_input_token = 1
        initial_input_ids = [fake_input_token] * (context_len + text_seq_len) + [self.start_mel_token]
        
        for i in range(num_beams):
            # All beams start with the same start_mel_token
            beam_codes.append(mx.array([[self.start_mel_token]], dtype=mx.int32))
            
            # PyTorch trick: only first beam has score 0, others have -1e9
            # This ensures first expansion only considers first beam's candidates
            if i == 0:
                beam_scores.append(0.0)
            else:
                beam_scores.append(-1e9)
            
            # All beams share the same initial KV cache (will diverge after first step)
            beam_past_kvs.append([kv for kv in initial_past_kvs])
            beam_finished.append(False)
            # Initialize input_ids with fake_inputs + start_mel_token
            beam_input_ids.append(initial_input_ids.copy())
        
        print(f">> [Beam Search] Initialized {num_beams} beams (PyTorch-style: scores=[0, -1e9, ...])")
        
        # Main beam search loop
        repetition_penalty = kwargs.get('repetition_penalty', 10.0)
        
        for step in range(max_length - 1):
            if all(beam_finished):
                break
            
            # Debug: print beam_scores at step 1
            if step == 1:
                print(f"\n>> [Step 1 START] Beam scores:")
                for i in range(len(beam_scores)):
                    print(f"   Beam {i}: beam_score={beam_scores[i]:.4f}, tokens={beam_codes[i][0].tolist()}")
            
            if step % 10 == 0 and step > 0:
                finished_count = sum(beam_finished)
                print(f">> [Beam Search] Step {step}/{max_length}, {finished_count}/{num_beams} beams finished", flush=True)
            
            # 🔧 PERFORMANCE FIX: Batch process all active beams together (15x faster!)
            # Collect active beams
            active_beam_indices = [i for i in range(num_beams) if not beam_finished[i]]
            
            if not active_beam_indices:
                break  # All beams finished
            
            # Batch prepare embeddings for all active beams
            batch_token_ids = []
            batch_positions = []
            for beam_idx in active_beam_indices:
                last_token_id = int(beam_codes[beam_idx][0, -1])
                batch_token_ids.append(last_token_id)
                # 🔧 FIX: Use relative position (matches PyTorch)
                # beam_codes includes start_mel, so shape[1]=1 means only start_mel
                # position should be: step=1→pos=2, step=2→pos=3, ...
                current_pos = beam_codes[beam_idx].shape[1] + 1  # +1 because start_mel is at pos 0
                batch_positions.append(current_pos)
            
            # Batch embedding (num_active_beams, 1, model_dim)
            batch_token_ids_mx = mx.array([[tid] for tid in batch_token_ids], dtype=mx.int32)
            batch_token_emb = self.mel_embedding(batch_token_ids_mx)  # (num_active, 1, D)
            
            # Batch position embedding (use relative positions)
            batch_pos_emb = mx.stack([self.mel_pos_embedding.weight[pos:pos+1][0] for pos in batch_positions], axis=0)
            batch_pos_emb = batch_pos_emb.reshape(len(active_beam_indices), 1, self.model_dim)
            batch_hidden = batch_token_emb + batch_pos_emb  # (num_active, 1, D)
            
            # Batch forward through transformer (ALL active beams at once!)
            batch_new_kvs = [[] for _ in range(len(active_beam_indices))]
            for layer_idx, block in enumerate(self.transformer_blocks):
                # Process all beams together
                layer_hiddens = []
                layer_new_kvs = []
                for i, beam_idx in enumerate(active_beam_indices):
                    h = batch_hidden[i:i+1]  # (1, 1, D)
                    h, kv = block(h, causal_mask=self.causal_mask,
                                past_kv=beam_past_kvs[beam_idx][layer_idx], use_cache=True)
                    layer_hiddens.append(h)
                    layer_new_kvs.append(kv)
                
                # Stack results
                batch_hidden = mx.concatenate(layer_hiddens, axis=0)  # (num_active, 1, D)
                for i, kv in enumerate(layer_new_kvs):
                    batch_new_kvs[i].append(kv)
            
            # Batch final norms and mel_head
            batch_hidden = self.gpt_ln_f(batch_hidden)  # (num_active, 1, D)
            batch_hidden = self.final_norm(batch_hidden)  # (num_active, 1, D)
            batch_logits = self.mel_head(batch_hidden[:, 0, :])  # (num_active, vocab)
            
            # 🔥 P0优化: 强制计算logits
            mx.eval(batch_logits)
            
            # Batch log_softmax
            batch_max_logits = mx.max(batch_logits, axis=-1, keepdims=True)
            batch_exp_logits = mx.exp(batch_logits - batch_max_logits)
            batch_log_sum_exp = mx.log(mx.sum(batch_exp_logits, axis=-1, keepdims=True)) + batch_max_logits
            batch_log_probs = batch_logits - batch_log_sum_exp  # (num_active, vocab)
            
            # Distribute results back to all beams
            all_log_probs = []
            all_new_kvs = []
            active_idx = 0
            for beam_idx in range(num_beams):
                if beam_finished[beam_idx]:
                    vocab_size = self.mel_head.weight.shape[0]
                    all_log_probs.append(mx.full((vocab_size,), -1e9))
                    all_new_kvs.append(None)
                else:
                    all_log_probs.append(batch_log_probs[active_idx])
                    all_new_kvs.append(batch_new_kvs[active_idx])
                    active_idx += 1
            
            # 🔧 CRITICAL: Strict PyTorch beam search replication
            # PyTorch: log_softmax -> logits_processor -> add beam_scores
            import numpy as np
            vocab_size = all_log_probs[0].shape[0]
            
            # Get parameters
            temperature = kwargs.get('temperature', 0.8)
            top_k = kwargs.get('top_k', 30)
            top_p = kwargs.get('top_p', 0.8)
            
            # Step 1: all_log_probs are already log_softmax results (WITHOUT beam_scores)
            # Step 2: Apply logits_processor (repetition_penalty)
            # Step 3: Add beam_scores
            processed_scores = []
            for beam_idx in range(num_beams):
                if beam_finished[beam_idx]:
                    # Finished beam: use -inf probs
                    processed_scores.append(mx.full((vocab_size,), -1e9))
                    continue
                
                # Start with log_probs (already log_softmax, WITHOUT beam_scores)
                log_probs_np = np.array(all_log_probs[beam_idx])
                
                # 🔧 Apply repetition_penalty (PyTorch's RepetitionPenaltyLogitsProcessor)
                # PyTorch applies to ALL unique tokens in input_ids (including fake_inputs)
                if repetition_penalty != 1.0:
                    # Use complete input_ids (fake_inputs + start_mel + generated_mels)
                    prev_tokens = set(beam_input_ids[beam_idx])
                    for token_id in prev_tokens:
                        if token_id < len(log_probs_np):
                            if log_probs_np[token_id] < 0:
                                log_probs_np[token_id] *= repetition_penalty
                            else:
                                log_probs_np[token_id] /= repetition_penalty
                
                # Add beam_score (PyTorch: next_token_scores_processed + beam_scores)
                processed = mx.array(log_probs_np) + beam_scores[beam_idx]
                processed_scores.append(processed)
            
            # Step 4: Reshape to (batch=1, num_beams * vocab_size) - PyTorch line 3537
            all_scores_flat = mx.concatenate([scores.reshape(-1) for scores in processed_scores])
            
            # Step 3: PyTorch sampling (lines 3546-3550)
            # Debug: analyze scores distribution at step 1
            if step == 1:
                scores_flat_np = np.array(all_scores_flat)
                print(f"\n>> [Step 1] All scores stats (before softmax):")
                print(f"   Total candidates: {len(scores_flat_np)}")
                print(f"   Min: {scores_flat_np.min():.4f}, Max: {scores_flat_np.max():.4f}")
                for beam_idx in range(num_beams):
                    start = beam_idx * vocab_size
                    end = start + vocab_size
                    beam_scores_slice = scores_flat_np[start:end]
                    print(f"   Beam {beam_idx} range: [{beam_scores_slice.min():.4f}, {beam_scores_slice.max():.4f}], " + 
                          f"top token score: {beam_scores_slice.max():.4f}")
            
            # Softmax over all candidates
            all_probs_flat = mx.softmax(all_scores_flat)
            
            # PyTorch sampling (lines 3546-3550): multinomial with sorting
            n_tokens_to_keep = 2 * num_beams  # PyTorch: max(2, 1 + n_eos_tokens) * num_beams
            
            probs_np = np.array(all_probs_flat)
            scores_np = np.array(all_scores_flat)
            
            # 🔧 FIX: Use Top-K instead of multinomial for stability
            # Multinomial has RNG differences between PyTorch and NumPy
            # Top-K guarantees the best candidates are selected
            use_topk = True  # Set to False to use multinomial (for debugging)
            
            if use_topk:
                # Deterministic top-k selection
                sorted_order = np.argsort(scores_np)[::-1]  # Descending
                sampled_indices_sorted = sorted_order[:n_tokens_to_keep]
                sampled_scores_sorted = scores_np[sampled_indices_sorted]
                
                if step in [0, 1]:
                    print(f"   [Top-K Selection] Selected top-{n_tokens_to_keep} candidates (deterministic)")
            else:
                # Original multinomial sampling (has RNG differences)
                sampled_indices = np.random.choice(
                    len(probs_np),
                    size=min(n_tokens_to_keep, len(probs_np)),
                    replace=False,
                    p=probs_np / probs_np.sum()
                )
                
                # Gather sampled scores and sort descending (PyTorch: gather + sort)
                sampled_scores = scores_np[sampled_indices]
                sorted_order = np.argsort(sampled_scores)[::-1]
                sampled_indices_sorted = sampled_indices[sorted_order]
                sampled_scores_sorted = sampled_scores[sorted_order]
            
            # Keep more candidates to handle finished beams (PyTorch logic)
            # We need to select num_beams active beams from potentially n_tokens_to_keep candidates
            candidate_indices_flat = mx.array(sampled_indices_sorted, dtype=mx.int32)
            candidate_scores_flat = mx.array(sampled_scores_sorted)
            
            
            
            # Build new beams - PyTorch beam_scorer.process logic
            # Separate finished beams (with eos) from active beams
            new_beam_codes = []
            new_beam_scores = []
            new_beam_past_kvs = []
            new_beam_finished = []
            new_beam_input_ids = []
            
            # Debug: print first and second step details
            if step in [0, 1]:
                print(f"\n>> [Step {step}] Token selection:")
                for i in range(min(num_beams * 2, len(candidate_indices_flat))):
                    source_beam = int(candidate_indices_flat[i] // vocab_size)
                    token = int(candidate_indices_flat[i] % vocab_size)
                    score = float(candidate_scores_flat[i])
                    print(f"   Candidate {i}: source_beam={source_beam}, token={token}, score={score:.4f}")
            
            # PyTorch: if token is eos, add to beam_hyps; otherwise add to active beams
            # We select top num_beams from NON-FINISHED candidates
            beam_idx = 0
            for i in range(len(candidate_indices_flat)):  # Iterate over all candidates
                if beam_idx >= num_beams:
                    break
                    
                source_beam_idx = int(candidate_indices_flat[i] // vocab_size)
                token_id = int(candidate_indices_flat[i] % vocab_size)
                score = float(candidate_scores_flat[i])
                
                # If this is a stop token, mark as finished and add to beams
                if token_id == self.stop_mel_token:
                    new_codes = mx.concatenate([beam_codes[source_beam_idx], 
                                               mx.array([[token_id]], dtype=mx.int32)], axis=1)
                    new_beam_codes.append(new_codes)
                    new_beam_scores.append(score)
                    new_beam_past_kvs.append(beam_past_kvs[source_beam_idx])  # Doesn't matter, won't use
                    new_beam_finished.append(True)
                    new_input_ids = beam_input_ids[source_beam_idx] + [token_id]
                    new_beam_input_ids.append(new_input_ids)
                    beam_idx += 1
                # If source beam was already finished, skip this candidate
                elif beam_finished[source_beam_idx]:
                    continue  # Skip finished beams
                # Otherwise, add as active beam
                else:
                    new_codes = mx.concatenate([beam_codes[source_beam_idx], 
                                               mx.array([[token_id]], dtype=mx.int32)], axis=1)
                    new_beam_codes.append(new_codes)
                    new_beam_scores.append(score)
                    new_beam_past_kvs.append(all_new_kvs[source_beam_idx])
                    new_beam_finished.append(False)
                    new_input_ids = beam_input_ids[source_beam_idx] + [token_id]
                    new_beam_input_ids.append(new_input_ids)
                    beam_idx += 1
            
            # Update beams
            beam_codes = new_beam_codes
            beam_scores = new_beam_scores
            beam_past_kvs = new_beam_past_kvs
            beam_finished = new_beam_finished
            beam_input_ids = new_beam_input_ids
            
            # Debug: print step 0 and 1 results
            if step in [0, 1]:
                print(f"\n>> [Step {step}] After beam update:")
                for i in range(len(beam_codes)):
                    tokens = beam_codes[i][0].tolist()
                    print(f"   Beam {i+1}: tokens={tokens}, score={beam_scores[i]:.4f}")
        
        # Select best beam (PyTorch uses generated_len, not total length)
        # decoder_prompt_len = 1 (start_mel_token)
        decoder_prompt_len = 1
        
        print(f"\n>> [Beam Search] Final beam selection:")
        for i in range(num_beams):
            total_length = beam_codes[i].shape[1]
            generated_len = total_length - decoder_prompt_len  # PyTorch: cur_len - decoder_prompt_len
            raw_score = beam_scores[i]
            normalized_score = raw_score / (generated_len ** length_penalty) if generated_len > 0 else raw_score
            print(f"   Beam {i+1}: total_len={total_length}, generated_len={generated_len}, raw_score={raw_score:.4f}, normalized={normalized_score:.4f}, finished={beam_finished[i]}")
            # Print first 10 tokens
            tokens_preview = beam_codes[i][0, :min(10, total_length)].tolist()
            print(f"           First {min(10, total_length)} tokens: {tokens_preview}")
        
        best_idx = 0
        generated_len_0 = beam_codes[0].shape[1] - decoder_prompt_len
        best_score = beam_scores[0] / (generated_len_0 ** length_penalty) if generated_len_0 > 0 else beam_scores[0]
        
        for i in range(1, num_beams):
            generated_len = beam_codes[i].shape[1] - decoder_prompt_len
            score = beam_scores[i] / (generated_len ** length_penalty) if generated_len > 0 else beam_scores[i]
            if score > best_score:
                best_score = score
                best_idx = i
        
        # 🔧 Save all beams if requested (for debugging)
        save_all_beams = kwargs.get('save_all_beams', False)
        if save_all_beams:
            print(f"\n>> [Beam Search] Saving all {num_beams} beams for comparison...")
            all_beams_cleaned = []
            for i in range(num_beams):
                codes = beam_codes[i]
                codes_list = codes[0].tolist()
                # Remove stop token if present
                if self.stop_mel_token in codes_list:
                    stop_idx = codes_list.index(self.stop_mel_token)
                    codes = codes[:, :stop_idx]
                all_beams_cleaned.append(codes)
                print(f"   Beam {i+1}: {codes.shape[1]} tokens (after removing stop)")
        
        best_codes = beam_codes[best_idx]
        
        # Remove stop token if present
        codes_list = best_codes[0].tolist()
        if self.stop_mel_token in codes_list:
            stop_idx = codes_list.index(self.stop_mel_token)
            best_codes = best_codes[:, :stop_idx]
        
        print(f">> [Beam Search] Complete: selected beam {best_idx+1}, {best_codes.shape[1]} tokens (normalized_score={best_score:.4f})")
        
        if save_all_beams:
            return best_codes, all_beams_cleaned
        else:
            return best_codes
    
    def simple_forward(self, text_tokens, conditioning=None, max_length=1500, **kwargs):
        """
        Full forward pass with transformer layers for generation.
        完全遵循PyTorch transformers的generation流程
        
        Args:
            text_tokens: Text token IDs (batch, text_len) - MLX array
            conditioning: Conditioning latent (batch, cond_len, model_dim) - MLX array
            max_length: Maximum mel tokens to generate
            **kwargs: Extra arguments including:
                - temperature: float (default 1.0)
                - repetition_penalty: float (default 1.0) 
                - top_p: float (default 1.0)
                - top_k: int (default 0)
                - use_sampling: bool (default False)
                - debug_generation: bool (default False)
            
        Returns:
            Generated mel codes (batch, mel_len) - MLX array
        """
        # 🔧 修复：重置随机状态以避免推理间状态累积
        # 每次推理使用新的随机种子（基于时间），确保推理独立性
        import time
        import os
        # 🚀 使用优化版本的logits processors
        try:
            from indextts.gpt.mlx_transformers_generation_utils_optimized import LogitsProcessorList
            use_optimized = True
        except ImportError:
            from indextts.gpt.mlx_transformers_generation_utils import (
                LogitsProcessorList,
                TemperatureLogitsWarper,
                RepetitionPenaltyLogitsProcessor,
                TopPLogitsWarper,
                TopKLogitsWarper,
            )
            use_optimized = False
        
        seed = kwargs.get('seed', None)
        if seed is None:
            # 检查是否有固定种子的环境变量（用于调试/复现）
            fixed_seed_str = os.environ.get('MLX_FIXED_SEED', None)
            if fixed_seed_str:
                seed = int(fixed_seed_str)
            else:
                # 🔧 修复：默认使用随机种子，避免固定种子对某些文本不友好
                # 固定种子 (42) 会导致某些中文文本（如"今天天气真不错"）生成提前停止，丢字
                # 用户可通过 MLX_FIXED_SEED 环境变量设置固定种子用于调试
                seed = int(time.time() * 1000000) % (2**32)
        
        # Set random seed for reproducibility
        mx.random.seed(seed)
        
        # 🔧 按照PyTorch的方式构建logits_processor
        # 参考: transformers.generation_utils._get_logits_processor
        
        # 🔧 匹配PyTorch默认参数
        temperature = kwargs.get('temperature', 1.0)
        repetition_penalty = kwargs.get('repetition_penalty', 10.0)  # PyTorch默认10.0
        top_p = kwargs.get('top_p', 1.0)
        top_k = kwargs.get('top_k', 0)
        
        # 🚀 优化：使用优化版本的processor（自动选择最优组合）
        if use_optimized:
            logits_processor = LogitsProcessorList.create_optimized(
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                top_k=top_k
            )
            print(f">> [MLX] Logits processors (optimized): {[type(p).__name__ for p in logits_processor]}")
        else:
            # 回退到原始实现
            from indextts.gpt.mlx_transformers_generation_utils import (
                TemperatureLogitsWarper,
                RepetitionPenaltyLogitsProcessor,
                TopPLogitsWarper,
                TopKLogitsWarper,
            )
            logits_processor = LogitsProcessorList()
            if temperature is not None and temperature != 1.0:
                logits_processor.append(TemperatureLogitsWarper(temperature))
            if repetition_penalty is not None and repetition_penalty != 1.0:
                logits_processor.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
            if top_k is not None and top_k > 0:
                logits_processor.append(TopKLogitsWarper(top_k))
            if top_p is not None and top_p < 1.0:
                logits_processor.append(TopPLogitsWarper(top_p))
            print(f">> [MLX] Logits processors: {[type(p).__name__ for p in logits_processor]}")
        
        # Create default conditioning if not provided
        if conditioning is None:
            batch_size = text_tokens.shape[0]
            cond_len = 32
            conditioning = mx.zeros((batch_size, cond_len, self.model_dim))
        
        # 🔧 CRITICAL FIX: Match PyTorch's prepare_gpt_inputs logic
        # PyTorch filters out stop/start tokens and adds them back at boundaries
        # This is essential for correct text-to-mel alignment
        batch_size = text_tokens.shape[0]
        
        # Process each batch (usually batch_size=1 for inference)
        processed_text_embs = []
        for i in range(batch_size):
            text_input = text_tokens[i]  # (T,)
            
            # 1. Filter out stop_text_token (1) and start_text_token (0)
            # Use list comprehension (more reliable than MLX boolean indexing)
            text_input_list = text_input.tolist()
            text_input_filtered_list = [t for t in text_input_list if t != 0 and t != 1]
            
            # 2. Add start_text_token at beginning and stop_text_token at end
            # PyTorch: F.pad(text_input, (1, 0), value=0) then F.pad(..., (0, 1), value=1)
            text_input_processed_list = [0] + text_input_filtered_list + [1]
            text_input_processed = mx.array(text_input_processed_list, dtype=mx.int32)
            
            # 3. Get text embeddings + positional embeddings
            text_seq_len = text_input_processed.shape[0]
            max_text_pos = self.text_pos_embedding.emb.shape[0]
            
            # 检查是否越界
            if text_seq_len > max_text_pos:
                print(f"⚠️  WARNING: text_seq_len ({text_seq_len}) > max_text_pos ({max_text_pos})")
                text_seq_len = max_text_pos
                text_input_processed = text_input_processed[:text_seq_len]
            
            text_emb = self.text_embedding(text_input_processed.reshape(1, -1))  # (1, T, D)
            text_pos_emb = mx.stack([self.text_pos_embedding.emb[j] for j in range(text_seq_len)], axis=0)  # (T, D)
            text_emb = text_emb[0] + text_pos_emb  # (T, D)
            
            processed_text_embs.append(text_emb)
        
        # Stack back to batch
        text_emb = mx.stack(processed_text_embs, axis=0)  # (B, T, D)
        
        # Combine conditioning + text
        context = mx.concatenate([conditioning, text_emb], axis=1)  # (B, C+T, D)
        
        # CRITICAL: Add start_mel_token embedding (like PyTorch version)
        batch_size = text_tokens.shape[0]
        start_token_ids = mx.full((batch_size, 1), self.start_mel_token, dtype=mx.int32)
        start_token_emb = self.mel_embedding(start_token_ids)  # (B, 1, D)
        
        # CRITICAL: PyTorch uses RELATIVE position (from 0) for mel tokens, NOT absolute!
        # PyTorch: start_mel uses position 0, first generated mel uses position 2, etc.
        # See GPT2InferenceModel.forward line 148 and 158
        context_len = context.shape[1]  # Length of conditioning + text (for reference only)
        
        # 🔧 FIX: Use relative position 0 for start_mel_token (matches PyTorch)
        start_pos = 0  # PyTorch uses position 0 for start_mel_token
        start_token_emb = start_token_emb + self.mel_pos_embedding.weight[start_pos:start_pos+1]  # (B, 1, D) + (1, D)
        
        # Full initial sequence: [context] + [start_mel_token]
        sequence = mx.concatenate([context, start_token_emb], axis=1)
        
        # Autoregressive generation with KV caching for efficiency
        generated = []
        
        debug_generation = kwargs.get('debug_generation', False)
        
        if debug_generation:
            print(f"\n[DEBUG] Generation Setup:")
            print(f"  Text tokens shape: {text_tokens.shape} -> Processed: {text_emb.shape}")
            print(f"  Conditioning shape: {conditioning.shape}")
            print(f"  Context length: {context.shape[1]}")
            print(f"  Start mel token ID: {self.start_mel_token} at position {start_pos}")
        
        print(f">> [MLX] Starting autoregressive loop with KV cache (max_length={max_length})...")
        print(f"   Initial sequence: {sequence.shape[1]} tokens (context={context.shape[1]} + start_token=1)")
        print(f"   Start token position: {start_pos}")
        
        # First pass: process full context (including start_mel_token) and initialize KV cache
        hidden = sequence
        past_kvs = []
        for block in self.transformer_blocks:
            hidden, kv = block(hidden, causal_mask=self.causal_mask, past_kv=None, use_cache=True)
            past_kvs.append(kv)
        hidden = self.gpt_ln_f(hidden)  # GPT2Model's final LayerNorm
        hidden = self.final_norm(hidden)  # lm_head's LayerNorm
        
        # Get first token - 完全按照PyTorch流程
        # 1. 获取原始logits
        next_token_logits = self.mel_head(hidden[:, -1:, :])  # (B, 1, vocab)
        next_token_logits = next_token_logits[:, 0, :]  # (B, vocab) - 去掉seq维度
        
        # 🔥 P0优化: 强制计算，避免lazy evaluation堆积
        mx.eval(next_token_logits)
        
        # 2. 构建input_ids用于logits_processor (目前只有fake_inputs + start_mel_token)
        # PyTorch在第一步时input_ids包含所有fake_inputs(conditioning) + start_mel_token
        current_input_ids = mx.full((batch_size, context_len + 1), 1, dtype=mx.int32)  # fake inputs
        current_input_ids[:, -1] = self.start_mel_token  # 最后一个是start_mel_token
        
        # 3. 应用logits_processor (temperature, repetition_penalty, top_p, top_k等)
        next_token_scores = logits_processor(current_input_ids, next_token_logits)
        
        # 4. 根据use_sampling决定采样或argmax
        use_sampling = kwargs.get('use_sampling', False)
        
        if use_sampling:
            # 🔧 完全匹配PyTorch: probs = softmax(scores), then multinomial(probs)
            probs = mx.softmax(next_token_scores[0], axis=-1)  # (vocab,)
            # MLX categorical接受log_probs，所以用log(probs)或直接用scores
            next_token_id = mx.random.categorical(mx.log(probs + 1e-10))
            next_token = mx.array([[next_token_id]])
        else:
            # Greedy (argmax)
            next_token_id = mx.argmax(next_token_scores[0])
            next_token = mx.array([[next_token_id]])
            # 🔥 P0优化: 强制计算token
            mx.eval(next_token)
        
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
            
            # CRITICAL: PyTorch uses RELATIVE position (from 0) for mel tokens!
            # PyTorch logic (GPT2InferenceModel line 158):
            #   position = attention_mask.shape[1] - mel_len
            # Where mel_len = cached_mel_emb.shape[1] (不变)
            # attention_mask grows: mel_len+1, mel_len+2, mel_len+3, ...
            # So position grows: 1, 2, 3, ...
            # But start_mel used position 0, so:
            #   step=1 (first generated) → position = 1 + 1 = 2
            #   step=2 → position = 2 + 1 = 3
            relative_pos = step + 1  # +1 because start_mel used position 0
            if relative_pos < self.mel_pos_embedding.weight.shape[0]:
                mel_pos_enc = self.mel_pos_embedding.weight[relative_pos:relative_pos+1]  # (1, D)
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
            
            hidden = self.gpt_ln_f(hidden)  # GPT2Model's final LayerNorm
            hidden = self.final_norm(hidden)  # lm_head's LayerNorm
            
            # Get next token logits - 完全按照PyTorch流程
            # 1. 获取原始logits
            next_token_logits = self.mel_head(hidden)  # (B, 1, vocab)
            next_token_logits = next_token_logits[:, 0, :]  # (B, vocab)
            
            # 🔥 P0优化: 强制计算logits
            mx.eval(next_token_logits)
            
            # 2. 构建当前的input_ids (用于logits_processor)
            # PyTorch: input_ids = [fake_inputs... + start_mel + generated_tokens...]
            # 我们需要维护完整的input_ids序列
            generated_ids = []
            for gen_tok in generated:
                generated_ids.append(int(gen_tok[0, 0]))
            
            # 完整input_ids = fake_inputs(context_len个) + start_mel + generated
            full_input_ids_list = [1] * context_len + [self.start_mel_token] + generated_ids
            current_input_ids = mx.array([full_input_ids_list], dtype=mx.int32)  # (1, total_len)
            
            # 🔍 DEBUG: Token-by-token comparison (controlled by env var)
            import os
            debug_token_by_token = os.environ.get('DEBUG_TOKEN_BY_TOKEN', '0') == '1'
            if debug_token_by_token and step <= 10:
                print(f"\n🔍 [MLX Step {step}] absolute_pos={absolute_pos}")
                print(f"   generated tokens so far: {generated_ids}")
                
                # Show top-5 logits BEFORE any processing
                logits_np = next_token_logits[0].tolist()
                top5_indices = sorted(range(len(logits_np)), key=lambda i: logits_np[i], reverse=True)[:5]
                print(f"   Top-5 logits (before processor): {[(idx, logits_np[idx]) for idx in top5_indices]}")
            
            # 3. 应用logits_processor (这里包含temperature, repetition_penalty, top_p, top_k等)
            next_token_scores = logits_processor(current_input_ids, next_token_logits)
            
            # 🔍 DEBUG: Show top-5 after processor
            if debug_token_by_token and step <= 10:
                scores_np = next_token_scores[0].tolist()
                top5_indices = sorted(range(len(scores_np)), key=lambda i: scores_np[i], reverse=True)[:5]
                print(f"   Top-5 scores (after processor): {[(idx, scores_np[idx]) for idx in top5_indices]}")
            
            # 4. 根据use_sampling决定采样或argmax
            if use_sampling:
                # 🔧 完全匹配PyTorch: probs = softmax(scores), then multinomial(probs)
                probs = mx.softmax(next_token_scores[0], axis=-1)
                next_token_id = mx.random.categorical(mx.log(probs + 1e-10))
                next_token = mx.array([[next_token_id]])  # (1, 1)
            else:
                # Greedy (argmax)
                next_token_id = mx.argmax(next_token_scores[0])
                next_token = mx.array([[next_token_id]])
                # 🔥 P0优化: 强制计算
                mx.eval(next_token)
            
            # Check stop token
            token_val = int(next_token[0, 0])
            
            # 🔍 DEBUG: Show selected token
            if debug_token_by_token and step <= 10:
                print(f"   Selected token: {token_val}")
            
            
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
            hidden = self.gpt_ln_f(inputs_mlx)  # GPT2Model's final LayerNorm
            hidden = self.final_norm(hidden)  # lm_head's LayerNorm
            
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
        
        # Convert cond_mel_lengths to MLX format
        lengths_mlx = None
        if cond_mel_lengths is not None:
            lengths_mlx = torch_to_mlx(cond_mel_lengths)
        
        # 🔥 CRITICAL: Use full MLX conditioning pipeline (Conformer + Perceiver)
        # This ensures identical processing to PyTorch version
        if self.use_mlx_conditioning and self.conditioning_module is not None:
            # Use native MLX Conformer + Perceiver (matches PyTorch architecture)
            conds = self.conditioning_module(speech_mlx, lengths_mlx)  # (b, 32, 1280)
            print(f">> [MLX Conditioning] Using full Conformer+Perceiver pipeline for consistency")
        else:
            # Fallback: simplified projection + pooling (legacy)
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
            print(f">> [MLX Conditioning] Using simplified pooling (fallback)")
        
        # Return as PyTorch
        return mlx_to_torch(conds, device='mps')
    
    def get_emo_conditioning(self, speech_conditioning_input, cond_mel_lengths=None):
        """
        Get emotion conditioning vector from semantic features using MLX Conformer + Perceiver.
        
        Matches model_v2.py: uses Conformer + Perceiver to extract emotion features.
        
        Args:
            speech_conditioning_input: Semantic features (b, time, 1024) PyTorch tensor
            cond_mel_lengths: Lengths tensor
            
        Returns:
            Emotion vector (b, 1024) PyTorch tensor (before emovec_layer projection)
        """
        # 🔥 Use pure MLX implementation with fixed Conv2d
        if not self.use_mlx_conditioning or self.emo_conditioning_module is None:
            raise RuntimeError("MLX emotion conditioning not enabled")
        
        from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch
        
        # Ensure correct shape (b, time, 1024)
        if speech_conditioning_input.shape[-1] != 1024:
            # Input is (b, 1024, time), transpose it
            speech_conditioning_input = speech_conditioning_input.transpose(1, 2)
        
        # Convert to MLX
        speech_mlx = torch_to_mlx(speech_conditioning_input.cpu())  # (b, time, 1024)
        cond_lengths_mlx = torch_to_mlx(cond_mel_lengths.cpu()) if cond_mel_lengths is not None else None
        
        # 🔥 CRITICAL: Match PyTorch step-by-step processing exactly
        # Step 1: Conformer encoder (matches PyTorch emo_conditioning_encoder)
        conformer_result = self.emo_conditioning_module.conformer(speech_mlx, cond_lengths_mlx)
        if isinstance(conformer_result, tuple):
            conformer_out, mask = conformer_result
        else:
            conformer_out = conformer_result
            # Create dummy mask if not returned
            batch_size_temp, seq_len_temp, _ = conformer_out.shape
            mask = mx.ones((batch_size_temp, 1, seq_len_temp), dtype=mx.bool_)
        
        # Step 2: Mask processing (matches PyTorch emo_cond_mask_pad)
        # PyTorch: conds_mask = self.emo_cond_mask_pad(mask.squeeze(1))
        # ConstantPad1d((1, 0), True) adds one True value at the beginning
        if mask is not None:
            mask_squeezed = mask.squeeze(1) if len(mask.shape) > 2 and mask.shape[1] == 1 else mask
            # Add padding: one True at the beginning
            conds_mask = mx.concatenate([
                mx.ones((mask_squeezed.shape[0], 1), dtype=mx.bool_),
                mask_squeezed
            ], axis=1)
        else:
            # Fallback if no mask - create appropriate mask for emotion conditioning
            batch_size_fb, seq_len_fb, _ = conformer_out.shape
            # For emotion: 1 latent + seq_len positions, all True
            conds_mask = mx.ones((batch_size_fb, seq_len_fb + 1), dtype=mx.bool_)
        
        # Step 3: Perceiver encoder (matches PyTorch emo_perceiver_encoder)
        emo_latent = self.emo_conditioning_module.perceiver(conformer_out, conds_mask)
        
        # Step 4: Squeeze to (b, 1024) (matches PyTorch return conds.squeeze(1))
        emo_cond = mx.squeeze(emo_latent, axis=1)  # (b, 1024)
        
        # Return as PyTorch
        return mlx_to_torch(emo_cond, device='mps')
    
    def get_emovec(self, emo_speech_conditioning_latent, emo_cond_lengths):
        """
        Get emotion vector using MLX Conformer + Perceiver (matches model_v2.py).
        
        Pipeline:
        1. get_emo_conditioning: Conformer + Perceiver → (b, 1024)
        2. emovec_layer: 1024 → model_dim (1280)
        3. emo_layer: model_dim → model_dim
        
        Args:
            emo_speech_conditioning_latent: Semantic features (b, 1024, time) PyTorch tensor
            emo_cond_lengths: Lengths tensor
            
        Returns:
            Emotion vector (b, model_dim) PyTorch tensor
        """
        if not self.use_mlx_conditioning:
            raise RuntimeError("MLX conditioning not enabled")
        
        # Get emotion conditioning using Conformer + Perceiver
        # Returns (b, 1024) PyTorch tensor
        emo_vec_syn_ori = self.get_emo_conditioning(
            emo_speech_conditioning_latent,
            emo_cond_lengths
        )
        
        # Apply emotion projection layers in MLX
        from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch
        
        emo_mlx = torch_to_mlx(emo_vec_syn_ori)  # (b, 1024)
        emo_vec_syn = self.emovec_layer(emo_mlx)  # (b, 1024) → (b, 1280)
        emo_vec = self.emo_layer(emo_vec_syn)  # (b, 1280) → (b, 1280)
        
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
        return_conditioning_mlx=False, **kwargs
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
        
        # 🔥 P0优化: 批量转换，减少.cpu()调用
        # Convert to MLX for pure MLX conditioning
        with torch.no_grad():
            # 一次性转到CPU（如果还在GPU/MPS上）
            if speech_condition.device.type != 'cpu':
                speech_condition_cpu = speech_condition.cpu()
                emo_speech_condition_cpu = emo_speech_condition.cpu()
                cond_lengths_cpu = cond_lengths.cpu() if cond_lengths is not None else None
            else:
                speech_condition_cpu = speech_condition
                emo_speech_condition_cpu = emo_speech_condition
                cond_lengths_cpu = cond_lengths
        
        # 批量转换为MLX
        speech_condition_mlx = torch_to_mlx(speech_condition_cpu)
        if speech_condition_mlx.ndim == 3 and speech_condition_mlx.shape[1] == 1024:
            speech_condition_mlx = speech_condition_mlx.transpose(0, 2, 1)  # (b, 1024, t) -> (b, t, 1024)
        
        emo_speech_condition_mlx = torch_to_mlx(emo_speech_condition_cpu)
        if emo_speech_condition_mlx.ndim == 3 and emo_speech_condition_mlx.shape[1] == 1024:
            emo_speech_condition_mlx = emo_speech_condition_mlx.transpose(0, 2, 1)
        
        cond_lengths_mlx = torch_to_mlx(cond_lengths_cpu) if cond_lengths_cpu is not None else None
        
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
        
        # 🔥 P0优化: 批量转换text
        # Convert text to MLX
        with torch.no_grad():
            text_cpu = text_inputs.cpu() if text_inputs.device.type != 'cpu' else text_inputs
        text_mlx = torch_to_mlx(text_cpu)
        
        # Generate codes using beam search or greedy/sampling
        num_beams = kwargs.get('num_beams', 1)  # 🔧 Changed to 1 for faster debugging (use 15 for production)
        
        if num_beams > 1:
            # Use beam search (most stable, matches PyTorch behavior)
            print(f">> [MLX] Starting generation with beam search (num_beams={num_beams})")
            # Prepare beam search kwargs, removing already-passed parameters
            beam_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['max_generate_length', 'num_beams', 'length_penalty', 'save_all_beams']}
            beam_result = self.beam_search_forward(
                text_mlx,
                conds_mlx,
                max_length=kwargs.get('max_generate_length', 1500),
                num_beams=num_beams,
                length_penalty=kwargs.get('length_penalty', 1.0),
                save_all_beams=kwargs.get('save_all_beams', False),
                **beam_kwargs
            )
            # Handle tuple return if save_all_beams=True
            if isinstance(beam_result, tuple):
                codes_mlx, all_beams_mlx = beam_result
            else:
                codes_mlx = beam_result
                all_beams_mlx = None
        else:
            # Use greedy/sampling (faster but less stable)
            # 🔧 UPDATED: Force greedy (argmax) when num_beams=1 for deterministic comparison
            # This matches the updated PyTorch behavior (do_sample=False for num_beams=1)
            use_sampling = kwargs.get('use_sampling', False)  # 🔧 Changed to False for deterministic debugging
            print(f">> [MLX] Starting generation with {'sampling (multinomial)' if use_sampling else 'greedy (argmax)'}")
            codes_mlx = self.simple_forward(
                text_mlx,
                conds_mlx,
                max_length=kwargs.get('max_generate_length', 1500),
                temperature=kwargs.get('temperature', 0.8),
                use_sampling=use_sampling,
                debug_generation=kwargs.get('debug_generation', False),
            )
            all_beams_mlx = None  # No beams in greedy/sampling mode
        
        print(f">> [MLX] Generation complete")
        
        # 🔥 P0优化: 强制计算codes_mlx，然后一次性转换
        mx.eval(codes_mlx)
        
        # Convert to PyTorch with correct dtype
        codes = mlx_to_torch(codes_mlx, device='cpu').long().to(speech_condition.device)
        speech_conditioning_latent_torch = mlx_to_torch(speech_conditioning_latent_mlx, device=speech_condition.device)
        
        # Convert all beams if requested
        if all_beams_mlx is not None:
            all_beams_torch = []
            for beam_mlx in all_beams_mlx:
                beam_torch = mlx_to_torch(beam_mlx, device='cpu').long().to(speech_condition.device)
                all_beams_torch.append(beam_torch)
            print(f">> [MLX Native] Generated {len(all_beams_torch)} beams for comparison")
        else:
            all_beams_torch = None
        
        print(f">> [MLX Native] Generated {codes.shape[1]} mel tokens with pure MLX conditioning")
        
        # 🔧 修复内存泄漏和状态累积：清理 MLX 中间结果
        try:
            # 删除大的中间 MLX 数组（保留conds_mlx如果需要缓存）
            del speech_condition_mlx, emo_speech_condition_mlx, cond_lengths_mlx
            del speech_conditioning_latent_mlx  # 不再需要缓存这个
            del emo_vec_mlx
            if not return_conditioning_mlx:
                del conds_mlx  # 如果不需要缓存，删除conds_mlx
            del text_mlx, codes_mlx
            if all_beams_mlx is not None:
                del all_beams_mlx
            # 清理 MLX 缓存和Metal资源
            mx.clear_cache()
            # 强制垃圾回收
            import gc
            gc.collect()
        except Exception as e:
            # 静默失败，不影响返回结果
            pass
        
        if all_beams_torch is not None:
            if return_conditioning_mlx:
                # 🔥 关键修复：返回完整的conds_mlx（包含emotion+duration），而不是speech_conditioning_latent_mlx
                return codes, speech_conditioning_latent_torch, all_beams_torch, conds_mlx
            else:
                return codes, speech_conditioning_latent_torch, all_beams_torch
        else:
            if return_conditioning_mlx:
                # 🔥 关键修复：返回完整的conds_mlx（包含emotion+duration），而不是speech_conditioning_latent_mlx
                return codes, speech_conditioning_latent_torch, conds_mlx
            else:
                return codes, speech_conditioning_latent_torch

    def _fix_key_name(self, key):
        """
        Fix problematic key names that cause MLX parameter loading issues.
        
        Args:
            key: Original key name
            
        Returns:
            Fixed key name that is a valid Python identifier
        """
        # 实际上，大多数键名都是正确的，不需要修复
        # 只有在真正有问题时才进行修复
        
        # 检查是否有特殊字符问题
        if any(c in key for c in ['[', ']', '(', ')', ' ', '\t']):
            # 只修复特殊字符，不修复数字开头的键
            parts = key.split('.')
            fixed_parts = []
            for part in parts:
                # 移除特殊字符
                clean_part = part.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
                clean_part = clean_part.replace(' ', '_').replace('\t', '_')
                
                # 确保不是空字符串
                if clean_part:
                    fixed_parts.append(clean_part)
            
            return '.'.join(fixed_parts)
        
        # 对于数字开头的键，MLX 模型通常能够处理
        # 只有在真正出错时才进行修复
        return key


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

