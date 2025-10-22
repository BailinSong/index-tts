"""
MLX Implementation of Conditional Flow Matching (CFM)
Complete implementation with GPT-fast Transformer and WaveNet
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional
import math

from indextts.s2mel.modules.mlx_gpt_fast import (
    MLXTransformerGPTFast,
    MLXAdaptiveLayerNorm,
    create_mlx_transformer_from_config
)
from indextts.s2mel.modules.mlx_wavenet import MLXWaveNet


def mlx_modulate(x, shift, scale):
    """
    Modulation function for AdaLN.
    x: (batch, seq_len, dim)
    shift, scale: (batch, dim)
    """
    return x * (1 + scale.reshape(-1, 1, scale.shape[-1])) + shift.reshape(-1, 1, shift.shape[-1])


class MLXTimestepEmbedder(nn.Module):
    """
    MLX implementation of Timestep Embedder.
    Converts scalar timesteps to sinusoidal embeddings.
    """
    
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = 10000
        self.scale = 1000
        
        # MLP
        self.mlp_0 = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.mlp_2 = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Precompute frequencies
        half = frequency_embedding_size // 2
        freqs = mx.exp(
            -math.log(self.max_period) * mx.arange(0, half, dtype=mx.float32) / half
        )
        self.freqs = freqs
    
    def timestep_embedding(self, t):
        """
        Create sinusoidal timestep embeddings.
        t: (batch,) - timestep indices
        Returns: (batch, frequency_embedding_size)
        """
        # t: (batch,)
        t_scaled = self.scale * t.reshape(-1, 1) * self.freqs.reshape(1, -1)
        
        # Concatenate cos and sin
        embedding = mx.concatenate([mx.cos(t_scaled), mx.sin(t_scaled)], axis=-1)
        
        # Pad if frequency_embedding_size is odd
        if self.frequency_embedding_size % 2:
            embedding = mx.concatenate(
                [embedding, mx.zeros((embedding.shape[0], 1))],
                axis=-1
            )
        
        return embedding
    
    def __call__(self, t):
        """
        t: (batch,) - timestep indices (can be fractional)
        Returns: (batch, hidden_size)
        """
        t_freq = self.timestep_embedding(t)  # (batch, freq_dim)
        t_emb = self.mlp_0(t_freq)  # (batch, hidden_size)
        t_emb = nn.silu(t_emb)
        t_emb = self.mlp_2(t_emb)  # (batch, hidden_size)
        return t_emb


class MLXStyleEmbedder(nn.Module):
    """
    MLX implementation of Style Embedder for CFG.
    """
    
    def __init__(self, input_size: int, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        
        use_cfg_embedding = dropout_prob > 0
        if use_cfg_embedding:
            self.embedding_table = nn.Embedding(1, hidden_size)
        else:
            self.embedding_table = None
        
        self.style_in = nn.Linear(input_size, hidden_size, bias=True)
    
    def __call__(self, labels, train=False, force_drop_ids=None):
        """
        labels: (batch, input_size) - style vectors
        train: bool - training mode
        force_drop_ids: optional mask for classifier-free guidance
        Returns: (batch, hidden_size)
        """
        use_dropout = self.dropout_prob > 0
        
        if (train and use_dropout) or (force_drop_ids is not None):
            # Token drop for CFG (not implemented in inference)
            embeddings = self.style_in(labels)
        else:
            embeddings = self.style_in(labels)
        
        return embeddings


class MLXFinalLayer(nn.Module):
    """
    MLX implementation of DiT's final layer with AdaLN.
    """
    
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        
        # LayerNorm without learnable parameters (elementwise_affine=False)
        self.norm_final = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        
        # Linear projection
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        
        # AdaLN modulation
        self.adaLN_0 = nn.Linear(hidden_size, 2 * hidden_size, bias=True)
    
    def __call__(self, x, c):
        """
        x: (batch, seq_len, hidden_size)
        c: (batch, hidden_size) - conditioning vector (timestep embedding)
        Returns: (batch, seq_len, out_channels)
        """
        # AdaLN modulation
        ada_out = self.adaLN_0(c)  # (batch, 2*hidden_size)
        ada_out = nn.silu(ada_out)
        
        # Split into shift and scale
        shift = ada_out[:, :self.hidden_size]
        scale = ada_out[:, self.hidden_size:]
        
        # Apply modulation
        x = mlx_modulate(self.norm_final(x), shift, scale)
        
        # Final projection
        x = self.linear(x)
        
        return x


class MLXDiT(nn.Module):
    """
    MLX implementation of Diffusion Transformer (DiT).
    Uses custom GPT-fast style Transformer.
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Extract config
        dit_cfg = config.DiT
        style_cfg = config.style_encoder
        
        self.in_channels = dit_cfg.in_channels  # 80 (mel bins)
        self.out_channels = dit_cfg.in_channels
        self.hidden_dim = dit_cfg.hidden_dim  # 512
        self.num_heads = dit_cfg.num_heads  # 8
        self.depth = dit_cfg.depth  # 13 layers
        
        self.time_as_token = dit_cfg.get('time_as_token', False)
        self.style_as_token = dit_cfg.get('style_as_token', False)
        self.long_skip_connection = dit_cfg.long_skip_connection
        self.transformer_style_condition = dit_cfg.style_condition
        self.is_causal = dit_cfg.is_causal
        self.final_layer_type = dit_cfg.final_layer_type
        
        # Embedders
        self.x_embedder = nn.Linear(dit_cfg.in_channels, dit_cfg.hidden_dim, bias=True)
        
        # Content embedding
        self.content_type = dit_cfg.content_type
        if self.content_type == 'discrete':
            self.cond_embedder = nn.Embedding(dit_cfg.content_codebook_size, dit_cfg.hidden_dim)
        else:
            self.cond_projection = nn.Linear(dit_cfg.content_dim, dit_cfg.hidden_dim, bias=True)
        
        # Timestep embedder
        self.t_embedder = MLXTimestepEmbedder(dit_cfg.hidden_dim)
        
        # GPT-fast style Transformer
        self.transformer = create_mlx_transformer_from_config(config)
        
        # Merge layer (combines x, prompt_x, cond, style)
        merge_input_dim = (
            dit_cfg.in_channels * 2 +  # for x and prompt_x
            dit_cfg.hidden_dim +  # for cond (already projected)
            style_cfg.dim * self.transformer_style_condition * (not self.style_as_token)  # for style
        )
        self.cond_x_merge_linear = nn.Linear(merge_input_dim, dit_cfg.hidden_dim, bias=True)
        
        # Style as token mode
        if self.style_as_token:
            self.style_in = nn.Linear(style_cfg.dim, dit_cfg.hidden_dim, bias=True)
        
        # Long skip connection
        if self.long_skip_connection:
            self.skip_linear = nn.Linear(dit_cfg.hidden_dim + dit_cfg.in_channels, dit_cfg.hidden_dim, bias=True)
        
        # Final layer
        if self.final_layer_type == 'wavenet':
            # WaveNet configuration
            wavenet_cfg = config.wavenet
            self.t_embedder2 = MLXTimestepEmbedder(wavenet_cfg.hidden_dim)
            self.conv1 = nn.Linear(dit_cfg.hidden_dim, wavenet_cfg.hidden_dim, bias=True)
            self.conv2 = nn.Conv1d(wavenet_cfg.hidden_dim, dit_cfg.in_channels, kernel_size=1, padding=0, bias=True)
            
            self.wavenet = MLXWaveNet(
                hidden_channels=wavenet_cfg.hidden_dim,
                kernel_size=wavenet_cfg.kernel_size,
                dilation_rate=wavenet_cfg.dilation_rate,
                n_layers=wavenet_cfg.num_layers,
                gin_channels=wavenet_cfg.hidden_dim,
                p_dropout=wavenet_cfg.p_dropout
            )
            
            self.final_layer = MLXFinalLayer(
                wavenet_cfg.hidden_dim, 1, wavenet_cfg.hidden_dim
            )
            self.res_projection = nn.Linear(dit_cfg.hidden_dim, wavenet_cfg.hidden_dim, bias=True)
            self.wavenet_style_condition = wavenet_cfg.style_condition
        else:
            # MLP final layer
            self.final_mlp_0 = nn.Linear(dit_cfg.hidden_dim, dit_cfg.hidden_dim, bias=True)
            self.final_mlp_2 = nn.Linear(dit_cfg.hidden_dim, dit_cfg.in_channels, bias=True)
        
        # Content masking for CFG
        self.class_dropout_prob = dit_cfg.class_dropout_prob
        self.content_mask_embedder = nn.Embedding(1, dit_cfg.hidden_dim)
        
        # Input positions buffer
        self.input_pos = mx.arange(16384)
        
        print(f">> MLX DiT initialized:")
        print(f"   Transformer: {self.depth} layers, {self.num_heads} heads, dim={self.hidden_dim}")
        print(f"   Final layer type: {self.final_layer_type}")
        print(f"   In/Out channels: {self.in_channels}")
    
    def __call__(self, x, prompt_x, x_lens, t, style, cond, mask_content=False):
        """
        Forward pass of DiT.
        
        Args:
            x: Noisy mel (batch, in_channels, seq_len) - PyTorch format
            prompt_x: Reference mel (batch, in_channels, seq_len) - PyTorch format
            x_lens: Sequence lengths (batch,) - can be torch or mlx
            t: Timesteps (batch,) - can be torch or mlx
            style: Style vectors (batch, style_dim) - can be torch or mlx
            cond: Semantic conditioning (batch, seq_len, content_dim)
            mask_content: Whether to mask content for CFG
        
        Returns:
            Predicted velocity (batch, out_channels, seq_len) - PyTorch format
        """
        import torch
        
        # Convert inputs to MLX if needed
        if isinstance(x, torch.Tensor):
            from indextts.utils.mlx_utils import torch_to_mlx
            x = torch_to_mlx(x.cpu())
            prompt_x = torch_to_mlx(prompt_x.cpu())
            x_lens = torch_to_mlx(x_lens.cpu())
            t = torch_to_mlx(t.cpu())
            style = torch_to_mlx(style.cpu())
            cond = torch_to_mlx(cond.cpu())
            convert_back = True
        else:
            convert_back = False
        
        batch, in_ch, seq_len = x.shape
        
        # Get timestep embedding
        t_emb = self.t_embedder(t)  # (batch, hidden_dim)
        
        # Project conditioning
        if self.content_type == 'discrete':
            # Cond is already semantic embeddings from quantizer
            # Need to transpose from (batch, content_dim, seq_len) to (batch, seq_len, content_dim)
            cond_t = cond.transpose(0, 2, 1)  # (batch, seq_len, content_dim)
            cond_proj = cond_t  # Keep as is (will be merged later)
        else:
            cond_proj = self.cond_projection(cond)  # (batch, seq_len, hidden_dim)
        
        # Transpose x and prompt_x to (batch, seq_len, channels)
        x_t = x.transpose(0, 2, 1)  # (batch, seq_len, in_channels)
        prompt_x_t = prompt_x.transpose(0, 2, 1)  # (batch, seq_len, in_channels)
        
        # Concatenate inputs: [x, prompt_x, cond]
        x_in = mx.concatenate([x_t, prompt_x_t, cond_proj], axis=-1)
        
        # Add style conditioning if not using style_as_token
        if self.transformer_style_condition and not self.style_as_token:
            # Broadcast style to all timesteps
            style_broadcast = mx.broadcast_to(
                style.reshape(batch, 1, -1),
                (batch, seq_len, style.shape[-1])
            )
            x_in = mx.concatenate([x_in, style_broadcast], axis=-1)
        
        # Apply masking for CFG
        if mask_content:
            # Zero out conditioning (keep only x and prompt_x)
            x_in = mx.concatenate([
                x_in[:, :, :self.in_channels * 2],  # Keep x and prompt_x
                mx.zeros_like(x_in[:, :, self.in_channels * 2:])  # Zero out rest
            ], axis=-1)
        
        # Merge inputs to hidden_dim
        x_in = self.cond_x_merge_linear(x_in)  # (batch, seq_len, hidden_dim)
        
        # Add style/time as tokens if needed
        if self.style_as_token:
            style_tok = self.style_in(style).reshape(batch, 1, -1)
            if mask_content:
                style_tok = mx.zeros_like(style_tok)
            x_in = mx.concatenate([style_tok, x_in], axis=1)
        
        if self.time_as_token:
            t_tok = t_emb.reshape(batch, 1, -1)
            x_in = mx.concatenate([t_tok, x_in], axis=1)
        
        # Get input positions
        input_pos = self.input_pos[:x_in.shape[1]]
        
        # Create attention mask (non-causal: length-based mask)
        if not self.is_causal:
            max_len = x_in.shape[1]
            actual_lens = x_lens + int(self.style_as_token) + int(self.time_as_token)
            
            # Create mask: (batch, 1, max_len, max_len)
            positions = mx.arange(max_len).reshape(1, 1, 1, -1)
            lens_reshaped = actual_lens.reshape(-1, 1, 1, 1)
            mask_expanded = positions < lens_reshaped
            mask_expanded = mx.broadcast_to(mask_expanded, (batch, 1, max_len, max_len))
        else:
            mask_expanded = None
        
        # Forward through Transformer
        x_res = self.transformer(
            x_in,
            t_emb,
            input_pos=input_pos,
            mask=mask_expanded
        )
        
        # Remove added tokens
        if self.time_as_token:
            x_res = x_res[:, 1:, :]
        if self.style_as_token:
            x_res = x_res[:, 1:, :]
        
        # Long skip connection
        if self.long_skip_connection:
            x_res = self.skip_linear(mx.concatenate([x_res, x_t], axis=-1))
        
        # Final layer
        if self.final_layer_type == 'wavenet':
            # WaveNet path
            x_out = self.conv1(x_res)  # (batch, seq_len, wavenet_dim)
            
            # WaveNet expects (batch, seq_len, channels), mask is (batch, 1, seq_len)
            # Create mask
            positions = mx.arange(x_out.shape[1]).reshape(1, 1, -1)
            lens_for_mask = x_lens.reshape(-1, 1, 1)
            x_mask = positions < lens_for_mask  # (batch, 1, seq_len)
            
            # Get timestep embedding for WaveNet
            t2_emb = self.t_embedder2(t)  # (batch, wavenet_dim)
            t2_emb_expanded = t2_emb.reshape(batch, 1, -1)  # (batch, 1, wavenet_dim)
            t2_emb_expanded = mx.broadcast_to(t2_emb_expanded, (batch, x_out.shape[1], t2_emb.shape[-1]))
            
            # WaveNet forward
            x_out = self.wavenet(x_out, x_mask, g=t2_emb_expanded)
            
            # Add residual from transformer
            x_out = x_out + self.res_projection(x_res)
            
            # Final layer with AdaLN
            x_out = self.final_layer(x_out, t_emb)  # (batch, seq_len, wavenet_dim)
            
            # Final conv (1x1)
            x_out = self.conv2(x_out)  # (batch, seq_len, in_channels)
        else:
            # MLP path
            x_out = self.final_mlp_0(x_res)
            x_out = nn.silu(x_out)
            x_out = self.final_mlp_2(x_out)
        
        # Transpose back to (batch, out_channels, seq_len)
        x_out = x_out.transpose(0, 2, 1)
        
        # Convert back to PyTorch if needed
        if convert_back:
            from indextts.utils.mlx_utils import mlx_to_torch
            x_out = mlx_to_torch(x_out, device='mps')
        
        return x_out


class MLXCFM(nn.Module):
    """
    MLX implementation of Conditional Flow Matching.
    """
    
    def __init__(self, config):
        super().__init__()
        self.sigma_min = 1e-6
        self.in_channels = config.DiT.in_channels
        
        # DiT estimator
        self.estimator = MLXDiT(config)
        
        # Check if zero_prompt_speech_token is set
        self.zero_prompt_speech_token = config.DiT.get('zero_prompt_speech_token', False)
        
        print(">> MLX CFM initialized with DiT estimator")
    
    def solve_euler(self, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.5):
        """
        Euler solver for ODE.
        
        Args:
            x: Random noise (batch, in_channels, seq_len) - MLX array
            x_lens: Sequence lengths (batch,) - MLX array
            prompt: Reference mel (batch, in_channels, prompt_len) - MLX array
            mu: Semantic conditioning (batch, seq_len, content_dim) - MLX array
            style: Style vector (batch, style_dim) - MLX array
            f0: F0 (not used currently)
            t_span: Time steps (num_steps+1,) - MLX array
            inference_cfg_rate: CFG weight
        
        Returns:
            Generated mel (batch, in_channels, seq_len) - MLX array
        """
        # Initialize
        prompt_len = prompt.shape[-1]
        
        # Prepare prompt
        prompt_x = mx.zeros_like(x)
        prompt_x[:, :, :prompt_len] = prompt[:, :, :prompt_len]
        x[:, :, :prompt_len] = 0
        
        # Zero out prompt speech tokens if needed
        if self.zero_prompt_speech_token:
            mu[:, :prompt_len, :] = 0
        
        # Euler iteration with progress
        num_steps = len(t_span) - 1
        print(f">> [MLX CFM] Starting Euler solver ({num_steps} steps)...")
        
        for step in range(1, len(t_span)):
            if step % 5 == 0 or step == 1:
                print(f"   Step {step}/{num_steps}", end='\r')
            
            t_current = t_span[step - 1]
            dt = t_span[step] - t_span[step - 1]
            
            if inference_cfg_rate > 0:
                # Classifier-free guidance: stack original and null inputs
                stacked_prompt_x = mx.concatenate([prompt_x, mx.zeros_like(prompt_x)], axis=0)
                stacked_style = mx.concatenate([style, mx.zeros_like(style)], axis=0)
                stacked_mu = mx.concatenate([mu, mx.zeros_like(mu)], axis=0)
                stacked_x = mx.concatenate([x, x], axis=0)
                
                # Create timestep tensor for both batches
                t_batch = mx.full((x.shape[0],), float(t_current))
                stacked_t = mx.concatenate([t_batch, t_batch], axis=0)
                
                # Duplicate x_lens for both batches
                stacked_x_lens = mx.concatenate([x_lens, x_lens], axis=0)
                
                # Forward pass
                stacked_dphi_dt = self.estimator(
                    stacked_x, stacked_prompt_x, stacked_x_lens, 
                    stacked_t, stacked_style, stacked_mu,
                    mask_content=False  # First half uses content
                )
                
                # Split and apply CFG
                dphi_dt = stacked_dphi_dt[:x.shape[0]]  # Conditioned
                dphi_dt_null = stacked_dphi_dt[x.shape[0]:]  # Unconditioned
                dphi_dt = dphi_dt_null + inference_cfg_rate * (dphi_dt - dphi_dt_null)
            else:
                # No CFG
                t_batch = mx.full((x.shape[0],), float(t_current))
                dphi_dt = self.estimator(x, prompt_x, x_lens, t_batch, style, mu, mask_content=False)
            
            # Euler step
            x = x + dt * dphi_dt
            
            # Keep prompt unchanged
            x[:, :, :prompt_len] = 0
            
            # Force evaluation to avoid graph buildup
            mx.eval(x)
        
        print(f"\n>> [MLX CFM] Euler solver completed")
        return x
    
    def inference(self, mu, x_lens, prompt, style, f0, n_timesteps, temperature=1.0, inference_cfg_rate=0.5):
        """
        Inference method (entry point).
        
        Args:
            mu: Semantic conditioning (batch, seq_len, content_dim) - PyTorch or MLX
            x_lens: Sequence lengths (batch,) - PyTorch or MLX
            prompt: Reference mel (batch, in_channels, prompt_len) - PyTorch or MLX
            style: Style vector (batch, style_dim) - PyTorch or MLX
            f0: F0 (not used)
            n_timesteps: Number of diffusion steps
            temperature: Noise temperature
            inference_cfg_rate: CFG weight
        
        Returns:
            Generated mel (batch, in_channels, seq_len) - Same format as input
        """
        import torch
        
        # Convert to MLX if needed
        if isinstance(mu, torch.Tensor):
            from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch
            mu_mlx = torch_to_mlx(mu.cpu())
            x_lens_mlx = torch_to_mlx(x_lens.cpu())
            prompt_mlx = torch_to_mlx(prompt.cpu())
            style_mlx = torch_to_mlx(style.cpu())
            convert_back = True
            device = mu.device
        else:
            mu_mlx = mu
            x_lens_mlx = x_lens
            prompt_mlx = prompt
            style_mlx = style
            convert_back = False
        
        batch, seq_len, _ = mu_mlx.shape
        
        # Initialize noise
        z = mx.random.normal((batch, self.in_channels, seq_len)) * temperature
        
        # Create time span
        t_span = mx.linspace(0, 1, n_timesteps + 1)
        
        # Solve ODE
        result = self.solve_euler(z, x_lens_mlx, prompt_mlx, mu_mlx, style_mlx, f0, t_span, inference_cfg_rate)
        
        # Convert back if needed
        if convert_back:
            result = mlx_to_torch(result, device=device)
        
        return result


    def load_weights_from_pytorch(self, pytorch_state_dict, prefix="models.cfm."):
        """
        Load CFM weights from PyTorch.
        
        Args:
            pytorch_state_dict: dict with numpy arrays
            prefix: key prefix
        
        Returns:
            Number of weights loaded
        """
        from indextts.s2mel.modules.mlx_dit_weights import load_dit_weights
        
        # Load DiT/estimator weights
        estimator_prefix = f"{prefix}estimator."
        loaded = load_dit_weights(self.estimator, pytorch_state_dict, estimator_prefix)
        
        print(f">> MLX CFM loaded {loaded} weights total")
        return loaded
    
    def extract_weights_for_cache(self):
        """
        Extract all MLX weights from CFM for caching using model's parameters() method.
        
        Returns:
            dict of {name: mx.array} suitable for mx.savez()
        """
        def flatten_parameters(params, prefix=""):
            """Recursively flatten nested parameter dict"""
            flat = {}
            for name, value in params.items():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(value, dict):
                    # Recurse into nested dict
                    flat.update(flatten_parameters(value, full_name))
                elif isinstance(value, mx.array):
                    # Leaf parameter
                    flat[full_name] = value
            return flat
        
        # Get all parameters from estimator (DiT)
        params = self.estimator.parameters()
        weights = flatten_parameters(params, "estimator")
        
        print(f">> Extracted {len(weights)} weight arrays for caching")
        return weights
    
    def load_from_cache(self, cache_dict):
        """
        Load CFM weights from cached MLX arrays.
        
        Args:
            cache_dict: dict from mx.load()
        
        Returns:
            Number of weights loaded
        """
        # Reconstruct nested parameter dict
        def unflatten_parameters(flat_dict, prefix="estimator"):
            """Reconstruct nested dict from flattened parameters"""
            nested = {}
            for key, value in flat_dict.items():
                if not key.startswith(prefix + "."):
                    continue
                # Remove prefix
                rel_key = key[len(prefix) + 1:]
                parts = rel_key.split('.')
                
                # Navigate/create nested structure
                current = nested
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Set leaf value
                current[parts[-1]] = value
            
            return nested
        
        # Unflatten parameters
        nested_params = unflatten_parameters(cache_dict, "estimator")
        
        # Use MLX's update method to load weights
        self.estimator.update(nested_params)
        
        loaded = len(cache_dict)
        print(f">> Loaded {loaded} weights from cache")
        return loaded


def create_mlx_cfm_from_pytorch(pytorch_cfm, config):
    """
    Create MLX CFM from PyTorch CFM model.
    
    Args:
        pytorch_cfm: PyTorch CFM model
        config: Configuration
    
    Returns:
        MLX CFM model with loaded weights
    """
    mlx_cfm = MLXCFM(config)
    
    # Load weights from PyTorch
    state_dict = pytorch_cfm.state_dict()
    state_dict_np = {k: v.cpu().numpy() for k, v in state_dict.items()}
    
    # Load weights
    mlx_cfm.load_weights_from_pytorch(state_dict_np, prefix="")
    
    return mlx_cfm

