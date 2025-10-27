"""
重写的MLX CFM实现 - 与PyTorch版本完全一致
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
    Modulation function for AdaLN - 与PyTorch版本完全一致
    x: (batch, seq_len, dim)
    shift, scale: (batch, dim)
    """
    # 使用与PyTorch完全相同的形状操作
    # PyTorch: scale.unsqueeze(1) -> (batch, 1, dim)
    # MLX: scale.reshape(batch, 1, dim)
    batch_size = scale.shape[0]
    scale_reshaped = scale.reshape(batch_size, 1, -1)
    shift_reshaped = shift.reshape(batch_size, 1, -1)
    return x * (1 + scale_reshaped) + shift_reshaped


class MLXTimestepEmbedderRewritten(nn.Module):
    """
    重写的MLX Timestep Embedder - 与PyTorch版本完全一致
    """
    
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = 10000
        self.scale = 1000
        
        # MLP - 与PyTorch版本完全一致
        self.mlp_0 = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.mlp_2 = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Precompute frequencies - 与PyTorch版本完全一致
        half = frequency_embedding_size // 2
        freqs = mx.exp(
            -math.log(self.max_period) * mx.arange(0, half, dtype=mx.float32) / half
        )
        self.freqs = freqs
    
    def timestep_embedding(self, t):
        """
        Create sinusoidal timestep embeddings - 与PyTorch版本完全一致
        t: (batch,) - timestep indices
        Returns: (batch, frequency_embedding_size)
        """
        # 与PyTorch版本完全一致的计算
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


class MLXFinalLayerRewritten(nn.Module):
    """
    重写的MLX Final Layer - 与PyTorch版本完全一致
    """
    
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        
        # 与PyTorch版本完全一致 - 修复affine参数
        self.norm_final = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        # 修复：使用与PyTorch完全一致的adaLN_modulation结构
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    
    def __call__(self, x, c):
        """
        x: (batch, seq_len, hidden_size)
        c: (batch, hidden_size)
        Returns: (batch, seq_len, patch_size * patch_size * out_channels)
        """
        # 逐层调试：记录final_layer输入
        print(f"   c: {c.shape}, min={float(c.min()):.6f}, max={float(c.max()):.6f}, mean={float(c.mean()):.6f}")
        
        # 与PyTorch版本完全一致
        c_emb = self.adaLN_modulation(c)
        print(f"   c_emb: {c_emb.shape}, min={float(c_emb.min()):.6f}, max={float(c_emb.max()):.6f}, mean={float(c_emb.mean()):.6f}")
        
        # 确保split操作与PyTorch的chunk操作一致
        # PyTorch: .chunk(2, dim=1) 在最后一个维度分割
        # MLX: mx.split(..., 2, axis=-1) 在最后一个维度分割
        shift, scale = mx.split(c_emb, 2, axis=-1)
        print(f"   shift: {shift.shape}, min={float(shift.min()):.6f}, max={float(shift.max()):.6f}, mean={float(shift.mean()):.6f}")
        print(f"   scale: {scale.shape}, min={float(scale.min()):.6f}, max={float(scale.max()):.6f}, mean={float(scale.mean()):.6f}")
        
        # Apply modulation
        x_norm = self.norm_final(x)
        print(f"   x_norm: {x_norm.shape}, min={float(x_norm.min()):.6f}, max={float(x_norm.max()):.6f}, mean={float(x_norm.mean()):.6f}")
        
        x_modulated = mlx_modulate(x_norm, shift, scale)
        print(f"   x_modulated: {x_modulated.shape}, min={float(x_modulated.min()):.6f}, max={float(x_modulated.max()):.6f}, mean={float(x_modulated.mean()):.6f}")
        
        x = self.linear(x_modulated)
        print(f"   x_output: {x.shape}, min={float(x.min()):.6f}, max={float(x.max()):.6f}, mean={float(x.mean()):.6f}")
        return x


class MLXDiTRewritten(nn.Module):
    """
    重写的MLX DiT - 与PyTorch版本完全一致
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
        
        # 确保推理模式 - 与PyTorch版本一致
        # MLX的training属性是只读的，我们通过其他方式确保推理模式
        self._inference_mode = True
        self.long_skip_connection = dit_cfg.long_skip_connection
        self.transformer_style_condition = dit_cfg.style_condition
        self.is_causal = dit_cfg.is_causal
        self.final_layer_type = dit_cfg.final_layer_type
        
        # Embedders - 与PyTorch版本完全一致
        self.x_embedder = nn.Linear(dit_cfg.in_channels, dit_cfg.hidden_dim, bias=True)
        
        # Content embedding - 与PyTorch版本完全一致
        self.content_type = dit_cfg.content_type
        self.cond_embedder = nn.Embedding(dit_cfg.content_codebook_size, dit_cfg.hidden_dim)
        self.cond_projection = nn.Linear(dit_cfg.content_dim, dit_cfg.hidden_dim, bias=True)
        
        # Timestep embedder
        self.t_embedder = MLXTimestepEmbedderRewritten(dit_cfg.hidden_dim)
        
        # GPT-fast style Transformer
        self.transformer = create_mlx_transformer_from_config(config)
        
        # Merge layer - 与PyTorch版本完全一致
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
        
        # Final layer - 与PyTorch版本完全一致
        if self.final_layer_type == 'wavenet':
            # WaveNet configuration
            wavenet_cfg = config.wavenet
            self.t_embedder2 = MLXTimestepEmbedderRewritten(wavenet_cfg.hidden_dim)
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
            
            self.final_layer = MLXFinalLayerRewritten(
                wavenet_cfg.hidden_dim, 1, wavenet_cfg.hidden_dim
            )
            self.res_projection = nn.Linear(dit_cfg.hidden_dim, wavenet_cfg.hidden_dim, bias=True)
            self.wavenet_style_condition = wavenet_cfg.style_condition
        else:
            # MLP final layer - 与PyTorch版本完全一致
            self.final_mlp_0 = nn.Linear(dit_cfg.hidden_dim, dit_cfg.hidden_dim, bias=True)
            self.final_mlp_2 = nn.Linear(dit_cfg.hidden_dim, dit_cfg.in_channels, bias=True)
        
        # Content masking for CFG
        self.class_dropout_prob = dit_cfg.class_dropout_prob
        self.content_mask_embedder = nn.Embedding(1, dit_cfg.hidden_dim)
        
        # Input positions buffer
        self.input_pos = mx.arange(16384, dtype=mx.int32)
        
        print(f">> MLX DiT Rewritten initialized:")
        print(f"   Transformer: {self.depth} layers, {self.num_heads} heads, dim={self.hidden_dim}")
        print(f"   Final layer type: {self.final_layer_type}")
        print(f"   In/Out channels: {self.in_channels}")
    
    def __call__(self, x, prompt_x, x_lens, t, style, cond, mask_content=False):
        """
        Forward pass of DiT - 与PyTorch版本完全一致
        """
        import torch
        
        # 逐层调试：记录输入
        print(f"   prompt_x: {prompt_x.shape}, min={float(prompt_x.min()):.6f}, max={float(prompt_x.max()):.6f}")
        print(f"   t: {t.shape}, min={float(t.min()):.6f}, max={float(t.max()):.6f}")
        print(f"   style: {style.shape}, min={float(style.min()):.6f}, max={float(style.max()):.6f}")
        print(f"   cond: {cond.shape}, min={float(cond.min()):.6f}, max={float(cond.max()):.6f}")
        print(f"   mask_content: {mask_content}")
        
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
        
        # Get timestep embedding - 与PyTorch版本完全一致
        if t.ndim == 0:
            t = mx.array([t.item()])
        t_emb = self.t_embedder(t)  # (batch, hidden_dim)
                # Project conditioning - 与PyTorch版本完全一致
        # 注意：PyTorch版本总是使用cond_projection，不管content_type
        cond_proj = self.cond_projection(cond)  # (batch, seq_len, hidden_dim)
                # Transpose x and prompt_x to (batch, seq_len, channels) - 与PyTorch版本完全一致
        x_t = x.transpose(0, 2, 1)  # (batch, seq_len, in_channels)
        prompt_x_t = prompt_x.transpose(0, 2, 1)  # (batch, seq_len, in_channels)
                        # Concatenate inputs: [x, prompt_x, cond] - 与PyTorch版本完全一致
        x_in = mx.concatenate([x_t, prompt_x_t, cond_proj], axis=-1)
                # Add style conditioning if not using style_as_token - 与PyTorch版本完全一致
        if self.transformer_style_condition and not self.style_as_token:
            # Broadcast style to all timesteps
            style_broadcast = mx.broadcast_to(
                style.reshape(batch, 1, -1),
                (batch, seq_len, style.shape[-1])
            )
            x_in = mx.concatenate([x_in, style_broadcast], axis=-1)
                    # Apply masking for CFG - 与PyTorch版本完全一致
        # 在推理模式下，class_dropout应该为False
        class_dropout = False
        if mask_content:
            class_dropout = True
            
        if class_dropout:
            # Zero out conditioning (keep only x and prompt_x)
            x_in = mx.concatenate([
                x_in[:, :, :self.in_channels * 2],  # Keep x and prompt_x
                mx.zeros_like(x_in[:, :, self.in_channels * 2:])  # Zero out rest
            ], axis=-1)
                    # Merge inputs to hidden_dim - 与PyTorch版本完全一致
        x_in = self.cond_x_merge_linear(x_in)  # (batch, seq_len, hidden_dim)
                # Add style/time as tokens if needed - 与PyTorch版本完全一致
        if self.style_as_token:
            style_tok = self.style_in(style).reshape(batch, 1, -1)
            if mask_content:
                style_tok = mx.zeros_like(style_tok)
            x_in = mx.concatenate([style_tok, x_in], axis=1)
        
        if self.time_as_token:
            t_tok = t_emb.reshape(batch, 1, -1)
            x_in = mx.concatenate([t_tok, x_in], axis=1)
        
        # Get input positions - 与PyTorch版本完全一致
        input_pos = self.input_pos[:x_in.shape[1]]
        
        # Create attention mask - 与PyTorch版本完全一致
        if not self.is_causal:
            max_len = x_in.shape[1]
            if x_lens.ndim == 0:
                x_lens = mx.array([x_lens.item()])
            actual_lens = x_lens + int(self.style_as_token) + int(self.time_as_token)
            
            # Create mask: (batch, 1, max_len, max_len)
            positions = mx.arange(max_len).reshape(1, 1, 1, -1)
            lens_reshaped = actual_lens.reshape(-1, 1, 1, 1)
            mask_expanded = positions < lens_reshaped
            mask_expanded = mx.broadcast_to(mask_expanded, (batch, 1, max_len, max_len))
        else:
            mask_expanded = None
        
        # Forward through Transformer - 与PyTorch版本完全一致
        # 调试输出已移除
        
        x_res = self.transformer(
            x_in,
            t_emb,
            input_pos=input_pos,
            mask=mask_expanded
        )
        
        # 调试输出已移除
        
        # Remove added tokens - 与PyTorch版本完全一致
        if self.time_as_token:
            x_res = x_res[:, 1:, :]
        if self.style_as_token:
            x_res = x_res[:, 1:, :]
        
        # Long skip connection - 与PyTorch版本完全一致
        if self.long_skip_connection:
            x_res = self.skip_linear(mx.concatenate([x_res, x_t], axis=-1))
        
        # Final layer - 与PyTorch版本完全一致
        if self.final_layer_type == 'wavenet':
            # WaveNet path
            x_out = self.conv1(x_res)  # (batch, seq_len, wavenet_dim)
                        # Create mask for WaveNet
            positions = mx.arange(x_out.shape[1]).reshape(1, 1, -1)
            if x_lens.ndim == 0:
                x_lens = mx.array([x_lens.item()])
            lens_for_mask = x_lens.reshape(-1, 1, 1)
            x_mask = positions < lens_for_mask  # (batch, 1, seq_len)
            
            # Get timestep embedding for WaveNet
            t2_emb = self.t_embedder2(t)  # (batch, wavenet_dim)
            t2_emb_expanded = mx.broadcast_to(t2_emb[:, None, :], (batch, x_out.shape[1], t2_emb.shape[-1]))
                        # WaveNet forward
            x_out = self.wavenet(x_out, x_mask, g=t2_emb_expanded)
                        # Add residual from transformer
            x_out = x_out + self.res_projection(x_res)
            # Final layer with AdaLN
            x_out = self.final_layer(x_out, t_emb)  # (batch, seq_len, wavenet_dim)
            # Final conv (1x1)
            x_out = self.conv2(x_out)  # (batch, seq_len, in_channels)
        else:
            # MLP path - 与PyTorch版本完全一致
            x_out = self.final_mlp_0(x_res)
            x_out = nn.silu(x_out)
            x_out = self.final_mlp_2(x_out)
        # Transpose back to (batch, out_channels, seq_len) - 与PyTorch版本完全一致
        x_out = x_out.transpose(0, 2, 1)
                # Convert back to PyTorch if needed
        if convert_back:
            from indextts.utils.mlx_utils import mlx_to_torch
            x_out = mlx_to_torch(x_out, device='mps')
        
        return x_out


class MLXCFMRewritten(nn.Module):
    """
    重写的MLX CFM - 与PyTorch版本完全一致
    """
    
    def __init__(self, config):
        super().__init__()
        self.sigma_min = 1e-6
        self.in_channels = config.DiT.in_channels
        
        # DiT estimator - 使用重写的版本
        self.estimator = MLXDiTRewritten(config)
        
        # Check if zero_prompt_speech_token is set
        self.zero_prompt_speech_token = config.DiT.get('zero_prompt_speech_token', False)
        
        print(">> MLX CFM Rewritten initialized with DiT estimator")
    
    def solve_euler(self, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.5, debug_layers=False):
        """
        Euler solver for ODE - 与PyTorch版本完全一致
        """
        # Initialize
        prompt_len = prompt.shape[-1]
        
        # Prepare prompt - 与PyTorch版本完全一致
        prompt_x = mx.zeros_like(x)
        prompt_x[:, :, :prompt_len] = prompt[:, :, :prompt_len]
        x[:, :, :prompt_len] = 0
        
        # Zero out prompt speech tokens if needed
        if self.zero_prompt_speech_token:
            mu[:, :prompt_len, :] = 0
        
        # 初始化时间变量 - 与PyTorch版本完全一致
        t = t_span[0]
        
        # Euler iteration with progress
        num_steps = len(t_span) - 1
        print(f">> [MLX CFM Rewritten] Starting Euler solver ({num_steps} steps)...")
        
        for step in range(1, len(t_span)):
            if step % 5 == 0 or step == 1:
                print(f"   Step {step}/{num_steps}", end='\r')
            
            # 计算时间步长 - 与PyTorch版本完全一致
            dt = t_span[step] - t_span[step - 1]
            
            # 调试输出已移除
            
            if inference_cfg_rate > 0:
                # Classifier-free guidance - 与PyTorch版本完全一致
                stacked_prompt_x = mx.concatenate([prompt_x, mx.zeros_like(prompt_x)], axis=0)
                stacked_style = mx.concatenate([style, mx.zeros_like(style)], axis=0)
                stacked_mu = mx.concatenate([mu, mx.zeros_like(mu)], axis=0)
                stacked_x = mx.concatenate([x, x], axis=0)
                
                # Create timestep tensor for both batches - 与PyTorch版本完全一致
                t_scalar = mx.array([float(t)])
                stacked_t = mx.concatenate([t_scalar, t_scalar], axis=0)
                
                # Duplicate x_lens for both batches
                stacked_x_lens = mx.concatenate([x_lens, x_lens], axis=0)
                
                # Forward pass
                stacked_dphi_dt = self.estimator(
                    stacked_x, stacked_prompt_x, stacked_x_lens, 
                    stacked_t, stacked_style, stacked_mu,
                    mask_content=False  # First half uses content
                )
                
                # Split and apply CFG - 与PyTorch版本完全一致
                dphi_dt, cfg_dphi_dt = mx.split(stacked_dphi_dt, 2, axis=0)
                dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
            else:
                # No CFG - 与PyTorch版本完全一致
                t_scalar = mx.array([float(t)])
                dphi_dt = self.estimator(x, prompt_x, x_lens, t_scalar, style, mu)
            
            # 逐层调试输出
            if debug_layers:
                print(f"   dphi_dt: min={float(dphi_dt.min()):.6f}, max={float(dphi_dt.max()):.6f}, mean={float(dphi_dt.mean()):.6f}")
            
            # Euler step - 与PyTorch版本完全一致
            x = x + dt * dphi_dt
            
            # 时间更新 - 与PyTorch版本完全一致
            t = t + dt
            
            # Keep prompt unchanged
            x[:, :, :prompt_len] = 0
            
            # 计算下一步的dt - 与PyTorch版本完全一致
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
            
            # Force evaluation to avoid graph buildup
            x = mx.eval(x)
        
        print(f"\n>> [MLX CFM Rewritten] Euler solver completed")
        return x
    
    def inference(self, mu, x_lens, prompt, style, f0, n_timesteps, temperature=1.0, inference_cfg_rate=0.5, unified_random=None):
        """
        Inference method - 与PyTorch版本完全一致
        """
        import torch
        
        # Convert inputs to MLX if needed
        if isinstance(mu, torch.Tensor):
            from indextts.utils.mlx_utils import torch_to_mlx
            mu_mlx = torch_to_mlx(mu.cpu())
            x_lens_mlx = torch_to_mlx(x_lens.cpu())
            prompt_mlx = torch_to_mlx(prompt.cpu())
            style_mlx = torch_to_mlx(style.cpu())
            f0_mlx = torch_to_mlx(f0.cpu()) if f0 is not None else None
            convert_back = True
        else:
            mu_mlx = mu
            x_lens_mlx = x_lens
            prompt_mlx = prompt
            style_mlx = style
            f0_mlx = f0
            convert_back = False
        
        batch, seq_len, _ = mu_mlx.shape
        
        # Initialize noise - 使用统一随机数生成器
        import torch
        torch.manual_seed(42)  # 确保与PyTorch版本同步
        mx.random.seed(42)
        
        if unified_random is not None:
            # 使用统一随机数生成器生成噪声
            z = unified_random.generate_noise_mlx((batch, self.in_channels, seq_len)) * temperature
        else:
            # 回退到原始方法
            z = mx.random.normal((batch, self.in_channels, seq_len)) * temperature
        
        # Create time span
        t_span = mx.linspace(0, 1, n_timesteps + 1)
        
        # Solve ODE
        result = self.solve_euler(z, x_lens_mlx, prompt_mlx, mu_mlx, style_mlx, f0_mlx, t_span, inference_cfg_rate, debug_layers=True)
        
        # Convert back if needed
        if convert_back:
            result = mlx_to_torch(result, device='mps')
        
        return result
