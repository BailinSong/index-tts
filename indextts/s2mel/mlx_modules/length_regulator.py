"""
MLX Implementation of InterpolateRegulator (Length Regulator)
Optimized for Apple Silicon M4
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Tuple, Optional


def sequence_mask(length: mx.array, max_length: Optional[int] = None) -> mx.array:
    """
    Create a sequence mask.
    
    Args:
        length: Shape (batch_size,)
        max_length: Maximum length (if None, use max of length)
    
    Returns:
        mask: Shape (batch_size, max_length)
    """
    if max_length is None:
        max_length = int(mx.max(length).item())
    
    # Create range [0, 1, 2, ..., max_length-1]
    x = mx.arange(max_length)
    # Broadcast and compare
    return x[None, :] < length[:, None]


def mlx_interpolate_nearest_1d(x: mx.array, target_length: int) -> mx.array:
    """
    MLX implementation of F.interpolate with mode='nearest' for 1D signals.
    
    Uses PyTorch-compatible nearest neighbor interpolation.
    Formula derived from PyTorch's implementation:
    index = min(floor(i * scale_factor), input_length - 1)
    where scale_factor = input_length / output_length
    
    Args:
        x: Input tensor (batch, channels, length)
        target_length: Target sequence length
    
    Returns:
        Interpolated tensor (batch, channels, target_length)
    """
    batch, channels, length = x.shape
    
    if length == target_length:
        return x
    
    # Compute indices using PyTorch's formula for nearest interpolation
    # This matches torch.nn.functional.interpolate(..., mode='nearest')
    scale_factor = length / target_length
    
    # For each output position, compute the input position
    # PyTorch uses: floor(output_idx * scale_factor)
    out_indices = mx.arange(target_length, dtype=mx.float32)
    in_indices = mx.floor(out_indices * scale_factor).astype(mx.int32)
    
    # Clamp to valid range
    in_indices = mx.clip(in_indices, 0, length - 1)
    
    # Gather values
    result = x[:, :, in_indices]  # (batch, channels, target_length)
    
    return result


class MLXInterpolateRegulator(nn.Module):
    """
    MLX implementation of InterpolateRegulator.
    
    Length regulator with optional interpolation for variable-length sequences.
    Supports discrete (codebook) and continuous inputs.
    """
    
    def __init__(
        self,
        channels: int,
        sampling_ratios: Tuple,
        is_discrete: bool = False,
        in_channels: int = None,
        vector_quantize: bool = False,
        codebook_size: int = 1024,
        out_channels: int = None,
        groups: int = 1,
        n_codebooks: int = 1,
        quantizer_dropout: float = 0.0,
        f0_condition: bool = False,
        n_f0_bins: int = 512,
    ):
        super().__init__()
        
        self.channels = channels
        self.sampling_ratios = sampling_ratios
        self.is_discrete = is_discrete
        self.n_codebooks = n_codebooks
        self.quantizer_dropout = quantizer_dropout
        self.f0_condition = f0_condition
        self.n_f0_bins = n_f0_bins
        
        out_channels = out_channels or channels
        
        # Build model layers
        model = []
        if len(sampling_ratios) > 0:
            self.interpolate = True
            for _ in sampling_ratios:
                # Conv1d -> GroupNorm -> Mish
                model.append(nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1))
                model.append(nn.GroupNorm(groups, channels))
                model.append(nn.Mish())
        else:
            self.interpolate = False
        
        # Final 1x1 conv
        model.append(nn.Conv1d(channels, out_channels, kernel_size=1, stride=1))
        
        self.model = model  # Store as list for sequential application
        
        # Embedding for discrete inputs
        if is_discrete:
            self.embedding = nn.Embedding(codebook_size, channels)
            if n_codebooks > 1:
                self.extra_codebooks = [
                    nn.Embedding(codebook_size, channels) for _ in range(n_codebooks - 1)
                ]
            else:
                self.extra_codebooks = []
        else:
            # Linear projection for continuous inputs
            self.content_in_proj = nn.Linear(in_channels, channels)
        
        # Mask token (learnable parameter)
        self.mask_token = mx.zeros((1, channels))
        
        # F0 conditioning
        if f0_condition:
            self.f0_embedding = nn.Embedding(n_f0_bins, channels)
            self.f0_mask = mx.zeros((1, channels))
            # F0 bins for bucketing
            self.f0_bins = mx.arange(2, 1024, 1024 // n_f0_bins)
    
    def __call__(
        self,
        x: mx.array,
        ylens: mx.array,
        n_quantizers: Optional[int] = None,
        f0: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array, None, None, None]:
        """
        Forward pass of MLX InterpolateRegulator.
        
        Args:
            x: Input tensor
               - If is_discrete: (batch, time) or (batch, n_codebooks, time)
               - If continuous: (batch, time, in_channels)
            ylens: Target lengths (batch,)
            n_quantizers: Number of quantizers to use (for multi-codebook)
            f0: F0 sequence (batch, time) - optional
        
        Returns:
            out: Output tensor (batch, target_len, out_channels)
            olens: Output lengths (batch,)
            None, None, None: Placeholder for compatibility
        """
        batch_size = x.shape[0]
        
        # Handle n_quantizers
        if n_quantizers is None:
            n_quantizers = mx.ones((batch_size,)) * self.n_codebooks
        else:
            if not isinstance(n_quantizers, mx.array):
                n_quantizers = mx.ones((batch_size,)) * n_quantizers
        
        # Process input based on type
        if self.is_discrete:
            # Discrete input: use embedding
            if self.n_codebooks > 1:
                # Multi-codebook: x shape (batch, n_codebooks, time)
                assert len(x.shape) == 3, f"Expected 3D input for multi-codebook, got {x.shape}"
                x_emb = self.embedding(x[:, 0])  # First codebook
                # Add extra codebooks
                for i, emb in enumerate(self.extra_codebooks):
                    # Conditional add based on n_quantizers
                    use_codebook = n_quantizers > (i + 1)
                    x_emb = x_emb + use_codebook[:, None, None] * emb(x[:, i + 1])
                x = x_emb
            elif self.n_codebooks == 1:
                # Single codebook
                if len(x.shape) == 2:
                    x = self.embedding(x)
                else:
                    x = self.embedding(x[:, 0])
            # x shape: (batch, time, channels)
        else:
            # Continuous input: use linear projection
            x = self.content_in_proj(x)
            # x shape: (batch, time, channels)
        
        # Create mask
        max_target_len = int(mx.max(ylens).item())
        mask = sequence_mask(ylens, max_target_len)  # (batch, max_target_len)
        mask = mask[:, :, None]  # (batch, max_target_len, 1) for broadcasting
        
        # x shape: (batch, time, channels)
        # NOTE: MLX Conv1d in this codebase expects input as (batch, length, channels)
        # But the Conv1d weights are stored as (out_channels, in_channels, kernel_size)
        # which matches PyTorch format
        
        # Interpolate if needed (work on time dimension)
        if self.interpolate:
            # Interpolate along time dimension
            # Need to transpose for interpolate function: (batch, time, channels) -> (batch, channels, time)
            x_for_interp = mx.transpose(x, [0, 2, 1])
            x_for_interp = mlx_interpolate_nearest_1d(x_for_interp, max_target_len)
            # Transpose back: (batch, channels, time) -> (batch, time, channels)
            x = mx.transpose(x_for_interp, [0, 2, 1])
        else:
            # Clip to max_target_len if input is longer
            if x.shape[1] > max_target_len:
                x = x[:, :max_target_len, :]
            # Update mask if input is shorter
            actual_len = x.shape[1]
            if actual_len < max_target_len:
                mask = mask[:, :actual_len, :]
                ylens = mx.clip(ylens, 0, actual_len)
        
        # Add F0 conditioning if enabled
        if self.f0_condition:
            if f0 is None:
                # No F0: add mask token
                # x shape: (batch, time, channels)
                f0_emb = self.f0_mask[None, None, :]  # (1, 1, channels)
                f0_emb = mx.broadcast_to(f0_emb, x.shape)
                x = x + f0_emb
            else:
                # Quantize F0
                # f0 shape: (batch, time)
                f0_coarse = f0_to_coarse_mlx(f0, self.n_f0_bins)
                f0_coarse = mx.clip(f0_coarse, 0, self.n_f0_bins - 1)
                
                # Get F0 embedding
                f0_emb = self.f0_embedding(f0_coarse)  # (batch, time, channels)
                
                # Interpolate F0 to target length if needed
                if f0_emb.shape[1] != max_target_len:
                    # Transpose for interpolation
                    f0_emb_t = mx.transpose(f0_emb, [0, 2, 1])  # (batch, channels, time)
                    f0_emb_t = mlx_interpolate_nearest_1d(f0_emb_t, max_target_len)
                    f0_emb = mx.transpose(f0_emb_t, [0, 2, 1])  # (batch, time, channels)
                
                x = x + f0_emb
        
        # Apply model layers sequentially
        # x shape: (batch, time, channels)
        # MLX Conv1d layers expect (batch, length, channels) format
        # The weights are (out_channels, in_channels, kernel_size) matching PyTorch
        for i, layer in enumerate(self.model):
            x = layer(x)
        
        # x shape: (batch, time, out_channels)
        out = x
        
        # Apply mask
        out = out * mask
        
        olens = ylens
        
        # Return format compatible with PyTorch version
        return out, olens, None, None, None


def f0_to_coarse_mlx(f0: mx.array, f0_bin: int) -> mx.array:
    """
    MLX implementation of f0_to_coarse.
    
    Converts continuous F0 to discrete bins.
    """
    f0_mel_min = 1127 * np.log(1 + 50.0 / 700)
    f0_mel_max = 1127 * np.log(1 + 1100.0 / 700)
    
    # Convert to mel scale
    f0_mel = 1127 * mx.log(1 + f0 / 700)
    
    # Linear mapping to bins
    a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
    b = f0_mel_min * a - 1.0
    
    f0_mel = mx.where(f0_mel > 0, f0_mel * a - b, f0_mel)
    
    # Round and convert to integer
    f0_coarse = mx.round(f0_mel).astype(mx.int32)
    
    # Clamp
    f0_coarse = f0_coarse * (f0_coarse > 0)  # Set negative to 0
    f0_coarse = f0_coarse + ((f0_coarse < 1) * 1)  # Set < 1 to 1
    f0_coarse = f0_coarse * (f0_coarse < f0_bin)  # Set >= f0_bin to 0
    f0_coarse = f0_coarse + ((f0_coarse >= f0_bin) * (f0_bin - 1))  # Set >= f0_bin to f0_bin-1
    
    return f0_coarse

