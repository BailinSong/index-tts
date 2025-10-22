"""
MLX Implementation of WaveNet for DiT Final Layer
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional


def fused_add_tanh_sigmoid_multiply_mlx(input_a, input_b, n_channels):
    """
    MLX implementation of fused gated activation.
    
    Args:
        input_a: (batch, 2*n_channels, seq_len)
        input_b: (batch, 2*n_channels, seq_len)
        n_channels: int
    
    Returns:
        output: (batch, n_channels, seq_len)
    """
    # Note: MLX uses (batch, seq_len, channels) format
    # So we need to adapt
    
    in_act = input_a + input_b
    
    # Split channels
    t_act = mx.tanh(in_act[:, :, :n_channels])
    s_act = mx.sigmoid(in_act[:, :, n_channels:])
    
    return t_act * s_act


class MLXWaveNet(nn.Module):
    """
    MLX implementation of WaveNet.
    Simplified version for DiT final layer (no streaming cache for inference).
    """
    
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0
    ):
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        
        # Input layers (dilated convolutions)
        self.in_layers = []
        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            
            layer = nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                bias=True
            )
            self.in_layers.append(layer)
        
        # Residual/skip layers
        self.res_skip_layers = []
        for i in range(n_layers):
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels
            
            layer = nn.Conv1d(
                hidden_channels,
                res_skip_channels,
                kernel_size=1,
                padding=0,
                bias=True
            )
            self.res_skip_layers.append(layer)
        
        # Conditioning layer (if using global conditioning)
        if gin_channels != 0:
            self.cond_layer = nn.Conv1d(
                gin_channels,
                2 * hidden_channels * n_layers,
                kernel_size=1,
                padding=0,
                bias=True
            )
        else:
            self.cond_layer = None
        
        # Dropout
        self.dropout = nn.Dropout(p_dropout)
        
        print(f">> MLX WaveNet initialized:")
        print(f"   Layers: {n_layers}, Channels: {hidden_channels}")
        print(f"   Kernel: {kernel_size}, Dilation rate: {dilation_rate}")
    
    def __call__(self, x, x_mask, g=None):
        """
        x: (batch, seq_len, hidden_channels) - MLX format
        x_mask: (batch, 1, seq_len) - mask (True = keep)
        g: (batch, seq_len, gin_channels) - global conditioning
        
        Returns:
            output: (batch, seq_len, hidden_channels)
        """
        # Note: MLX Conv1d expects (batch, seq_len, channels)
        # PyTorch WaveNet uses (batch, channels, seq_len)
        # We keep MLX format internally
        
        # Reshape mask for MLX broadcast: (batch, 1, seq_len) -> (batch, seq_len, 1)
        x_mask_mlx = x_mask.transpose(0, 2, 1)  # (batch, seq_len, 1)
        
        output = mx.zeros_like(x)
        
        # Process global conditioning
        if g is not None and self.cond_layer is not None:
            g_cond = self.cond_layer(g)  # (batch, seq_len, 2*hidden_channels*n_layers)
        else:
            g_cond = None
        
        for i in range(self.n_layers):
            # Apply mask
            x_in = self.in_layers[i](x * x_mask_mlx)  # (batch, seq_len, 2*hidden_channels)
            
            # Add global conditioning
            if g_cond is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g_cond[:, :, cond_offset:cond_offset + 2 * self.hidden_channels]
            else:
                g_l = mx.zeros_like(x_in)
            
            # Fused gated activation
            acts = fused_add_tanh_sigmoid_multiply_mlx(x_in, g_l, self.hidden_channels)
            
            # Dropout
            acts = self.dropout(acts)
            
            # Residual/skip connection
            res_skip_acts = self.res_skip_layers[i](acts)
            
            if i < self.n_layers - 1:
                # Split into residual and skip
                res_acts = res_skip_acts[:, :, :self.hidden_channels]
                skip_acts = res_skip_acts[:, :, self.hidden_channels:]
                
                x = (x + res_acts) * x_mask_mlx
                output = output + skip_acts
            else:
                # Last layer: only skip
                output = output + res_skip_acts
        
        return output * x_mask_mlx

