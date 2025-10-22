"""
MLX Implementation of BigVGAN Vocoder for Apple Silicon
Uses official MLX nn.Conv1d and nn.ConvTranspose1d for optimal performance
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional
import json


class MLXSnake(nn.Module):
    """
    MLX implementation of Snake activation function.
    Snake(x) = x + (1/alpha) * sin^2(x * alpha)
    """
    
    def __init__(self, channels: int, alpha: float = 1.0, alpha_logscale: bool = False):
        super().__init__()
        self.channels = channels
        self.alpha_logscale = alpha_logscale
        self.no_div_by_zero = 1e-9
        
        # Initialize alpha parameter
        if alpha_logscale:
            self.alpha = mx.zeros((channels,)) * alpha
        else:
            self.alpha = mx.ones((channels,)) * alpha
    
    def __call__(self, x):
        """
        Forward pass.
        x: (batch, time, channels) - MLX format
        """
        # Reshape alpha to broadcast: (1, 1, channels)
        alpha = self.alpha.reshape(1, 1, -1)
        
        if self.alpha_logscale:
            alpha = mx.exp(alpha)
        
        # Snake formula: x + (1/alpha) * sin^2(x * alpha)
        sin_term = mx.sin(x * alpha)
        result = x + (1.0 / (alpha + self.no_div_by_zero)) * (sin_term ** 2)
        
        return result


class MLXSnakeBeta(nn.Module):
    """
    MLX implementation of SnakeBeta activation function.
    SnakeBeta(x) = x + (1/beta) * sin^2(x * alpha)
    """
    
    def __init__(self, channels: int, alpha: float = 1.0, alpha_logscale: bool = False):
        super().__init__()
        self.channels = channels
        self.alpha_logscale = alpha_logscale
        self.no_div_by_zero = 1e-9
        
        # Initialize alpha and beta parameters
        if alpha_logscale:
            self.alpha = mx.zeros((channels,)) * alpha
            self.beta = mx.zeros((channels,)) * alpha
        else:
            self.alpha = mx.ones((channels,)) * alpha
            self.beta = mx.ones((channels,)) * alpha
    
    def __call__(self, x):
        """
        Forward pass.
        x: (batch, time, channels) - MLX format
        """
        # Reshape to broadcast: (1, 1, channels)
        alpha = self.alpha.reshape(1, 1, -1)
        beta = self.beta.reshape(1, 1, -1)
        
        if self.alpha_logscale:
            alpha = mx.exp(alpha)
            beta = mx.exp(beta)
        
        # SnakeBeta formula: x + (1/beta) * sin^2(x * alpha)
        sin_term = mx.sin(x * alpha)
        result = x + (1.0 / (beta + self.no_div_by_zero)) * (sin_term ** 2)
        
        return result


class MLXAMPBlock1(nn.Module):
    """
    MLX implementation of AMPBlock1 (Anti-Aliased Multi-Periodicity Block).
    Uses residual connections with Snake/SnakeBeta activations.
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: tuple = (1, 3, 5),
        activation: str = "snakebeta",
        alpha_logscale: bool = True
    ):
        super().__init__()
        self.channels = channels
        self.num_layers = len(dilations) * 2  # Each dilation has 2 conv layers
        
        # First set of convolutions (with dilation)
        self.convs1 = []
        for dilation in dilations:
            padding = (kernel_size * dilation - dilation) // 2
            conv = nn.Conv1d(
                channels, channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                bias=True
            )
            self.convs1.append(conv)
        
        # Second set of convolutions (no dilation)
        self.convs2 = []
        for _ in dilations:
            padding = (kernel_size - 1) // 2
            conv = nn.Conv1d(
                channels, channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=1,
                bias=True
            )
            self.convs2.append(conv)
        
        # Activation functions
        if activation == "snake":
            self.activations = [
                MLXSnake(channels, alpha_logscale=alpha_logscale)
                for _ in range(self.num_layers)
            ]
        elif activation == "snakebeta":
            self.activations = [
                MLXSnakeBeta(channels, alpha_logscale=alpha_logscale)
                for _ in range(self.num_layers)
            ]
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def __call__(self, x):
        """
        Forward pass with residual connection.
        x: (batch, time, channels) - MLX format
        """
        for i, (c1, c2) in enumerate(zip(self.convs1, self.convs2)):
            # Get corresponding activations
            a1 = self.activations[i * 2]
            a2 = self.activations[i * 2 + 1]
            
            # Residual block: act -> conv -> act -> conv -> add
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x  # Residual connection
        
        return x


class MLXBigVGAN(nn.Module):
    """
    MLX implementation of BigVGAN vocoder using official MLX components.
    
    Architecture:
    - Conv1d pre-processing
    - ConvTranspose1d upsampling (6 stages)
    - AMPBlock residual blocks with Snake activations
    - Conv1d post-processing
    
    Args:
        config: Configuration dict with hyperparameters
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # Extract hyperparameters
        num_mels = config.get("num_mels", 80)
        upsample_rates = config.get("upsample_rates", [4, 4, 2, 2, 2, 2])
        upsample_kernel_sizes = config.get("upsample_kernel_sizes", [8, 8, 4, 4, 4, 4])
        upsample_initial_channel = config.get("upsample_initial_channel", 1536)
        resblock_kernel_sizes = config.get("resblock_kernel_sizes", [3, 7, 11])
        resblock_dilation_sizes = config.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
        activation = config.get("activation", "snakebeta")
        snake_logscale = config.get("snake_logscale", True)
        use_tanh_at_final = config.get("use_tanh_at_final", False)
        use_bias_at_final = config.get("use_bias_at_final", False)
        
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.use_tanh_at_final = use_tanh_at_final
        
        # Pre-conv (80 mel -> 1536 channels)
        self.conv_pre = nn.Conv1d(
            num_mels,
            upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=True
        )
        
        # Upsampling layers (ConvTranspose1d)
        self.ups = []
        for i, (rate, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_ch = upsample_initial_channel // (2 ** i)
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            padding = (kernel_size - rate) // 2
            
            up = nn.ConvTranspose1d(
                in_ch, out_ch,
                kernel_size=kernel_size,
                stride=rate,
                padding=padding,
                bias=True
            )
            self.ups.append([up])  # Wrap in list for compatibility
        
        # Residual blocks (AMPBlock1)
        self.resblocks = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilations in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                block = MLXAMPBlock1(
                    ch,
                    kernel_size=kernel_size,
                    dilations=dilations,
                    activation=activation,
                    alpha_logscale=snake_logscale
                )
                self.resblocks.append(block)
        
        # Post-conv activation
        final_channels = upsample_initial_channel // (2 ** len(upsample_rates))
        if activation == "snake":
            self.activation_post = MLXSnake(final_channels, alpha_logscale=snake_logscale)
        elif activation == "snakebeta":
            self.activation_post = MLXSnakeBeta(final_channels, alpha_logscale=snake_logscale)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Post-conv (final_channels -> 1 waveform)
        self.conv_post = nn.Conv1d(
            final_channels, 1,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=use_bias_at_final
        )
        
        print(f">> MLX BigVGAN initialized:")
        print(f"   Upsampling: {self.num_upsamples} stages, rates={upsample_rates}")
        print(f"   Residual blocks: {len(self.resblocks)} total")
        print(f"   Activation: {activation}")
    
    def __call__(self, x):
        """
        Forward pass: mel-spectrogram -> waveform.
        
        Args:
            x: Mel-spectrogram (batch, num_mels, time) - PyTorch format
               Will be converted to (batch, time, num_mels) - MLX format
            
        Returns:
            waveform: Audio waveform (batch, 1, time * prod(upsample_rates)) - PyTorch format
        """
        # Convert from PyTorch format (B, C, T) to MLX format (B, T, C)
        x = x.transpose(0, 2, 1)  # (batch, time, num_mels)
        
        # Pre-conv
        x = self.conv_pre(x)  # (batch, time, 1536)
        
        # Upsampling + residual blocks
        for i in range(self.num_upsamples):
            # Upsample
            for up_layer in self.ups[i]:
                x = up_layer(x)
            
            # Apply residual blocks (average outputs)
            xs = None
            for j in range(self.num_kernels):
                block_output = self.resblocks[i * self.num_kernels + j](x)
                if xs is None:
                    xs = block_output
                else:
                    xs = xs + block_output
            x = xs / self.num_kernels
        
        # Post-conv
        x = self.activation_post(x)
        x = self.conv_post(x)  # (batch, time, 1)
        
        # Final activation
        if self.use_tanh_at_final:
            x = mx.tanh(x)
        else:
            x = mx.clip(x, -1.0, 1.0)
        
        # Convert back to PyTorch format (B, C, T)
        x = x.transpose(0, 2, 1)  # (batch, 1, time)
        
        return x
    
    def load_weights_from_pytorch(self, pytorch_state_dict: dict):
        """
        Load weights from PyTorch BigVGAN checkpoint.
        
        NOTE: Expects state_dict AFTER remove_weight_norm() has been called!
        The weights should be in regular format, not weight_g/weight_v.
        
        Handles weight format conversion:
        - PyTorch Conv1d: (out_channels, in_channels, kernel_size)
        - MLX Conv1d: (out_channels, kernel_size, in_channels)
        """
        import numpy as np
        
        loaded = 0
        
        # Helper functions to convert weights
        def convert_conv1d_weight(w):
            """PyTorch Conv1d (O, I, K) -> MLX (O, K, I)"""
            return w.transpose(0, 2, 1)
        
        def convert_convtranspose1d_weight(w):
            """PyTorch ConvTranspose1d (I, O, K) -> MLX (O, K, I)"""
            # PyTorch: (in_channels, out_channels, kernel_size)
            # MLX: (out_channels, kernel_size, in_channels)
            return w.transpose(1, 2, 0)  # (I, O, K) -> (O, K, I)
        
        # Load conv_pre (regular weight format after remove_weight_norm)
        if "conv_pre.weight" in pytorch_state_dict:
            w = pytorch_state_dict["conv_pre.weight"]
            self.conv_pre.weight = mx.array(convert_conv1d_weight(w))
            loaded += 1
        
        if "conv_pre.bias" in pytorch_state_dict:
            self.conv_pre.bias = mx.array(pytorch_state_dict["conv_pre.bias"])
            loaded += 1
        
        # Load upsampling layers
        for i in range(self.num_upsamples):
            prefix = f"ups.{i}.0"
            if f"{prefix}.weight" in pytorch_state_dict:
                w = pytorch_state_dict[f"{prefix}.weight"]
                # ConvTranspose1d: different format!
                self.ups[i][0].weight = mx.array(convert_convtranspose1d_weight(w))
                loaded += 1
            if f"{prefix}.bias" in pytorch_state_dict:
                self.ups[i][0].bias = mx.array(pytorch_state_dict[f"{prefix}.bias"])
                loaded += 1
        
        # Load residual blocks
        for i in range(len(self.resblocks)):
            block = self.resblocks[i]
            prefix = f"resblocks.{i}"
            
            # Load convs1
            for j in range(len(block.convs1)):
                conv_prefix = f"{prefix}.convs1.{j}"
                if f"{conv_prefix}.weight" in pytorch_state_dict:
                    w = pytorch_state_dict[f"{conv_prefix}.weight"]
                    block.convs1[j].weight = mx.array(convert_conv1d_weight(w))
                    loaded += 1
                if f"{conv_prefix}.bias" in pytorch_state_dict:
                    block.convs1[j].bias = mx.array(pytorch_state_dict[f"{conv_prefix}.bias"])
                    loaded += 1
            
            # Load convs2
            for j in range(len(block.convs2)):
                conv_prefix = f"{prefix}.convs2.{j}"
                if f"{conv_prefix}.weight" in pytorch_state_dict:
                    w = pytorch_state_dict[f"{conv_prefix}.weight"]
                    block.convs2[j].weight = mx.array(convert_conv1d_weight(w))
                    loaded += 1
                if f"{conv_prefix}.bias" in pytorch_state_dict:
                    block.convs2[j].bias = mx.array(pytorch_state_dict[f"{conv_prefix}.bias"])
                    loaded += 1
            
            # Load activation parameters (alpha, beta)
            for j in range(len(block.activations)):
                act_prefix = f"{prefix}.activations.{j}.activation"
                act = block.activations[j]
                
                if f"{act_prefix}.alpha" in pytorch_state_dict:
                    act.alpha = mx.array(pytorch_state_dict[f"{act_prefix}.alpha"])
                    loaded += 1
                
                # SnakeBeta has beta parameter
                if hasattr(act, 'beta') and f"{act_prefix}.beta" in pytorch_state_dict:
                    act.beta = mx.array(pytorch_state_dict[f"{act_prefix}.beta"])
                    loaded += 1
        
        # Load post activation
        if "activation_post.activation.alpha" in pytorch_state_dict:
            self.activation_post.alpha = mx.array(pytorch_state_dict["activation_post.activation.alpha"])
            loaded += 1
        if hasattr(self.activation_post, 'beta') and "activation_post.activation.beta" in pytorch_state_dict:
            self.activation_post.beta = mx.array(pytorch_state_dict["activation_post.activation.beta"])
            loaded += 1
        
        # Load conv_post
        if "conv_post.weight" in pytorch_state_dict:
            w = pytorch_state_dict["conv_post.weight"]
            self.conv_post.weight = mx.array(convert_conv1d_weight(w))
            loaded += 1
        if "conv_post.bias" in pytorch_state_dict:
            self.conv_post.bias = mx.array(pytorch_state_dict["conv_post.bias"])
            loaded += 1
        
        print(f">> MLX BigVGAN loaded {loaded} weight tensors from PyTorch checkpoint")
        
        return loaded


def create_mlx_bigvgan_from_pytorch(pytorch_model, config: dict = None):
    """
    Create MLX BigVGAN from PyTorch BigVGAN model.
    
    Args:
        pytorch_model: PyTorch BigVGAN model instance
        config: Optional config dict (will use pytorch_model.h if not provided)
        
    Returns:
        MLX BigVGAN model with loaded weights
    """
    if config is None:
        config = pytorch_model.h
    
    mlx_model = MLXBigVGAN(config)
    
    # Get PyTorch state dict (without weight_norm)
    state_dict = pytorch_model.state_dict()
    
    # Convert to numpy
    state_dict_np = {k: v.cpu().numpy() for k, v in state_dict.items()}
    
    # Load weights
    mlx_model.load_weights_from_pytorch(state_dict_np)
    
    return mlx_model

