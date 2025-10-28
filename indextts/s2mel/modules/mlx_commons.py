"""
MLX Implementation of S2MEL Components
Uses official MLX components for maximum performance and compatibility
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
import math


def init_linear_pytorch_compatible(linear_layer: nn.Linear, std: float = 0.01):
    """
    Initialize MLX Linear layer to match PyTorch initialization.
    
    Args:
        linear_layer: MLX Linear layer
        std: Standard deviation for normal initialization (default: 0.01)
    """
    # PyTorch uses normal_(mean=0.0, std=std) for Linear layers
    linear_layer.weight = mx.random.normal(linear_layer.weight.shape) * std
    # MLX Linear layers don't have bias attribute, they have bias parameter
    if hasattr(linear_layer, 'bias') and linear_layer.bias is not None:
        linear_layer.bias = mx.zeros_like(linear_layer.bias)


def init_embedding_pytorch_compatible(embedding_layer: nn.Embedding, std: float = 0.01):
    """
    Initialize MLX Embedding layer to match PyTorch initialization.
    
    Args:
        embedding_layer: MLX Embedding layer
        std: Standard deviation for normal initialization (default: 0.01)
    """
    # PyTorch uses normal_(mean=0.0, std=std) for Embedding layers
    embedding_layer.weight = mx.random.normal(embedding_layer.weight.shape) * std


def init_conv_pytorch_compatible(conv_layer: nn.Conv1d, std: float = 0.01):
    """
    Initialize MLX Conv1d layer to match PyTorch initialization.
    
    Args:
        conv_layer: MLX Conv1d layer
        std: Standard deviation for normal initialization (default: 0.01)
    """
    # PyTorch uses normal_(mean=0.0, std=std) for Conv1d layers
    conv_layer.weight = mx.random.normal(conv_layer.weight.shape) * std
    # MLX Conv1d layers don't have bias attribute, they have bias parameter
    if hasattr(conv_layer, 'bias') and conv_layer.bias is not None:
        conv_layer.bias = mx.zeros_like(conv_layer.bias)


class MLXGPTLayer(nn.Module):
    """
    MLX implementation of S2MEL's gpt_layer.
    Simple 3-layer MLP: 1280 -> 256 -> 128 -> 1024
    """
    
    def __init__(self):
        super().__init__()
        self.layer0 = nn.Linear(1280, 256, bias=True)
        self.layer1 = nn.Linear(256, 128, bias=True)
        self.layer2 = nn.Linear(128, 1024, bias=True)
        
        # Initialize with PyTorch-compatible weights
        init_linear_pytorch_compatible(self.layer0)
        init_linear_pytorch_compatible(self.layer1)
        init_linear_pytorch_compatible(self.layer2)
    
    def __call__(self, x):
        """
        Args:
            x: (batch, seq_len, 1280)
        Returns:
            x: (batch, seq_len, 1024)
        """
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    
    def load_weights_from_pytorch(self, pytorch_state_dict: dict, prefix="models.gpt_layer."):
        """Load weights from PyTorch gpt_layer"""
        loaded = 0
        
        # Layer 0
        if f"{prefix}0.weight" in pytorch_state_dict:
            self.layer0.weight = mx.array(pytorch_state_dict[f"{prefix}0.weight"])
            loaded += 1
        if f"{prefix}0.bias" in pytorch_state_dict:
            self.layer0.bias = mx.array(pytorch_state_dict[f"{prefix}0.bias"])
            loaded += 1
        
        # Layer 1
        if f"{prefix}1.weight" in pytorch_state_dict:
            self.layer1.weight = mx.array(pytorch_state_dict[f"{prefix}1.weight"])
            loaded += 1
        if f"{prefix}1.bias" in pytorch_state_dict:
            self.layer1.bias = mx.array(pytorch_state_dict[f"{prefix}1.bias"])
            loaded += 1
        
        # Layer 2
        if f"{prefix}2.weight" in pytorch_state_dict:
            self.layer2.weight = mx.array(pytorch_state_dict[f"{prefix}2.weight"])
            loaded += 1
        if f"{prefix}2.bias" in pytorch_state_dict:
            self.layer2.bias = mx.array(pytorch_state_dict[f"{prefix}2.bias"])
            loaded += 1
        
        print(f">> MLX GPTLayer loaded {loaded} weight tensors")
        return loaded


class MLXLengthRegulator(nn.Module):
    """
    MLX implementation of InterpolateRegulator.
    Handles semantic token interpolation and upsampling.
    """
    
    def __init__(
        self,
        channels: int,
        sampling_ratios: Tuple,
        is_discrete: bool = False,
        in_channels: int = None,
        codebook_size: int = 1024,
        out_channels: int = None,
        groups: int = 1,
        n_codebooks: int = 1,
        f0_condition: bool = False,
        n_f0_bins: int = 512,
    ):
        super().__init__()
        self.channels = channels
        self.sampling_ratios = sampling_ratios
        self.is_discrete = is_discrete
        self.n_codebooks = n_codebooks
        self.f0_condition = f0_condition
        
        out_channels = out_channels or channels
        
        # Build model layers
        layers = []
        if len(sampling_ratios) > 0:
            self.interpolate = True
            for _ in sampling_ratios:
                # Conv1d
                layers.append(nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1))
                # GroupNorm
                layers.append(nn.GroupNorm(groups, channels))
                # Mish activation
                layers.append(nn.Mish())
        else:
            self.interpolate = False
        
        # Final 1x1 conv
        layers.append(nn.Conv1d(channels, out_channels, kernel_size=1, stride=1, padding=0))
        
        self.model_layers = layers
        
        # Embeddings
        self.embedding = nn.Embedding(codebook_size, channels)
        init_embedding_pytorch_compatible(self.embedding)
        self.is_discrete = is_discrete
        
        # Mask token (learnable parameter)
        self.mask_token = mx.zeros((1, channels))
        
        # Extra codebooks
        if n_codebooks > 1:
            self.extra_codebooks = [
                nn.Embedding(codebook_size, channels)
                for _ in range(n_codebooks - 1)
            ]
            for extra_emb in self.extra_codebooks:
                init_embedding_pytorch_compatible(extra_emb)
            self.extra_codebook_mask_tokens = [
                mx.zeros((1, channels))
                for _ in range(n_codebooks - 1)
            ]
        
        # F0 conditioning
        if f0_condition:
            self.f0_embedding = nn.Embedding(n_f0_bins, channels)
            init_embedding_pytorch_compatible(self.f0_embedding)
            self.f0_mask = mx.zeros((1, channels))
        
        # Non-discrete mode
        if not is_discrete and in_channels is not None:
            self.content_in_proj = nn.Linear(in_channels, channels)
            init_linear_pytorch_compatible(self.content_in_proj)
        else:
            self.content_in_proj = None
        
        print(f">> MLX LengthRegulator initialized:")
        print(f"   Channels: {channels}")
        print(f"   Sampling ratios: {sampling_ratios}")
        print(f"   Discrete mode: {is_discrete}")
    
    def __call__(self, x, ylens=None, n_quantizers=None, f0=None):
        """
        Args:
            x: Input codes/features
                - If discrete: (batch, n_codebooks, seq_len) int32
                - If continuous: (batch, seq_len, in_channels) float32
            ylens: Target lengths (batch,) - can be mx.array or torch.Tensor
            n_quantizers: Number of quantizers to use
            f0: F0 values (optional)
        
        Returns:
            Tuple of (output, olens, codes, commitment_loss, codebook_loss)
            output: (batch, seq_len, out_channels)
        """
        import torch
        
        # Convert ylens to MLX if it's a torch tensor
        if isinstance(ylens, torch.Tensor):
            ylens_mx = mx.array(ylens.cpu().numpy())
        else:
            ylens_mx = ylens
        
        batch_size = x.shape[0]
        
        if self.is_discrete:
            # Discrete mode: embed codes
            # x: (batch, n_codebooks, seq_len)
            # Use first codebook
            embedded = self.embedding(x[:, 0, :])  # (batch, seq_len, channels)
            
            # Add extra codebooks if needed
            if self.n_codebooks > 1 and x.shape[1] > 1:
                # Handle n_quantizers (number of codebooks to use)
                if n_quantizers is None:
                    n_quant_val = self.n_codebooks
                else:
                    n_quant_val = int(n_quantizers) if isinstance(n_quantizers, (int, float)) else int(n_quantizers[0])
                
                for i in range(min(x.shape[1] - 1, len(self.extra_codebooks))):
                    if i + 1 < n_quant_val:  # Only add if within n_quantizers
                        extra_emb = self.extra_codebooks[i](x[:, i+1, :])
                        embedded = embedded + extra_emb
        else:
            # Continuous mode: project features
            if self.content_in_proj is not None:
                embedded = self.content_in_proj(x)  # (batch, seq_len, channels)
            else:
                embedded = x  # No projection
        
        # Add F0 conditioning if needed
        if self.f0_condition and f0 is not None:
            # F0 conditioning implementation pending
            pass
        
        # Interpolate BEFORE applying model layers (matches PyTorch)
        if self.interpolate and ylens_mx is not None:
            # PyTorch: F.interpolate(x.transpose(1,2), size=ylens.max(), mode='nearest')
            # Input: (batch, seq, channels) -> transpose to (batch, channels, seq)
            # Then interpolate along seq dimension
            
            target_len = int(ylens_mx.max())
            current_len = embedded.shape[1]
            
            if target_len != current_len:
                # 手动实现nearest插值以精确匹配PyTorch
                # PyTorch在(B,C,T)格式下沿T轴插值
                # 我们需要在(B,T,C)格式下沿T轴插值
                
                # 计算每个输出位置对应的输入位置
                scale = current_len / target_len
                indices = mx.arange(target_len) * scale
                indices = mx.floor(indices).astype(mx.int32)
                indices = mx.minimum(indices, current_len - 1)  # Clamp to valid range
                
                # 使用索引选择
                embedded = embedded[:, indices, :]  # (batch, target_len, channels)
        
        # Apply model layers (Conv1d, GroupNorm, Mish, final Conv1d)
        out = embedded
        for layer in self.model_layers:
            out = layer(out)
        
        # Create mask
        if ylens_mx is not None:
            # Sequence mask: True for valid positions
            max_len = out.shape[1]
            positions = mx.arange(max_len).reshape(1, -1)  # (1, max_len)
            lengths = ylens_mx.reshape(-1, 1)  # (batch, 1)
            mask = positions < lengths  # (batch, max_len)
            mask = mask.reshape(batch_size, max_len, 1)  # (batch, seq_len, 1)
            
            # Apply mask
            out = out * mask
            olens = ylens_mx
        else:
            olens = mx.array([out.shape[1]] * batch_size)
        
        # Return format matching PyTorch
        return out, olens, None, None, None
    
    def load_weights_from_pytorch(self, pytorch_state_dict: dict, prefix="models.length_regulator."):
        """Load weights from PyTorch length_regulator"""
        loaded = 0
        
        # Helper for Conv1d weight conversion
        def convert_conv1d_weight(w):
            """PyTorch (O, I, K) -> MLX (O, K, I)"""
            return w.transpose(0, 2, 1)
        
        # Embeddings
        if f"{prefix}embedding.weight" in pytorch_state_dict:
            self.embedding.weight = mx.array(pytorch_state_dict[f"{prefix}embedding.weight"])
            loaded += 1
        
        # Extra codebooks
        if self.n_codebooks > 1:
            for i in range(len(self.extra_codebooks)):
                key = f"{prefix}extra_codebooks.{i}.weight"
                if key in pytorch_state_dict:
                    self.extra_codebooks[i].weight = mx.array(pytorch_state_dict[key])
                    loaded += 1
        
        # Mask tokens
        if f"{prefix}mask_token" in pytorch_state_dict:
            self.mask_token = mx.array(pytorch_state_dict[f"{prefix}mask_token"])
            loaded += 1
        
        # F0 conditioning
        if self.f0_condition:
            if f"{prefix}f0_embedding.weight" in pytorch_state_dict:
                self.f0_embedding.weight = mx.array(pytorch_state_dict[f"{prefix}f0_embedding.weight"])
                loaded += 1
            if f"{prefix}f0_mask" in pytorch_state_dict:
                self.f0_mask = mx.array(pytorch_state_dict[f"{prefix}f0_mask"])
                loaded += 1
        
        # Non-discrete projection
        if not self.is_discrete and self.content_in_proj is not None:
            if f"{prefix}content_in_proj.weight" in pytorch_state_dict:
                self.content_in_proj.weight = mx.array(pytorch_state_dict[f"{prefix}content_in_proj.weight"])
                loaded += 1
            if f"{prefix}content_in_proj.bias" in pytorch_state_dict:
                self.content_in_proj.bias = mx.array(pytorch_state_dict[f"{prefix}content_in_proj.bias"])
                loaded += 1
        
        # Model layers
        # Count conv layers and other layers separately
        conv_idx = 0
        norm_idx = 0
        
        for i, layer in enumerate(self.model_layers):
            if isinstance(layer, nn.Conv1d):
                # Conv1d weights
                w_key = f"{prefix}model.{i}.weight"
                b_key = f"{prefix}model.{i}.bias"
                
                if w_key in pytorch_state_dict:
                    w = pytorch_state_dict[w_key]
                    layer.weight = mx.array(convert_conv1d_weight(w))
                    loaded += 1
                if b_key in pytorch_state_dict:
                    layer.bias = mx.array(pytorch_state_dict[b_key])
                    loaded += 1
                conv_idx += 1
                
            elif isinstance(layer, nn.GroupNorm):
                # GroupNorm weights
                w_key = f"{prefix}model.{i}.weight"
                b_key = f"{prefix}model.{i}.bias"
                
                if w_key in pytorch_state_dict:
                    layer.weight = mx.array(pytorch_state_dict[w_key])
                    loaded += 1
                if b_key in pytorch_state_dict:
                    layer.bias = mx.array(pytorch_state_dict[b_key])
                    loaded += 1
                norm_idx += 1
        
        print(f">> MLX LengthRegulator loaded {loaded} weight tensors")
        return loaded


def create_mlx_gpt_layer():
    """Create MLX gpt_layer component"""
    return MLXGPTLayer()


def create_mlx_length_regulator(config):
    """Create MLX length_regulator from config"""
    return MLXLengthRegulator(
        channels=config.get('channels', 1024),
        sampling_ratios=config.get('sampling_ratios', []),
        is_discrete=config.get('is_discrete', True),
        in_channels=config.get('in_channels', 1024),
        codebook_size=config.get('content_codebook_size', 8192),
        n_codebooks=config.get('n_codebooks', 1),
        f0_condition=config.get('f0_condition', False),
        n_f0_bins=config.get('n_f0_bins', 512),
    )

