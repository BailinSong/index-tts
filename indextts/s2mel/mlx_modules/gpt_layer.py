"""
MLX Implementation of GPT Layer for S2MEL
Simple 3-layer MLP: 1280 → 256 → 128 → 1024
"""

import mlx.core as mx
import mlx.nn as nn


class MLXGPTLayer(nn.Module):
    """
    MLX implementation of S2MEL's GPT projection layer.
    
    Converts GPT latent representation (1280D) to S2MEL conditioning (1024D)
    through a 3-layer bottleneck MLP.
    
    Architecture:
        Linear(1280 → 256) → Linear(256 → 128) → Linear(128 → 1024)
    
    Note: No activation functions between layers in original PyTorch version.
    """
    
    def __init__(self):
        super().__init__()
        
        self.layer1 = nn.Linear(1280, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 1024)
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, 1280)
        
        Returns:
            Output tensor (batch, seq_len, 1024)
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

