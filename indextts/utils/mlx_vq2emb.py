"""
MLX implementation of Semantic Codec vq2emb for Apple Silicon optimization.
This module provides a fast MLX-based vector quantization embedding lookup.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Optional
import torch


class MLXCodebookEmbedding(nn.Module):
    """
    MLX implementation of codebook embedding lookup.
    Equivalent to PyTorch's F.embedding(vq, embed_weight).
    """
    
    def __init__(self, embed_weight):
        """
        Args:
            embed_weight: Embedding weight (codebook_size, codebook_dim) - numpy or MLX array
        """
        super().__init__()
        # Convert to MLX array if needed
        if isinstance(embed_weight, torch.Tensor):
            # Use detach() to handle tensors with requires_grad=True
            self.embed = mx.array(embed_weight.detach().cpu().numpy())
        elif isinstance(embed_weight, np.ndarray):
            self.embed = mx.array(embed_weight)
        elif isinstance(embed_weight, mx.array):
            self.embed = embed_weight
        else:
            raise TypeError(f"Unsupported embed_weight type: {type(embed_weight)}")
        
        self.codebook_size, self.codebook_dim = self.embed.shape
    
    def __call__(self, vq):
        """
        Embedding lookup.
        
        Args:
            vq: Vector quantized indices (B, T) or (B, 1, T) - PyTorch or MLX array
        
        Returns:
            Embedding vectors (B, T, codebook_dim) or (B, 1, T, codebook_dim) - MLX array
        """
        # Convert to MLX if needed
        if isinstance(vq, torch.Tensor):
            vq_mlx = mx.array(vq.cpu().numpy().astype(np.int32))
            convert_back = True
            device = vq.device
        else:
            vq_mlx = vq if isinstance(vq, mx.array) else mx.array(vq)
            convert_back = False
        
        # Handle different input shapes
        original_shape = vq_mlx.shape
        if len(original_shape) == 3:
            # (B, 1, T) -> flatten to (B, T)
            B, _, T = original_shape
            vq_flat = vq_mlx.reshape(B, T)
        elif len(original_shape) == 2:
            # (B, T)
            vq_flat = vq_mlx
            B, T = original_shape
        else:
            raise ValueError(f"Unsupported vq shape: {original_shape}")
        
        # Embedding lookup using mx.take (equivalent to F.embedding)
        # vq_flat: (B, T) with indices in [0, codebook_size)
        # self.embed: (codebook_size, codebook_dim)
        # Output: (B, T, codebook_dim)
        emb = mx.take(self.embed, vq_flat, axis=0)
        
        # Always return (B, T, codebook_dim) format
        # The caller will handle any needed shape transformations
        
        # Convert back to PyTorch if needed
        if convert_back:
            from indextts.utils.mlx_utils import mlx_to_torch
            emb = mlx_to_torch(emb, device=device)
        
        return emb


class MLXWeightNormConv1d(nn.Module):
    """
    MLX implementation of WeightNorm Conv1d.
    Equivalent to PyTorch's weight_norm(nn.Conv1d(...)).
    
    Note: This loads weights that have already had weight_norm removed.
    """
    
    def __init__(self, weight, bias=None):
        """
        Args:
            weight: Conv1d weight (out_channels, in_channels, kernel_size) - PyTorch tensor or numpy array
            bias: Conv1d bias (out_channels,) - PyTorch tensor or numpy array, optional
        """
        super().__init__()
        
        # Convert weight to MLX format
        if isinstance(weight, torch.Tensor):
            # Use detach() to handle tensors with requires_grad=True
            weight_np = weight.detach().cpu().numpy()
        elif isinstance(weight, np.ndarray):
            weight_np = weight
        elif isinstance(weight, mx.array):
            weight_np = np.array(weight)
        else:
            raise TypeError(f"Unsupported weight type: {type(weight)}")
        
        # Extract dimensions
        out_channels, in_channels, kernel_size = weight_np.shape
        
        # Create MLX Conv1d layer
        # MLX Conv1d weight format is (out_channels, kernel_size, in_channels)
        # PyTorch format is (out_channels, in_channels, kernel_size)
        # So we need to transpose: (O, I, K) -> (O, K, I)
        weight_mlx = mx.array(np.transpose(weight_np, (0, 2, 1)))
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias is not None)
        self.conv.weight = weight_mlx
        
        if bias is not None:
            if isinstance(bias, torch.Tensor):
                # Use detach() to handle tensors with requires_grad=True
                self.conv.bias = mx.array(bias.detach().cpu().numpy())
            elif isinstance(bias, np.ndarray):
                self.conv.bias = mx.array(bias)
            elif isinstance(bias, mx.array):
                self.conv.bias = bias
            else:
                raise TypeError(f"Unsupported bias type: {type(bias)}")
        else:
            self.conv.bias = None
        
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
    
    def __call__(self, x):
        """
        Conv1d forward pass.
        
        Args:
            x: Input (B, C, T) - PyTorch format or (B, T, C) - MLX format
        
        Returns:
            Output (B, O, T) - PyTorch format or (B, T, O) - MLX format
        """
        # Convert to MLX if needed
        if isinstance(x, torch.Tensor):
            x_mlx = mx.array(x.cpu().numpy())
            convert_back = True
            device = x.device
            # PyTorch format (B, C, T) -> MLX format (B, T, C)
            x_mlx = x_mlx.transpose(0, 2, 1)
        else:
            x_mlx = x if isinstance(x, mx.array) else mx.array(x)
            convert_back = False
            # Assume already in MLX format (B, T, C)
        
        # MLX Conv1d expects (B, T, C) and outputs (B, T, O)
        output = self.conv(x_mlx)
        
        # Convert back to PyTorch format if needed
        if convert_back:
            # (B, T, O) -> (B, O, T)
            output = output.transpose(0, 2, 1)
            from indextts.utils.mlx_utils import mlx_to_torch
            output = mlx_to_torch(output, device=device)
        
        return output


class MLXVQ2EmbSingle(nn.Module):
    """
    MLX implementation of a single VectorQuantize vq2emb.
    This handles one quantizer in a ResidualVQ.
    """
    
    def __init__(self, quantizer):
        """
        Args:
            quantizer: PyTorch VectorQuantize instance
        """
        super().__init__()
        
        # Extract codebook embedding
        # Handle different quantizer types
        quantizer_type = type(quantizer).__name__
        
        if quantizer_type == 'FactorizedVectorQuantize':
            # FactorizedVectorQuantize uses nn.Embedding directly
            codebook = quantizer.codebook
            if isinstance(codebook, torch.nn.Embedding):
                embed_weight = codebook.weight  # (codebook_size, codebook_dim)
            else:
                raise ValueError(f"FactorizedVectorQuantize.codebook should be nn.Embedding, got {type(codebook)}")
        else:
            # VectorQuantize uses a codebook wrapper (SimpleCodebook or EuclideanCodebook)
            codebook = quantizer.codebook
            codebook_type = type(codebook).__name__
            
            # Handle different codebook types
            if codebook_type == 'SimpleCodebook':
                # SimpleCodebook uses nn.Embedding, need to access .weight
                if hasattr(codebook.embed, 'weight'):
                    embed_weight = codebook.embed.weight  # (codebook_size, codebook_dim)
                else:
                    # Fallback: might be direct Embedding
                    embed_weight = codebook.embed
            elif codebook_type == 'EuclideanCodebook':
                # EuclideanCodebook uses register_buffer("embed", ...)
                embed_weight = codebook.embed  # (codebook_size, codebook_dim)
            else:
                # Try to get embed attribute, fallback to weight
                if hasattr(codebook, 'embed'):
                    embed_weight = codebook.embed
                    if hasattr(embed_weight, 'weight'):
                        # It's an Embedding object
                        embed_weight = embed_weight.weight
                elif hasattr(codebook, 'weight'):
                    embed_weight = codebook.weight
                else:
                    raise ValueError(f"Cannot extract embedding weight from codebook type: {codebook_type}")
        
        # Create MLX codebook embedding
        self.codebook_emb = MLXCodebookEmbedding(embed_weight)
        
        # Extract out_project if it exists and is not Identity
        if quantizer.input_dim != quantizer.codebook_dim:
            # Has out_project (WNConv1d)
            out_project = quantizer.out_project
            
            # Ensure the module is in eval mode for correct weight_norm handling
            # In eval mode, PyTorch's weight_norm applies the normalization
            # We need to get the actual weight that will be used during inference
            original_training = out_project.training
            out_project.eval()  # Ensure eval mode for weight_norm computation
            
            # Trigger weight computation by calling forward once (if weight_norm is active)
            # This ensures weight_g and weight_v are properly normalized
            # Create a dummy input to trigger weight computation
            dummy_input = torch.zeros(1, quantizer.codebook_dim, 1, device=quantizer.codebook.weight.device)
            _ = out_project(dummy_input)  # This triggers weight_norm computation
            
            # Get weight and bias from the Conv1d layer
            # In eval mode, weight_norm modules compute weight = weight_g * weight_v / ||weight_v||
            # This is stored in .weight attribute after forward pass
            if hasattr(out_project, 'weight'):
                weight = out_project.weight.detach().clone()  # (O, I, K) - already normalized in eval mode
                bias = out_project.bias.detach().clone() if out_project.bias is not None else None
            else:
                # Might be wrapped in another module
                raise ValueError("Cannot extract out_project weight")
            
            # Restore original training state
            out_project.train(original_training)
            
            self.out_project = MLXWeightNormConv1d(weight, bias)
            self.has_out_project = True
        else:
            self.out_project = None
            self.has_out_project = False
        
        self.codebook_dim = quantizer.codebook_dim
        self.input_dim = quantizer.input_dim
    
    def __call__(self, vq, out_proj=True):
        """
        Single quantizer vq2emb.
        
        Args:
            vq: Vector quantized indices (B, T) - PyTorch tensor
               Note: In ResidualVQ, this is called with vq[idx] which is (B, T), not (B, 1, T)
            out_proj: Whether to apply out_project (default: True)
        
        Returns:
            Embedding vectors (B, input_dim, T) - PyTorch tensor
        """
        # Convert to MLX if needed
        if isinstance(vq, torch.Tensor):
            vq_mlx = mx.array(vq.cpu().numpy().astype(np.int32))
            convert_back = True
            device = vq.device
        else:
            vq_mlx = vq if isinstance(vq, mx.array) else mx.array(vq)
            convert_back = False
        
        # Handle input shape: PyTorch VectorQuantize.vq2emb receives (B, T) from ResidualVQ
        # So we expect (B, T) format
        original_shape = vq_mlx.shape
        if len(original_shape) == 2:
            # (B, T) - correct format
            vq_flat = vq_mlx
            B, T = original_shape
        elif len(original_shape) == 3:
            # (B, 1, T) - flatten to (B, T)
            B, _, T = original_shape
            vq_flat = vq_mlx.reshape(B, T)
        else:
            raise ValueError(f"Unsupported vq shape: {original_shape}")
        
        # Step 1: Embedding lookup (all in MLX)
        # Input vq_flat: (B, T)
        # PyTorch: F.embedding(vq, embed) with vq=(B, T) -> (B, T, codebook_dim)
        # MLX: mx.take with vq=(B, T) -> (B, T, codebook_dim)
        emb_mlx = self.codebook_emb(vq_flat)  # (B, T, codebook_dim)
        
        # Step 2: Transpose to match PyTorch format
        # PyTorch: (B, T, codebook_dim) -> transpose(1, 2) -> (B, codebook_dim, T)
        # MLX: (B, T, codebook_dim) -> transpose(0, 2, 1) -> (B, codebook_dim, T)
        emb_mlx = emb_mlx.transpose(0, 2, 1)  # (B, T, codebook_dim) -> (B, codebook_dim, T)
        
        # Step 3: Convert to PyTorch format for out_project
        from indextts.utils.mlx_utils import mlx_to_torch
        
        if convert_back:
            target_device = device
        else:
            target_device = 'cpu'
        
        emb_torch = mlx_to_torch(emb_mlx, device=target_device)
        
        # Step 4: Apply out_project if needed
        # emb_torch is now (B, codebook_dim, T) - correct format for Conv1d
        if out_proj and self.has_out_project:
            # out_project: (B, codebook_dim, T) -> (B, input_dim, T)
            # MLXWeightNormConv1d expects (B, C, T) which we have
            emb_torch = self.out_project(emb_torch)  # Returns (B, input_dim, T)
        
        return emb_torch


class MLXVQ2Emb(nn.Module):
    """
    MLX implementation of Semantic Codec vq2emb.
    
    This supports both VectorQuantize and ResidualVQ.
    """
    
    def __init__(self, quantizer):
        """
        Args:
            quantizer: PyTorch VectorQuantize or ResidualVQ instance (from semantic_codec.quantizer)
        """
        super().__init__()
        
        # Check if it's ResidualVQ or VectorQuantize
        quantizer_type = type(quantizer).__name__
        
        if quantizer_type == 'ResidualVQ':
            # ResidualVQ: contains multiple quantizers
            self.is_residual = True
            self.num_quantizers = quantizer.num_quantizers
            
            # Create MLX vq2emb for each quantizer
            self.mlx_quantizers = []
            for i, q in enumerate(quantizer.quantizers):
                try:
                    mlx_q = MLXVQ2EmbSingle(q)
                    self.mlx_quantizers.append(mlx_q)
                except Exception as e:
                    print(f">> Warning: Failed to create MLX vq2emb for quantizer {i}: {e}")
                    print(f">> Falling back to PyTorch for quantizer {i}")
                    self.mlx_quantizers.append(None)  # Use PyTorch fallback
            
            print(f">> Created MLX vq2emb for {sum(1 for q in self.mlx_quantizers if q is not None)}/{self.num_quantizers} quantizers")
        else:
            # Single VectorQuantize
            self.is_residual = False
            try:
                self.mlx_quantizer = MLXVQ2EmbSingle(quantizer)
            except Exception as e:
                raise ValueError(f"Failed to create MLX vq2emb for single quantizer: {e}")
        
        # Store original quantizer for fallback
        self.quantizer = quantizer
    
    def __call__(self, vq, n_quantizers=None, out_proj=True):
        """
        Vector quantization to embedding lookup.
        
        Args:
            vq: Vector quantized indices 
                - For ResidualVQ: (n_quantizers, B, T) or (B, 1, T) with n_quantizers inferred
                - For single VectorQuantize: (B, 1, T) or (B, T)
            n_quantizers: Number of quantizers to use (for ResidualVQ, default: all)
            out_proj: Whether to apply out_project (default: True)
        
        Returns:
            Embedding vectors (B, input_dim, T) - PyTorch tensor
        """
        if self.is_residual:
            # ResidualVQ: handle multiple quantizers
            if n_quantizers is None:
                n_quantizers = self.num_quantizers
            
            # Handle input shape for ResidualVQ
            # vq might be (B, 1, T) - we need to check how it's called
            # Based on ResidualVQ.vq2emb: it expects vq[idx] for each quantizer
            # So vq should be (n_quantizers, B, T) or we need to infer
            
            if isinstance(vq, torch.Tensor):
                vq_shape = vq.shape
                if len(vq_shape) == 3:
                    # Check if it's (n_quantizers, B, T) or (B, 1, T)
                    if vq_shape[0] == self.num_quantizers:
                        # (n_quantizers, B, T) - proper format
                        vq_per_quantizer = [vq[i] for i in range(min(n_quantizers, self.num_quantizers))]
                    elif vq_shape[1] == 1:
                        # (B, 1, T) - same vq for all quantizers (common case)
                        # ResidualVQ.vq2emb does: quantizer.vq2emb(vq[idx]) where idx is quantizer index
                        # If vq is (B, 1, T), then vq[0] is (1, T), vq[1] would be index error if only 1 quantizer
                        # Actually, if vq is (1, 1, T), then vq[0] = (1, T) which is correct
                        # But we need to handle the batch dimension properly
                        # Extract the first batch: vq[:, 0, :] -> (B, T), or vq[0] -> (1, T) if B=1
                        if vq_shape[0] == 1:
                            # Single batch: vq[0] -> (1, T)
                            vq_single = vq[0]  # (1, T)
                        else:
                            # Multiple batches: extract first channel
                            vq_single = vq[:, 0, :]  # (B, T)
                        vq_per_quantizer = [vq_single] * min(n_quantizers, self.num_quantizers)
                    else:
                        # Assume (B, C, T) format - extract first channel
                        vq_per_quantizer = [vq[:, 0, :] for _ in range(min(n_quantizers, self.num_quantizers))]
                elif len(vq_shape) == 2:
                    # (B, T) - same for all quantizers, extract first batch
                    vq_per_quantizer = [vq[0] if vq.shape[0] == 1 else vq] * min(n_quantizers, self.num_quantizers)
                else:
                    raise ValueError(f"Unsupported vq shape for ResidualVQ: {vq_shape}")
            else:
                # Non-tensor input - not supported for ResidualVQ
                raise ValueError(f"Unsupported vq type for ResidualVQ: {type(vq)}")
            
            # Process each quantizer and accumulate
            quantized_out = 0.0
            for idx in range(min(n_quantizers, self.num_quantizers)):
                if self.mlx_quantizers[idx] is not None:
                    # Use MLX version
                    emb = self.mlx_quantizers[idx](vq_per_quantizer[idx], out_proj=out_proj)
                else:
                    # Fallback to PyTorch
                    emb = self.quantizer.quantizers[idx].vq2emb(vq_per_quantizer[idx], out_proj=out_proj)
                
                quantized_out = quantized_out + emb
            
            return quantized_out
        else:
            # Single VectorQuantize
            return self.mlx_quantizer(vq, out_proj=out_proj)

