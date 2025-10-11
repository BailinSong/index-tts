"""
MLX Utility Functions for Apple Silicon M4 Optimization

Provides essential utilities for MLX framework integration,
including PyTorch-MLX data conversion and model helpers.
"""

import numpy as np


def torch_to_mlx(tensor):
    """
    Convert PyTorch tensor to MLX array.
    
    Uses unified memory on Apple Silicon for efficient conversion.
    
    Args:
        tensor: PyTorch tensor or nested structure
        
    Returns:
        MLX array or nested structure with MLX arrays
    """
    try:
        import torch
        import mlx.core as mx
        
        if tensor is None:
            return None
        
        # Handle nested structures
        if isinstance(tensor, (list, tuple)):
            return type(tensor)(torch_to_mlx(t) for t in tensor)
        elif isinstance(tensor, dict):
            return {k: torch_to_mlx(v) for k, v in tensor.items()}
        
        # Convert tensor
        if isinstance(tensor, torch.Tensor):
            numpy_array = tensor.detach().cpu().numpy()
            return mx.array(numpy_array)
        
        return tensor
        
    except ImportError as e:
        raise ImportError(f"Required libraries not available: {e}")


def mlx_to_torch(array, device='cpu'):
    """
    Convert MLX array to PyTorch tensor.
    
    Args:
        array: MLX array or nested structure
        device: Target PyTorch device
        
    Returns:
        PyTorch tensor or nested structure with PyTorch tensors
    """
    try:
        import torch
        import mlx.core as mx
        
        if array is None:
            return None
        
        # Handle nested structures
        if isinstance(array, (list, tuple)):
            return type(array)(mlx_to_torch(a, device) for a in array)
        elif isinstance(array, dict):
            return {k: mlx_to_torch(v, device) for k, v in array.items()}
        
        # Convert array
        if isinstance(array, mx.array):
            numpy_array = np.array(array)
            return torch.from_numpy(numpy_array).to(device)
        
        return array
        
    except ImportError as e:
        raise ImportError(f"Required libraries not available: {e}")


def check_mlx_available():
    """
    Check if MLX framework is available.
    
    Returns:
        bool: True if MLX is available and system supports it
    """
    try:
        import mlx.core as mx
        import torch
        
        # Check for Apple Silicon MPS support
        if not (hasattr(torch, "mps") and torch.backends.mps.is_available()):
            print(">> MLX requires Apple Silicon (MPS support not found)")
            return False
        
        return True
        
    except ImportError:
        return False

