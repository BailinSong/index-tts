"""
MLX 工具模块

提供 MLX 相关的工具函数和缓存管理
"""

__version__ = "2.0.0"

# 导出主要组件
from .cache import MLXModelCache
from .utils import torch_to_mlx, mlx_to_torch, check_mlx_available

__all__ = [
    'MLXModelCache',
    'torch_to_mlx',
    'mlx_to_torch',
    'check_mlx_available',
]

