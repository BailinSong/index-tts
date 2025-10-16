"""
MLX 工具模块

提供 MLX 相关的工具函数和缓存管理
"""

__version__ = "2.0.0"

# 导出主要组件
from .cache import MLXCache
from .utils import torch_to_mlx, mlx_to_torch

__all__ = [
    'MLXCache',
    'torch_to_mlx',
    'mlx_to_torch',
]

