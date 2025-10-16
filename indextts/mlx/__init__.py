"""
MLX 优化模块

提供 IndexTTS2 的 MLX 优化版本实现
"""

__version__ = "2.0.0"

# 直接导入主要组件
from .infer_mlx import IndexTTS2MLX
from .model_loader import MLXModelLoader
from .memory_optimizer import MemoryOptimizer, LazySemanticModel, LazyQwenEmotion

__all__ = [
    'IndexTTS2MLX',
    'MLXModelLoader',
    'MemoryOptimizer',
    'LazySemanticModel',
    'LazyQwenEmotion',
]

