"""
MLX 优化模块

提供 IndexTTS2 的 MLX 优化版本实现
"""

__version__ = "2.0.0"
__all__ = []

# 延迟导入，避免不必要的加载
def _lazy_import():
    """延迟导入主要组件"""
    global IndexTTS2MLX, MLXModelLoader, MemoryOptimizer
    
    from .infer_mlx import IndexTTS2MLX
    from .model_loader import MLXModelLoader
    from .memory_optimizer import MemoryOptimizer
    
    __all__.extend(['IndexTTS2MLX', 'MLXModelLoader', 'MemoryOptimizer'])

# 模块级别的延迟导入
def __getattr__(name):
    if name in ['IndexTTS2MLX', 'MLXModelLoader', 'MemoryOptimizer']:
        _lazy_import()
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

