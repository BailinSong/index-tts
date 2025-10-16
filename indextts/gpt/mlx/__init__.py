"""
GPT MLX 实现模块

包含 UnifiedVoice 的完整 MLX 实现
"""

__version__ = "2.0.0"

# 导出主要组件
from .model import UnifiedVoiceMLX
from .conditioning import MLXConditioningModule, MLXConformerEncoder, MLXPerceiverResampler
from .logits_processors import (
    RepetitionPenaltyLogitsProcessorOptimized,
    TopKLogitsWarperOptimized,
    TopPLogitsWarperOptimized,
    LogitsProcessorList
)

__all__ = [
    'UnifiedVoiceMLX',
    'MLXConditioningModule',
    'MLXConformerEncoder',
    'MLXPerceiverResampler',
    'RepetitionPenaltyLogitsProcessorOptimized',
    'TopKLogitsWarperOptimized',
    'TopPLogitsWarperOptimized',
    'LogitsProcessorList',
]

