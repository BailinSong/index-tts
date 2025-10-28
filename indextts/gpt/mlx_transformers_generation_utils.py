"""
MLX Logits Processors - 完全遵循PyTorch transformers的实现
"""

import mlx.core as mx
import numpy as np
from typing import List, Set, Optional


class LogitsProcessor:
    """Base class for all logits processors"""
    
    def __call__(self, input_ids: mx.array, logits: mx.array) -> mx.array:
        """
        Args:
            input_ids: (batch_size, seq_len) - 当前已生成的token序列
            logits: (batch_size, vocab_size) - 原始logits
        Returns:
            processed_logits: (batch_size, vocab_size) - 处理后的logits
        """
        raise NotImplementedError


class TemperatureLogitsWarper(LogitsProcessor):
    """
    Temperature scaling for logits
    对应PyTorch: transformers.generation.logits_process.TemperatureLogitsWarper
    """
    
    def __init__(self, temperature: float):
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = temperature
    
    def __call__(self, input_ids: mx.array, logits: mx.array) -> mx.array:
        """Apply temperature scaling"""
        return logits / self.temperature


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    """
    Repetition penalty for previously generated tokens
    对应PyTorch: transformers.generation.logits_process.RepetitionPenaltyLogitsProcessor
    
    PyTorch实现逻辑：
    - 对input_ids中出现的所有token应用penalty
    - 如果 score < 0: score *= penalty  (更负)
    - 如果 score > 0: score /= penalty  (更小)
    """
    
    def __init__(self, penalty: float):
        if penalty <= 0:
            raise ValueError(f"penalty must be positive, got {penalty}")
        self.penalty = penalty
    
    def __call__(self, input_ids: mx.array, logits: mx.array) -> mx.array:
        """
        Apply repetition penalty to previously generated tokens
        
        PyTorch源码参考：
        score = torch.gather(logits, 1, input_ids)
        score = torch.where(score < 0, score * penalty, score / penalty)
        logits.scatter_(1, input_ids, score)
        """
        if self.penalty == 1.0:
            return logits
        
        batch_size, vocab_size = logits.shape
        
        # MLX实现：逐batch处理
        processed_logits = []
        for batch_idx in range(batch_size):
            batch_logits = logits[batch_idx]  # (vocab_size,)
            batch_input_ids = input_ids[batch_idx]  # (seq_len,)
            
            # 转为numpy便于操作
            logits_np = np.array(batch_logits)
            input_ids_list = batch_input_ids.tolist()
            
            # 获取所有已出现的unique tokens
            unique_tokens = set(input_ids_list)
            
            # 对每个已出现的token应用penalty
            for token_id in unique_tokens:
                if 0 <= token_id < vocab_size:
                    score = logits_np[token_id]
                    if score < 0:
                        logits_np[token_id] = score * self.penalty
                    else:
                        logits_np[token_id] = score / self.penalty
            
            processed_logits.append(mx.array(logits_np))
        
        return mx.stack(processed_logits, axis=0)


class TopPLogitsWarper(LogitsProcessor):
    """
    Top-p (nucleus) sampling
    对应PyTorch: transformers.generation.logits_process.TopPLogitsWarper
    
    只保留累积概率 <= top_p 的tokens
    """
    
    def __init__(self, top_p: float, filter_value: float = -float("inf"), min_tokens_to_keep: int = 1):
        if not (0 <= top_p <= 1.0):
            raise ValueError(f"top_p must be in [0, 1], got {top_p}")
        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep
    
    def __call__(self, input_ids: mx.array, logits: mx.array) -> mx.array:
        """Apply top-p filtering"""
        if self.top_p >= 1.0:
            return logits
        
        batch_size, vocab_size = logits.shape
        processed_logits = []
        
        for batch_idx in range(batch_size):
            batch_logits = logits[batch_idx]  # (vocab_size,)
            logits_np = np.array(batch_logits)
            
            # Sort in descending order
            sorted_indices = np.argsort(logits_np)[::-1]
            sorted_logits = logits_np[sorted_indices]
            
            # Compute softmax probabilities
            sorted_probs = np.exp(sorted_logits - sorted_logits.max())
            sorted_probs = sorted_probs / sorted_probs.sum()
            
            # Compute cumulative probabilities
            cumulative_probs = np.cumsum(sorted_probs)
            
            # Find cutoff: keep tokens until cumulative prob > top_p
            # But keep at least min_tokens_to_keep
            cutoff_idx = max(
                self.min_tokens_to_keep,
                np.searchsorted(cumulative_probs, self.top_p) + 1
            )
            
            # Filter out tokens beyond cutoff
            filtered_logits = np.full_like(logits_np, self.filter_value)
            filtered_logits[sorted_indices[:cutoff_idx]] = logits_np[sorted_indices[:cutoff_idx]]
            
            processed_logits.append(mx.array(filtered_logits))
        
        return mx.stack(processed_logits, axis=0)


class TopKLogitsWarper(LogitsProcessor):
    """
    Top-k sampling
    对应PyTorch: transformers.generation.logits_process.TopKLogitsWarper
    
    只保留top-k个最高分数的tokens
    """
    
    def __init__(self, top_k: int, filter_value: float = -float("inf"), min_tokens_to_keep: int = 1):
        if top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {top_k}")
        self.top_k = top_k
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep
    
    def __call__(self, input_ids: mx.array, logits: mx.array) -> mx.array:
        """Apply top-k filtering"""
        if self.top_k == 0:
            return logits
        
        batch_size, vocab_size = logits.shape
        top_k = min(max(self.top_k, self.min_tokens_to_keep), vocab_size)
        
        processed_logits = []
        
        for batch_idx in range(batch_size):
            batch_logits = logits[batch_idx]  # (vocab_size,)
            logits_np = np.array(batch_logits)
            
            # Find top-k indices
            top_k_indices = np.argsort(logits_np)[-top_k:]
            
            # Filter
            filtered_logits = np.full_like(logits_np, self.filter_value)
            filtered_logits[top_k_indices] = logits_np[top_k_indices]
            
            processed_logits.append(mx.array(filtered_logits))
        
        return mx.stack(processed_logits, axis=0)


class LogitsProcessorList(list):
    """
    List of logits processors that are applied sequentially
    对应PyTorch: transformers.generation.logits_process.LogitsProcessorList
    """
    
    def __call__(self, input_ids: mx.array, logits: mx.array) -> mx.array:
        """
        Apply all processors in sequence
        
        Args:
            input_ids: (batch_size, seq_len)
            logits: (batch_size, vocab_size)
        Returns:
            processed_logits: (batch_size, vocab_size)
        """
        for processor in self:
            logits = processor(input_ids, logits)
        return logits






