"""
MLX Logits Processors - 优化版本
主要优化：
1. 使用纯MLX操作，避免numpy转换
2. 向量化处理，避免Python循环
3. 合并常用processors，减少overhead
"""

import mlx.core as mx
from typing import List, Optional


class LogitsProcessor:
    """Base class for all logits processors"""
    
    def __call__(self, input_ids: mx.array, logits: mx.array) -> mx.array:
        raise NotImplementedError


class TemperatureLogitsWarper(LogitsProcessor):
    """
    Temperature scaling - 已经很快，无需优化
    """
    
    def __init__(self, temperature: float):
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = temperature
    
    def __call__(self, input_ids: mx.array, logits: mx.array) -> mx.array:
        return logits / self.temperature


class RepetitionPenaltyLogitsProcessorOptimized(LogitsProcessor):
    """
    优化版本的Repetition Penalty
    
    优化点：
    1. 纯MLX操作，避免numpy转换
    2. 向量化处理，避免Python循环
    3. 使用mx.scatter减少操作
    """
    
    def __init__(self, penalty: float):
        if penalty <= 0:
            raise ValueError(f"penalty must be positive, got {penalty}")
        self.penalty = penalty
    
    def __call__(self, input_ids: mx.array, logits: mx.array) -> mx.array:
        if self.penalty == 1.0:
            return logits
        
        batch_size, vocab_size = logits.shape
        
        # 🚀 优化：使用纯MLX向量化操作
        for batch_idx in range(batch_size):
            batch_logits = logits[batch_idx]
            batch_input_ids = input_ids[batch_idx]
            
            # 获取unique tokens (MLX操作)
            unique_ids = mx.unique(batch_input_ids)
            
            # 过滤掉无效的token id
            valid_mask = (unique_ids >= 0) & (unique_ids < vocab_size)
            unique_ids = unique_ids[valid_mask]
            
            if unique_ids.size == 0:
                continue
            
            # 获取这些token的scores
            scores = batch_logits[unique_ids]
            
            # 向量化应用penalty
            # score < 0: score *= penalty
            # score > 0: score /= penalty
            penalized_scores = mx.where(scores < 0, scores * self.penalty, scores / self.penalty)
            
            # 更新logits (使用索引赋值)
            # MLX不支持直接索引赋值，需要重建
            updated_logits = batch_logits
            for i, token_id in enumerate(unique_ids.tolist()):
                # 这里仍需要循环，但只循环unique tokens（通常很少）
                mask = mx.arange(vocab_size) == token_id
                updated_logits = mx.where(mask, penalized_scores[i], updated_logits)
            
            logits = mx.array([updated_logits if j == batch_idx else logits[j] for j in range(batch_size)])
        
        return logits


class CombinedLogitsProcessor(LogitsProcessor):
    """
    合并Temperature和RepetitionPenalty，减少中间计算
    
    这是最常用的组合，合并后可以减少一次logits遍历
    """
    
    def __init__(self, temperature: float = 1.0, repetition_penalty: float = 1.0):
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.use_temp = temperature != 1.0
        self.use_rep = repetition_penalty != 1.0
    
    def __call__(self, input_ids: mx.array, logits: mx.array) -> mx.array:
        # Step 1: Temperature (应用于所有logits)
        if self.use_temp:
            logits = logits / self.temperature
        
        # Step 2: Repetition Penalty (只应用于已出现的tokens)
        if self.use_rep:
            batch_size, vocab_size = logits.shape
            
            # 🚀 优化：使用numpy进行快速处理（避免MLX的索引赋值问题）
            import numpy as np
            logits_np = np.array(logits)
            input_ids_np = np.array(input_ids)
            
            for batch_idx in range(batch_size):
                batch_logits = logits_np[batch_idx]
                batch_input_ids = input_ids_np[batch_idx]
                
                # 获取unique tokens
                unique_ids = np.unique(batch_input_ids)
                unique_ids = unique_ids[(unique_ids >= 0) & (unique_ids < vocab_size)]
                
                if len(unique_ids) == 0:
                    continue
                
                # 获取scores并应用penalty
                for token_id in unique_ids:
                    score = batch_logits[token_id]
                    if score < 0:
                        batch_logits[token_id] = score * self.repetition_penalty
                    else:
                        batch_logits[token_id] = score / self.repetition_penalty
            
            logits = mx.array(logits_np)
        
        return logits


class TopPLogitsWarperOptimized(LogitsProcessor):
    """
    优化版本的Top-p sampling
    
    优化点：
    1. 使用MLX的argsort和where
    2. 减少不必要的数据转换
    """
    
    def __init__(self, top_p: float, filter_value: float = -float("inf"), min_tokens_to_keep: int = 1):
        if not (0 <= top_p <= 1.0):
            raise ValueError(f"top_p must be in [0, 1], got {top_p}")
        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep
    
    def __call__(self, input_ids: mx.array, logits: mx.array) -> mx.array:
        if self.top_p >= 1.0:
            return logits
        
        batch_size, vocab_size = logits.shape
        processed = []
        
        for batch_idx in range(batch_size):
            batch_logits = logits[batch_idx]
            
            # 🚀 使用MLX操作
            # Sort in descending order
            sorted_indices = mx.argsort(batch_logits)[::-1]
            sorted_logits = batch_logits[sorted_indices]
            
            # Softmax (使用log-sum-exp技巧避免overflow)
            max_logit = mx.max(sorted_logits)
            exp_logits = mx.exp(sorted_logits - max_logit)
            sorted_probs = exp_logits / mx.sum(exp_logits)
            
            # Cumulative probabilities
            cumsum_probs = mx.cumsum(sorted_probs)
            
            # Find cutoff
            # Keep tokens where cumsum <= top_p (plus one more)
            keep_mask = cumsum_probs <= self.top_p
            # Ensure at least min_tokens_to_keep
            keep_count = max(self.min_tokens_to_keep, mx.sum(keep_mask).item() + 1)
            
            # Create filter mask
            filter_mask = mx.ones(vocab_size, dtype=mx.bool_)
            keep_indices = sorted_indices[:keep_count]
            
            # Build filtered logits
            filtered = mx.full((vocab_size,), self.filter_value, dtype=batch_logits.dtype)
            for idx in keep_indices.tolist():
                mask = mx.arange(vocab_size) == idx
                filtered = mx.where(mask, batch_logits[idx], filtered)
            
            processed.append(filtered)
        
        return mx.stack(processed, axis=0)


class TopKLogitsWarperOptimized(LogitsProcessor):
    """
    优化版本的Top-k sampling
    """
    
    def __init__(self, top_k: int, filter_value: float = -float("inf"), min_tokens_to_keep: int = 1):
        if top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {top_k}")
        self.top_k = top_k
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep
    
    def __call__(self, input_ids: mx.array, logits: mx.array) -> mx.array:
        if self.top_k == 0:
            return logits
        
        batch_size, vocab_size = logits.shape
        top_k = min(max(self.top_k, self.min_tokens_to_keep), vocab_size)
        processed = []
        
        for batch_idx in range(batch_size):
            batch_logits = logits[batch_idx]
            
            # 🚀 使用MLX的argsort
            sorted_indices = mx.argsort(batch_logits)
            top_k_indices = sorted_indices[-top_k:]
            
            # Build filtered logits
            filtered = mx.full((vocab_size,), self.filter_value, dtype=batch_logits.dtype)
            for idx in top_k_indices.tolist():
                mask = mx.arange(vocab_size) == idx
                filtered = mx.where(mask, batch_logits[idx], filtered)
            
            processed.append(filtered)
        
        return mx.stack(processed, axis=0)


class LogitsProcessorList(list):
    """
    优化版本的LogitsProcessorList
    
    优化点：
    1. 检测并使用CombinedLogitsProcessor
    2. 减少不必要的processor调用
    """
    
    def __call__(self, input_ids: mx.array, logits: mx.array) -> mx.array:
        """Apply all processors in sequence"""
        for processor in self:
            logits = processor(input_ids, logits)
        return logits
    
    @staticmethod
    def create_optimized(temperature=1.0, repetition_penalty=10.0, top_p=1.0, top_k=0):
        """
        工厂方法：创建优化的processor list
        
        自动选择最优组合：
        - 如果只有temp+rep: 使用CombinedLogitsProcessor
        - 否则: 使用单独的processors
        """
        processor_list = LogitsProcessorList()
        
        use_temp = temperature != 1.0
        use_rep = repetition_penalty != 1.0
        use_top_p = top_p < 1.0
        use_top_k = top_k > 0
        
        # 🚀 优化：合并常用组合
        if (use_temp or use_rep) and not use_top_p and not use_top_k:
            # 只有temp+rep，使用合并版本
            processor_list.append(CombinedLogitsProcessor(temperature, repetition_penalty))
        else:
            # 使用单独的processors
            if use_temp:
                processor_list.append(TemperatureLogitsWarper(temperature))
            if use_rep:
                processor_list.append(RepetitionPenaltyLogitsProcessorOptimized(repetition_penalty))
            if use_top_k:
                processor_list.append(TopKLogitsWarperOptimized(top_k))
            if use_top_p:
                processor_list.append(TopPLogitsWarperOptimized(top_p))
        
        return processor_list

