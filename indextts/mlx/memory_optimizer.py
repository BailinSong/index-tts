"""
Memory Optimizer - 统一管理所有内存优化策略

职责:
1. Semantic Model 按需加载/卸载
2. Qwen Emotion 延迟加载
3. 内存清理和监控
"""

import gc
import torch
from typing import Any, Optional


class MemoryOptimizer:
    """
    内存优化管理器
    
    通过延迟加载和按需卸载策略，最大化降低内存占用
    """
    
    def __init__(self):
        self.semantic_wrapper = None
        self.qwen_wrapper = None
        print(">> [Memory Optimizer] Initialized")
    
    def optimize_semantic_model(self, tts_instance):
        """
        优化 Semantic Model 的内存使用
        
        策略: 按需加载/卸载
        节省: ~1.0GB
        
        Args:
            tts_instance: IndexTTS2 或 IndexTTS2MLX 实例
        """
        print(">> [Memory Opt] Applying Semantic Model optimization...")
        
        # 保存原始的提取器和模型路径
        original_extract_features = tts_instance.extract_features
        model_path = "facebook/w2v-bert-2.0"
        stat_path = tts_instance.semantic_stat_path if hasattr(tts_instance, 'semantic_stat_path') else None
        device = tts_instance.device
        
        # 如果模型已加载，先卸载
        if hasattr(tts_instance, 'semantic_model') and tts_instance.semantic_model is not None:
            print(">> [Memory Opt] Unloading existing Semantic Model...")
            del tts_instance.semantic_model
            if hasattr(tts_instance, 'semantic_mean'):
                del tts_instance.semantic_mean
            if hasattr(tts_instance, 'semantic_std'):
                del tts_instance.semantic_std
            gc.collect()
            torch.mps.empty_cache()
        
        # 创建延迟加载包装器
        self.semantic_wrapper = LazySemanticModel(
            model_path=model_path,
            stat_path=stat_path,
            device=device,
            extract_features=original_extract_features
        )
        
        # 替换实例的方法
        tts_instance.semantic_model = None
        tts_instance.semantic_mean = None
        tts_instance.semantic_std = None
        tts_instance._ensure_semantic_loaded = self.semantic_wrapper.ensure_loaded
        tts_instance._unload_semantic = self.semantic_wrapper.unload
        tts_instance.get_emb = self.semantic_wrapper.get_emb
        
        print(">> [Memory Opt] ✓ Semantic Model: Lazy loading enabled (saves ~1.0GB)")
    
    def optimize_qwen_emotion(self, tts_instance):
        """
        优化 Qwen Emotion 的内存使用
        
        策略: 延迟加载
        节省: ~1.2GB
        
        Args:
            tts_instance: IndexTTS2 或 IndexTTS2MLX 实例
        """
        print(">> [Memory Opt] Applying Qwen Emotion optimization...")
        
        # 保存模型路径
        if hasattr(tts_instance, 'qwen_emo_path'):
            qwen_path = tts_instance.qwen_emo_path
        else:
            import os
            qwen_path = os.path.join(tts_instance.model_dir, tts_instance.cfg.qwen_emo_path)
        
        # 如果模型已加载，先卸载
        if hasattr(tts_instance, 'qwen_emo') and tts_instance.qwen_emo is not None:
            print(">> [Memory Opt] Unloading existing Qwen Emotion...")
            del tts_instance.qwen_emo
            gc.collect()
            torch.mps.empty_cache()
        
        # 创建延迟加载包装器
        self.qwen_wrapper = LazyQwenEmotion(qwen_path)
        
        # 替换实例的属性和方法
        tts_instance.qwen_emo = None
        tts_instance.qwen_emo_path = qwen_path
        tts_instance._ensure_qwen_loaded = self.qwen_wrapper.ensure_loaded
        
        print(">> [Memory Opt] ✓ Qwen Emotion: Lazy loading enabled (saves ~1.2GB)")
    
    def get_memory_stats(self) -> dict:
        """
        获取内存使用统计
        
        Returns:
            内存统计字典
        """
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_gb': memory_info.rss / 1024**3,
            'semantic_loaded': self.semantic_wrapper.loaded if self.semantic_wrapper else False,
            'qwen_loaded': self.qwen_wrapper.loaded if self.qwen_wrapper else False,
        }


class LazySemanticModel:
    """
    延迟加载的 Semantic Model 包装器
    
    自动管理模型的加载和卸载
    """
    
    def __init__(self, model_path: str, stat_path: Optional[str], device: str, extract_features):
        self.model_path = model_path
        self.stat_path = stat_path
        self.device = device
        self.extract_features = extract_features
        
        self.model = None
        self.mean = None
        self.std = None
        self.loaded = False
    
    def ensure_loaded(self):
        """确保模型已加载"""
        if not self.loaded:
            self._load()
    
    def _load(self):
        """加载模型"""
        print(">> Loading Semantic Model (W2V-BERT) for feature extraction...")
        
        from transformers import Wav2Vec2BertModel
        
        self.model = Wav2Vec2BertModel.from_pretrained(self.model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 加载统计数据
        if self.stat_path:
            import numpy as np
            stat = np.load(self.stat_path)
            self.mean = torch.from_numpy(stat['mean']).to(self.device)
            self.std = torch.from_numpy(stat['std']).to(self.device)
        
        self.loaded = True
        print(">> Semantic Model loaded (~1.0GB)")
    
    def unload(self):
        """卸载模型，释放内存"""
        if self.loaded:
            del self.model
            del self.mean
            del self.std
            self.model = None
            self.mean = None
            self.std = None
            gc.collect()
            torch.mps.empty_cache()
            self.loaded = False
            print(">> Semantic Model unloaded (~1.0GB freed)")
    
    def get_emb(self, input_features, attention_mask):
        """
        提取语义嵌入
        
        自动加载模型（如果未加载）
        """
        self.ensure_loaded()
        
        with torch.no_grad():
            vq_emb = self.model(
                input_features=input_features,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            feat = vq_emb.hidden_states[17]  # (B, T, C)
            
            # 归一化
            if self.mean is not None and self.std is not None:
                feat = (feat - self.mean) / self.std
            
            return feat
    
    def __enter__(self):
        """支持 context manager"""
        self.ensure_loaded()
        return self
    
    def __exit__(self, *args):
        self.unload()


class LazyQwenEmotion:
    """
    延迟加载的 Qwen Emotion 包装器
    
    只在需要时加载模型
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.loaded = False
    
    def ensure_loaded(self):
        """确保模型已加载"""
        if not self.loaded:
            self._load()
    
    def _load(self):
        """加载模型"""
        print(">> Loading Qwen Emotion model (first use)...")
        
        from indextts.qwen_emo.qwen_emotion import QwenEmotion
        
        self.model = QwenEmotion(self.model_path)
        self.loaded = True
        print(">> Qwen Emotion loaded (~1.2GB)")
    
    def inference(self, text: str) -> dict:
        """
        推理情感
        
        自动加载模型（如果未加载）
        """
        self.ensure_loaded()
        return self.model.inference(text)
    
    def unload(self):
        """卸载模型"""
        if self.loaded:
            del self.model
            self.model = None
            gc.collect()
            torch.mps.empty_cache()
            self.loaded = False
            print(">> Qwen Emotion unloaded (~1.2GB freed)")

