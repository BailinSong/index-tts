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
        
        完全替换父类已加载的 semantic_model
        """
        print(">> [Memory Opt] Applying Semantic Model optimization...")
        
        # 获取必要信息
        stat_path = os.path.join(tts_instance.model_dir, tts_instance.cfg.w2v_stat)
        device = tts_instance.device
        
        # 删除父类加载的 Semantic Model
        if hasattr(tts_instance, 'semantic_model') and tts_instance.semantic_model is not None:
            del tts_instance.semantic_model
            del tts_instance.semantic_mean
            del tts_instance.semantic_std
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            print(">> [Memory Opt] Existing Semantic Model removed")
        
        # 创建延迟加载包装器
        self.semantic_wrapper = LazySemanticModel(
            model_path="facebook/w2v-bert-2.0",
            stat_path=stat_path,
            device=device
        )
        
        # 替换实例的属性和方法
        tts_instance.semantic_model = self.semantic_wrapper
        tts_instance.semantic_mean = None
        tts_instance.semantic_std = None
        tts_instance.semantic_model_loaded = False  # 兼容性标志
        tts_instance._ensure_semantic_loaded = self.semantic_wrapper.ensure_loaded
        tts_instance._unload_semantic = self.semantic_wrapper.unload
        
        # 重写 get_emb 方法
        original_get_emb = tts_instance.get_emb
        def get_emb_with_lazy_loading(input_features, attention_mask):
            return self.semantic_wrapper.get_emb(input_features, attention_mask)
        tts_instance.get_emb = get_emb_with_lazy_loading
        
        print(">> [Memory Opt] ✓ Semantic Model: Lazy loading enabled (saves ~1.0GB)")
    
    def optimize_qwen_emotion(self, tts_instance):
        """
        优化 Qwen Emotion 的内存使用
        
        策略: 延迟加载
        节省: ~1.2GB
        
        完全替换父类已加载的 qwen_emo
        """
        print(">> [Memory Opt] Applying Qwen Emotion optimization...")
        
        # 获取路径
        qwen_path = os.path.join(tts_instance.model_dir, tts_instance.cfg.qwen_emo_path)
        
        # 删除父类加载的 Qwen Emotion
        if hasattr(tts_instance, 'qwen_emo') and tts_instance.qwen_emo is not None:
            del tts_instance.qwen_emo
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            print(">> [Memory Opt] Existing Qwen Emotion removed")
        
        # 创建延迟加载包装器
        self.qwen_wrapper = LazyQwenEmotion(qwen_path)
        
        # 替换实例的属性
        tts_instance.qwen_emo = self.qwen_wrapper
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
    
    完全替换 IndexTTS2 的 semantic_model,实现按需加载/卸载
    """
    
    def __init__(self, model_path: str, stat_path: str, device: str):
        self.model_path = model_path
        self.stat_path = stat_path
        self.device = device
        
        self.model = None
        self.mean = None
        self.std = None
        self.loaded = False
    
    def ensure_loaded(self):
        """延迟加载 Semantic Model（仅在提取特征时加载）"""
        if not self.loaded:
            print(">> Loading Semantic Model (W2V-BERT) for feature extraction...")
            from indextts.utils.maskgct_utils import build_semantic_model
            
            self.model, self.mean, self.std = build_semantic_model(self.stat_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.mean = self.mean.to(self.device)
            self.std = self.std.to(self.device)
            self.loaded = True
            print(">> Semantic Model loaded (~1.0GB)")
    
    def unload(self):
        """卸载 Semantic Model，释放内存"""
        if self.loaded:
            print(">> Unloading Semantic Model...")
            del self.model
            del self.mean
            del self.std
            self.model = None
            self.mean = None
            self.std = None
            self.loaded = False
            
            # 清理内存
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            print(">> Semantic Model unloaded (~1.0GB freed)")
    
    def get_emb(self, input_features, attention_mask):
        """
        提取语义嵌入
        
        这个方法会被父类的 get_emb 调用
        """
        # 确保模型已加载
        self.ensure_loaded()
        
        # 调用实际模型
        with torch.no_grad():
            vq_emb = self.model(
                input_features=input_features,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            feat = vq_emb.hidden_states[17]  # (B, T, C)
            
            # Normalize
            feat = (feat - self.mean) / self.std
            
            return feat
    
    def __call__(self, *args, **kwargs):
        """支持直接调用"""
        self.ensure_loaded()
        return self.model(*args, **kwargs)
    
    def to(self, device):
        """支持 .to() 调用（兼容性）"""
        self.device = device
        if self.loaded:
            self.model = self.model.to(device)
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        return self
    
    def eval(self):
        """支持 .eval() 调用（兼容性）"""
        if self.loaded:
            self.model.eval()
        return self


class LazyQwenEmotion:
    """
    延迟加载的 Qwen Emotion 包装器
    
    完全替换 IndexTTS2 的 qwen_emo,实现延迟加载
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.loaded = False
    
    def ensure_loaded(self):
        """延迟加载 Qwen Emotion 模型（仅在需要时加载）"""
        if not self.loaded:
            print(">> Loading Qwen Emotion model (first use)...")
            from indextts.qwen_emo.qwen_emotion import QwenEmotion
            self.model = QwenEmotion(self.model_path)
            self.loaded = True
            print(">> Qwen Emotion loaded (~1.2GB)")
    
    def inference(self, text: str) -> dict:
        """
        推理情感
        
        自动加载模型（如果未加载）
        这个方法会被父类调用
        """
        self.ensure_loaded()
        return self.model.inference(text)
    
    def unload(self):
        """卸载模型"""
        if self.loaded:
            del self.model
            self.model = None
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            self.loaded = False
            print(">> Qwen Emotion unloaded (~1.2GB freed)")

