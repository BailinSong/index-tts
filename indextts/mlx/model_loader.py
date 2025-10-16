"""
MLX Model Loader - 统一管理所有 MLX 模型加载

职责:
1. 加载所有 MLX 模型（GPT, S2MEL, BigVGAN等）
2. 管理 MLX 缓存
3. 处理加载失败和降级
"""

from typing import Any, Optional, Dict
import os
import torch


class MLXModelLoader:
    """
    MLX 模型加载器
    
    通过此类统一管理所有 MLX 模型的加载逻辑，
    使得主类（IndexTTS2）的代码保持简洁。
    """
    
    def __init__(self, model_dir: str, config: Any, device: str = "mps"):
        """
        初始化 MLX 模型加载器
        
        Args:
            model_dir: 模型目录路径
            config: 配置对象
            device: 设备（默认 mps）
        """
        self.model_dir = model_dir
        self.config = config
        self.device = device
        self._init_mlx_cache()
        
        # 记录已加载的模型
        self.loaded_models = {}
    
    def _init_mlx_cache(self):
        """初始化 MLX 缓存"""
        from indextts.utils.mlx.cache import MLXModelCache
        
        cache_dir = os.path.join(self.model_dir, "mlx")
        self.mlx_cache = MLXModelCache(cache_dir=cache_dir)
        print(f">> [MLX Loader] Cache Directory: {cache_dir}")
    
    def load_gpt(self, checkpoint_path: str) -> tuple:
        """
        加载 MLX GPT 模型
        
        Args:
            checkpoint_path: GPT checkpoint 路径
            
        Returns:
            (model, is_mlx): 模型实例和是否为 MLX 标志
        """
        print(">> [MLX Loader] Loading GPT model...")
        
        try:
            from indextts.gpt.mlx.model import UnifiedVoiceMLX
            
            # 获取或转换权重
            mlx_gpt_weights = self.mlx_cache.get_or_convert("gpt", checkpoint_path)
            
            # 创建 MLX 模型
            model = UnifiedVoiceMLX(
                use_mlx_conditioning=True,
                **self.config.gpt
            )
            
            # 加载权重
            loaded_count = model.load_weights_from_dict(mlx_gpt_weights)
            print(f">> [MLX Loader] Loaded {loaded_count} GPT weights")
            
            # JIT 预热
            print(">> [MLX Loader] JIT Warmup: Pre-compiling Metal kernels...")
            import time
            start = time.time()
            model.jit_warmup()
            elapsed = time.time() - start
            print(f">> [MLX Loader] JIT Warmup completed in {elapsed:.2f}s")
            
            self.loaded_models['gpt'] = model
            print(">> [MLX Loader] ✓ GPT model loaded successfully")
            
            return model, True
            
        except Exception as e:
            print(f">> [MLX Loader] Failed to load GPT: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def load_s2mel(self, checkpoint_path: str) -> tuple:
        """
        加载 MLX S2MEL 模型
        
        Args:
            checkpoint_path: S2MEL checkpoint 路径
            
        Returns:
            (model, has_mlx_modules): S2MEL 模型和 MLX 模块标志
        """
        print(">> [MLX Loader] Loading S2MEL model...")
        
        try:
            from indextts.s2mel.models import S2Mel
            
            # 获取或转换 S2MEL 权重
            mlx_s2mel_weights = self.mlx_cache.get_or_convert("s2mel", checkpoint_path)
            
            # 创建 S2MEL 模型
            model = S2Mel(**self.config.s2mel)
            
            # 创建 MLX 模块
            from indextts.s2mel.mlx_modules.gpt_layer import MLXGPTLayer
            from indextts.s2mel.mlx_modules.length_regulator import MLXLengthRegulator
            
            # 加载 MLX GPT Layer
            model.mlx_gpt_layer = MLXGPTLayer(self.config.s2mel)
            model.mlx_gpt_layer.load_weights(mlx_s2mel_weights.get('gpt_layer', {}))
            print(">> [MLX Loader] ✓ MLX GPT Layer loaded")
            
            # 加载 MLX Length Regulator
            model.mlx_length_regulator = MLXLengthRegulator(self.config.s2mel)
            model.mlx_length_regulator.load_weights(mlx_s2mel_weights.get('length_regulator', {}))
            print(">> [MLX Loader] ✓ MLX Length Regulator loaded")
            
            # 加载其他 PyTorch 组件
            from indextts.utils.checkpoint import load_checkpoint
            load_checkpoint(model, checkpoint_path)
            model = model.to(self.device)
            model.eval()
            
            self.loaded_models['s2mel'] = model
            print(">> [MLX Loader] ✓ S2MEL model loaded successfully")
            
            return model, True
            
        except Exception as e:
            print(f">> [MLX Loader] Failed to load S2MEL: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def load_vocoder(self, vocoder_name: str = "nvidia/bigvgan_v2_22khz_80band_256x") -> Any:
        """
        加载 MLX BigVGAN 声码器
        
        Args:
            vocoder_name: 声码器名称
            
        Returns:
            BigVGAN 模型实例
        """
        print(">> [MLX Loader] Loading BigVGAN vocoder...")
        
        try:
            from indextts.s2mel.modules.bigvgan.bigvgan import BigVGAN
            
            # 加载 BigVGAN
            print(f"Loading weights from {vocoder_name}")
            model = BigVGAN.from_pretrained(
                vocoder_name,
                use_cuda_kernel=False  # MLX 模式不使用 CUDA
            )
            model = model.to(self.device)
            model.eval()
            
            # 缓存模型
            model.cache_model()
            print(">> [MLX Loader] BigVGAN already cached")
            
            self.loaded_models['vocoder'] = model
            print(">> [MLX Loader] ✓ BigVGAN loaded successfully")
            
            return model
            
        except Exception as e:
            print(f">> [MLX Loader] Failed to load BigVGAN: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_loaded_model(self, model_name: str) -> Optional[Any]:
        """
        获取已加载的模型
        
        Args:
            model_name: 模型名称 (gpt, s2mel, vocoder)
            
        Returns:
            模型实例或 None
        """
        return self.loaded_models.get(model_name)
    
    def unload_model(self, model_name: str):
        """
        卸载模型释放内存
        
        Args:
            model_name: 模型名称
        """
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            import gc
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            print(f">> [MLX Loader] Model '{model_name}' unloaded")
    
    def unload_all(self):
        """卸载所有模型"""
        model_names = list(self.loaded_models.keys())
        for name in model_names:
            self.unload_model(name)
        print(">> [MLX Loader] All models unloaded")

