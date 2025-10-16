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
        
        # 检查 MLX 是否可用
        from indextts.utils.mlx.utils import check_mlx_available
        if not check_mlx_available():
            raise RuntimeError("MLX not available on this system")
        
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
    
    def load_s2mel_mlx_modules(self) -> tuple:
        """
        加载 S2MEL 的 MLX 优化模块
        
        Returns:
            (mlx_gpt_layer, mlx_length_regulator): MLX 模块或 None
        """
        print("\n>> [Model 2/4] Loading S2MEL with MLX optimization...")
        
        try:
            s2mel_path = os.path.join(self.model_dir, self.config.s2mel_checkpoint)
            mlx_s2mel_weights = self.mlx_cache.get_or_convert("s2mel", s2mel_path)
            print(">> MLX S2MEL weights ready")
            
            from indextts.s2mel.mlx_modules import MLXGPTLayer, MLXInterpolateRegulator
            
            # MLX GPT Layer
            print(">> Creating MLX GPT Layer...")
            mlx_gpt_layer = MLXGPTLayer()
            
            # 加载 weights
            prefix = 'gpt_layer.'
            if f'{prefix}0.weight' in mlx_s2mel_weights:
                mlx_gpt_layer.layer1.weight = mlx_s2mel_weights[f'{prefix}0.weight']
                mlx_gpt_layer.layer1.bias = mlx_s2mel_weights[f'{prefix}0.bias']
                mlx_gpt_layer.layer2.weight = mlx_s2mel_weights[f'{prefix}1.weight']
                mlx_gpt_layer.layer2.bias = mlx_s2mel_weights[f'{prefix}1.bias']
                mlx_gpt_layer.layer3.weight = mlx_s2mel_weights[f'{prefix}2.weight']
                mlx_gpt_layer.layer3.bias = mlx_s2mel_weights[f'{prefix}2.bias']
                print("   ✓ MLX GPT Layer weights loaded")
            else:
                print("   ⚠️  GPT Layer weights not found, skipping")
                mlx_gpt_layer = None
            
            # MLX Length Regulator
            print(">> Creating MLX Length Regulator...")
            mlx_length_regulator = MLXInterpolateRegulator(
                channels=self.config.s2mel.length_regulator.channels,
                sampling_ratios=self.config.s2mel.length_regulator.sampling_ratios,
                is_discrete=self.config.s2mel.length_regulator.is_discrete,
                in_channels=getattr(self.config.s2mel.length_regulator, "in_channels", None),
                vector_quantize=getattr(self.config.s2mel.length_regulator, "vector_quantize", False),
                codebook_size=self.config.s2mel.length_regulator.content_codebook_size,
                n_codebooks=getattr(self.config.s2mel.length_regulator, "n_codebooks", 1),
                f0_condition=getattr(self.config.s2mel.length_regulator, "f0_condition", False),
                n_f0_bins=getattr(self.config.s2mel.length_regulator, "n_f0_bins", 512),
            )
            
            # 加载 Length Regulator weights
            lr_prefix = 'length_regulator.'
            lr_weights_found = False
            if f'{lr_prefix}content_in_proj.weight' in mlx_s2mel_weights:
                mlx_length_regulator.content_in_proj.weight = mlx_s2mel_weights[f'{lr_prefix}content_in_proj.weight']
                mlx_length_regulator.content_in_proj.bias = mlx_s2mel_weights[f'{lr_prefix}content_in_proj.bias']
                lr_weights_found = True
            
            # 加载 model layers
            layer_idx = 0
            while f'{lr_prefix}model.{layer_idx}.weight' in mlx_s2mel_weights:
                if layer_idx < len(mlx_length_regulator.model):
                    mlx_layer = mlx_length_regulator.model[layer_idx]
                    if hasattr(mlx_layer, 'weight'):
                        mlx_layer.weight = mlx_s2mel_weights[f'{lr_prefix}model.{layer_idx}.weight']
                        if f'{lr_prefix}model.{layer_idx}.bias' in mlx_s2mel_weights:
                            mlx_layer.bias = mlx_s2mel_weights[f'{lr_prefix}model.{layer_idx}.bias']
                        lr_weights_found = True
                layer_idx += 1
            
            if lr_weights_found:
                print("   ✓ MLX Length Regulator weights loaded")
            else:
                print("   ⚠️  Length Regulator weights not found, skipping")
                mlx_length_regulator = None
            
            if mlx_gpt_layer or mlx_length_regulator:
                print(">> ✓ S2MEL MLX modules ready")
            
            return mlx_gpt_layer, mlx_length_regulator
            
        except Exception as e:
            print(f">> [MLX Loader] S2MEL MLX modules creation failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def cache_bigvgan(self, bigvgan_model) -> None:
        """
        缓存 BigVGAN 权重到 MLX 格式
        
        Args:
            bigvgan_model: 已加载的 BigVGAN 模型
        """
        print("\n>> [Model 3/4] Loading BigVGAN with MLX optimization...")
        
        try:
            if not self.mlx_cache.is_cached("bigvgan"):
                print(">> Caching BigVGAN weights in MLX format...")
                self.mlx_cache.convert_and_cache("bigvgan", state_dict=bigvgan_model.state_dict())
            else:
                print(">> BigVGAN already cached")
            
            print(">> BigVGAN: Running on MPS with MLX optimizations")
        except Exception as e:
            print(f">> BigVGAN caching skipped: {e}")
    
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

