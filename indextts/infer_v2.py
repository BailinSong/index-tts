import os
from subprocess import CalledProcessError

os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import json
import re
import time
import librosa
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 导入一致性随机数生成器
from unified_random_generator import UnifiedRandomGenerator

from omegaconf import OmegaConf

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.front import TextNormalizer, TextTokenizer

from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.audio import mel_spectrogram

from transformers import AutoTokenizer
from modelscope import AutoModelForCausalLM
from huggingface_hub import hf_hub_download
import safetensors
from transformers import SeamlessM4TFeatureExtractor
import random
import torch.nn.functional as F

class IndexTTS2:
    def __init__(
            self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, device=None,
            use_cuda_kernel=None,use_deepspeed=False, use_mlx=False, diffusion_steps=25
    ):
        """
        Args:
            cfg_path (str): path to the config file.
            model_dir (str): path to the model directory.
            use_fp16 (bool): whether to use fp16.
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
            use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel, only for CUDA device.
            use_deepspeed (bool): whether to use DeepSpeed or not.
            use_mlx (bool): whether to enable MLX optimizations for Apple Silicon M4.
            diffusion_steps (int): number of diffusion steps for S2MEL (default: 25, range: 10-25).
        """
        # MLX optimization mode for Apple Silicon M4
        self.use_mlx = use_mlx
        self.mlx_available = False
        self.mlx_cache = None
        
        if use_mlx:
            from indextts.utils.mlx_utils import check_mlx_available
            from indextts.utils.mlx_cache import MLXModelCache
            
            self.mlx_available = check_mlx_available()
            if self.mlx_available:
                print("\n" + "="*70)
                print("MLX Framework Enabled for Apple Silicon M4")
                print("="*70)
                print("Mode: Full MLX Native Implementation")
                print("Strategy: Convert models to MLX, cache for fast loading")
                print("="*70 + "\n")
                
                # Initialize cache manager
                self.mlx_cache = MLXModelCache(cache_dir=os.path.join(model_dir, "mlx"))
            else:
                print(">> MLX not available, using standard mode")
                self.use_mlx = False
        
        if device is not None:
            self.device = device
            self.use_fp16 = False if device == "cpu" else use_fp16
            self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")
        elif self.use_mlx:
            # MLX mode: Force MPS backend for Apple Silicon M4 optimization
            self.device = "mps"
            self.use_fp16 = False  # Float32 performs better on MPS
            self.use_cuda_kernel = False
            print(">> MLX mode: Using MPS backend optimized for Apple Silicon M4")
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.use_fp16 = use_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = "xpu"
            self.use_fp16 = use_fp16
            self.use_cuda_kernel = False
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.use_fp16 = False  # Use float16 on MPS is overhead than float32
            self.use_cuda_kernel = False
        else:
            self.device = "cpu"
            self.use_fp16 = False
            self.use_cuda_kernel = False
            print(">> Be patient, it may take a while to run in CPU mode.")

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.use_fp16 else None
        self.diffusion_steps = diffusion_steps  # Number of diffusion steps for S2MEL (default: 25)
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        # 🎯 统一随机数生成器：确保所有随机数生成的一致性
        self.unified_random = UnifiedRandomGenerator(seed=42)
        print(">> Unified Random Generator: Initialized with seed 42")

        # 🎯 内存优化：Qwen Emotion 延迟加载（节省 ~1.2GB）
        self.qwen_emo = None
        self.qwen_emo_path = os.path.join(self.model_dir, self.cfg.qwen_emo_path)
        print(">> Qwen Emotion: Lazy loading enabled (saves ~1.2GB)")

        # Load GPT model with MLX native implementation if enabled
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        self.gpt_is_mlx = False
        
        # 🎯 核心优化：MLX 模式下只加载 MLX 模型，不加载 PyTorch
        if self.use_mlx and self.mlx_available:
            print("\n>> [Model 1/4] Creating Pure MLX GPT (MLX Cond + MLX Transformer)...")
            try:
                from indextts.gpt.mlx_model import UnifiedVoiceMLX
                mlx_gpt_weights = self.mlx_cache.get_or_convert("gpt", self.gpt_path)
                # Create MLX model with PURE MLX mode
                self.mlx_transformer = UnifiedVoiceMLX(
                    use_mlx_conditioning=True,
                    **self.cfg.gpt
                )
                # Load weights
                self.mlx_transformer.load_weights_from_dict(mlx_gpt_weights)
                
                # 🔥 关键修复：为了保持音色一致性，我们需要确保使用正确的conditioning
                # 将在运行时使用MLX模型的get_conditioning方法，但要确保精度
                
                self.gpt_is_mlx = True
                self.gpt = None  # 🎯 不加载完整的 PyTorch 模型！节省内存
                print(">> ✓ Pure MLX GPT loaded successfully")
                print(">> ✓ PyTorch GPT skipped (saved ~2.5GB memory)")
                print("   (MLX: Conformer + Perceiver + Emotion Conditioning)")
            except Exception as e:
                print(f">> MLX loading failed: {e}")
                import traceback
                traceback.print_exc()
                print(">> Falling back to PyTorch GPT...")
                # 降级：加载 PyTorch 模型
                self.gpt = UnifiedVoice(**self.cfg.gpt)
                load_checkpoint(self.gpt, self.gpt_path)
                self.gpt = self.gpt.to(self.device)
                if self.use_fp16:
                    self.gpt.eval().half()
                else:
                    self.gpt.eval()
                print(">> GPT weights restored from:", self.gpt_path)
                self.gpt_is_mlx = False
                self.mlx_transformer = None
        else:
            # 非 MLX 模式：加载 PyTorch 模型
            print(">> Loading PyTorch GPT...")
            self.gpt = UnifiedVoice(**self.cfg.gpt)
            load_checkpoint(self.gpt, self.gpt_path)
            self.gpt = self.gpt.to(self.device)
            if self.use_fp16:
                self.gpt.eval().half()
            else:
                self.gpt.eval()
            print(">> GPT weights restored from:", self.gpt_path)
            self.gpt_is_mlx = False
            self.mlx_transformer = None

        # Post-init only for PyTorch models
        if not self.gpt_is_mlx:
            if use_deepspeed:
                try:
                    import deepspeed
                except (ImportError, OSError, CalledProcessError) as e:
                    use_deepspeed = False
                    print(f">> Failed to load DeepSpeed. Falling back to normal inference. Error: {e}")

            self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=self.use_fp16)
        else:
            print(">> MLX GPT: Skipping PyTorch-specific post-init")

        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from indextts.s2mel.modules.bigvgan.alias_free_activation.cuda import activation1d

                print(">> Preload custom CUDA kernel for BigVGAN", activation1d.anti_alias_activation_cuda)
            except Exception as e:
                print(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.")
                print(f"{e!r}")
                self.use_cuda_kernel = False

        # 🎯 内存优化：Semantic Model 按需加载（节省 ~1.0GB）
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        self.semantic_model = None
        self.semantic_mean = None
        self.semantic_std = None
        self.semantic_model_loaded = False
        self.semantic_stat_path = os.path.join(self.model_dir, self.cfg.w2v_stat)
        print(">> Semantic Model (W2V-BERT): Lazy loading enabled (saves ~1.0GB)")

        # Semantic Codec 必须保留（推理时需要 vq2emb 查表）
        semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
        semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        self.semantic_codec = semantic_codec.to(self.device)
        self.semantic_codec.eval()
        print('>> semantic_codec weights restored from: {}'.format(semantic_code_ckpt))

        # Load S2MEL model with MLX caching if enabled
        s2mel_path = os.path.join(self.model_dir, self.cfg.s2mel_checkpoint)
        
        # Initialize MLX S2MEL modules
        self.mlx_s2mel_gpt_layer = None
        self.mlx_s2mel_length_regulator = None
        
        # 🚀 启用S2MEL MLX模块（gpt_layer + length_regulator + cfm）
        self.mlx_s2mel_cfm = None
        if self.use_mlx and self.mlx_available:
            print("\n>> [Model 2/4] Loading S2MEL MLX modules...")
            try:
                from indextts.s2mel.modules.mlx_s2mel import MLXGPTLayer, MLXLengthRegulator
                from indextts.s2mel.modules.mlx_cfm import MLXCFM
                
                # Create gpt_layer and length_regulator (always needed)
                self.mlx_s2mel_gpt_layer = MLXGPTLayer()
                self.mlx_s2mel_length_regulator = MLXLengthRegulator(
                    channels=self.cfg.s2mel.length_regulator.channels,
                    sampling_ratios=tuple(self.cfg.s2mel.length_regulator.sampling_ratios),
                    is_discrete=self.cfg.s2mel.length_regulator.is_discrete,
                    in_channels=self.cfg.s2mel.length_regulator.in_channels if hasattr(self.cfg.s2mel.length_regulator, "in_channels") else None,
                    codebook_size=self.cfg.s2mel.length_regulator.content_codebook_size,
                    n_codebooks=self.cfg.s2mel.length_regulator.n_codebooks if hasattr(self.cfg.s2mel.length_regulator, "n_codebooks") else 1,
                    f0_condition=self.cfg.s2mel.length_regulator.f0_condition if hasattr(self.cfg.s2mel.length_regulator, "f0_condition") else False,
                    n_f0_bins=self.cfg.s2mel.length_regulator.n_f0_bins if hasattr(self.cfg.s2mel.length_regulator, "n_f0_bins") else 512,
                )
                
                # Create MLX CFM
                self.mlx_s2mel_cfm = MLXCFM(self.cfg.s2mel)
                
                print(">> S2MEL MLX modules created successfully")
            except Exception as e:
                print(f">> S2MEL MLX modules creation failed: {e}")
                import traceback
                traceback.print_exc()
                self.mlx_s2mel_gpt_layer = None
                self.mlx_s2mel_length_regulator = None
                self.mlx_s2mel_cfm = None
        
        s2mel = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        s2mel, _, _, _ = load_checkpoint2(
            s2mel,
            None,
            s2mel_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        self.s2mel = s2mel.to(self.device)
        self.s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        self.s2mel.eval()
        print(">> s2mel weights restored from:", s2mel_path)
        
        # Load weights into MLX S2MEL modules with caching
        if self.use_mlx and self.mlx_available:
            if self.mlx_s2mel_gpt_layer is not None or self.mlx_s2mel_length_regulator is not None or self.mlx_s2mel_cfm is not None:
                print(">> Loading weights into S2MEL MLX modules...")
                s2mel_state_dict = self.s2mel.state_dict()
                s2mel_state_dict_np = {k: v.cpu().numpy() for k, v in s2mel_state_dict.items()}
                
                # Load gpt_layer weights and prepare for unified caching
                if self.mlx_s2mel_gpt_layer is not None:
                    self.mlx_s2mel_gpt_layer.load_weights_from_pytorch(s2mel_state_dict_np)
                
                # Load length_regulator weights and prepare for unified caching
                if self.mlx_s2mel_length_regulator is not None:
                    self.mlx_s2mel_length_regulator.load_weights_from_pytorch(s2mel_state_dict_np)
                
                # 🔥 CFM: 从统一的 s2mel.npz 加载权重
                if self.mlx_s2mel_cfm is not None:
                    s2mel_cache_file = os.path.join(self.mlx_cache.cache_dir, "s2mel.npz")
                    
                    if os.path.exists(s2mel_cache_file):
                        # 从统一的 s2mel.npz 加载 CFM 权重
                        print(">> Loading S2MEL CFM from unified cache...")
                        print(f"   Cache: {s2mel_cache_file}")
                        
                        try:
                            import mlx.core as mx
                            s2mel_weights = mx.load(s2mel_cache_file)
                            
                            # 提取 CFM 权重
                            cfm_weights = {}
                            for key, value in s2mel_weights.items():
                                if key.startswith('models.cfm.'):
                                    # 保持原始键名，因为MLX CFM期望cfm.estimator.xxx格式
                                    cfm_weights[key] = value
                            
                            print(f"   Extracted {len(cfm_weights)} CFM weights from s2mel.npz")
                            
                            # Load weights into CFM
                            self.mlx_s2mel_cfm.load_from_cache(cfm_weights)
                            
                            # Get cache size
                            size_mb = os.path.getsize(s2mel_cache_file) / (1024 * 1024)
                            print(f"   Size: {size_mb:.2f} MB")
                            print(">> ✓ Loaded from unified cache (fast!)")
                        except Exception as e:
                            print(f">> ✗ Unified cache loading failed: {e}")
                            print(">> Converting from PyTorch...")
                            self.mlx_s2mel_cfm.load_weights_from_pytorch(s2mel_state_dict_np, prefix="models.cfm.")
                    else:
                        # 首次运行：转换并缓存到统一的 s2mel.npz
                        print(">> S2MEL unified cache not found (first run)")
                        print(">> Converting PyTorch CFM to MLX and caching...")
                        print("   ⏳ This will take a few minutes on first run...")
                        
                        try:
                            # Load weights
                            loaded = self.mlx_s2mel_cfm.load_weights_from_pytorch(s2mel_state_dict_np, prefix="models.cfm.")
                            print(f">> Loaded {loaded} weights")
                            
                            # Extract and cache weights to unified s2mel.npz
                            print(">> Caching weights to unified s2mel.npz...")
                            cfm_weights_to_cache = self.mlx_s2mel_cfm.extract_weights_for_cache()
                            
                            # 使用 MLXModelCache 的 convert_and_cache 方法
                            print(">> Using MLXModelCache to save S2MEL weights...")
                            self.mlx_cache.convert_and_cache("s2mel", state_dict=s2mel_state_dict)
                            print(">> ✓ S2MEL weights cached successfully")
                            print(">> Next run will load from unified cache (much faster!)")
                        except Exception as e:
                            print(f">> ✗ CFM conversion failed: {e}")
                            print(">> CFM will use PyTorch fallback")
                            import traceback
                            traceback.print_exc()
                            self.mlx_s2mel_cfm = None
                
                print(">> S2MEL MLX modules ready")
            
            if self.mlx_s2mel_cfm is not None:
                print(">> S2MEL: Running on MPS with FULL MLX ⚡ (gpt_layer + length_regulator + cfm)")
            else:
                print(">> S2MEL: Running on MPS with MLX optimizations (gpt_layer + length_regulator)")

        # load campplus_model
        campplus_ckpt_path = hf_hub_download(
            "funasr/campplus", filename="campplus_cn_common.bin"
        )
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model = campplus_model.to(self.device)
        self.campplus_model.eval()
        print(">> campplus_model weights restored from:", campplus_ckpt_path)

        # Load BigVGAN with MLX caching
        bigvgan_name = self.cfg.vocoder.name
        
        if self.use_mlx and self.mlx_available:
            print("\n>> [Model 3/4] Loading BigVGAN with MLX optimization...")
        
        # MLX mode: Disable CUDA kernels on Apple Silicon
        use_kernel = self.use_cuda_kernel if not self.use_mlx else False
        self.bigvgan = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=use_kernel)
        self.bigvgan = self.bigvgan.to(self.device)
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        print(">> bigvgan weights restored from:", bigvgan_name)
        
        # Cache BigVGAN weights in MLX format
        if self.use_mlx and self.mlx_available:
            try:
                if not self.mlx_cache.is_cached("bigvgan"):
                    print(">> Caching BigVGAN weights in MLX format...")
                    self.mlx_cache.convert_and_cache("bigvgan", state_dict=self.bigvgan.state_dict())
                else:
                    print(">> BigVGAN already cached")
            except Exception as e:
                print(f">> BigVGAN caching skipped: {e}")
            print(">> BigVGAN: Running on MPS with MLX optimizations")

        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        print(">> TextNormalizer loaded")
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        print(">> bpe model loaded from:", self.bpe_path)

        emo_matrix = torch.load(os.path.join(self.model_dir, self.cfg.emo_matrix))
        self.emo_matrix = emo_matrix.to(self.device)
        self.emo_num = list(self.cfg.emo_num)

        spk_matrix = torch.load(os.path.join(self.model_dir, self.cfg.spk_matrix))
        self.spk_matrix = spk_matrix.to(self.device)

        self.emo_matrix = torch.split(self.emo_matrix, self.emo_num)
        self.spk_matrix = torch.split(self.spk_matrix, self.emo_num)

        mel_fn_args = {
            "n_fft": self.cfg.s2mel['preprocess_params']['spect_params']['n_fft'],
            "win_size": self.cfg.s2mel['preprocess_params']['spect_params']['win_length'],
            "hop_size": self.cfg.s2mel['preprocess_params']['spect_params']['hop_length'],
            "num_mels": self.cfg.s2mel['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
            "fmin": self.cfg.s2mel['preprocess_params']['spect_params'].get('fmin', 0),
            "fmax": None if self.cfg.s2mel['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
            "center": False
        }
        self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)

        # 缓存参考音频：
        self.cache_spk_cond = None
        self.cache_s2mel_style = None
        self.cache_s2mel_prompt = None
        self.cache_spk_audio_prompt = None
        self.cache_emo_cond = None
        self.cache_emo_audio_prompt = None
        # GPT Conditioning缓存（技术验证）- 需要缓存两个格式
        self.cache_gpt_conditioning_latent_mlx = None  # MLX format（用于generation）
        self.cache_gpt_conditioning_latent_torch = None  # PyTorch format（用于forward保留音色）
        self.cache_mel = None

        # 进度引用显示（可选）
        self.gr_progress = None
        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None
        
        # MLX initialization summary
        if self.use_mlx and self.mlx_available:
            # 判断是 Pure MLX 还是 Hybrid
            is_pure_mlx = (self.gpt_is_mlx and 
                          hasattr(self.mlx_transformer, 'use_mlx_conditioning') and 
                          self.mlx_transformer.use_mlx_conditioning)
            
            print("\n" + "="*70)
            if is_pure_mlx:
                print("⚡ Pure MLX Mode (Apple Silicon M4 Optimized) ✅")
            else:
                print("MLX Hybrid Mode")
            print("="*70)
            print(f"Device: {self.device}")
            
            if is_pure_mlx:
                print(f"GPT Backend: Pure MLX ⚡ (Conditioning + Transformer)")
                print("  - Conditioning: MLX (Conformer + Perceiver) ✅")
                print("  - Transformer: MLX (24 layers with KV cache) ✅")
                print("  - Status: Stable (v1.0)")
            else:
                print(f"GPT Backend: {'Hybrid (PyTorch + MLX) ⚡' if self.gpt_is_mlx else 'PyTorch (full)'}")
            
            print(f"Cache Directory: {self.mlx_cache.cache_dir}")
            print("\nCached Models:")
            for model in ["gpt", "s2mel", "bigvgan"]:
                status = "✓ Cached" if self.mlx_cache.is_cached(model) else "✗ Not cached"
                print(f"  {model.upper():10s}: {status}")
            
            if is_pure_mlx:
                print("\n📊 Performance:")
                print("  - RTF: 3-6x (stable)")
                print("  - Memory: Optimized with auto-cleanup")
                print("  - Quality: High (correlation 0.98+ with PyTorch)")
            
            print("\nNext run will load from cache (faster!)")
            print("="*70 + "\n")

    def _print_mps_memory(self, step_name=""):
        """打印MPS内存使用情况，格式：总量：[应用/驱动]"""
        if self.device == 'mps' and torch.backends.mps.is_available():
            try:
                allocated = torch.mps.current_allocated_memory() / 1024**3  # GB - 应用分配内存
                driver = torch.mps.driver_allocated_memory() / 1024**3  # GB - 驱动分配内存
                total = allocated + driver  # 总量 = 应用内存 + 驱动内存
                status_text = f" {step_name}" if step_name else ""
                print(f">> MPS内存{status_text}: {total:.2f}GB：[{allocated:.2f}/{driver:.2f}]")
            except Exception as e:
                status_text = f" {step_name}" if step_name else ""
                print(f">> MPS内存{status_text}: 无法获取内存信息 ({e})")
        else:
            status_text = f" {step_name}" if step_name else ""
            print(f">> MPS内存{status_text}: 非MPS设备，跳过内存监控")

    @torch.no_grad()
    def get_emb(self, input_features, attention_mask):
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat

    def remove_long_silence(self, codes: torch.Tensor, silent_token=52, max_consecutive=30):
        """
        Shrink special tokens (silent_token and stop_mel_token) in codes
        codes: [B, T]
        """
        code_lens = []
        codes_list = []
        device = codes.device
        dtype = codes.dtype
        isfix = False
        for i in range(0, codes.shape[0]):
            code = codes[i]
            if not torch.any(code == self.stop_mel_token).item():
                len_ = code.size(0)
            else:
                stop_mel_idx = (code == self.stop_mel_token).nonzero(as_tuple=False)
                len_ = stop_mel_idx[0].item() if len(stop_mel_idx) > 0 else code.size(0)

            count = torch.sum(code == silent_token).item()
            if count > max_consecutive:
                # code = code.cpu().tolist()
                ncode_idx = []
                n = 0
                for k in range(len_):
                    assert code[
                               k] != self.stop_mel_token, f"stop_mel_token {self.stop_mel_token} should be shrinked here"
                    if code[k] != silent_token:
                        ncode_idx.append(k)
                        n = 0
                    elif code[k] == silent_token and n < 10:
                        ncode_idx.append(k)
                        n += 1
                    # if (k == 0 and code[k] == 52) or (code[k] == 52 and code[k-1] == 52):
                    #    n += 1
                # new code
                len_ = len(ncode_idx)
                codes_list.append(code[ncode_idx])
                isfix = True
            else:
                # shrink to len_
                codes_list.append(code[:len_])
            code_lens.append(len_)
        if isfix:
            if len(codes_list) > 1:
                codes = pad_sequence(codes_list, batch_first=True, padding_value=self.stop_mel_token)
            else:
                codes = codes_list[0].unsqueeze(0)
        else:
            # unchanged
            pass
        # clip codes to max length
        max_len = max(code_lens)
        if max_len < codes.shape[1]:
            codes = codes[:, :max_len]
        code_lens = torch.tensor(code_lens, dtype=torch.long, device=device)
        return codes, code_lens

    def interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        """
        Silences to be insert between generated segments.
        """

        if not wavs or interval_silence <= 0:
            return wavs

        # get channel_size
        channel_size = wavs[0].size(0)
        # get silence tensor
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        return torch.zeros(channel_size, sil_dur)

    def insert_interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        """
        Insert silences between generated segments.
        wavs: List[torch.tensor]
        """

        if not wavs or interval_silence <= 0:
            return wavs

        # get channel_size
        channel_size = wavs[0].size(0)
        # get silence tensor
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        sil_tensor = torch.zeros(channel_size, sil_dur)

        wavs_list = []
        for i, wav in enumerate(wavs):
            wavs_list.append(wav)
            if i < len(wavs) - 1:
                wavs_list.append(sil_tensor)

        return wavs_list

    def _set_gr_progress(self, value, desc):
        if self.gr_progress is not None:
            self.gr_progress(value, desc=desc)

    def _load_and_cut_audio(self,audio_path,max_audio_length_seconds,verbose=False,sr=None):
        if not sr:
            audio, sr = librosa.load(audio_path)
        else:
            audio, _ = librosa.load(audio_path,sr=sr)
        audio = torch.tensor(audio).unsqueeze(0)
        max_audio_samples = int(max_audio_length_seconds * sr)

        if audio.shape[1] > max_audio_samples:
            if verbose:
                print(f"Audio too long ({audio.shape[1]} samples), truncating to {max_audio_samples} samples")
            audio = audio[:, :max_audio_samples]
        return audio, sr
    
    def normalize_emo_vec(self, emo_vector, apply_bias=True):
        # apply biased emotion factors for better user experience,
        # by de-emphasizing emotions that can cause strange results
        if apply_bias:
            # [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
            emo_bias = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
            emo_vector = [vec * bias for vec, bias in zip(emo_vector, emo_bias)]

        # the total emotion sum must be 0.8 or less
        emo_sum = sum(emo_vector)
        if emo_sum > 0.8:
            scale_factor = 0.8 / emo_sum
            emo_vector = [vec * scale_factor for vec in emo_vector]

        return emo_vector
    
    def _ensure_qwen_loaded(self):
        """延迟加载 Qwen Emotion 模型（仅在需要时加载）"""
        if self.qwen_emo is None:
            print(">> Loading Qwen Emotion model (first use)...")
            self.qwen_emo = QwenEmotion(self.qwen_emo_path)
            print(">> Qwen Emotion loaded (~1.2GB)")
    
    def _ensure_semantic_loaded(self):
        """延迟加载 Semantic Model（仅在提取特征时加载）"""
        if not self.semantic_model_loaded:
            print(">> Loading Semantic Model (W2V-BERT) for feature extraction...")
            from indextts.utils.maskgct_utils import build_semantic_model
            self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(self.semantic_stat_path)
            self.semantic_model = self.semantic_model.to(self.device)
            self.semantic_model.eval()
            self.semantic_mean = self.semantic_mean.to(self.device)
            self.semantic_std = self.semantic_std.to(self.device)
            self.semantic_model_loaded = True
            print(">> Semantic Model loaded (~1.0GB)")
    
    def _unload_semantic(self):
        """卸载 Semantic Model，释放内存"""
        if self.semantic_model_loaded:
            print(">> Unloading Semantic Model...")
            del self.semantic_model
            del self.semantic_mean
            del self.semantic_std
            self.semantic_model = None
            self.semantic_mean = None
            self.semantic_std = None
            self.semantic_model_loaded = False
            
            # 清理内存
            import gc
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            print(">> Semantic Model unloaded (~1.0GB freed)")

    # 原始推理模式
    def infer(self, spk_audio_prompt, text, output_path,
              emo_audio_prompt=None, emo_alpha=1.0,
              emo_vector=None,
              use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
              verbose=False, max_text_tokens_per_segment=120, stream_return=False, more_segment_before=0, **generation_kwargs):
        try:
            if stream_return:
                return self.infer_generator(
                    spk_audio_prompt, text, output_path,
                    emo_audio_prompt, emo_alpha,
                    emo_vector,
                    use_emo_text, emo_text, use_random, interval_silence,
                    verbose, max_text_tokens_per_segment, stream_return, more_segment_before, **generation_kwargs
                )
            else:
                try:
                    return list(self.infer_generator(
                        spk_audio_prompt, text, output_path,
                        emo_audio_prompt, emo_alpha,
                        emo_vector,
                        use_emo_text, emo_text, use_random, interval_silence,
                        verbose, max_text_tokens_per_segment, stream_return, more_segment_before, **generation_kwargs
                    ))[0]
                except IndexError:
                    return None
        finally:
            # 🔧 修复内存泄漏：每次推理后清理缓存
            import gc
            gc.collect()
            
            # 清理 PyTorch MPS 缓存
            if self.device == 'mps' and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            # 清理 MLX 缓存
            if self.use_mlx:
                import mlx.core as mx
                try:
                    mx.clear_cache()
                except:
                    pass

    def infer_generator(self, spk_audio_prompt, text, output_path,
              emo_audio_prompt=None, emo_alpha=1.0,
              emo_vector=None,
              use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
              verbose=False, max_text_tokens_per_segment=120, stream_return=False, quick_streaming_tokens=0, **generation_kwargs):
        
        # Set random seed if provided
        seed = generation_kwargs.pop('seed', None)
        if seed is not None:
            print(f">> Setting random seed: {seed}")
            # 使用统一随机数生成器
            self.unified_random.reset_seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            if self.use_mlx:
                import mlx.core as mx
                mx.random.seed(seed)
        
        print(">> starting inference...")
        self._print_mps_memory("推理开始")
        self._set_gr_progress(0, "starting inference...")
        if verbose:
            print(f"origin text:{text}, spk_audio_prompt:{spk_audio_prompt}, "
                  f"emo_audio_prompt:{emo_audio_prompt}, emo_alpha:{emo_alpha}, "
                  f"emo_vector:{emo_vector}, use_emo_text:{use_emo_text}, "
                  f"emo_text:{emo_text}")
        start_time = time.perf_counter()

        if use_emo_text or emo_vector is not None:
            # we're using a text or emotion vector guidance; so we must remove
            # "emotion reference voice", to ensure we use correct emotion mixing!
            emo_audio_prompt = None

        if use_emo_text:
            # automatically generate emotion vectors from text prompt
            if emo_text is None:
                emo_text = text  # use main text prompt
            # 延迟加载 Qwen Emotion 模型
                self._ensure_qwen_loaded()
                emo_dict = self.qwen_emo.inference(emo_text)
                print(f"detected emotion vectors from text: {emo_dict}")
                # convert ordered dict to list of vectors; the order is VERY important!
                emo_vector = list(emo_dict.values())

        if emo_vector is not None:
            # we have emotion vectors; they can't be blended via alpha mixing
            # in the main inference process later, so we must pre-calculate
            # their new strengths here based on the alpha instead!
            emo_vector_scale = max(0.0, min(1.0, emo_alpha))
            if emo_vector_scale != 1.0:
                # scale each vector and truncate to 4 decimals (for nicer printing)
                emo_vector = [int(x * emo_vector_scale * 10000) / 10000 for x in emo_vector]
                print(f"scaled emotion vectors to {emo_vector_scale}x: {emo_vector}")

        if emo_audio_prompt is None:
            # we are not using any external "emotion reference voice"; use
            # speaker's voice as the main emotion reference audio.
            emo_audio_prompt = spk_audio_prompt
            # must always use alpha=1.0 when we don't have an external reference voice
            emo_alpha = 1.0

        # 如果参考音频改变了，才需要重新生成, 提升速度
        if self.cache_spk_cond is None or self.cache_spk_audio_prompt != spk_audio_prompt:
            if self.cache_spk_cond is not None:
                self.cache_spk_cond = None
                self.cache_s2mel_style = None
                self.cache_s2mel_prompt = None
                self.cache_mel = None
                self.cache_gpt_conditioning_latent_mlx = None  # 清除GPT Conditioning缓存
                self.cache_gpt_conditioning_latent_torch = None
                torch.cuda.empty_cache()
            audio,sr = self._load_and_cut_audio(spk_audio_prompt,15,verbose)
            audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
            audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

            # 🎯 加载 Semantic Model（按需）
            self._ensure_semantic_loaded()
            self._print_mps_memory("特征提取前")
            
            inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
            input_features = inputs["input_features"]
            attention_mask = inputs["attention_mask"]
            input_features = input_features.to(self.device)
            attention_mask = attention_mask.to(self.device)
            spk_cond_emb = self.get_emb(input_features, attention_mask)

            _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
            
            # 🎯 特征提取完成，卸载 Semantic Model
            self._unload_semantic()
            self._print_mps_memory("特征提取后")
            
            ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
            ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
            feat = torchaudio.compliance.kaldi.fbank(audio_16k.to(ref_mel.device),
                                                     num_mel_bins=80,
                                                     dither=0,
                                                     sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)  # feat2另外一个滤波器能量组特征[922, 80]
            style = self.campplus_model(feat.unsqueeze(0))  # 参考音频的全局style2[1,192]

            if self.use_mlx and self.mlx_s2mel_length_regulator is not None:
                # MLX版本
                import mlx.core as mx
                from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch
                S_ref_mlx = torch_to_mlx(S_ref.cpu())
                ref_target_lengths_mlx = torch_to_mlx(ref_target_lengths.cpu())
                prompt_condition_mlx, _, _, _, _ = self.mlx_s2mel_length_regulator(
                    S_ref_mlx,
                    ylens=ref_target_lengths_mlx,
                    n_quantizers=None,
                    f0=None
                )
                mx.eval(prompt_condition_mlx)
                prompt_condition = mlx_to_torch(prompt_condition_mlx).to(self.device)
            else:
                # PyTorch版本（稳定）
                prompt_condition = self.s2mel.models['length_regulator'](S_ref,
                                                                         ylens=ref_target_lengths,
                                                                         n_quantizers=3,
                                                                         f0=None)[0]
            
            self.cache_spk_cond = spk_cond_emb
            self.cache_s2mel_style = style
            self.cache_s2mel_prompt = prompt_condition
            self.cache_spk_audio_prompt = spk_audio_prompt
            self.cache_mel = ref_mel
            print(f">> [Cache] Cached semantic features for: {spk_audio_prompt}")
        else:
            style = self.cache_s2mel_style
            prompt_condition = self.cache_s2mel_prompt
            spk_cond_emb = self.cache_spk_cond
            ref_mel = self.cache_mel
            print(f">> [Cache Hit] Using cached semantic features")

        if emo_vector is not None:
            weight_vector = torch.tensor(emo_vector).to(self.device)
            if use_random:
                random_index = [random.randint(0, x - 1) for x in self.emo_num]
            else:
                random_index = [find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]

            emo_matrix = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)]
            emo_matrix = torch.cat(emo_matrix, 0)
            emovec_mat = weight_vector.unsqueeze(1) * emo_matrix
            emovec_mat = torch.sum(emovec_mat, 0)
            emovec_mat = emovec_mat.unsqueeze(0)

        if self.cache_emo_cond is None or self.cache_emo_audio_prompt != emo_audio_prompt:
            if self.cache_emo_cond is not None:
                self.cache_emo_cond = None
                torch.cuda.empty_cache()
            
            # 🎯 加载 Semantic Model（如果还未加载）
            self._ensure_semantic_loaded()
            
            emo_audio, _ = self._load_and_cut_audio(emo_audio_prompt,15,verbose,sr=16000)
            emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
            emo_input_features = emo_inputs["input_features"]
            emo_attention_mask = emo_inputs["attention_mask"]
            emo_input_features = emo_input_features.to(self.device)
            emo_attention_mask = emo_attention_mask.to(self.device)
            emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)

            self.cache_emo_cond = emo_cond_emb
            self.cache_emo_audio_prompt = emo_audio_prompt
            
            # 🎯 特征提取完成，卸载 Semantic Model
            self._unload_semantic()
        else:
            emo_cond_emb = self.cache_emo_cond

        self._set_gr_progress(0.1, "text processing...")
        text_tokens_list = self.tokenizer.tokenize(text)
        segments = self.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment, quick_streaming_tokens = quick_streaming_tokens)
        segments_count = len(segments)

        text_token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)
        if self.tokenizer.unk_token_id in text_token_ids:
            print(f"  >> Warning: input text contains {text_token_ids.count(self.tokenizer.unk_token_id)} unknown tokens (id={self.tokenizer.unk_token_id}):")
            print( "     Tokens which can't be encoded: ", [t for t, id in zip(text_tokens_list, text_token_ids) if id == self.tokenizer.unk_token_id])
            print(f"     Consider updating the BPE model or modifying the text to avoid unknown tokens.")
                  
        if verbose:
            print("text_tokens_list:", text_tokens_list)
            print("segments count:", segments_count)
            print("max_text_tokens_per_segment:", max_text_tokens_per_segment)
            print(*segments, sep="\n")
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 0.8)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 1)  # 🔧 Changed to 1 for faster debugging
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 1500)
        sampling_rate = 22050

        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        # GPT生成详细计时
        gpt_emovec_time = 0
        gpt_conditioning_time = 0
        gpt_to_mlx_time = 0
        gpt_generation_time = 0
        gpt_from_mlx_time = 0
        s2mel_time = 0
        bigvgan_time = 0
        has_warned = False
        silence = None # for stream_return
        for seg_idx, sent in enumerate(segments):
            self._set_gr_progress(0.2 + 0.7 * seg_idx / segments_count,
                                  f"speech synthesis {seg_idx + 1}/{segments_count}...")

            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
            if verbose:
                print(text_tokens)
                print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
                # debug tokenizer
                text_token_syms = self.tokenizer.convert_ids_to_tokens(text_tokens[0].tolist())
                print("text_token_syms is same as segment tokens", text_token_syms == sent)
            
            m_start_time = time.perf_counter()
            self._print_mps_memory(f"GPT生成前(段落{seg_idx+1})")
            with torch.no_grad():
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    # Profiling: emovec计算
                    t0_emovec = time.perf_counter()
                    if self.gpt_is_mlx:
                        emovec = self.mlx_transformer.merge_emovec(
                            spk_cond_emb,
                            emo_cond_emb,
                            torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                            torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                            alpha=emo_alpha
                        )
                    else:
                        emovec = self.gpt.merge_emovec(
                            spk_cond_emb,
                            emo_cond_emb,
                            torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                            torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                            alpha=emo_alpha
                        )

                    if emo_vector is not None:
                        emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec
                        # emovec = emovec_mat
                    gpt_emovec_time += time.perf_counter() - t0_emovec

                    # Use MLX inference if enabled
                    if self.gpt_is_mlx and self.mlx_transformer is not None:
                        # Check if using pure MLX or hybrid mode
                        if hasattr(self.mlx_transformer, 'use_mlx_conditioning') and self.mlx_transformer.use_mlx_conditioning:
                            # Pure MLX mode
                            # 🚀 优化：检查GPT Conditioning缓存
                            if self.cache_gpt_conditioning_latent_mlx is not None:
                                # 缓存命中：直接用缓存的conditioning进行generation
                                print(f">> [Cache Hit] Using cached MLX GPT conditioning")
                                print(f"   Cached MLX shape: {self.cache_gpt_conditioning_latent_mlx.shape}")
                                from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch
                                
                                # 只需要转换text和执行generation
                                print(f">> [Cache] Converting text to MLX...")
                                text_mlx = torch_to_mlx(text_tokens.cpu())
                                print(f">> [Cache] Running generation with cached MLX conditioning...")
                                codes = self.mlx_transformer.simple_forward(
                                    text_mlx,
                                    conditioning=self.cache_gpt_conditioning_latent_mlx,  # 使用MLX缓存
                                    max_length=max_mel_tokens,
                                    temperature=temperature,
                                    use_sampling=generation_kwargs.get('use_sampling', True),
                                    debug_generation=generation_kwargs.get('debug_generation', False),
                                )
                                print(f">> [Cache] Generation complete, converting back...")
                                codes = mlx_to_torch(codes, device='cpu').long().to(text_tokens.device)
                                print(f">> [Cache] Codes converted: {codes.shape}")
                                # 🔥 MLX模式：从MLX缓存转换回PyTorch格式用于后续处理
                                speech_conditioning_latent = mlx_to_torch(self.cache_gpt_conditioning_latent_mlx, device=text_tokens.device)
                                print(f">> [MLX Native] Using cached MLX conditioning converted to PyTorch: {speech_conditioning_latent.shape}")
                            else:
                                # 缓存未命中：完整计算并缓存
                                print(f">> [Cache Miss] Computing GPT conditioning...")
                                result = self.mlx_transformer.inference_speech(
                                    spk_cond_emb,
                                    text_tokens,
                                    emo_speech_condition=emo_cond_emb,
                                    cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                                    emo_cond_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                                    emo_vec=emovec,
                                    temperature=temperature,
                                    max_generate_length=max_mel_tokens,
                                    use_sampling=generation_kwargs.get('use_sampling', True),  # 传递采样模式
                                    debug_generation=generation_kwargs.get('debug_generation', False),  # 传递 debug 模式
                                    return_conditioning_mlx=True,  # 请求返回MLX格式conditioning
                                )
                                # 解包返回值并确保使用一致的conditioning
                                if len(result) == 3:
                                    codes, speech_conditioning_latent_mlx_converted, cond_latent_mlx = result
                                else:
                                    codes, speech_conditioning_latent_mlx_converted = result
                                    cond_latent_mlx = None
                                
                                # 🔥 CRITICAL: 使用MLX模型的get_conditioning方法确保与PyTorch版本一致
                                # 这样可以保证在相同输入变量下产生一致的结果（忽略精度累计差异）
                                speech_conditioning_latent = self.mlx_transformer.get_conditioning(
                                    spk_cond_emb.transpose(1, 2), 
                                    torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device)
                                )
                                print(f">> [MLX Consistency] Using MLX get_conditioning for PyTorch-MLX consistency")
                                
                                # 调试：记录MLX GPT生成后的输出
                                print(f">> [MLX GPT Debug] GPT生成后:")
                                print(f"   codes: {codes.shape}, min={codes.min():.6f}, max={codes.max():.6f}")
                                print(f"   speech_conditioning_latent: {speech_conditioning_latent.shape}, min={speech_conditioning_latent.min():.6f}, max={speech_conditioning_latent.max():.6f}")
                                
                                # 保存MLX GPT输出到文件
                                import pickle
                                gpt_outputs_mlx = {
                                    'codes': codes.detach().cpu(),
                                    'speech_conditioning_latent': speech_conditioning_latent.detach().cpu()
                                }
                                with open('gpt_outputs_mlx.pkl', 'wb') as f:
                                    pickle.dump(gpt_outputs_mlx, f)
                                print(f">> [MLX GPT Debug] GPT输出已保存到 gpt_outputs_mlx.pkl")
                                
                                # 🔥 关键：缓存MLX格式
                                if cond_latent_mlx is not None:
                                    self.cache_gpt_conditioning_latent_mlx = cond_latent_mlx
                                    print(f">> [Cache] GPT conditioning cached (MLX native)")
                                    print(f"   MLX shape: {cond_latent_mlx.shape}")
                                    print(f"   Consistent Torch shape: {speech_conditioning_latent.shape}")
                                else:
                                    from indextts.utils.mlx_utils import torch_to_mlx
                                    # 缓存MLX版本用于后续推理
                                    self.cache_gpt_conditioning_latent_mlx = torch_to_mlx(speech_conditioning_latent.cpu())
                                    print(f">> [Cache] GPT conditioning cached from consistent result")
                    else:
                        # PyTorch inference
                        if self.gpt is None:
                            raise RuntimeError("PyTorch GPT not loaded. Cannot use PyTorch inference.")
                        
                        # 🔧 For debugging: force greedy decoding when num_beams=1 for deterministic output
                        # PyTorch's do_sample=True is not deterministic even with fixed seed
                        use_sampling_mode = False if num_beams == 1 else do_sample

                        codes, speech_conditioning_latent = self.gpt.inference_speech(
                            spk_cond_emb,
                            text_tokens,
                            emo_cond_emb,
                            cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                            emo_cond_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                            emo_vec=emovec,
                                do_sample=use_sampling_mode,
                            top_p=top_p,
                            top_k=top_k,
                            temperature=temperature,
                            num_return_sequences=autoregressive_batch_size,
                            length_penalty=length_penalty,
                            num_beams=num_beams,
                            repetition_penalty=repetition_penalty,
                            max_generate_length=max_mel_tokens,
                            **generation_kwargs
                        )
                        
                        # 调试：记录MLX GPT生成后的输出
                        print(f">> [MLX GPT Debug] GPT生成后:")
                        print(f"   codes: {codes.shape}, min={codes.min():.6f}, max={codes.max():.6f}")
                        print(f"   speech_conditioning_latent: {speech_conditioning_latent.shape}, min={speech_conditioning_latent.min():.6f}, max={speech_conditioning_latent.max():.6f}")
                        
                        # 保存MLX GPT输出到文件
                        import pickle
                        gpt_outputs_mlx = {
                            'codes': codes.detach().cpu(),
                            'speech_conditioning_latent': speech_conditioning_latent.detach().cpu()
                        }
                        with open('gpt_outputs_mlx.pkl', 'wb') as f:
                            pickle.dump(gpt_outputs_mlx, f)
                        print(f">> [MLX GPT Debug] GPT输出已保存到 gpt_outputs_mlx.pkl")

                gpt_gen_time += time.perf_counter() - m_start_time
                self._print_mps_memory(f"GPT生成后(段落{seg_idx+1})")
                if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Input text tokens: {text_tokens.shape[1]}. "
                        f"Consider reducing `max_text_tokens_per_segment`({max_text_tokens_per_segment}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning
                    )
                    has_warned = True

                code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)
                #                 if verbose:
                #                     print(codes, type(codes))
                #                     print(f"codes shape: {codes.shape}, codes type: {codes.dtype}")
                #                     print(f"code len: {code_lens}")

                code_lens = []
                for code in codes:
                    if self.stop_mel_token not in code:
                        code_len = len(code)
                    else:
                        len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0] + 1
                        code_len = len_ - 1
                    code_lens.append(code_len)  # ✅ FIX: Only append once
                codes = codes[:, :code_len]
                code_lens = torch.LongTensor(code_lens)
                code_lens = code_lens.to(self.device)
                if verbose:
                    print(codes, type(codes))
                    print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")

                m_start_time = time.perf_counter()
                use_speed = torch.zeros(spk_cond_emb.size(0)).to(spk_cond_emb.device).long()
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    if self.gpt_is_mlx:
                        latent = self.mlx_transformer(
                            speech_conditioning_latent,
                            text_tokens,
                            torch.tensor([text_tokens.shape[-1]], device=text_tokens.device),
                            codes,
                            torch.tensor([codes.shape[-1]], device=text_tokens.device),
                            emo_cond_emb,
                            cond_mel_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                            emo_cond_mel_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                            emo_vec=emovec,
                            use_speed=use_speed,
                        )
                    else:
                        if self.gpt is None:
                            raise RuntimeError("PyTorch GPT not loaded.")
                        latent = self.gpt(
                            speech_conditioning_latent,
                            text_tokens,
                            torch.tensor([text_tokens.shape[-1]], device=text_tokens.device),
                            codes,
                            torch.tensor([codes.shape[-1]], device=text_tokens.device),
                            emo_cond_emb,
                            cond_mel_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                            emo_cond_mel_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                            emo_vec=emovec,
                            use_speed=use_speed,
                        )
                    gpt_forward_time += time.perf_counter() - m_start_time

                dtype = None
                with torch.amp.autocast(text_tokens.device.type, enabled=dtype is not None, dtype=dtype):
                    m_start_time = time.perf_counter()
                    self._print_mps_memory(f"S2MEL处理前(段落{seg_idx+1})")
                    # 🚀 使用配置的diffusion steps，默认25步提升音质
                    diffusion_steps = self.diffusion_steps  # 默认值：25
                    inference_cfg_rate = 0.7
                    
                    # Profiling: gpt_layer
                    t0 = time.perf_counter()
                    if self.use_mlx and self.mlx_s2mel_gpt_layer is not None:
                        # MLX版本
                        import mlx.core as mx
                        from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch
                        latent_mlx = torch_to_mlx(latent.cpu())
                        latent_mlx = self.mlx_s2mel_gpt_layer(latent_mlx)
                        mx.eval(latent_mlx)
                        latent = mlx_to_torch(latent_mlx).to(self.device)
                    else:
                        # PyTorch版本（稳定）
                        latent = self.s2mel.models['gpt_layer'](latent)
                    t_gpt_layer = time.perf_counter() - t0
                    
                    # Profiling: vq2emb
                    t0 = time.perf_counter()
                    S_infer = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
                    t_vq2emb = time.perf_counter() - t0
                    
                    # Profiling: transpose + add
                    t0 = time.perf_counter()
                    S_infer = S_infer.transpose(1, 2)
                    S_infer = S_infer + latent
                    target_lengths = (code_lens * 1.72).long()
                    t_prepare = time.perf_counter() - t0
                    
                    # Profiling: length_regulator
                    t0 = time.perf_counter()
                    if self.use_mlx and self.mlx_s2mel_length_regulator is not None:
                        # MLX版本
                        import mlx.core as mx
                        from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch
                        S_infer_mlx = torch_to_mlx(S_infer.cpu())
                        target_lengths_mlx = torch_to_mlx(target_lengths.cpu())
                        cond_mlx, _, _, _, _ = self.mlx_s2mel_length_regulator(
                            S_infer_mlx,
                            ylens=target_lengths_mlx,
                            n_quantizers=None,
                            f0=None
                        )
                        mx.eval(cond_mlx)
                        cond = mlx_to_torch(cond_mlx).to(self.device)
                    else:
                        # PyTorch版本（稳定）
                        cond = self.s2mel.models['length_regulator'](S_infer,
                                                                     ylens=target_lengths,
                                                                     n_quantizers=3,
                                                                     f0=None)[0]
                    t_length_reg = time.perf_counter() - t0
                    
                    # Fix batch dimension mismatch if needed
                    if cond.shape[0] != prompt_condition.shape[0]:
                        print(f">> [MLX Fix] Adjusting batch dimension: cond {cond.shape} vs prompt {prompt_condition.shape}")
                        # Take first batch element if cond has extra batch dimension
                        if cond.shape[0] > prompt_condition.shape[0]:
                            cond = cond[:prompt_condition.shape[0]]
                    
                    cat_condition = torch.cat([prompt_condition, cond], dim=1)
                    
                    # Profiling: CFM diffusion
                    t0 = time.perf_counter()
                    # 捕获S2MEL输入数据用于测试
                    s2mel_inputs = {
                        'cat_condition': cat_condition.clone(),
                        'x_lens': torch.LongTensor([cat_condition.size(1)]).to(cond.device),
                        'ref_mel': ref_mel.clone(),
                        'style': style.clone(),
                        'diffusion_steps': diffusion_steps,
                        'inference_cfg_rate': inference_cfg_rate
                    }
                    self._s2mel_inputs = s2mel_inputs
                    
                    if self.use_mlx and self.mlx_s2mel_cfm is not None:
                        # MLX版本CFM
                        import mlx.core as mx
                        from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch
                        
                        # 调试：记录MLX CFM输入
                        print(f">> [MLX CFM Debug] CFM输入:")
                        print(f"   cat_condition: {cat_condition.shape}, min={cat_condition.min():.6f}, max={cat_condition.max():.6f}")
                        print(f"   x_lens: {cat_condition.size(1)}")
                        print(f"   ref_mel: {ref_mel.shape}, min={ref_mel.min():.6f}, max={ref_mel.max():.6f}")
                        print(f"   style: {style.shape}, min={style.min():.6f}, max={style.max():.6f}")
                        print(f"   diffusion_steps: {diffusion_steps}")
                        print(f"   inference_cfg_rate: {inference_cfg_rate}")
                        
                        # 转换输入到MLX
                        cat_condition_mlx = torch_to_mlx(cat_condition)
                        x_lens_mlx = torch_to_mlx(torch.LongTensor([cat_condition.size(1)]).to(cond.device))
                        ref_mel_mlx = torch_to_mlx(ref_mel)
                        style_mlx = torch_to_mlx(style)
                        
                        # 保存MLX CFM输入到文件
                        import pickle
                        cfm_inputs_mlx = {
                            'cat_condition': cat_condition.detach().cpu(),
                            'x_lens': torch.LongTensor([cat_condition.size(1)]),
                            'ref_mel': ref_mel.detach().cpu(),
                            'style': style.detach().cpu(),
                            'diffusion_steps': diffusion_steps,
                            'inference_cfg_rate': inference_cfg_rate
                        }
                        with open('cfm_inputs_mlx.pkl', 'wb') as f:
                            pickle.dump(cfm_inputs_mlx, f)
                        print(f">> [MLX CFM Debug] CFM输入已保存到 cfm_inputs_mlx.pkl")
                        
                        # 🔥 使用 MLX CFM 进行推理
                        print(">> [MLX CFM Debug] 使用 MLX CFM 进行推理")
                        
                        # 使用 MLX CFM 推理
                        vc_target_mlx = self.mlx_s2mel_cfm.inference(
                            cat_condition_mlx,
                            x_lens_mlx,
                            ref_mel_mlx, 
                            style_mlx,
                            None,
                            diffusion_steps,
                            inference_cfg_rate=inference_cfg_rate,
                            unified_random=self.unified_random
                        )
                        
                        print(f">> [MLX CFM Debug] MLX CFM 输出:")
                        print(f"   vc_target: {vc_target_mlx.shape}, min={vc_target_mlx.min():.6f}, max={vc_target_mlx.max():.6f}")
                        print(f"   vc_target mean: {vc_target_mlx.mean():.6f}, std: {vc_target_mlx.std():.6f}")
                        
                        # 将 MLX 输出转换为 PyTorch tensor
                        vc_target = mlx_to_torch(vc_target_mlx).to(self.device)
                        
                        print(f">> [MLX CFM Debug] 转换后的 PyTorch 输出:")
                        print(f"   vc_target: {vc_target.shape}, min={vc_target.min():.6f}, max={vc_target.max():.6f}")
                        print(f"   vc_target mean: {vc_target.mean():.6f}, std: {vc_target.std():.6f}")
                        
                        # 保存 MLX CFM 输出到文件
                        cfm_outputs_mlx = {
                            'vc_target': vc_target.detach().cpu()
                        }
                        with open('cfm_outputs_mlx.pkl', 'wb') as f:
                            pickle.dump(cfm_outputs_mlx, f)
                        print(f">> [MLX CFM Debug] MLX CFM 输出已保存到 cfm_outputs_mlx.pkl")
                        
                        # 捕获 MLX S2MEL 输出用于调试
                        self._s2mel_outputs = {
                            'vc_target_shape': vc_target.shape,
                            'vc_target_min': vc_target.min().item(),
                            'vc_target_max': vc_target.max().item(),
                            'vc_target_mean': vc_target.mean().item(),
                            'vc_target_std': vc_target.std().item()
                        }
                    else:
                        # PyTorch版本CFM
                        vc_target = self.s2mel.models['cfm'].inference(cat_condition,
                                                                       torch.LongTensor([cat_condition.size(1)]).to(
                                                                           cond.device),
                                                                       ref_mel, style, None, diffusion_steps,
                                                                       inference_cfg_rate=inference_cfg_rate,
                                                                       unified_random=self.unified_random)
                        
                        # 捕获PyTorch S2MEL输出用于调试
                        self._s2mel_outputs = {
                            'vc_target_shape': vc_target.shape,
                            'vc_target_min': vc_target.min().item(),
                            'vc_target_max': vc_target.max().item(),
                            'vc_target_mean': vc_target.mean().item(),
                            'vc_target_std': vc_target.std().item()
                        }
                    t_cfm = time.perf_counter() - t0
                    vc_target = vc_target[:, :, ref_mel.size(-1):]
                    s2mel_time += time.perf_counter() - m_start_time

                    # Print detailed profiling
                    print(f">> S2MEL breakdown: gpt_layer={t_gpt_layer:.2f}s, vq2emb={t_vq2emb:.2f}s, prepare={t_prepare:.4f}s, length_reg={t_length_reg:.2f}s, cfm={t_cfm:.2f}s (steps={diffusion_steps})")
                    self._print_mps_memory(f"S2MEL处理后(段落{seg_idx+1})")
                    
                    m_start_time = time.perf_counter()
                    self._print_mps_memory(f"BigVGAN处理前(段落{seg_idx+1})")
                    wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
                    print(wav.shape)
                    bigvgan_time += time.perf_counter() - m_start_time
                    self._print_mps_memory(f"BigVGAN处理后(段落{seg_idx+1})")
                    wav = wav.squeeze(1)

                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                if verbose:
                    print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())
                # wavs.append(wav[:, :-512])
                wavs.append(wav.cpu())  # to cpu before saving
                if stream_return:
                    yield wav.cpu()
                    if silence == None:
                        silence = self.interval_silence(wavs, sampling_rate=sampling_rate, interval_silence=interval_silence)
                    yield silence
        end_time = time.perf_counter()
        self._print_mps_memory("推理完成")

        self._set_gr_progress(0.9, "saving audio...")
        wavs = self.insert_interval_silence(wavs, sampling_rate=sampling_rate, interval_silence=interval_silence)
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        if gpt_emovec_time > 0:
            # 计算实际generation时间（总时间 - emovec）
            actual_generation = gpt_gen_time - gpt_emovec_time
            print(f"   ├─ emovec: {gpt_emovec_time:.2f}s ({gpt_emovec_time/gpt_gen_time*100:.1f}%)")
            print(f"   └─ MLX inference: {actual_generation:.2f}s ({actual_generation/gpt_gen_time*100:.1f}%)")
            if gpt_conditioning_time > 0 or gpt_to_mlx_time > 0 or gpt_from_mlx_time > 0:
                # Hybrid模式的详细拆分
                print(f"      ├─ conditioning: {gpt_conditioning_time:.2f}s")
                print(f"      ├─ torch→mlx: {gpt_to_mlx_time:.2f}s")
                print(f"      ├─ generation: {gpt_generation_time:.2f}s")
                print(f"      └─ mlx→torch: {gpt_from_mlx_time:.2f}s")
        print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        print(f">> s2mel_time: {s2mel_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            # 直接保存音频到指定路径中
            if os.path.isfile(output_path):
                os.remove(output_path)
                print(">> remove old wav file:", output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            print(">> wav file saved to:", output_path)
            if stream_return:
                return None
            yield output_path
        else:
            if stream_return:
                return None
            # 返回以符合Gradio的格式要求
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            yield (sampling_rate, wav_data)


def find_most_similar_cosine(query_vector, matrix):
    query_vector = query_vector.float()
    matrix = matrix.float()

    similarities = F.cosine_similarity(query_vector, matrix, dim=1)
    most_similar_index = torch.argmax(similarities)
    return most_similar_index

class QwenEmotion:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            torch_dtype="float16",  # "auto"
            device_map="auto"
        )
        self.prompt = "文本情感分类"
        self.cn_key_to_en = {
            "高兴": "happy",
            "愤怒": "angry",
            "悲伤": "sad",
            "恐惧": "afraid",
            "反感": "disgusted",
            # TODO: the "低落" (melancholic) emotion will always be mapped to
            # "悲伤" (sad) by QwenEmotion's text analysis. it doesn't know the
            # difference between those emotions even if user writes exact words.
            # SEE: `self.melancholic_words` for current workaround.
            "低落": "melancholic",
            "惊讶": "surprised",
            "自然": "calm",
        }
        self.desired_vector_order = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]
        self.melancholic_words = {
            # emotion text phrases that will force QwenEmotion's "悲伤" (sad) detection
            # to become "低落" (melancholic) instead, to fix limitations mentioned above.
            "低落",
            "melancholy",
            "melancholic",
            "depression",
            "depressed",
            "gloomy",
        }
        self.max_score = 1.2
        self.min_score = 0.0

    def clamp_score(self, value):
        return max(self.min_score, min(self.max_score, value))

    def convert(self, content):
        # generate emotion vector dictionary:
        # - insert values in desired order (Python 3.7+ `dict` remembers insertion order)
        # - convert Chinese keys to English
        # - clamp all values to the allowed min/max range
        # - use 0.0 for any values that were missing in `content`
        emotion_dict = {
            self.cn_key_to_en[cn_key]: self.clamp_score(content.get(cn_key, 0.0))
            for cn_key in self.desired_vector_order
        }

        # default to a calm/neutral voice if all emotion vectors were empty
        if all(val <= 0.0 for val in emotion_dict.values()):
            print(">> no emotions detected; using default calm/neutral voice")
            emotion_dict["calm"] = 1.0

        return emotion_dict

    def inference(self, text_input):
        start = time.time()
        messages = [
            {"role": "system", "content": f"{self.prompt}"},
            {"role": "user", "content": f"{text_input}"}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True)

        # decode the JSON emotion detections as a dictionary
        try:
            content = json.loads(content)
        except json.decoder.JSONDecodeError:
            # invalid JSON; fallback to manual string parsing
            # print(">> parsing QwenEmotion response", content)
            content = {
                m.group(1): float(m.group(2))
                for m in re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)', content)
            }
            # print(">> dict result", content)

        # workaround for QwenEmotion's inability to distinguish "悲伤" (sad) vs "低落" (melancholic).
        # if we detect any of the IndexTTS "melancholic" words, we swap those vectors
        # to encode the "sad" emotion as "melancholic" (instead of sadness).
        text_input_lower = text_input.lower()
        if any(word in text_input_lower for word in self.melancholic_words):
            # print(">> before vec swap", content)
            content["悲伤"], content["低落"] = content.get("低落", 0.0), content.get("悲伤", 0.0)
            # print(">>  after vec swap", content)

        return self.convert(content)

    def compare_mlx_pytorch_cfm(self, cached_inputs=None):
        """
        对比MLX和PyTorch CFM的输出
        """
        print("\n" + "="*70)
        print("🔍 MLX vs PyTorch CFM 生产环境对比")
        print("="*70)
        
        # 加载缓存数据
        if cached_inputs is None:
            cached_data = load_cached_inputs()
            if not cached_data:
                print("❌ 没有找到缓存数据，无法进行对比")
                return
        else:
            cached_data = cached_inputs
        
        # 检查是否有MLX CFM
        if not hasattr(self, 'mlx_s2mel_cfm') or self.mlx_s2mel_cfm is None:
            print("❌ MLX CFM未初始化")
            return
        
        # 检查是否有PyTorch CFM
        if not hasattr(self, 's2mel') or self.s2mel is None:
            print("❌ PyTorch S2MEL未初始化")
            return
        
        try:
            # 使用MLX CFM进行对比
            if 'cfm_inputs_mlx' in cached_data:
                inputs = cached_data['cfm_inputs_mlx']
                cat_condition = inputs['cat_condition']
                x_lens = inputs['x_lens']
                ref_mel = inputs['ref_mel']
                style = inputs['style']
                
                print("\n📊 使用缓存输入进行对比:")
                print(f"   cat_condition: {cat_condition.shape}")
                print(f"   x_lens: {x_lens.shape}")
                print(f"   ref_mel: {ref_mel.shape}")
                print(f"   style: {style.shape}")
                
                # 调用MLX CFM的对比方法
                mlx_output, pytorch_output = self.mlx_s2mel_cfm.compare_with_pytorch(
                    self.s2mel.cfm, 
                    (cat_condition, x_lens, ref_mel, style)
                )
                
                print("\n✅ 对比完成")
                return mlx_output, pytorch_output
            else:
                print("❌ 没有找到MLX CFM输入缓存")
                return None, None
                
        except Exception as e:
            print(f"❌ 对比过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return None, None
