#!/usr/bin/env python3
"""
CFM 推理一致性验证工具
集成到 IndexTTS2 推理流程中，用于验证 PyTorch 和 MLX CFM 的一致性
"""

import os
import sys
import pickle
import numpy as np
import torch
import mlx.core as mx
from typing import Dict, Any, Optional
import time

# 添加项目路径
sys.path.insert(0, '.')

from cfm_data_cache import CFMDataCache
from indextts.infer_v2 import IndexTTS2


class CFMConsistencyValidator:
    """CFM 一致性验证器"""
    
    def __init__(self, cache_dir: str = "cfm_consistency_cache"):
        self.cache = CFMDataCache(cache_dir)
        self.torch_inputs_cache = None
        self.mlx_inputs_cache = None
    
    def cache_torch_cfm_inputs(self, mu, x_lens, prompt, style, f0, 
                              n_timesteps, temperature, inference_cfg_rate,
                              unified_random=None):
        """
        缓存 PyTorch CFM 输入数据
        
        Args:
            mu, x_lens, prompt, style, f0: CFM 输入参数
            n_timesteps, temperature, inference_cfg_rate: CFM 参数
            unified_random: 统一随机数生成器
        """
        torch_inputs = {
            'mu': mu,
            'x_lens': x_lens,
            'prompt': prompt,
            'style': style,
            'f0': f0,
            'n_timesteps': n_timesteps,
            'temperature': temperature,
            'inference_cfg_rate': inference_cfg_rate,
            'unified_random_seed': unified_random.seed if unified_random else None,
            'timestamp': time.time()
        }
        
        filename = f"cfm_torch_inputs_{int(time.time())}.pkl"
        self.torch_inputs_cache = self.cache.save_torch_inputs(torch_inputs, filename)
        
        print(f"🔍 Cached PyTorch CFM inputs: {filename}")
        return torch_inputs
    
    def cache_mlx_cfm_inputs(self, mu, x_lens, prompt, style, f0,
                            n_timesteps, temperature, inference_cfg_rate,
                            unified_random=None):
        """
        缓存 MLX CFM 输入数据
        
        Args:
            mu, x_lens, prompt, style, f0: CFM 输入参数 (MLX 格式)
            n_timesteps, temperature, inference_cfg_rate: CFM 参数
            unified_random: 统一随机数生成器
        """
        mlx_inputs = {
            'mu': mu,
            'x_lens': x_lens,
            'prompt': prompt,
            'style': style,
            'f0': f0,
            'n_timesteps': n_timesteps,
            'temperature': temperature,
            'inference_cfg_rate': inference_cfg_rate,
            'unified_random_seed': unified_random.seed if unified_random else None,
            'timestamp': time.time()
        }
        
        filename = f"cfm_mlx_inputs_{int(time.time())}.pkl"
        self.mlx_inputs_cache = self.cache.save_mlx_inputs(mlx_inputs, filename)
        
        print(f"🔍 Cached MLX CFM inputs: {filename}")
        return mlx_inputs
    
    def validate_cfm_consistency(self, torch_output: torch.Tensor, mlx_output: mx.array,
                              tolerance: float = 1e-6) -> bool:
        """
        验证 PyTorch 和 MLX CFM 输出的一致性
        
        Args:
            torch_output: PyTorch CFM 输出
            mlx_output: MLX CFM 输出
            tolerance: 数值容差
            
        Returns:
            是否一致
        """
        print("\n🔍 CFM Consistency Validation")
        print("=" * 40)
        
        # 比较输出
        is_consistent = self.cache.compare_outputs(
            torch_output, mlx_output, tolerance, "CFM Output"
        )
        
        if is_consistent:
            print("✅ CFM PyTorch and MLX outputs are consistent!")
        else:
            print("❌ CFM PyTorch and MLX outputs are inconsistent!")
            print("🔧 Check the following:")
            print("   - Random seed initialization")
            print("   - Weight loading")
            print("   - Numerical precision differences")
            print("   - Algorithm implementation differences")
        
        return is_consistent
    
    def run_consistency_test(self, text: str, spk_audio_prompt: str, 
                           output_dir: str = "cfm_test_outputs",
                           seed: int = 42):
        """
        运行完整的 CFM 一致性测试
        
        Args:
            text: 要合成的文本
            spk_audio_prompt: 说话人音频提示
            output_dir: 输出目录
            seed: 随机种子
        """
        print(f"🧪 Running CFM Consistency Test")
        print(f"   Text: {text}")
        print(f"   Speaker: {spk_audio_prompt}")
        print(f"   Seed: {seed}")
        print("=" * 50)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建 PyTorch 模型
        print("\n1. Creating PyTorch model...")
        torch_model = IndexTTS2(use_mlx=False)
        
        # 创建 MLX 模型
        print("\n2. Creating MLX model...")
        mlx_model = IndexTTS2(use_mlx=True)
        
        # 设置相同的随机种子
        print(f"\n3. Setting random seed: {seed}")
        torch_model.unified_random.reset_seed(seed)
        mlx_model.unified_random.reset_seed(seed)
        
        # 运行 PyTorch 推理
        print("\n4. Running PyTorch inference...")
        torch_output_path = os.path.join(output_dir, "torch_output.wav")
        
        # 这里需要修改 infer 方法来支持缓存 CFM 输入
        # 暂时使用现有的推理方法
        try:
            torch_model.infer(
                spk_audio_prompt=spk_audio_prompt,
                text=text,
                output_path=torch_output_path,
                seed=seed
            )
            print("✅ PyTorch inference completed")
        except Exception as e:
            print(f"❌ PyTorch inference failed: {e}")
            return False
        
        # 运行 MLX 推理
        print("\n5. Running MLX inference...")
        mlx_output_path = os.path.join(output_dir, "mlx_output.wav")
        
        try:
            mlx_model.infer(
                spk_audio_prompt=spk_audio_prompt,
                text=text,
                output_path=mlx_output_path,
                seed=seed
            )
            print("✅ MLX inference completed")
        except Exception as e:
            print(f"❌ MLX inference failed: {e}")
            return False
        
        # 比较输出音频文件
        print("\n6. Comparing output files...")
        try:
            import librosa
            
            # 加载音频文件
            torch_audio, _ = librosa.load(torch_output_path, sr=22050)
            mlx_audio, _ = librosa.load(mlx_output_path, sr=22050)
            
            # 比较音频长度
            if len(torch_audio) != len(mlx_audio):
                print(f"⚠️  Audio length mismatch:")
                print(f"   PyTorch: {len(torch_audio)} samples")
                print(f"   MLX:     {len(mlx_audio)} samples")
                # 截断到较短的长度
                min_len = min(len(torch_audio), len(mlx_audio))
                torch_audio = torch_audio[:min_len]
                mlx_audio = mlx_audio[:min_len]
            
            # 计算音频差异
            audio_diff = np.abs(torch_audio - mlx_audio)
            max_diff = np.max(audio_diff)
            mean_diff = np.mean(audio_diff)
            
            print(f"🔍 Audio comparison:")
            print(f"   Max difference:  {max_diff:.8f}")
            print(f"   Mean difference: {mean_diff:.8f}")
            
            # 音频容差通常更大
            audio_tolerance = 1e-3
            if max_diff < audio_tolerance:
                print("✅ Audio outputs are consistent!")
                return True
            else:
                print("❌ Audio outputs are inconsistent!")
                return False
                
        except Exception as e:
            print(f"❌ Audio comparison failed: {e}")
            return False


def patch_cfm_methods():
    """
    为 IndexTTS2 类添加 CFM 输入缓存功能
    """
    from indextts.s2mel.modules.flow_matching import BASECFM
    from indextts.s2mel.modules.mlx_flow_matching import MLXCFM
    
    # 保存原始方法
    original_torch_inference = BASECFM.inference
    original_mlx_inference = MLXCFM.inference
    
    # 创建全局验证器
    validator = CFMConsistencyValidator()
    
    def patched_torch_inference(self, mu, x_lens, prompt, style, f0, 
                               n_timesteps, temperature=1.0, 
                               inference_cfg_rate=0.5, unified_random=None):
        """带缓存的 PyTorch CFM inference"""
        # 缓存输入
        validator.cache_torch_cfm_inputs(
            mu, x_lens, prompt, style, f0, n_timesteps, 
            temperature, inference_cfg_rate, unified_random
        )
        
        # 调用原始方法
        return original_torch_inference(
            self, mu, x_lens, prompt, style, f0, n_timesteps,
            temperature, inference_cfg_rate, unified_random
        )
    
    def patched_mlx_inference(self, mu, x_lens, prompt, style, f0,
                             n_timesteps, temperature=1.0,
                             inference_cfg_rate=0.5, unified_random=None):
        """带缓存的 MLX CFM inference"""
        # 缓存输入
        validator.cache_mlx_cfm_inputs(
            mu, x_lens, prompt, style, f0, n_timesteps,
            temperature, inference_cfg_rate, unified_random
        )
        
        # 调用原始方法
        return original_mlx_inference(
            self, mu, x_lens, prompt, style, f0, n_timesteps,
            temperature, inference_cfg_rate, unified_random
        )
    
    # 应用补丁
    BASECFM.inference = patched_torch_inference
    MLXCFM.inference = patched_mlx_inference
    
    print("✅ CFM methods patched for input caching")
    return validator


def main():
    """主函数"""
    print("🔧 CFM Consistency Validator")
    print("=" * 40)
    
    # 创建验证器
    validator = CFMConsistencyValidator()
    
    # 运行测试
    test_text = "Hello, this is a test for CFM consistency validation."
    test_speaker = "examples/speaker.wav"  # 需要实际的音频文件
    
    # 检查测试文件是否存在
    if not os.path.exists(test_speaker):
        print(f"⚠️  Test speaker file not found: {test_speaker}")
        print("Please provide a valid speaker audio file.")
        return
    
    success = validator.run_consistency_test(
        text=test_text,
        spk_audio_prompt=test_speaker,
        seed=42
    )
    
    if success:
        print("\n🎉 CFM consistency test passed!")
    else:
        print("\n❌ CFM consistency test failed!")
        print("Check the cached inputs for debugging.")


if __name__ == "__main__":
    main()
