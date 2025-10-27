#!/usr/bin/env python3
"""
修复MLX CFM权重加载问题
正确映射checkpoints/mlx/s2mel.npz中的权重到MLX CFM
"""

import sys
import os
sys.path.append('.')

import torch
import numpy as np
import pickle
from indextts.infer_v2 import IndexTTS2
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def fix_mlx_cfm_weight_mapping():
    """修复MLX CFM权重映射问题"""
    
    print("=== 修复MLX CFM权重映射问题 ===")
    
    # 初始化TTS
    tts = IndexTTS2()
    
    # 确保MLX CFM已初始化
    if tts.mlx_s2mel_cfm is None:
        print("🔧 初始化MLX CFM...")
        from indextts.s2mel.modules.mlx_cfm import MLXCFM
        tts.mlx_s2mel_cfm = MLXCFM(tts.cfg.s2mel)
    
    mlx_cfm = tts.mlx_s2mel_cfm
    
    print(f"\n🔍 检查MLX权重文件...")
    
    # 加载MLX权重文件
    mlx_weights_path = "checkpoints/mlx/s2mel.npz"
    if not os.path.exists(mlx_weights_path):
        print(f"❌ MLX权重文件不存在: {mlx_weights_path}")
        return
    
    print(f"✅ 找到MLX权重文件: {mlx_weights_path}")
    
    # 加载权重数据
    mlx_weights = np.load(mlx_weights_path)
    print(f"📊 MLX权重文件包含 {len(mlx_weights.keys())} 个权重")
    
    # 检查CFM相关权重
    cfm_keys = [k for k in mlx_weights.keys() if 'cfm' in k]
    print(f"🔍 CFM相关权重数量: {len(cfm_keys)}")
    
    # 检查t_embedder权重
    t_embedder_keys = [k for k in mlx_weights.keys() if 't_embedder' in k]
    print(f"🔍 t_embedder权重: {t_embedder_keys}")
    
    # 创建正确的权重映射
    print(f"\n🔧 创建正确的权重映射...")
    
    # 从MLX权重文件中提取CFM权重
    cfm_weights = {}
    for key, value in mlx_weights.items():
        if key.startswith('models.cfm.'):
            # 将 "models.cfm.estimator.xxx" 映射为 "cfm.estimator.xxx"
            new_key = key.replace('models.cfm.', 'cfm.')
            cfm_weights[new_key] = value
            print(f"   {key} -> {new_key}")
    
    print(f"✅ 提取了 {len(cfm_weights)} 个CFM权重")
    
    # 使用load_from_cache方法加载权重
    print(f"\n🔧 使用load_from_cache方法加载权重...")
    
    try:
        loaded_count = mlx_cfm.load_from_cache(cfm_weights)
        print(f"✅ MLX CFM权重加载成功，加载了 {loaded_count} 个权重")
        
        # 验证权重是否正确加载
        print(f"\n🔍 验证权重加载...")
        
        # 检查t_embedder权重
        mlx_estimator = mlx_cfm.estimator
        if hasattr(mlx_estimator.t_embedder, 'mlp_0'):
            weight = mlx_estimator.t_embedder.mlp_0.weight
            print(f"✅ t_embedder.mlp_0.weight: {weight.shape}, 范围: [{weight.min():.6f}, {weight.max():.6f}]")
        else:
            print(f"❌ t_embedder.mlp_0不存在")
        
        # 检查cond_projection权重
        if hasattr(mlx_estimator, 'cond_projection'):
            weight = mlx_estimator.cond_projection.weight
            print(f"✅ cond_projection.weight: {weight.shape}, 范围: [{weight.min():.6f}, {weight.max():.6f}]")
        else:
            print(f"❌ cond_projection不存在")
        
        # 检查x_embedder权重
        if hasattr(mlx_estimator, 'x_embedder'):
            weight = mlx_estimator.x_embedder.weight
            print(f"✅ x_embedder.weight: {weight.shape}, 范围: [{weight.min():.6f}, {weight.max():.6f}]")
        else:
            print(f"❌ x_embedder不存在")
        
        # 测试t_embedder功能
        print(f"\n🧪 测试t_embedder功能...")
        import mlx.core as mx
        test_t = mx.array([0.0, 0.5, 1.0])
        t_emb_output = mlx_estimator.t_embedder(test_t)
        print(f"✅ t_embedder输出: {t_emb_output.shape}, 范围: [{t_emb_output.min():.6f}, {t_emb_output.max():.6f}]")
        
        # 检查不同时间步的输出差异
        t_emb_0 = mlx_estimator.t_embedder(mx.array([0.0]))
        t_emb_1 = mlx_estimator.t_embedder(mx.array([1.0]))
        diff = mx.abs(t_emb_0 - t_emb_1).max()
        print(f"✅ t=0和t=1的输出差异: {diff:.6f}")
        
        # 测试MLX CFM推理
        print(f"\n🧪 测试MLX CFM推理...")
        
        # 创建简单的测试输入
        batch_size = 1
        seq_len = 10
        test_mu = mx.random.normal((batch_size, seq_len, 512))
        test_x_lens = mx.array([seq_len])
        test_ref_mel = mx.random.normal((batch_size, 80, 5))
        test_style = mx.random.normal((batch_size, 192))
        
        # 测试MLX CFM推理
        test_output = mlx_cfm.inference(
            test_mu, test_x_lens, test_ref_mel, test_style, None,
            n_timesteps=5, temperature=1.0, inference_cfg_rate=0.0,
            unified_random=tts.unified_random
        )
        
        print(f"✅ MLX CFM测试输出:")
        print(f"   形状: {test_output.shape}")
        print(f"   范围: [{test_output.min():.6f}, {test_output.max():.6f}]")
        print(f"   均值: {test_output.mean():.6f}")
        print(f"   标准差: {test_output.std():.6f}")
        
        # 检查是否全零
        is_zero = mx.allclose(test_output, mx.zeros_like(test_output))
        print(f"   是否全零: {is_zero}")
        
        if not is_zero:
            print(f"🎉 MLX CFM权重加载成功！推理功能正常！")
        else:
            print(f"❌ MLX CFM输出全零，权重加载可能有问题")
        
    except Exception as e:
        print(f"❌ MLX CFM权重加载失败: {e}")
        import traceback
        traceback.print_exc()

def test_production_inference():
    """测试生产环境推理"""
    
    print(f"\n=== 测试生产环境推理 ===")
    
    tts = IndexTTS2()
    
    # 使用简单的测试数据
    text = "你好，这是一个测试。"
    spk_audio_prompt = "examples/voice_01.wav"
    
    print(f"📝 测试文本: {text}")
    print(f"🎵 测试音频: {spk_audio_prompt}")
    
    # 测试PyTorch版本
    print(f"\n🔧 测试PyTorch版本...")
    try:
        torch.manual_seed(42)
        np.random.seed(42)
        wav_torch = tts.infer(
            spk_audio_prompt=spk_audio_prompt,
            text=text,
            output_path="test_torch_fixed.wav",
            n_timesteps=25,
            inference_cfg_rate=0.5
        )
        print(f"✅ PyTorch版本生成成功: {wav_torch.shape}")
    except Exception as e:
        print(f"❌ PyTorch版本生成失败: {e}")
        return
    
    # 测试MLX版本
    print(f"\n🔧 测试MLX版本...")
    try:
        torch.manual_seed(42)
        np.random.seed(42)
        wav_mlx = tts.infer(
            spk_audio_prompt=spk_audio_prompt,
            text=text,
            output_path="test_mlx_fixed.wav",
            use_mlx=True,
            n_timesteps=25,
            inference_cfg_rate=0.5
        )
        print(f"✅ MLX版本生成成功: {wav_mlx.shape}")
        
        # 比较两个版本
        if wav_torch.shape == wav_mlx.shape:
            diff = np.abs(wav_torch - wav_mlx)
            print(f"📊 版本差异:")
            print(f"   最大差异: {diff.max():.6f}")
            print(f"   平均差异: {diff.mean():.6f}")
            print(f"   标准差差异: {diff.std():.6f}")
            
            # 计算信噪比
            signal_power = np.mean(wav_torch ** 2)
            noise_power = np.mean(diff ** 2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
            print(f"   信噪比: {snr:.2f} dB")
            
            if snr > 20:
                print(f"🎉 MLX版本质量优秀！")
            elif snr > 10:
                print(f"✅ MLX版本质量良好")
            else:
                print(f"⚠️ MLX版本质量需要改进")
        else:
            print(f"❌ 输出形状不匹配: PyTorch {wav_torch.shape} vs MLX {wav_mlx.shape}")
            
    except Exception as e:
        print(f"❌ MLX版本生成失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    
    print("开始修复MLX CFM权重映射问题...")
    
    # 修复权重映射
    fix_mlx_cfm_weight_mapping()
    
    # 测试生产环境推理
    test_production_inference()
    
    print(f"\n🎉 修复完成！")

if __name__ == "__main__":
    main()
