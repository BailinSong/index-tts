#!/usr/bin/env python3
"""
测试生产环境中的CFM权重加载调试信息
"""

import sys
import os
sys.path.append('.')

import torch
import numpy as np
from indextts.infer_v2 import IndexTTS2

def test_production_cfm_weight_loading():
    """测试生产环境中的CFM权重加载"""
    
    print("=== 测试生产环境中的CFM权重加载调试信息 ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 初始化TTS（这会触发CFM权重加载）
    print("🔧 初始化IndexTTS2...")
    tts = IndexTTS2(use_mlx=True)
    
    print(f"\n📊 CFM初始化状态:")
    print(f"  MLX CFM是否已初始化: {tts.mlx_s2mel_cfm is not None}")
    
    if tts.mlx_s2mel_cfm is not None:
        print(f"  MLX CFM类型: {type(tts.mlx_s2mel_cfm)}")
        
        # 检查CFM estimator的权重状态
        estimator = tts.mlx_s2mel_cfm.estimator
        print(f"  Estimator类型: {type(estimator)}")
        
        # 检查关键权重
        if hasattr(estimator, 't_embedder'):
            print(f"  t_embedder存在: True")
            if hasattr(estimator.t_embedder, 'mlp_0'):
                t_weight = estimator.t_embedder.mlp_0.weight
                print(f"  t_embedder.mlp_0.weight: {t_weight.shape}, 范围: [{t_weight.min():.6f}, {t_weight.max():.6f}]")
            else:
                print(f"  t_embedder.mlp_0: False")
        else:
            print(f"  t_embedder存在: False")
        
        if hasattr(estimator, 'cond_projection'):
            cond_weight = estimator.cond_projection.weight
            print(f"  cond_projection.weight: {cond_weight.shape}, 范围: [{cond_weight.min():.6f}, {cond_weight.max():.6f}]")
        else:
            print(f"  cond_projection存在: False")
    
    # 测试简单的推理
    print(f"\n🔧 测试简单推理...")
    text = "你好，这是一个测试。"
    spk_audio_prompt = "examples/voice_01.wav"
    
    try:
        wav = tts.infer(
            spk_audio_prompt=spk_audio_prompt,
            text=text,
            output_path="test_production_cfm.wav",
            use_mlx=True,
            n_timesteps=25,
            inference_cfg_rate=0.7
        )
        print(f"✅ 推理成功: {wav.shape}")
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    
    print("开始测试生产环境中的CFM权重加载调试信息...")
    
    test_production_cfm_weight_loading()
    
    print(f"\n🎉 测试完成！")

if __name__ == "__main__":
    main()
