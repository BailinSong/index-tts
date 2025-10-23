"""
缓存真实S2MEL输入数据用于一致性测试
"""

import torch
import numpy as np
import pickle
import os
import sys
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(__file__))

from indextts.infer_v2 import IndexTTS2

def cache_s2mel_inputs():
    """执行一次真实推理，缓存S2MEL的输入数据"""
    
    print("="*80)
    print("缓存真实S2MEL输入数据")
    print("="*80)
    
    # 创建TTS实例（使用PyTorch版本）
    print("创建TTS实例...")
    tts = IndexTTS2(use_mlx=False)  # 使用PyTorch版本
    
    # 准备测试数据
    spk_audio_prompt = 'examples/zh_vo_Main_Linaxita_2_4_24_6.wav'
    text = '测试语音合成'
    
    print(f"使用音频: {spk_audio_prompt}")
    print(f"文本: {text}")
    
    # 执行推理并捕获S2MEL输入
    print("\n执行推理并捕获S2MEL输入...")
    
    # 直接检查_s2mel_inputs属性
    print("检查TTS实例的_s2mel_inputs属性...")
    
    # 执行推理
    try:
        result = tts.infer(
            spk_audio_prompt=spk_audio_prompt,
            text=text,
            output_path='cache_test.wav',
            num_beams=1,
            seed=42
        )
        print("✅ 推理完成")
        
        # 检查_s2mel_inputs属性
        if hasattr(tts, '_s2mel_inputs') and tts._s2mel_inputs is not None:
            s2mel_inputs = tts._s2mel_inputs
            print(f"\n捕获到S2MEL输入:")
            for key, value in s2mel_inputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"  {key}: {type(value)} = {value}")
            
            # 保存到文件
            cache_file = 's2mel_inputs_cache.pkl'
            with open(cache_file, 'wb') as f:
                pickle.dump(s2mel_inputs, f)
            print(f"\nS2MEL输入已保存到: {cache_file}")
        else:
            print("❌ 未找到_s2mel_inputs属性")
            
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    cache_s2mel_inputs()
