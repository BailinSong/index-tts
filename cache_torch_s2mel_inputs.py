"""
缓存PyTorch版本的S2MEL输入数据
"""

import torch
import random
import mlx.core as mx
import pickle
import os

def cache_torch_s2mel_inputs():
    """缓存PyTorch版本的S2MEL输入数据"""
    
    print("="*80)
    print("缓存PyTorch版本的S2MEL输入数据")
    print("="*80)
    
    # 固定随机种子
    torch.manual_seed(42)
    random.seed(42)
    mx.random.seed(42)
    
    # 运行PyTorch版本并缓存S2MEL输入
    print("\n1. 运行PyTorch版本并缓存S2MEL输入...")
    from indextts.infer_v2_torch import IndexTTS2
    
    tts_torch = IndexTTS2()
    
    # 在推理过程中捕获S2MEL输入
    print("PyTorch版本推理开始...")
    result_torch = tts_torch.infer(
        spk_audio_prompt='examples/zh_vo_Main_Linaxita_2_4_24_6.wav',
        text='今天天气很好',
        output_path='test_torch_cache.wav',
        num_beams=1
    )
    print('PyTorch版本推理完成')
    
    # 检查是否成功缓存了S2MEL输入
    if hasattr(tts_torch, '_s2mel_inputs') and tts_torch._s2mel_inputs is not None:
        print("\n2. 保存S2MEL输入缓存...")
        
        # 保存S2MEL输入到文件
        s2mel_inputs = tts_torch._s2mel_inputs
        with open('s2mel_inputs_cache.pkl', 'wb') as f:
            pickle.dump(s2mel_inputs, f)
        
        print("S2MEL输入缓存内容:")
        for key, value in s2mel_inputs.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}, min={value.min():.6f}, max={value.max():.6f}")
            else:
                print(f"  {key}: {value}")
        
        print(f"\n✅ S2MEL输入已保存到 s2mel_inputs_cache.pkl")
        
    else:
        print("❌ 未找到S2MEL输入缓存")
        print("需要在PyTorch版本中添加S2MEL输入捕获逻辑")
    
    print(f"\n{'='*80}")
    print("S2MEL输入缓存完成")
    print(f"{'='*80}")

if __name__ == "__main__":
    cache_torch_s2mel_inputs()

