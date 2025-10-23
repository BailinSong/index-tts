#!/usr/bin/env python3
"""
调试 CFM PyTorch 和 MLX 实现的差异
"""

import sys
sys.path.append('.')
import torch
import mlx.core as mx
import numpy as np
import os
from indextts.infer_v2 import IndexTTS2

def debug_cfm_implementations():
    """调试 CFM 实现的差异"""
    
    print("=== CFM 实现差异调试 ===\n")
    
    # 设置固定种子
    torch.manual_seed(42)
    np.random.seed(42)
    mx.random.seed(42)
    
    print(f"固定种子: 42")
    print(f"PyTorch 种子: {torch.initial_seed()}")
    print(f"MLX 种子: 42")
    
    # 初始化 MLX 版本
    print("\n1. 初始化 MLX 版本...")
    tts_mlx = IndexTTS2(
        cfg_path='checkpoints/config.yaml',
        model_dir='checkpoints',
        use_mlx=True,
        device='mps'
    )
    
    # 准备测试数据
    test_text = '今天天气真不错'
    test_voice = 'examples/zh_vo_Main_Linaxita_2_4_24_6.wav'
    
    print(f"\n2. 测试数据: {test_text}")
    
    # 重新设置种子确保一致性
    torch.manual_seed(42)
    np.random.seed(42)
    mx.random.seed(42)
    
    print("\n3. 开始推理并对比中间输出...")
    
    try:
        # 进行推理
        result = tts_mlx.infer(
            spk_audio_prompt=test_voice,
            text=test_text,
            output_path='debug_cfm_output.wav',
            seed=42
        )
        
        print("\n✅ 推理完成")
        
        # 检查输出文件
        if os.path.exists('debug_cfm_output.wav'):
            import librosa
            audio, sr = librosa.load('debug_cfm_output.wav', sr=None)
            print(f"\n=== 输出音频信息 ===")
            print(f"文件: debug_cfm_output.wav")
            print(f"采样率: {sr}Hz")
            print(f"长度: {len(audio)} 样本")
            print(f"时长: {len(audio)/sr:.2f}s")
            print(f"范围: [{audio.min():.6f}, {audio.max():.6f}]")
            print(f"均值: {audio.mean():.6f}")
            print(f"标准差: {audio.std():.6f}")
            
            # 检查音频质量
            if np.abs(audio).max() > 0.01:
                print("✅ 音频有正常音量")
            else:
                print("❌ 音频音量过低")
                
            if np.std(audio) > 0.01:
                print("✅ 音频有正常变化")
            else:
                print("❌ 音频变化较小")
        else:
            print("❌ 输出文件不存在")
            
    except Exception as e:
        print(f"❌ 推理过程出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 调试文件检查 ===")
    
    # 检查调试文件
    debug_files = [
        'cfm_inputs_mlx.pkl',
        'cfm_outputs_pytorch.pkl',
        'gpt_outputs_mlx.pkl'
    ]
    
    for file in debug_files:
        if os.path.exists(file):
            print(f"✅ {file} 存在")
            try:
                import pickle
                with open(file, 'rb') as f:
                    data = pickle.load(f)
                print(f"   内容: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            except Exception as e:
                print(f"   读取失败: {e}")
        else:
            print(f"❌ {file} 不存在")
    
    print("\n=== 关键差异分析 ===")
    print("1. 随机数生成: PyTorch 和 MLX 的随机数生成器可能产生不同结果")
    print("2. 数据类型转换: MLX 需要多次转换，可能导致精度损失")
    print("3. 数值计算: MLX 和 PyTorch 在某些数值计算上可能有微小差异")
    print("4. 设备差异: 不同设备上的计算可能有微小差异")
    
    print("\n=== 建议修复方案 ===")
    print("1. 统一随机数生成策略")
    print("2. 减少数据类型转换")
    print("3. 确保数值计算精度一致性")
    print("4. 添加详细的调试输出来对比中间结果")

if __name__ == "__main__":
    debug_cfm_implementations()
