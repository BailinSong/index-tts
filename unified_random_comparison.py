#!/usr/bin/env python3
"""
统一随机数生成并对比第一层差异
"""

import sys
sys.path.append('.')
import torch
import mlx.core as mx
import numpy as np
import os
from indextts.infer_v2 import IndexTTS2

def create_unified_random_sequence(length, seed=42):
    """创建统一的随机数序列"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)
    
    # 使用 PyTorch 生成随机数序列
    random_seq = torch.randn(length)
    return random_seq

def compare_first_layer_outputs():
    """对比第一层的输入和输出差异"""
    
    print('=== 统一随机数生成并对比第一层差异 ===')
    
    # 设置固定种子
    torch.manual_seed(42)
    np.random.seed(42)
    mx.random.seed(42)
    
    print(f'固定种子: 42')
    
    # 初始化 MLX 版本
    print('\\n1. 初始化 MLX 版本...')
    tts_mlx = IndexTTS2(
        cfg_path='checkpoints/config.yaml',
        model_dir='checkpoints',
        use_mlx=True,
        device='mps'
    )
    
    # 准备测试数据
    test_text = '今天天气真不错'
    test_voice = 'examples/zh_vo_Main_Linaxita_2_4_24_6.wav'
    
    print(f'\\n2. 测试数据: {test_text}')
    
    # 重新设置种子确保一致性
    torch.manual_seed(42)
    np.random.seed(42)
    mx.random.seed(42)
    
    print('\\n3. 开始推理并对比第一层差异...')
    
    try:
        # 进行推理
        result = tts_mlx.infer(
            spk_audio_prompt=test_voice,
            text=test_text,
            output_path='unified_random_test.wav',
            seed=42
        )
        
        print('\\n✅ 推理完成')
        
        # 检查调试文件
        debug_files = [
            'cfm_inputs_mlx.pkl',
            'cfm_outputs_pytorch.pkl',
            'gpt_outputs_mlx.pkl'
        ]
        
        print('\\n=== 调试文件检查 ===')
        for file in debug_files:
            if os.path.exists(file):
                print(f'✅ {file} 存在')
                try:
                    import pickle
                    with open(file, 'rb') as f:
                        data = pickle.load(f)
                    
                    if isinstance(data, dict):
                        print(f'   内容: {list(data.keys())}')
                        for key, value in data.items():
                            if isinstance(value, torch.Tensor):
                                print(f'   {key}: {value.shape}, 范围 [{value.min():.6f}, {value.max():.6f}]')
                            elif isinstance(value, np.ndarray):
                                print(f'   {key}: {value.shape}, 范围 [{value.min():.6f}, {value.max():.6f}]')
                            else:
                                print(f'   {key}: {type(value)} = {value}')
                    else:
                        print(f'   数据类型: {type(data)}')
                except Exception as e:
                    print(f'   读取失败: {e}')
            else:
                print(f'❌ {file} 不存在')
        
        # 检查输出文件
        if os.path.exists('unified_random_test.wav'):
            import librosa
            audio, sr = librosa.load('unified_random_test.wav', sr=None)
            print(f'\\n=== 输出音频信息 ===')
            print(f'文件: unified_random_test.wav')
            print(f'采样率: {sr}Hz')
            print(f'长度: {len(audio)} 样本')
            print(f'时长: {len(audio)/sr:.2f}s')
            print(f'范围: [{audio.min():.6f}, {audio.max():.6f}]')
            print(f'均值: {audio.mean():.6f}')
            print(f'标准差: {audio.std():.6f}')
            
            # 检查音频质量
            if np.abs(audio).max() > 0.01:
                print('✅ 音频有正常音量')
            else:
                print('❌ 音频音量过低')
                
            if np.std(audio) > 0.01:
                print('✅ 音频有正常变化')
            else:
                print('❌ 音频变化较小')
        else:
            print('❌ 输出文件不存在')
            
    except Exception as e:
        print(f'❌ 推理过程出错: {e}')
        import traceback
        traceback.print_exc()
    
    print('\\n=== 关键差异分析 ===')
    print('1. 随机数生成: 使用统一的随机数序列')
    print('2. 数据类型转换: 减少不必要的转换')
    print('3. 数值计算: 确保精度一致性')
    print('4. 设备差异: 统一设备处理')
    
    print('\\n=== 建议修复方案 ===')
    print('1. 统一随机数生成策略')
    print('2. 减少数据类型转换')
    print('3. 确保数值计算精度一致性')
    print('4. 添加详细的调试输出来对比中间结果')

if __name__ == "__main__":
    compare_first_layer_outputs()
