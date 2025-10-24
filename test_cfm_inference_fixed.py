#!/usr/bin/env python3
"""
测试修复后的CFM推理
"""

import sys
sys.path.append('.')
import torch
import mlx.core as mx
import numpy as np
import os
from indextts.infer_v2 import IndexTTS2
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def test_cfm_inference_fixed():
    """测试修复后的CFM推理"""
    
    print('=== 测试修复后的CFM推理 ===')
    
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
    
    print('\\n=== 步骤1: 测试CFM推理 ===')
    
    # 准备测试数据
    batch_size = 1
    seq_len = 100
    in_channels = 80
    
    # 创建测试输入 - 确保所有输入形状一致
    seq_len = 80  # 使用更小的序列长度
    z = torch.randn(batch_size, in_channels, seq_len, device=tts_mlx.device)
    x_lens = torch.tensor([seq_len], device=tts_mlx.device)
    prompt = torch.randn(batch_size, in_channels, seq_len, device=tts_mlx.device)
    mu = torch.randn(batch_size, in_channels, seq_len, device=tts_mlx.device)
    style = torch.randn(batch_size, 512, device=tts_mlx.device)
    f0 = torch.randn(batch_size, seq_len, device=tts_mlx.device)
    
    print(f'测试输入形状:')
    print(f'  z: {z.shape}')
    print(f'  x_lens: {x_lens.shape}')
    print(f'  prompt: {prompt.shape}')
    print(f'  mu: {mu.shape}')
    print(f'  style: {style.shape}')
    print(f'  f0: {f0.shape}')
    
    # 测试PyTorch CFM推理
    print(f'\\n--- 测试PyTorch CFM推理 ---')
    try:
        pytorch_output = tts_mlx.s2mel.models['cfm'].inference(
            mu, x_lens, prompt, style, f0, n_timesteps=25,
            unified_random=tts_mlx.unified_random
        )
        print(f'PyTorch CFM 输出形状: {pytorch_output.shape}')
        print(f'PyTorch CFM 输出范围: [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]')
        pytorch_success = True
    except Exception as e:
        print(f'❌ PyTorch CFM推理失败: {e}')
        pytorch_success = False
    
    # 测试MLX CFM推理
    print(f'\\n--- 测试MLX CFM推理 ---')
    try:
        mlx_output = tts_mlx.mlx_s2mel_cfm.inference(
            mu, x_lens, prompt, style, f0, n_timesteps=25,
            unified_random=tts_mlx.unified_random
        )
        print(f'MLX CFM 输出形状: {mlx_output.shape}')
        print(f'MLX CFM 输出范围: [{mlx_output.min():.6f}, {mlx_output.max():.6f}]')
        mlx_success = True
    except Exception as e:
        print(f'❌ MLX CFM推理失败: {e}')
        mlx_success = False
    
    # 对比输出
    if pytorch_success and mlx_success:
        print(f'\\n--- 输出对比 ---')
        if pytorch_output.shape == mlx_output.shape:
            output_diff = torch.abs(pytorch_output - mlx_output)
            max_diff = torch.max(output_diff).item()
            mean_diff = torch.mean(output_diff).item()
            
            print(f'输出差异: 最大 {max_diff:.6f}, 平均 {mean_diff:.6f}')
            
            if max_diff < 1e-3:
                print(f'✅ CFM推理输出基本一致')
            elif max_diff < 1e-1:
                print(f'⚠️ CFM推理输出有差异但可接受')
            else:
                print(f'❌ CFM推理输出差异较大')
        else:
            print(f'❌ CFM推理输出形状不匹配')
    
    print(f'\\n=== 步骤2: 测试完整推理流程 ===')
    
    # 测试完整推理流程
    print(f'\\n--- 测试完整推理流程 ---')
    try:
        # 准备文本和音频输入
        text = "你好，这是一个测试。"
        ref_audio = "examples/ref_audio.wav"
        
        if os.path.exists(ref_audio):
            print(f'使用参考音频: {ref_audio}')
            
            # 执行完整推理
            output_path = "test_cfm_fixed_output.wav"
            tts_mlx.infer(
                text=text,
                ref_audio=ref_audio,
                output_path=output_path,
                seed=42
            )
            
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f'✅ 完整推理成功，输出文件: {output_path}')
                print(f'文件大小: {file_size / 1024:.2f} KB')
            else:
                print(f'❌ 输出文件未生成')
        else:
            print(f'⚠️ 参考音频文件不存在: {ref_audio}')
            print(f'跳过完整推理测试')
    
    except Exception as e:
        print(f'❌ 完整推理失败: {e}')
        import traceback
        traceback.print_exc()
    
    print(f'\\n=== 测试完成 ===')

if __name__ == "__main__":
    test_cfm_inference_fixed()
