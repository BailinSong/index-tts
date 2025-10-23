#!/usr/bin/env python3
"""
使用前级 input 缓存测试 PyTorch CFM 和 MLX CFM 的差异
"""

import sys
sys.path.append('.')
import torch
import mlx.core as mx
import numpy as np
import os
import pickle
import librosa
from indextts.infer_v2 import IndexTTS2
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def test_cfm_with_cached_inputs():
    """使用前级 input 缓存测试 PyTorch CFM 和 MLX CFM 的差异"""
    
    print('=== 使用前级 input 缓存测试 CFM 差异 ===')
    
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
    
    print('\\n3. 开始 MLX 推理以获取缓存输入...')
    
    try:
        # 进行 MLX 推理以获取缓存输入
        result_mlx = tts_mlx.infer(
            spk_audio_prompt=test_voice,
            text=test_text,
            output_path='mlx_cached_inputs.wav',
            seed=42
        )
        
        print('\\n✅ MLX 推理完成，获取缓存输入')
        
        # 检查是否有缓存输入文件
        if os.path.exists('cfm_inputs_mlx.pkl'):
            print('\\n=== 加载缓存输入数据 ===')
            with open('cfm_inputs_mlx.pkl', 'rb') as f:
                cached_inputs = pickle.load(f)
            
            print('缓存输入数据:')
            for key, value in cached_inputs.items():
                if isinstance(value, torch.Tensor):
                    print(f'  {key}: {value.shape}, 范围 [{value.min():.6f}, {value.max():.6f}]')
                else:
                    print(f'  {key}: {type(value)} = {value}')
            
            # 提取输入数据
            cat_condition = cached_inputs['cat_condition']
            x_lens = cached_inputs['x_lens']
            ref_mel = cached_inputs['ref_mel']
            style = cached_inputs['style']
            diffusion_steps = cached_inputs['diffusion_steps']
            inference_cfg_rate = cached_inputs['inference_cfg_rate']
            
            print(f'\\n=== 使用相同输入测试 PyTorch CFM ===')
            
            # 重新设置种子确保一致性
            torch.manual_seed(42)
            np.random.seed(42)
            mx.random.seed(42)
            
            # 使用 PyTorch CFM 进行推理
            print('使用 PyTorch CFM 推理...')
            # 确保所有输入都在正确的设备上
            cat_condition_device = cat_condition.to(tts_mlx.device)
            ref_mel_device = ref_mel.to(tts_mlx.device)
            style_device = style.to(tts_mlx.device)
            
            vc_target_pytorch = tts_mlx.s2mel.models['cfm'].inference(
                cat_condition_device,
                torch.LongTensor([cat_condition_device.size(1)]).to(tts_mlx.device),
                ref_mel_device, style_device, None, diffusion_steps,
                inference_cfg_rate=inference_cfg_rate
            )
            
            print(f'PyTorch CFM 输出:')
            print(f'  形状: {vc_target_pytorch.shape}')
            print(f'  范围: [{vc_target_pytorch.min():.6f}, {vc_target_pytorch.max():.6f}]')
            print(f'  均值: {vc_target_pytorch.mean():.6f}')
            print(f'  标准差: {vc_target_pytorch.std():.6f}')
            
            # 保存 PyTorch CFM 输出
            pytorch_outputs = {
                'vc_target': vc_target_pytorch.detach().cpu()
            }
            with open('cfm_outputs_pytorch_cached.pkl', 'wb') as f:
                pickle.dump(pytorch_outputs, f)
            print('PyTorch CFM 输出已保存到 cfm_outputs_pytorch_cached.pkl')
            
            print(f'\\n=== 使用相同输入测试 MLX CFM ===')
            
            # 重新设置种子确保一致性
            torch.manual_seed(42)
            np.random.seed(42)
            mx.random.seed(42)
            
            # 将输入转换为 MLX
            cat_condition_mlx = torch_to_mlx(cat_condition.cpu())
            x_lens_mlx = torch_to_mlx(x_lens.cpu())
            ref_mel_mlx = torch_to_mlx(ref_mel.cpu())
            style_mlx = torch_to_mlx(style.cpu())
            
            print('使用 MLX CFM 推理...')
            vc_target_mlx = tts_mlx.mlx_s2mel_cfm.inference(
                cat_condition_mlx,
                x_lens_mlx,
                ref_mel_mlx,
                style_mlx,
                None,
                diffusion_steps,
                inference_cfg_rate=inference_cfg_rate
            )
            
            print(f'MLX CFM 输出:')
            print(f'  形状: {vc_target_mlx.shape}')
            print(f'  范围: [{vc_target_mlx.min():.6f}, {vc_target_mlx.max():.6f}]')
            print(f'  均值: {vc_target_mlx.mean():.6f}')
            print(f'  标准差: {vc_target_mlx.std():.6f}')
            
            # 转换为 PyTorch 进行对比
            vc_target_mlx_torch = mlx_to_torch(vc_target_mlx).to(tts_mlx.device)
            
            print(f'转换后的 MLX CFM 输出:')
            print(f'  形状: {vc_target_mlx_torch.shape}')
            print(f'  范围: [{vc_target_mlx_torch.min():.6f}, {vc_target_mlx_torch.max():.6f}]')
            print(f'  均值: {vc_target_mlx_torch.mean():.6f}')
            print(f'  标准差: {vc_target_mlx_torch.std():.6f}')
            
            # 保存 MLX CFM 输出
            mlx_outputs = {
                'vc_target': vc_target_mlx_torch.detach().cpu()
            }
            with open('cfm_outputs_mlx_cached.pkl', 'wb') as f:
                pickle.dump(mlx_outputs, f)
            print('MLX CFM 输出已保存到 cfm_outputs_mlx_cached.pkl')
            
            print(f'\\n=== 对比 PyTorch CFM 和 MLX CFM 输出 ===')
            
            # 确保形状一致
            if vc_target_pytorch.shape == vc_target_mlx_torch.shape:
                # 计算差异
                diff = torch.abs(vc_target_pytorch - vc_target_mlx_torch)
                max_diff = torch.max(diff).item()
                mean_diff = torch.mean(diff).item()
                rms_diff = torch.sqrt(torch.mean(diff**2)).item()
                
                print(f'输出形状: {vc_target_pytorch.shape}')
                print(f'最大差异: {max_diff:.2e}')
                print(f'平均差异: {mean_diff:.2e}')
                print(f'RMS 差异: {rms_diff:.2e}')
                
                # 检查是否超过阈值
                threshold = 1e-5
                if max_diff > threshold:
                    print(f'\\n❌ 差异大于阈值 {threshold:.0e}')
                    print(f'最大差异位置: {torch.argmax(diff).item()}')
                    print(f'最大差异值: {max_diff:.2e}')
                    
                    # 显示差异分布
                    large_diff_count = torch.sum(diff > threshold).item()
                    total_elements = diff.numel()
                    print(f'超过阈值的元素数: {large_diff_count} / {total_elements} ({large_diff_count/total_elements*100:.2f}%)')
                    
                    # 显示前几个大差异的位置
                    large_diff_indices = torch.where(diff > threshold)
                    if len(large_diff_indices[0]) > 0:
                        print(f'\\n前10个大差异位置:')
                        for i in range(min(10, len(large_diff_indices[0]))):
                            idx = tuple(torch.tensor([large_diff_indices[j][i] for j in range(len(large_diff_indices))]))
                            pytorch_val = vc_target_pytorch[idx].item()
                            mlx_val = vc_target_mlx_torch[idx].item()
                            diff_val = diff[idx].item()
                            print(f'  位置 {idx}: PyTorch={pytorch_val:.6f}, MLX={mlx_val:.6f}, 差异={diff_val:.2e}')
                else:
                    print(f'\\n✅ 差异在可接受范围内 (≤ {threshold:.0e})')
                    
                    # 计算相关系数
                    correlation = torch.corrcoef(torch.stack([vc_target_pytorch.flatten(), vc_target_mlx_torch.flatten()]))[0, 1].item()
                    print(f'相关系数: {correlation:.6f}')
                    
                    if correlation > 0.99:
                        print('✅ 输出高度相关，质量一致')
                    elif correlation > 0.95:
                        print('⚠️  输出基本相关，存在轻微差异')
                    else:
                        print('❌ 输出相关性较低，存在明显差异')
            else:
                print(f'❌ 输出形状不匹配: PyTorch {vc_target_pytorch.shape} vs MLX {vc_target_mlx_torch.shape}')
            
            print(f'\\n=== 统计信息对比 ===')
            print(f'PyTorch CFM 输出范围: [{vc_target_pytorch.min():.6f}, {vc_target_pytorch.max():.6f}]')
            print(f'MLX CFM 输出范围: [{vc_target_mlx_torch.min():.6f}, {vc_target_mlx_torch.max():.6f}]')
            print(f'PyTorch CFM 输出均值: {vc_target_pytorch.mean():.6f}')
            print(f'MLX CFM 输出均值: {vc_target_mlx_torch.mean():.6f}')
            print(f'PyTorch CFM 输出标准差: {vc_target_pytorch.std():.6f}')
            print(f'MLX CFM 输出标准差: {vc_target_mlx_torch.std():.6f}')
            
        else:
            print('❌ 缓存输入文件不存在')
            
    except Exception as e:
        print(f'❌ 测试过程出错: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cfm_with_cached_inputs()
