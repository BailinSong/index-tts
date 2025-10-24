#!/usr/bin/env python3
"""
测试重写的MLX CFM estimator
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
from unified_random_generator import UnifiedRandomGenerator

def test_rewritten_mlx_cfm():
    """测试重写的MLX CFM estimator"""
    
    print('=== 测试重写的MLX CFM estimator ===')
    
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
            output_path='rewritten_mlx_cfm_test.wav',
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
            
            print(f'\\n=== 测试重写的MLX CFM estimator ===')
            
            # 重置生成器
            tts_mlx.unified_random.reset_seed(42)
            
            # 确保所有输入都在正确的设备上
            cat_condition_device = cat_condition.to(tts_mlx.device)
            ref_mel_device = ref_mel.to(tts_mlx.device)
            style_device = style.to(tts_mlx.device)
            
            # 转换为 MLX
            cat_condition_mlx = torch_to_mlx(cat_condition_device.cpu())
            x_lens_mlx = torch_to_mlx(x_lens.cpu())
            ref_mel_mlx = torch_to_mlx(ref_mel_device.cpu())
            style_mlx = torch_to_mlx(style_device.cpu())
            
            print(f'\\n--- 测试重写的MLX CFM estimator ---')
            
            # 重置生成器
            tts_mlx.unified_random.reset_seed(42)
            
            # 重写的MLX CFM 推理
            print('重写的MLX CFM 推理...')
            vc_target_mlx_rewritten = tts_mlx.mlx_s2mel_cfm.inference(
                cat_condition_mlx,
                x_lens_mlx,
                ref_mel_mlx,
                style_mlx,
                None,
                diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
                unified_random=tts_mlx.unified_random
            )
            
            print(f'重写的MLX CFM 输出: {vc_target_mlx_rewritten.shape}, 范围 [{vc_target_mlx_rewritten.min():.6f}, {vc_target_mlx_rewritten.max():.6f}]')
            
            print(f'\\n--- 对比原始MLX CFM和重写版本 ---')
            
            # 重置生成器
            tts_mlx.unified_random.reset_seed(42)
            
            # 原始MLX CFM 推理
            print('原始MLX CFM 推理...')
            vc_target_mlx_original = tts_mlx.mlx_s2mel_cfm.estimator.inference(
                cat_condition_mlx,
                x_lens_mlx,
                ref_mel_mlx,
                style_mlx,
                None,
                diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
                unified_random=tts_mlx.unified_random
            )
            
            print(f'原始MLX CFM 输出: {vc_target_mlx_original.shape}, 范围 [{vc_target_mlx_original.min():.6f}, {vc_target_mlx_original.max():.6f}]')
            
            # 对比输出
            vc_target_mlx_rewritten_torch = mlx_to_torch(vc_target_mlx_rewritten).to(tts_mlx.device)
            vc_target_mlx_original_torch = mlx_to_torch(vc_target_mlx_original).to(tts_mlx.device)
            
            rewritten_vs_original_diff = torch.abs(vc_target_mlx_rewritten_torch - vc_target_mlx_original_torch)
            print(f'\\n=== 重写版本 vs 原始版本对比 ===')
            print(f'输出差异: 最大 {torch.max(rewritten_vs_original_diff):.6f}, 平均 {torch.mean(rewritten_vs_original_diff):.6f}')
            
            # 检查是否超过阈值
            threshold = 1e-5
            if torch.max(rewritten_vs_original_diff) > threshold:
                print(f'\\n❌ 重写版本与原始版本差异大于阈值 {threshold:.0e}')
                print(f'最大差异位置: {torch.argmax(rewritten_vs_original_diff).item()}')
                print(f'最大差异值: {torch.max(rewritten_vs_original_diff):.6f}')
                
                # 显示差异分布
                large_diff_count = torch.sum(rewritten_vs_original_diff > threshold).item()
                total_elements = rewritten_vs_original_diff.numel()
                print(f'超过阈值的元素数: {large_diff_count} / {total_elements} ({large_diff_count/total_elements*100:.2f}%)')
                
                # 显示前几个大差异的位置
                large_diff_indices = torch.where(rewritten_vs_original_diff > threshold)
                if len(large_diff_indices[0]) > 0:
                    print(f'\\n前10个大差异位置:')
                    for i in range(min(10, len(large_diff_indices[0]))):
                        idx = tuple(torch.tensor([large_diff_indices[j][i] for j in range(len(large_diff_indices))]))
                        rewritten_val = vc_target_mlx_rewritten_torch[idx].item()
                        original_val = vc_target_mlx_original_torch[idx].item()
                        diff_val = rewritten_vs_original_diff[idx].item()
                        print(f'  位置 {idx}: 重写={rewritten_val:.6f}, 原始={original_val:.6f}, 差异={diff_val:.6f}')
            else:
                print(f'\\n✅ 重写版本与原始版本差异在可接受范围内 (≤ {threshold:.0e})')
                
                # 计算相关系数
                correlation = torch.corrcoef(torch.stack([vc_target_mlx_rewritten_torch.flatten(), vc_target_mlx_original_torch.flatten()]))[0, 1].item()
                print(f'相关系数: {correlation:.6f}')
                
                if correlation > 0.99:
                    print('✅ 重写版本与原始版本高度相关')
                elif correlation > 0.95:
                    print('⚠️  重写版本与原始版本基本相关，存在轻微差异')
                else:
                    print('❌ 重写版本与原始版本相关性较低，存在明显差异')
            
            # 保存详细对比结果
            comparison_results = {
                'rewritten_output': vc_target_mlx_rewritten_torch.detach().cpu(),
                'original_output': vc_target_mlx_original_torch.detach().cpu(),
                'difference': rewritten_vs_original_diff.detach().cpu(),
                'max_diff': torch.max(rewritten_vs_original_diff).item(),
                'mean_diff': torch.mean(rewritten_vs_original_diff).item(),
                'rms_diff': torch.sqrt(torch.mean(rewritten_vs_original_diff**2)).item()
            }
            
            with open('rewritten_mlx_cfm_comparison.pkl', 'wb') as f:
                pickle.dump(comparison_results, f)
            print('\\n详细对比结果已保存到 rewritten_mlx_cfm_comparison.pkl')
            
        else:
            print('❌ 缓存输入文件不存在')
            
    except Exception as e:
        print(f'❌ 测试过程出错: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rewritten_mlx_cfm()
