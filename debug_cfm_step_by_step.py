#!/usr/bin/env python3
"""
拆分 CFM 流程，逐个流程监控差异
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

def debug_cfm_step_by_step():
    """拆分 CFM 流程，逐个流程监控差异"""
    
    print('=== 拆分 CFM 流程，逐个流程监控差异 ===')
    
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
            output_path='debug_step_by_step.wav',
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
            
            print(f'\\n=== 步骤1: 初始化噪声 ===')
            
            # 重新设置种子确保一致性
            torch.manual_seed(42)
            np.random.seed(42)
            mx.random.seed(42)
            
            # PyTorch 噪声生成
            B, T = cat_condition.size(0), cat_condition.size(1)
            z_pytorch = torch.randn([B, 80, T], device=tts_mlx.device)
            
            # MLX 噪声生成
            z_mlx = mx.random.normal((B, 80, T))
            
            print(f'PyTorch 噪声: {z_pytorch.shape}, 范围 [{z_pytorch.min():.6f}, {z_pytorch.max():.6f}]')
            print(f'MLX 噪声: {z_mlx.shape}, 范围 [{z_mlx.min():.6f}, {z_mlx.max():.6f}]')
            
            # 对比噪声差异
            z_mlx_torch = mlx_to_torch(z_mlx).to(tts_mlx.device)
            noise_diff = torch.abs(z_pytorch - z_mlx_torch)
            print(f'噪声差异: 最大 {torch.max(noise_diff):.6f}, 平均 {torch.mean(noise_diff):.6f}')
            
            print(f'\\n=== 步骤2: 时间跨度生成 ===')
            
            # PyTorch 时间跨度
            t_span_pytorch = torch.linspace(0, 1, diffusion_steps + 1, device=tts_mlx.device)
            
            # MLX 时间跨度
            t_span_mlx = mx.linspace(0, 1, diffusion_steps + 1)
            
            print(f'PyTorch 时间跨度: {t_span_pytorch.shape}, 范围 [{t_span_pytorch.min():.6f}, {t_span_pytorch.max():.6f}]')
            print(f'MLX 时间跨度: {t_span_mlx.shape}, 范围 [{t_span_mlx.min():.6f}, {t_span_mlx.max():.6f}]')
            
            # 对比时间跨度差异
            t_span_mlx_torch = mlx_to_torch(t_span_mlx).to(tts_mlx.device)
            t_span_diff = torch.abs(t_span_pytorch - t_span_mlx_torch)
            print(f'时间跨度差异: 最大 {torch.max(t_span_diff):.6f}, 平均 {torch.mean(t_span_diff):.6f}')
            
            print(f'\\n=== 步骤3: 输入数据转换 ===')
            
            # 确保所有输入都在正确的设备上
            cat_condition_device = cat_condition.to(tts_mlx.device)
            ref_mel_device = ref_mel.to(tts_mlx.device)
            style_device = style.to(tts_mlx.device)
            
            # 转换为 MLX
            cat_condition_mlx = torch_to_mlx(cat_condition_device.cpu())
            x_lens_mlx = torch_to_mlx(x_lens.cpu())
            ref_mel_mlx = torch_to_mlx(ref_mel_device.cpu())
            style_mlx = torch_to_mlx(style_device.cpu())
            
            print(f'PyTorch 输入:')
            print(f'  cat_condition: {cat_condition_device.shape}, 范围 [{cat_condition_device.min():.6f}, {cat_condition_device.max():.6f}]')
            print(f'  ref_mel: {ref_mel_device.shape}, 范围 [{ref_mel_device.min():.6f}, {ref_mel_device.max():.6f}]')
            print(f'  style: {style_device.shape}, 范围 [{style_device.min():.6f}, {style_device.max():.6f}]')
            
            print(f'MLX 输入:')
            print(f'  cat_condition: {cat_condition_mlx.shape}, 范围 [{cat_condition_mlx.min():.6f}, {cat_condition_mlx.max():.6f}]')
            print(f'  ref_mel: {ref_mel_mlx.shape}, 范围 [{ref_mel_mlx.min():.6f}, {ref_mel_mlx.max():.6f}]')
            print(f'  style: {style_mlx.shape}, 范围 [{style_mlx.min():.6f}, {style_mlx.max():.6f}]')
            
            # 对比输入差异
            cat_condition_mlx_torch = mlx_to_torch(cat_condition_mlx).to(tts_mlx.device)
            ref_mel_mlx_torch = mlx_to_torch(ref_mel_mlx).to(tts_mlx.device)
            style_mlx_torch = mlx_to_torch(style_mlx).to(tts_mlx.device)
            
            cat_diff = torch.abs(cat_condition_device - cat_condition_mlx_torch)
            ref_mel_diff = torch.abs(ref_mel_device - ref_mel_mlx_torch)
            style_diff = torch.abs(style_device - style_mlx_torch)
            
            print(f'输入差异:')
            print(f'  cat_condition: 最大 {torch.max(cat_diff):.6f}, 平均 {torch.mean(cat_diff):.6f}')
            print(f'  ref_mel: 最大 {torch.max(ref_mel_diff):.6f}, 平均 {torch.mean(ref_mel_diff):.6f}')
            print(f'  style: 最大 {torch.max(style_diff):.6f}, 平均 {torch.mean(style_diff):.6f}')
            
            print(f'\\n=== 步骤4: 单步推理对比 ===')
            
            # 重新设置种子确保一致性
            torch.manual_seed(42)
            np.random.seed(42)
            mx.random.seed(42)
            
            # 测试单步推理
            t = 0.0
            dt = 0.04
            
            print(f'测试单步推理: t={t}, dt={dt}')
            
            # PyTorch 单步推理
            try:
                # 准备 PyTorch 输入
                prompt_len = ref_mel_device.size(-1)
                prompt_x_pytorch = torch.zeros_like(z_pytorch)
                prompt_x_pytorch[..., :prompt_len] = ref_mel_device[..., :prompt_len]
                x_pytorch = z_pytorch.clone()
                x_pytorch[..., :prompt_len] = 0
                
                # PyTorch CFM 单步
                t_tensor = torch.tensor([t], device=tts_mlx.device)
                dphi_dt_pytorch = tts_mlx.s2mel.models['cfm'].estimator(
                    x_pytorch, prompt_x_pytorch, x_lens, t_tensor, style_device, cat_condition_device
                )
                
                print(f'PyTorch 单步输出: {dphi_dt_pytorch.shape}, 范围 [{dphi_dt_pytorch.min():.6f}, {dphi_dt_pytorch.max():.6f}]')
                
            except Exception as e:
                print(f'PyTorch 单步推理失败: {e}')
                dphi_dt_pytorch = None
            
            # MLX 单步推理
            try:
                # 准备 MLX 输入
                prompt_len = ref_mel_mlx.shape[-1]
                prompt_x_mlx = mx.zeros_like(z_mlx)
                prompt_x_mlx[:, :, :prompt_len] = ref_mel_mlx[:, :, :prompt_len]
                x_mlx = z_mlx.copy()
                x_mlx[:, :, :prompt_len] = 0
                
                # MLX CFM 单步
                t_scalar = mx.array([t])
                dphi_dt_mlx = tts_mlx.mlx_s2mel_cfm.estimator(
                    x_mlx, prompt_x_mlx, x_lens_mlx, t_scalar, style_mlx, cat_condition_mlx
                )
                
                print(f'MLX 单步输出: {dphi_dt_mlx.shape}, 范围 [{dphi_dt_mlx.min():.6f}, {dphi_dt_mlx.max():.6f}]')
                
            except Exception as e:
                print(f'MLX 单步推理失败: {e}')
                dphi_dt_mlx = None
            
            # 对比单步输出
            if dphi_dt_pytorch is not None and dphi_dt_mlx is not None:
                dphi_dt_mlx_torch = mlx_to_torch(dphi_dt_mlx).to(tts_mlx.device)
                step_diff = torch.abs(dphi_dt_pytorch - dphi_dt_mlx_torch)
                print(f'单步输出差异: 最大 {torch.max(step_diff):.6f}, 平均 {torch.mean(step_diff):.6f}')
                
                # 显示前几个差异位置
                large_diff_indices = torch.where(step_diff > 0.1)
                if len(large_diff_indices[0]) > 0:
                    print(f'\\n前5个大差异位置:')
                    for i in range(min(5, len(large_diff_indices[0]))):
                        idx = tuple(torch.tensor([large_diff_indices[j][i] for j in range(len(large_diff_indices))]))
                        pytorch_val = dphi_dt_pytorch[idx].item()
                        mlx_val = dphi_dt_mlx_torch[idx].item()
                        diff_val = step_diff[idx].item()
                        print(f'  位置 {idx}: PyTorch={pytorch_val:.6f}, MLX={mlx_val:.6f}, 差异={diff_val:.6f}')
            
            print(f'\\n=== 步骤5: 完整推理对比 ===')
            
            # 重新设置种子确保一致性
            torch.manual_seed(42)
            np.random.seed(42)
            mx.random.seed(42)
            
            # PyTorch 完整推理
            print('PyTorch CFM 完整推理...')
            vc_target_pytorch = tts_mlx.s2mel.models['cfm'].inference(
                cat_condition_device,
                torch.LongTensor([cat_condition_device.size(1)]).to(tts_mlx.device),
                ref_mel_device, style_device, None, diffusion_steps,
                inference_cfg_rate=inference_cfg_rate
            )
            
            print(f'PyTorch CFM 输出: {vc_target_pytorch.shape}, 范围 [{vc_target_pytorch.min():.6f}, {vc_target_pytorch.max():.6f}]')
            
            # MLX 完整推理
            print('MLX CFM 完整推理...')
            vc_target_mlx = tts_mlx.mlx_s2mel_cfm.inference(
                cat_condition_mlx,
                x_lens_mlx,
                ref_mel_mlx,
                style_mlx,
                None,
                diffusion_steps,
                inference_cfg_rate=inference_cfg_rate
            )
            
            print(f'MLX CFM 输出: {vc_target_mlx.shape}, 范围 [{vc_target_mlx.min():.6f}, {vc_target_mlx.max():.6f}]')
            
            # 对比完整输出
            vc_target_mlx_torch = mlx_to_torch(vc_target_mlx).to(tts_mlx.device)
            final_diff = torch.abs(vc_target_pytorch - vc_target_mlx_torch)
            print(f'完整输出差异: 最大 {torch.max(final_diff):.6f}, 平均 {torch.mean(final_diff):.6f}')
            
            # 保存详细对比结果
            comparison_results = {
                'pytorch_output': vc_target_pytorch.detach().cpu(),
                'mlx_output': vc_target_mlx_torch.detach().cpu(),
                'difference': final_diff.detach().cpu(),
                'max_diff': torch.max(final_diff).item(),
                'mean_diff': torch.mean(final_diff).item(),
                'rms_diff': torch.sqrt(torch.mean(final_diff**2)).item()
            }
            
            with open('cfm_step_by_step_comparison.pkl', 'wb') as f:
                pickle.dump(comparison_results, f)
            print('详细对比结果已保存到 cfm_step_by_step_comparison.pkl')
            
        else:
            print('❌ 缓存输入文件不存在')
            
    except Exception as e:
        print(f'❌ 测试过程出错: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_cfm_step_by_step()
