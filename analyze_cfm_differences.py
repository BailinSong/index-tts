#!/usr/bin/env python3
"""
深度分析MLX CFM与PyTorch CFM差异的根本原因
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

def analyze_cfm_differences():
    """深度分析CFM差异的根本原因"""
    
    print('=== 深度分析MLX CFM与PyTorch CFM差异的根本原因 ===')
    
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
            output_path='analyze_differences_test.wav',
            seed=42
        )
        
        print('\\n✅ MLX 推理完成，获取缓存输入')
        
        # 检查是否有缓存输入文件
        if os.path.exists('cfm_inputs_mlx.pkl'):
            print('\\n=== 加载缓存输入数据 ===')
            with open('cfm_inputs_mlx.pkl', 'rb') as f:
                cached_inputs = pickle.load(f)
            
            # 提取输入数据
            cat_condition = cached_inputs['cat_condition']
            x_lens = cached_inputs['x_lens']
            ref_mel = cached_inputs['ref_mel']
            style = cached_inputs['style']
            diffusion_steps = cached_inputs['diffusion_steps']
            inference_cfg_rate = cached_inputs['inference_cfg_rate']
            
            print(f'\\n=== 步骤1: 分析权重差异 ===')
            
            # 分析PyTorch CFM权重
            pytorch_cfm = tts_mlx.s2mel.models['cfm']
            print(f'\\n--- PyTorch CFM 权重分析 ---')
            print(f'PyTorch CFM 类型: {type(pytorch_cfm)}')
            print(f'PyTorch CFM estimator 类型: {type(pytorch_cfm.estimator)}')
            
            # 分析MLX CFM权重
            mlx_cfm = tts_mlx.mlx_s2mel_cfm
            print(f'\\n--- MLX CFM 权重分析 ---')
            print(f'MLX CFM 类型: {type(mlx_cfm)}')
            print(f'MLX CFM estimator 类型: {type(mlx_cfm.estimator)}')
            
            # 检查权重是否相同
            print(f'\\n--- 权重对比分析 ---')
            
            # 获取PyTorch CFM的权重
            pytorch_state_dict = pytorch_cfm.state_dict()
            print(f'PyTorch CFM 权重数量: {len(pytorch_state_dict)}')
            
            # 获取MLX CFM的权重
            mlx_state_dict = mlx_cfm.estimator.state_dict()
            print(f'MLX CFM 权重数量: {len(mlx_state_dict)}')
            
            # 对比权重
            print(f'\\n--- 权重键对比 ---')
            pytorch_keys = set(pytorch_state_dict.keys())
            mlx_keys = set(mlx_state_dict.keys())
            
            print(f'PyTorch 权重键: {len(pytorch_keys)}')
            print(f'MLX 权重键: {len(mlx_keys)}')
            
            common_keys = pytorch_keys.intersection(mlx_keys)
            pytorch_only = pytorch_keys - mlx_keys
            mlx_only = mlx_keys - pytorch_keys
            
            print(f'共同键: {len(common_keys)}')
            print(f'仅PyTorch: {len(pytorch_only)}')
            print(f'仅MLX: {len(mlx_only)}')
            
            if pytorch_only:
                print(f'\\n仅PyTorch的键: {list(pytorch_only)[:10]}...')
            if mlx_only:
                print(f'\\n仅MLX的键: {list(mlx_only)[:10]}...')
            
            print(f'\\n=== 步骤2: 分析estimator结构差异 ===')
            
            # 分析PyTorch estimator结构
            print(f'\\n--- PyTorch Estimator 结构 ---')
            pytorch_estimator = pytorch_cfm.estimator
            print(f'PyTorch Estimator 类型: {type(pytorch_estimator)}')
            print(f'PyTorch Estimator 模块: {list(pytorch_estimator._modules.keys())}')
            
            # 分析MLX estimator结构
            print(f'\\n--- MLX Estimator 结构 ---')
            mlx_estimator = mlx_cfm.estimator
            print(f'MLX Estimator 类型: {type(mlx_estimator)}')
            print(f'MLX Estimator 模块: {list(mlx_estimator._modules.keys())}')
            
            print(f'\\n=== 步骤3: 分析单步推理差异 ===')
            
            # 准备单步推理的输入
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
            
            # 准备噪声
            B, T = cat_condition_device.size(0), cat_condition_device.size(1)
            z_pytorch = tts_mlx.unified_random.generate_noise((B, 80, T), device=tts_mlx.device)
            z_mlx = tts_mlx.unified_random.generate_noise_mlx((B, 80, T))
            
            # 准备prompt
            prompt_len = ref_mel_device.size(-1)
            prompt_x_pytorch = torch.zeros_like(z_pytorch)
            prompt_x_pytorch[..., :prompt_len] = ref_mel_device[..., :prompt_len]
            x_pytorch = z_pytorch.clone()
            x_pytorch[..., :prompt_len] = 0
            
            prompt_x_mlx = mx.zeros_like(z_mlx)
            prompt_x_mlx[:, :, :prompt_len] = ref_mel_mlx[:, :, :prompt_len]
            x_mlx = z_mlx.copy()
            x_mlx[:, :, :prompt_len] = 0
            
            # 时间步
            t = 0.0
            t_tensor = torch.tensor([t], device=tts_mlx.device)
            t_scalar = mx.array([t])
            
            print(f'\\n--- 单步推理输入对比 ---')
            print(f'PyTorch x: {x_pytorch.shape}, 范围 [{x_pytorch.min():.6f}, {x_pytorch.max():.6f}]')
            print(f'MLX x: {x_mlx.shape}, 范围 [{x_mlx.min():.6f}, {x_mlx.max():.6f}]')
            print(f'PyTorch prompt_x: {prompt_x_pytorch.shape}, 范围 [{prompt_x_pytorch.min():.6f}, {prompt_x_pytorch.max():.6f}]')
            print(f'MLX prompt_x: {prompt_x_mlx.shape}, 范围 [{prompt_x_mlx.min():.6f}, {prompt_x_mlx.max():.6f}]')
            
            # 单步推理
            print(f'\\n--- 单步推理输出对比 ---')
            
            # PyTorch 单步推理
            try:
                dphi_dt_pytorch = pytorch_estimator(
                    x_pytorch, prompt_x_pytorch, x_lens, t_tensor, style_device, cat_condition_device
                )
                print(f'PyTorch 单步输出: {dphi_dt_pytorch.shape}, 范围 [{dphi_dt_pytorch.min():.6f}, {dphi_dt_pytorch.max():.6f}]')
                print(f'PyTorch 单步输出 mean: {dphi_dt_pytorch.mean():.6f}, std: {dphi_dt_pytorch.std():.6f}')
            except Exception as e:
                print(f'PyTorch 单步推理失败: {e}')
                dphi_dt_pytorch = None
            
            # MLX 单步推理
            try:
                dphi_dt_mlx = mlx_estimator(
                    x_mlx, prompt_x_mlx, x_lens_mlx, t_scalar, style_mlx, cat_condition_mlx
                )
                print(f'MLX 单步输出: {dphi_dt_mlx.shape}, 范围 [{dphi_dt_mlx.min():.6f}, {dphi_dt_mlx.max():.6f}]')
                print(f'MLX 单步输出 mean: {dphi_dt_mlx.mean():.6f}, std: {dphi_dt_mlx.std():.6f}')
            except Exception as e:
                print(f'MLX 单步推理失败: {e}')
                dphi_dt_mlx = None
            
            # 对比单步输出
            if dphi_dt_pytorch is not None and dphi_dt_mlx is not None:
                dphi_dt_mlx_torch = mlx_to_torch(dphi_dt_mlx).to(tts_mlx.device)
                step_diff = torch.abs(dphi_dt_pytorch - dphi_dt_mlx_torch)
                print(f'\\n--- 单步输出差异分析 ---')
                print(f'单步输出差异: 最大 {torch.max(step_diff):.6f}, 平均 {torch.mean(step_diff):.6f}')
                print(f'PyTorch 数值范围: {torch.max(dphi_dt_pytorch) - torch.min(dphi_dt_pytorch):.6f}')
                print(f'MLX 数值范围: {torch.max(dphi_dt_mlx_torch) - torch.min(dphi_dt_mlx_torch):.6f}')
                print(f'数值范围比例: {(torch.max(dphi_dt_pytorch) - torch.min(dphi_dt_pytorch)) / (torch.max(dphi_dt_mlx_torch) - torch.min(dphi_dt_mlx_torch)):.2f}')
                
                # 分析差异分布
                large_diff_count = torch.sum(step_diff > 1.0).item()
                total_elements = step_diff.numel()
                print(f'大差异元素数 (>1.0): {large_diff_count} / {total_elements} ({large_diff_count/total_elements*100:.2f}%)')
                
                # 显示前几个大差异的位置
                large_diff_indices = torch.where(step_diff > 1.0)
                if len(large_diff_indices[0]) > 0:
                    print(f'\\n前10个大差异位置:')
                    for i in range(min(10, len(large_diff_indices[0]))):
                        idx = tuple(torch.tensor([large_diff_indices[j][i] for j in range(len(large_diff_indices))]))
                        pytorch_val = dphi_dt_pytorch[idx].item()
                        mlx_val = dphi_dt_mlx_torch[idx].item()
                        diff_val = step_diff[idx].item()
                        print(f'  位置 {idx}: PyTorch={pytorch_val:.6f}, MLX={mlx_val:.6f}, 差异={diff_val:.6f}')
            
            print(f'\\n=== 步骤4: 分析权重数值差异 ===')
            
            # 对比共同权重的数值差异
            if common_keys:
                print(f'\\n--- 权重数值对比 (前10个共同键) ---')
                for i, key in enumerate(list(common_keys)[:10]):
                    pytorch_weight = pytorch_state_dict[key]
                    mlx_weight = mlx_state_dict[key]
                    
                    # 转换MLX权重为PyTorch格式
                    if isinstance(mlx_weight, mx.array):
                        mlx_weight_torch = mlx_to_torch(mlx_weight).to(tts_mlx.device)
                    else:
                        mlx_weight_torch = mlx_weight
                    
                    # 确保形状一致
                    if pytorch_weight.shape == mlx_weight_torch.shape:
                        weight_diff = torch.abs(pytorch_weight - mlx_weight_torch)
                        print(f'{key}:')
                        print(f'  PyTorch: {pytorch_weight.shape}, 范围 [{pytorch_weight.min():.6f}, {pytorch_weight.max():.6f}]')
                        print(f'  MLX: {mlx_weight_torch.shape}, 范围 [{mlx_weight_torch.min():.6f}, {mlx_weight_torch.max():.6f}]')
                        print(f'  差异: 最大 {torch.max(weight_diff):.6f}, 平均 {torch.mean(weight_diff):.6f}')
                        
                        if torch.max(weight_diff) > 1e-5:
                            print(f'  ❌ 权重差异超过阈值')
                        else:
                            print(f'  ✅ 权重差异在阈值内')
                    else:
                        print(f'{key}: 形状不匹配 - PyTorch: {pytorch_weight.shape}, MLX: {mlx_weight_torch.shape}')
            
            print(f'\\n=== 步骤5: 分析estimator内部结构差异 ===')
            
            # 分析PyTorch estimator的内部结构
            print(f'\\n--- PyTorch Estimator 内部结构 ---')
            for name, module in pytorch_estimator.named_modules():
                if len(name) > 0:  # 跳过根模块
                    print(f'{name}: {type(module)}')
                    if hasattr(module, 'weight') and module.weight is not None:
                        print(f'  权重形状: {module.weight.shape}, 范围 [{module.weight.min():.6f}, {module.weight.max():.6f}]')
                    if hasattr(module, 'bias') and module.bias is not None:
                        print(f'  偏置形状: {module.bias.shape}, 范围 [{module.bias.min():.6f}, {module.bias.max():.6f}]')
            
            # 分析MLX estimator的内部结构
            print(f'\\n--- MLX Estimator 内部结构 ---')
            for name, module in mlx_estimator.named_modules():
                if len(name) > 0:  # 跳过根模块
                    print(f'{name}: {type(module)}')
                    if hasattr(module, 'weight') and module.weight is not None:
                        print(f'  权重形状: {module.weight.shape}, 范围 [{module.weight.min():.6f}, {module.weight.max():.6f}]')
                    if hasattr(module, 'bias') and module.bias is not None:
                        print(f'  偏置形状: {module.bias.shape}, 范围 [{module.bias.min():.6f}, {module.bias.max():.6f}]')
            
            # 保存分析结果
            analysis_results = {
                'pytorch_keys': list(pytorch_keys),
                'mlx_keys': list(mlx_keys),
                'common_keys': list(common_keys),
                'pytorch_only': list(pytorch_only),
                'mlx_only': list(mlx_only),
                'single_step_pytorch': dphi_dt_pytorch.detach().cpu() if dphi_dt_pytorch is not None else None,
                'single_step_mlx': dphi_dt_mlx_torch.detach().cpu() if dphi_dt_mlx is not None else None,
                'single_step_diff': step_diff.detach().cpu() if dphi_dt_pytorch is not None and dphi_dt_mlx is not None else None
            }
            
            with open('cfm_differences_analysis.pkl', 'wb') as f:
                pickle.dump(analysis_results, f)
            print('\\n详细分析结果已保存到 cfm_differences_analysis.pkl')
            
        else:
            print('❌ 缓存输入文件不存在')
            
    except Exception as e:
        print(f'❌ 分析过程出错: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_cfm_differences()
