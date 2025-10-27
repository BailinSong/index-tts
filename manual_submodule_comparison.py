#!/usr/bin/env python3
"""
手动收集和对比 PyTorch 和 MLX CFM 各个子模块的输入输出

使用正确的缓存输入数据，手动收集两个版本各个子模块的输入输出，
然后进行详细的对比分析
"""

import pickle
import numpy as np
import torch
import mlx.core as mx
from pathlib import Path
import sys
import os

# 添加项目路径
sys.path.append('/Users/bailin/index-tts')


def manual_submodule_comparison():
    """手动收集和对比子模块数据"""
    print("🔍 Manual CFM Submodule Comparison")
    print("="*60)
    
    # 加载正确的缓存输入数据
    cached_inputs = load_correct_cached_inputs()
    if not cached_inputs:
        print("❌ No correct cached inputs available!")
        return
    
    print("\n📊 Using Correct Cached Inputs:")
    print_cached_inputs_summary(cached_inputs)
    
    # 收集 PyTorch CFM 子模块数据
    print("\n🔍 Collecting PyTorch CFM Submodule Data:")
    pytorch_submodules = collect_pytorch_submodules_manual(cached_inputs)
    
    # 收集 MLX CFM 子模块数据
    print("\n🔍 Collecting MLX CFM Submodule Data:")
    mlx_submodules = collect_mlx_submodules_manual(cached_inputs)
    
    # 对比各个子模块
    print("\n📊 Submodule-by-Submodule Comparison:")
    comparison_results = compare_submodules_manual(pytorch_submodules, mlx_submodules)
    
    # 生成详细对比报告
    print("\n📄 Generating Manual Comparison Report:")
    generate_manual_report(cached_inputs, pytorch_submodules, mlx_submodules, comparison_results)
    
    print("\n🎉 Manual submodule comparison completed!")


def load_correct_cached_inputs():
    """加载正确的缓存输入数据"""
    cached_file = Path("cfm_debug_outputs/correct_cached_inputs.pkl")
    
    if not cached_file.exists():
        print("❌ Correct cached inputs file not found!")
        return None
    
    with open(cached_file, 'rb') as f:
        cached_inputs = pickle.load(f)
    
    print(f"✅ Correct cached inputs loaded from: {cached_file}")
    return cached_inputs


def print_cached_inputs_summary(cached_inputs):
    """打印缓存输入摘要"""
    for key, value in cached_inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"     {key}: {value.shape}, range: [{value.min():.6f}, {value.max():.6f}]")
        elif isinstance(value, np.ndarray):
            print(f"     {key}: {value.shape}, range: [{value.min():.6f}, {value.max():.6f}]")
        else:
            print(f"     {key}: {value}")


def collect_pytorch_submodules_manual(cached_inputs):
    """手动收集 PyTorch CFM 子模块数据"""
    try:
        # 导入 PyTorch CFM
        from indextts.s2mel.modules.flow_matching import CFM
        
        # 创建配置
        config_dict = {
            'DiT': {
                'in_channels': 80,
                'hidden_dim': 512,
                'num_heads': 8,
                'depth': 13,
                'time_as_token': False,
                'style_as_token': False,
                'long_skip_connection': True,
                'style_condition': True,
                'is_causal': True,
                'final_layer_type': 'wavenet',
                'content_type': 'continuous',
                'content_codebook_size': 1024,
                'content_dim': 512,
                'class_dropout_prob': 0.1
            },
            'style_encoder': {
                'dim': 192
            },
            'wavenet': {
                'hidden_dim': 512,
                'kernel_size': 3,
                'dilation_rate': 2,
                'num_layers': 8,
                'p_dropout': 0.1,
                'style_condition': True
            },
            'dit_type': 'DiT',
            'reg_loss_type': 'l2'
        }
        
        # 创建 SimpleConfig 类
        class SimpleConfig:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    if isinstance(value, dict):
                        setattr(self, key, SimpleConfig(value))
                    else:
                        setattr(self, key, value)
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        config = SimpleConfig(config_dict)
        
        # 初始化 PyTorch CFM
        pytorch_cfm = CFM(config)
        pytorch_cfm.eval()
        
        # 初始化 caches
        batch_size = cached_inputs['batch_size']
        seq_len = cached_inputs['seq_len']
        pytorch_cfm.estimator.setup_caches(max_batch_size=batch_size, max_seq_length=seq_len)
        
        # 准备输入数据
        x = cached_inputs['x']
        prompt_x = cached_inputs['prompt_x']
        x_lens = cached_inputs['x_lens']
        style = cached_inputs['style']
        mu = cached_inputs['mu']
        
        print(f"     Input shapes: x={x.shape}, prompt_x={prompt_x.shape}, x_lens={x_lens.shape}")
        print(f"     Input shapes: style={style.shape}, mu={mu.shape}")
        
        # 创建时间跨度
        t_span = torch.linspace(0, 1, 4)  # 3 steps + 1
        
        # 手动执行 CFM 步骤并收集子模块数据
        print(f"\n     🔍 Manual PyTorch CFM Execution with Submodule Tracking:")
        
        submodule_data = {}
        
        with torch.no_grad():
            # 应用 prompt
            prompt_len = prompt_x.size(-1)
            prompt_x_processed = torch.zeros_like(x)
            prompt_x_processed[..., :prompt_len] = prompt_x[..., :prompt_len]
            x_processed = x.clone()
            x_processed[..., :prompt_len] = 0
            
            print(f"       After prompt processing:")
            print(f"         x_processed: range=[{x_processed.min():.6f}, {x_processed.max():.6f}]")
            print(f"         prompt_x_processed: range=[{prompt_x_processed.min():.6f}, {prompt_x_processed.max():.6f}]")
            
            # 存储初始数据
            submodule_data['initial'] = {
                'x_processed': x_processed,
                'prompt_x_processed': prompt_x_processed,
                'x_lens': x_lens,
                'style': style,
                'mu': mu
            }
            
            # 执行 Euler 步骤
            for step in range(1, len(t_span)):
                dt = t_span[step] - t_span[step - 1]
                t_current = t_span[step - 1]
                
                print(f"\n       Step {step}/{len(t_span)-1}: t={t_current:.6f}, dt={dt:.6f}")
                
                # 调用 estimator (DiT)
                dphi_dt = pytorch_cfm.estimator(
                    x_processed, prompt_x_processed, x_lens, 
                    t_current.unsqueeze(0), style, mu
                )
                
                print(f"         dphi_dt: shape={dphi_dt.shape}, range=[{dphi_dt.min():.6f}, {dphi_dt.max():.6f}]")
                
                # 存储步骤数据
                submodule_data[f'step_{step}'] = {
                    't': t_current,
                    'dt': dt,
                    'dphi_dt': dphi_dt,
                    'x_before': x_processed.clone()
                }
                
                # 更新 x
                x_processed = x_processed + dt * dphi_dt
                
                print(f"         x_updated: range=[{x_processed.min():.6f}, {x_processed.max():.6f}]")
                
                # 存储更新后的数据
                submodule_data[f'step_{step}']['x_after'] = x_processed.clone()
            
            # 存储最终输出
            submodule_data['final'] = {
                'output': x_processed,
                'output_range': [x_processed.min().item(), x_processed.max().item()]
            }
            
            print(f"\n       Final PyTorch output: range=[{x_processed.min():.6f}, {x_processed.max():.6f}]")
        
        return {
            'submodule_data': submodule_data,
            'success': True,
            'config': config_dict
        }
        
    except Exception as e:
        print(f"     ❌ PyTorch manual collection failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'success': False
        }


def collect_mlx_submodules_manual(cached_inputs):
    """手动收集 MLX CFM 子模块数据"""
    try:
        # 导入 MLX CFM
        from indextts.s2mel.modules.mlx_cfm import MLXCFM
        
        # 创建配置
        config_dict = {
            'DiT': {
                'in_channels': 80,
                'hidden_dim': 512,
                'num_heads': 8,
                'depth': 13,
                'time_as_token': False,
                'style_as_token': False,
                'long_skip_connection': True,
                'style_condition': True,
                'is_causal': True,
                'final_layer_type': 'wavenet',
                'content_type': 'continuous',
                'content_codebook_size': 1024,
                'content_dim': 512,
                'class_dropout_prob': 0.1
            },
            'style_encoder': {
                'dim': 192
            },
            'wavenet': {
                'hidden_dim': 512,
                'kernel_size': 3,
                'dilation_rate': 2,
                'num_layers': 8,
                'p_dropout': 0.1,
                'style_condition': True
            }
        }
        
        # 初始化 MLX CFM
        mlx_cfm = MLXCFM(config_dict)
        
        # 准备输入数据（转换为 MLX）
        x_mlx = mx.array(cached_inputs['x'].numpy())
        prompt_x_mlx = mx.array(cached_inputs['prompt_x'].numpy())
        x_lens_mlx = mx.array(cached_inputs['x_lens'].numpy())
        style_mlx = mx.array(cached_inputs['style'].numpy())
        mu_mlx = mx.array(cached_inputs['mu'].numpy())
        
        print(f"     Input shapes: x={x_mlx.shape}, prompt_x={prompt_x_mlx.shape}, x_lens={x_lens_mlx.shape}")
        print(f"     Input shapes: style={style_mlx.shape}, mu={mu_mlx.shape}")
        
        # 创建时间跨度
        t_span_mlx = mx.array(np.linspace(0, 1, 4))  # 3 steps + 1
        
        # 手动执行 CFM 步骤并收集子模块数据
        print(f"\n     🔍 Manual MLX CFM Execution with Submodule Tracking:")
        
        submodule_data = {}
        
        # 应用 prompt
        prompt_len = prompt_x_mlx.shape[-1]
        prompt_x_processed_mlx = mx.zeros_like(x_mlx)
        prompt_x_processed_mlx = mx.concatenate([prompt_x_mlx, mx.zeros((x_mlx.shape[0], x_mlx.shape[1], x_mlx.shape[2] - prompt_len))], axis=-1)
        x_processed_mlx = mx.concatenate([mx.zeros((x_mlx.shape[0], x_mlx.shape[1], prompt_len)), x_mlx[:, :, prompt_len:]], axis=-1)
        
        print(f"       After prompt processing:")
        print(f"         x_processed: range=[{float(x_processed_mlx.min()):.6f}, {float(x_processed_mlx.max()):.6f}]")
        print(f"         prompt_x_processed: range=[{float(prompt_x_processed_mlx.min()):.6f}, {float(prompt_x_processed_mlx.max()):.6f}]")
        
        # 存储初始数据
        submodule_data['initial'] = {
            'x_processed': x_processed_mlx,
            'prompt_x_processed': prompt_x_processed_mlx,
            'x_lens': x_lens_mlx,
            'style': style_mlx,
            'mu': mu_mlx
        }
        
        # 执行 Euler 步骤
        for step in range(1, len(t_span_mlx)):
            dt = t_span_mlx[step] - t_span_mlx[step - 1]
            t_current = t_span_mlx[step - 1]
            
            print(f"\n       Step {step}/{len(t_span_mlx)-1}: t={float(t_current):.6f}, dt={float(dt):.6f}")
            
            # 调用 estimator (DiT)
            dphi_dt = mlx_cfm.estimator(
                x_processed_mlx, prompt_x_processed_mlx, x_lens_mlx, 
                t_current.reshape(1), style_mlx, mu_mlx
            )
            
            print(f"         dphi_dt: shape={dphi_dt.shape}, range=[{float(dphi_dt.min()):.6f}, {float(dphi_dt.max()):.6f}]")
            
            # 存储步骤数据
            submodule_data[f'step_{step}'] = {
                't': t_current,
                'dt': dt,
                'dphi_dt': dphi_dt,
                'x_before': x_processed_mlx
            }
            
            # 更新 x
            x_processed_mlx = x_processed_mlx + dt * dphi_dt
            
            print(f"         x_updated: range=[{float(x_processed_mlx.min()):.6f}, {float(x_processed_mlx.max()):.6f}]")
            
            # 存储更新后的数据
            submodule_data[f'step_{step}']['x_after'] = x_processed_mlx
        
        # 存储最终输出
        submodule_data['final'] = {
            'output': x_processed_mlx,
            'output_range': [float(x_processed_mlx.min()), float(x_processed_mlx.max())]
        }
        
        print(f"\n       Final MLX output: range=[{float(x_processed_mlx.min()):.6f}, {float(x_processed_mlx.max()):.6f}]")
        
        return {
            'submodule_data': submodule_data,
            'success': True,
            'config': config_dict
        }
        
    except Exception as e:
        print(f"     ❌ MLX manual collection failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'success': False
        }


def compare_submodules_manual(pytorch_data, mlx_data):
    """手动对比子模块数据"""
    print("\n   📋 Manual Submodule Comparison:")
    
    if not pytorch_data.get('success') or not mlx_data.get('success'):
        print("     ❌ Cannot compare: one or both data collection failed")
        return None
    
    pytorch_submodules = pytorch_data['submodule_data']
    mlx_submodules = mlx_data['submodule_data']
    
    comparison_results = {}
    
    # 对比初始数据
    print(f"\n     🔍 Initial Data Comparison:")
    pytorch_initial = pytorch_submodules['initial']
    mlx_initial = mlx_submodules['initial']
    
    # 对比 x_processed
    pytorch_x = pytorch_initial['x_processed']
    mlx_x = mlx_initial['x_processed']
    
    pytorch_x_np = pytorch_x.detach().cpu().numpy()
    mlx_x_np = np.array(mlx_x)
    
    x_diff = np.max(np.abs(pytorch_x_np - mlx_x_np))
    print(f"       x_processed difference: {x_diff:.2e}")
    
    comparison_results['initial'] = {
        'x_processed_diff': x_diff,
        'pytorch_x_range': [pytorch_x_np.min(), pytorch_x_np.max()],
        'mlx_x_range': [mlx_x_np.min(), mlx_x_np.max()]
    }
    
    # 对比各个步骤
    print(f"\n     🔍 Step-by-Step Comparison:")
    
    for step_key in pytorch_submodules.keys():
        if step_key.startswith('step_'):
            step_num = step_key.split('_')[1]
            
            if step_key in mlx_submodules:
                pytorch_step = pytorch_submodules[step_key]
                mlx_step = mlx_submodules[step_key]
                
                print(f"\n       Step {step_num}:")
                
                # 对比 dphi_dt
                pytorch_dphi_dt = pytorch_step['dphi_dt']
                mlx_dphi_dt = mlx_step['dphi_dt']
                
                pytorch_dphi_dt_np = pytorch_dphi_dt.detach().cpu().numpy()
                mlx_dphi_dt_np = np.array(mlx_dphi_dt)
                
                dphi_dt_diff = np.max(np.abs(pytorch_dphi_dt_np - mlx_dphi_dt_np))
                print(f"         dphi_dt difference: {dphi_dt_diff:.2e}")
                
                # 对比 x_after
                pytorch_x_after = pytorch_step['x_after']
                mlx_x_after = mlx_step['x_after']
                
                pytorch_x_after_np = pytorch_x_after.detach().cpu().numpy()
                mlx_x_after_np = np.array(mlx_x_after)
                
                x_after_diff = np.max(np.abs(pytorch_x_after_np - mlx_x_after_np))
                print(f"         x_after difference: {x_after_diff:.2e}")
                
                comparison_results[step_key] = {
                    'dphi_dt_diff': dphi_dt_diff,
                    'x_after_diff': x_after_diff,
                    'pytorch_dphi_dt_range': [pytorch_dphi_dt_np.min(), pytorch_dphi_dt_np.max()],
                    'mlx_dphi_dt_range': [mlx_dphi_dt_np.min(), mlx_dphi_dt_np.max()],
                    'pytorch_x_after_range': [pytorch_x_after_np.min(), pytorch_x_after_np.max()],
                    'mlx_x_after_range': [mlx_x_after_np.min(), mlx_x_after_np.max()]
                }
    
    # 对比最终输出
    print(f"\n     🔍 Final Output Comparison:")
    pytorch_final = pytorch_submodules['final']
    mlx_final = mlx_submodules['final']
    
    pytorch_output = pytorch_final['output']
    mlx_output = mlx_final['output']
    
    pytorch_output_np = pytorch_output.detach().cpu().numpy()
    mlx_output_np = np.array(mlx_output)
    
    final_diff = np.max(np.abs(pytorch_output_np - mlx_output_np))
    mean_diff = np.mean(np.abs(pytorch_output_np - mlx_output_np))
    relative_diff = mean_diff / (np.mean(np.abs(pytorch_output_np)) + 1e-8)
    
    print(f"       Final output max difference: {final_diff:.2e}")
    print(f"       Final output mean difference: {mean_diff:.2e}")
    print(f"       Final output relative difference: {relative_diff:.2e}")
    
    pytorch_range = [pytorch_output_np.min(), pytorch_output_np.max()]
    mlx_range = [mlx_output_np.min(), mlx_output_np.max()]
    
    print(f"       PyTorch final range: [{pytorch_range[0]:.6f}, {pytorch_range[1]:.6f}]")
    print(f"       MLX final range: [{mlx_range[0]:.6f}, {mlx_range[1]:.6f}]")
    
    # 检查音频范围
    pytorch_clipping = pytorch_range[0] < -1.0 or pytorch_range[1] > 1.0
    mlx_clipping = mlx_range[0] < -1.0 or mlx_range[1] > 1.0
    
    if pytorch_clipping:
        print(f"       ⚠️ PyTorch output exceeds audio range [-1, 1]")
    if mlx_clipping:
        print(f"       ❌ MLX output exceeds audio range [-1, 1]")
    
    if not pytorch_clipping and not mlx_clipping:
        print(f"       ✅ Both outputs within audio range [-1, 1]")
    
    # 判断差异严重程度
    if final_diff < 1e-5:
        severity = "微小"
        print(f"       ✅ Differences within acceptable range (< 1e-5)")
    elif final_diff < 1e-3:
        severity = "中等"
        print(f"       ⚠️ Differences moderate (< 1e-3)")
    else:
        severity = "严重"
        print(f"       ❌ Differences too large (> 1e-3)")
    
    comparison_results['final'] = {
        'max_diff': final_diff,
        'mean_diff': mean_diff,
        'relative_diff': relative_diff,
        'pytorch_range': pytorch_range,
        'mlx_range': mlx_range,
        'pytorch_clipping': pytorch_clipping,
        'mlx_clipping': mlx_clipping,
        'severity': severity,
        'pytorch_output': pytorch_output_np,
        'mlx_output': mlx_output_np
    }
    
    return comparison_results


def generate_manual_report(cached_inputs, pytorch_data, mlx_data, comparison_results):
    """生成手动对比报告"""
    report_lines = []
    report_lines.append("# PyTorch vs MLX CFM 手动子模块对比分析报告")
    report_lines.append("="*60)
    report_lines.append("")
    
    report_lines.append("## 🎯 分析目标")
    report_lines.append("")
    report_lines.append("使用 CFM 前级缓存数据作为输入，手动收集和对比 PyTorch 和 MLX 版本的 CFM 实现")
    report_lines.append("中各个子模块的输入输出，找出具体的差异点。")
    report_lines.append("")
    report_lines.append("**关键要求**: 在输入一致的情况下，输出差异应该小于 e-5")
    report_lines.append("")
    
    # 缓存输入分析
    report_lines.append("## 📊 缓存输入分析")
    report_lines.append("")
    report_lines.append("### 输入数据摘要")
    report_lines.append("")
    
    for key, value in cached_inputs.items():
        if isinstance(value, torch.Tensor):
            report_lines.append(f"- **{key}**: {value.shape}, 范围: [{value.min():.6f}, {value.max():.6f}]")
        elif isinstance(value, np.ndarray):
            report_lines.append(f"- **{key}**: {value.shape}, 范围: [{value.min():.6f}, {value.max():.6f}]")
        else:
            report_lines.append(f"- **{key}**: {value}")
    
    report_lines.append("")
    
    # 数据收集状态
    report_lines.append("## 🔍 数据收集状态")
    report_lines.append("")
    
    report_lines.append("### PyTorch CFM")
    report_lines.append("")
    if pytorch_data.get('success'):
        report_lines.append("✅ **数据收集成功**")
        pytorch_submodules = pytorch_data['submodule_data']
        report_lines.append(f"- 子模块数据: {len(pytorch_submodules)} 个阶段")
        
        # 列出各个阶段
        for stage_name in pytorch_submodules.keys():
            report_lines.append(f"  - {stage_name}")
        
        # 最终输出
        if 'final' in pytorch_submodules:
            final_output = pytorch_submodules['final']['output']
            report_lines.append(f"- 最终输出: {final_output.shape}, 范围: [{final_output.min():.6f}, {final_output.max():.6f}]")
    else:
        report_lines.append("❌ **数据收集失败**")
        report_lines.append(f"- 错误: {pytorch_data.get('error', 'Unknown error')}")
    
    report_lines.append("")
    
    report_lines.append("### MLX CFM")
    report_lines.append("")
    if mlx_data.get('success'):
        report_lines.append("✅ **数据收集成功**")
        mlx_submodules = mlx_data['submodule_data']
        report_lines.append(f"- 子模块数据: {len(mlx_submodules)} 个阶段")
        
        # 列出各个阶段
        for stage_name in mlx_submodules.keys():
            report_lines.append(f"  - {stage_name}")
        
        # 最终输出
        if 'final' in mlx_submodules:
            final_output = mlx_submodules['final']['output']
            report_lines.append(f"- 最终输出: {final_output.shape}, 范围: [{float(final_output.min()):.6f}, {float(final_output.max()):.6f}]")
    else:
        report_lines.append("❌ **数据收集失败**")
        report_lines.append(f"- 错误: {mlx_data.get('error', 'Unknown error')}")
    
    report_lines.append("")
    
    # 对比结果分析
    if comparison_results:
        report_lines.append("## 📊 对比结果分析")
        report_lines.append("")
        
        # 初始数据对比
        if 'initial' in comparison_results:
            initial_comp = comparison_results['initial']
            report_lines.append("### 初始数据对比")
            report_lines.append("")
            report_lines.append(f"- **x_processed 差异**: {initial_comp['x_processed_diff']:.2e}")
            report_lines.append(f"- **PyTorch x 范围**: [{initial_comp['pytorch_x_range'][0]:.6f}, {initial_comp['pytorch_x_range'][1]:.6f}]")
            report_lines.append(f"- **MLX x 范围**: [{initial_comp['mlx_x_range'][0]:.6f}, {initial_comp['mlx_x_range'][1]:.6f}]")
            report_lines.append("")
        
        # 步骤对比
        step_keys = [key for key in comparison_results.keys() if key.startswith('step_')]
        if step_keys:
            report_lines.append("### 步骤对比")
            report_lines.append("")
            
            for step_key in sorted(step_keys):
                step_comp = comparison_results[step_key]
                step_num = step_key.split('_')[1]
                
                report_lines.append(f"**步骤 {step_num}**:")
                report_lines.append("")
                report_lines.append(f"- **dphi_dt 差异**: {step_comp['dphi_dt_diff']:.2e}")
                report_lines.append(f"- **x_after 差异**: {step_comp['x_after_diff']:.2e}")
                report_lines.append("")
                report_lines.append(f"- **PyTorch dphi_dt 范围**: [{step_comp['pytorch_dphi_dt_range'][0]:.6f}, {step_comp['pytorch_dphi_dt_range'][1]:.6f}]")
                report_lines.append(f"- **MLX dphi_dt 范围**: [{step_comp['mlx_dphi_dt_range'][0]:.6f}, {step_comp['mlx_dphi_dt_range'][1]:.6f}]")
                report_lines.append("")
                report_lines.append(f"- **PyTorch x_after 范围**: [{step_comp['pytorch_x_after_range'][0]:.6f}, {step_comp['pytorch_x_after_range'][1]:.6f}]")
                report_lines.append(f"- **MLX x_after 范围**: [{step_comp['mlx_x_after_range'][0]:.6f}, {step_comp['mlx_x_after_range'][1]:.6f}]")
                report_lines.append("")
        
        # 最终输出对比
        if 'final' in comparison_results:
            final_comp = comparison_results['final']
            report_lines.append("### 最终输出对比")
            report_lines.append("")
            report_lines.append(f"- **最大差异**: {final_comp['max_diff']:.2e}")
            report_lines.append(f"- **平均差异**: {final_comp['mean_diff']:.2e}")
            report_lines.append(f"- **相对差异**: {final_comp['relative_diff']:.2e}")
            report_lines.append(f"- **差异严重程度**: {final_comp['severity']}")
            report_lines.append("")
            
            report_lines.append(f"- **PyTorch 输出范围**: [{final_comp['pytorch_range'][0]:.6f}, {final_comp['pytorch_range'][1]:.6f}]")
            report_lines.append(f"- **MLX 输出范围**: [{final_comp['mlx_range'][0]:.6f}, {final_comp['mlx_range'][1]:.6f}]")
            report_lines.append("")
            
            if final_comp['pytorch_clipping']:
                report_lines.append("- ⚠️ **PyTorch 输出超出音频范围 [-1, 1]**")
            if final_comp['mlx_clipping']:
                report_lines.append("- ❌ **MLX 输出超出音频范围 [-1, 1]**")
            
            if not final_comp['pytorch_clipping'] and not final_comp['mlx_clipping']:
                report_lines.append("- ✅ **两个版本输出都在音频范围 [-1, 1] 内**")
            
            report_lines.append("")
    
    else:
        report_lines.append("## ❌ 对比失败")
        report_lines.append("")
        report_lines.append("无法进行子模块对比，请检查数据收集过程。")
        report_lines.append("")
    
    # 关键发现
    report_lines.append("## 🎯 关键发现")
    report_lines.append("")
    
    if comparison_results and 'final' in comparison_results:
        final_comp = comparison_results['final']
        
        if final_comp['severity'] == "严重":
            report_lines.append("### ❌ 严重差异")
            report_lines.append("")
            report_lines.append(f"PyTorch 和 MLX 版本存在严重差异 (最大差异: {final_comp['max_diff']:.2e})，")
            report_lines.append("这可能导致音频质量问题。")
            report_lines.append("")
            
            if final_comp['mlx_clipping']:
                report_lines.append("**主要问题**: MLX 输出超出音频范围 [-1, 1]，这是爆音的根本原因。")
                report_lines.append("")
                report_lines.append("**解决方案**: 在 MLX CFM 的最终输出前添加范围限制:")
                report_lines.append("```python")
                report_lines.append("output = mx.clip(output, -1.0, 1.0)")
                report_lines.append("```")
                report_lines.append("")
        
        elif final_comp['severity'] == "中等":
            report_lines.append("### ⚠️ 中等差异")
            report_lines.append("")
            report_lines.append(f"PyTorch 和 MLX 版本存在中等差异 (最大差异: {final_comp['max_diff']:.2e})，")
            report_lines.append("建议进一步调查数值精度问题。")
            report_lines.append("")
        
        else:
            report_lines.append("### ✅ 微小差异")
            report_lines.append("")
            report_lines.append(f"PyTorch 和 MLX 版本差异微小 (最大差异: {final_comp['max_diff']:.2e})，")
            report_lines.append("在可接受范围内。")
            report_lines.append("")
    
    # 建议
    report_lines.append("## 💡 修复建议")
    report_lines.append("")
    
    if comparison_results and 'final' in comparison_results:
        final_comp = comparison_results['final']
        
        if final_comp['mlx_clipping']:
            report_lines.append("### 1. 立即修复音频爆音问题")
            report_lines.append("")
            report_lines.append("**问题**: MLX 输出超出音频范围 [-1, 1]")
            report_lines.append("")
            report_lines.append("**解决方案**:")
            report_lines.append("1. 在 MLX CFM 的最终输出前添加范围限制")
            report_lines.append("2. 使用 `mx.clip(output, -1.0, 1.0)` 或 `mx.tanh(output)`")
            report_lines.append("3. 添加音频范围检查")
            report_lines.append("")
        
        if final_comp['severity'] in ["严重", "中等"]:
            report_lines.append("### 2. 数值精度优化")
            report_lines.append("")
            report_lines.append("**问题**: 数值差异过大")
            report_lines.append("")
            report_lines.append("**解决方案**:")
            report_lines.append("1. 检查 MLX 和 PyTorch 的激活函数实现一致性")
            report_lines.append("2. 验证权重转换精度")
            report_lines.append("3. 使用更高精度的数值计算")
            report_lines.append("4. 添加数值稳定性检查")
            report_lines.append("")
    
    report_lines.append("### 3. 子模块监控")
    report_lines.append("")
    report_lines.append("**建议**:")
    report_lines.append("1. 建立子模块级别的数值监控")
    report_lines.append("2. 检测异常值和溢出")
    report_lines.append("3. 记录数值精度差异")
    report_lines.append("4. 建立音频质量评分")
    report_lines.append("")
    
    # 结论
    report_lines.append("## 📝 结论")
    report_lines.append("")
    
    if comparison_results and 'final' in comparison_results:
        final_comp = comparison_results['final']
        
        if final_comp['mlx_clipping']:
            report_lines.append("**主要发现**: MLX 输出超出音频范围 [-1, 1]，这是爆音的根本原因。")
            report_lines.append("")
            report_lines.append("**立即行动**: 在 MLX CFM 的最终输出前添加范围限制。")
            report_lines.append("")
            report_lines.append("**长期目标**: 确保 PyTorch 和 MLX 版本的数值一致性，")
            report_lines.append("使输出差异小于 e-5。")
        else:
            report_lines.append("**主要发现**: PyTorch 和 MLX 版本存在数值差异，")
            report_lines.append(f"差异严重程度为 {final_comp['severity']}。")
            report_lines.append("")
            report_lines.append("**建议**: 根据差异严重程度采取相应的修复措施。")
    else:
        report_lines.append("**状态**: 数据收集失败，需要重新收集数据进行对比分析。")
    
    report_lines.append("")
    
    # 保存报告
    report_path = Path("cfm_debug_outputs/manual_submodule_comparison_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"     ✅ Manual comparison report saved to: {report_path}")
    
    return report_path


def main():
    """主函数"""
    print("🔍 Manual CFM Submodule Comparison")
    print("="*60)
    
    try:
        # 手动收集和对比子模块数据
        manual_submodule_comparison()
        
        print("\n🎉 Manual submodule comparison completed!")
        print("\n📋 Generated files:")
        print("1. cfm_debug_outputs/manual_submodule_comparison_report.md - 手动子模块对比分析报告")
        
        print("\n📊 Key Features:")
        print("   🔍 使用 CFM 前级缓存数据作为输入")
        print("   📊 手动收集 PyTorch 和 MLX 各个子模块数据")
        print("   🎯 详细对比两个版本各个步骤的输入输出")
        print("   💡 找出具体的差异点")
        print("   🔧 提供详细的修复建议")
        
        print("\n🎯 Analysis Focus:")
        print("   📏 输出差异应该小于 e-5")
        print("   🔍 识别 PyTorch 和 MLX 的差异")
        print("   📊 按严重程度分类差异")
        print("   🛠️ 提供针对性修复建议")
        
    except Exception as e:
        print(f"\n❌ Error during manual comparison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
