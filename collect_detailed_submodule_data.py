#!/usr/bin/env python3
"""
使用 CFM 前级缓存数据对比 PyTorch 和 MLX CFM 各个子模块的输入输出

使用正确的缓存输入数据，分别运行 PyTorch 和 MLX 版本的 CFM 实现，
详细收集两个版本各个子模块的输入输出，然后进行对比分析
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


def collect_detailed_submodule_data():
    """收集详细的子模块数据"""
    print("🔍 Collecting Detailed CFM Submodule Data")
    print("="*60)
    
    # 加载正确的缓存输入数据
    cached_inputs = load_correct_cached_inputs()
    if not cached_inputs:
        print("❌ No correct cached inputs available!")
        return
    
    print("\n📊 Using Correct Cached Inputs:")
    print_cached_inputs_summary(cached_inputs)
    
    # 收集 PyTorch CFM 详细子模块数据
    print("\n🔍 Collecting PyTorch CFM Detailed Submodule Data:")
    pytorch_submodule_data = collect_pytorch_submodule_data_detailed(cached_inputs)
    
    # 收集 MLX CFM 详细子模块数据
    print("\n🔍 Collecting MLX CFM Detailed Submodule Data:")
    mlx_submodule_data = collect_mlx_submodule_data_detailed(cached_inputs)
    
    # 对比各个子模块
    print("\n📊 Detailed Submodule Comparison:")
    comparison_results = compare_detailed_submodules(pytorch_submodule_data, mlx_submodule_data)
    
    # 生成详细对比报告
    print("\n📄 Generating Detailed Comparison Report:")
    generate_detailed_report(cached_inputs, pytorch_submodule_data, mlx_submodule_data, comparison_results)
    
    print("\n🎉 Detailed submodule data collection completed!")


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


def collect_pytorch_submodule_data_detailed(cached_inputs):
    """收集 PyTorch CFM 详细子模块数据"""
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
        
        # 运行 CFM 推理并收集中间数据
        print(f"\n     🔍 Running PyTorch CFM with detailed submodule tracking:")
        
        # 创建调试器来收集子模块数据
        from indextts.utils.cfm_debugger import CFMDebugger
        debugger = CFMDebugger(debug_dir="cfm_debug_outputs")
        
        with torch.no_grad():
            # 启用调试模式
            pytorch_cfm._debug_layers = True
            
            # 运行 CFM 推理
            output = pytorch_cfm.solve_euler(
                x=x,
                x_lens=x_lens,
                prompt=prompt_x,
                mu=mu,
                style=style,
                f0=None,
                t_span=t_span,
                inference_cfg_rate=0.0,  # 禁用 CFG 模式
                debug_layers=True
            )
            
            print(f"     PyTorch CFM output shape: {output.shape}")
            print(f"     PyTorch CFM output range: [{output.min():.6f}, {output.max():.6f}]")
        
        # 保存调试数据
        debugger.save_debug_data("pytorch_detailed_submodules")
        
        return {
            'output': output,
            'config': config_dict,
            'success': True,
            'input_data': {
                'x': x,
                'prompt_x': prompt_x,
                'x_lens': x_lens,
                'style': style,
                'mu': mu,
                't_span': t_span
            },
            'debugger': debugger
        }
        
    except Exception as e:
        print(f"     ❌ PyTorch detailed data collection failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'success': False
        }


def collect_mlx_submodule_data_detailed(cached_inputs):
    """收集 MLX CFM 详细子模块数据"""
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
        
        # 运行 CFM 推理
        print(f"\n     🔍 Running MLX CFM with detailed submodule tracking:")
        
        # 创建调试器来收集子模块数据
        from indextts.utils.cfm_debugger import CFMDebugger
        debugger = CFMDebugger(debug_dir="cfm_debug_outputs")
        
        output = mlx_cfm.solve_euler(
            x=x_mlx,
            x_lens=x_lens_mlx,
            prompt=prompt_x_mlx,
            mu=mu_mlx,
            style=style_mlx,
            f0=None,
            t_span=t_span_mlx,
            inference_cfg_rate=0.0,  # 禁用 CFG 模式
            debug_layers=True
        )
        
        print(f"     MLX CFM output shape: {output.shape}")
        print(f"     MLX CFM output range: [{float(output.min()):.6f}, {float(output.max()):.6f}]")
        
        # 保存调试数据
        debugger.save_debug_data("mlx_detailed_submodules")
        
        return {
            'output': output,
            'config': config_dict,
            'success': True,
            'input_data': {
                'x': x_mlx,
                'prompt_x': prompt_x_mlx,
                'x_lens': x_lens_mlx,
                'style': style_mlx,
                'mu': mu_mlx,
                't_span': t_span_mlx
            },
            'debugger': debugger
        }
        
    except Exception as e:
        print(f"     ❌ MLX detailed data collection failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'success': False
        }


def compare_detailed_submodules(pytorch_data, mlx_data):
    """对比详细的子模块数据"""
    print("\n   📋 Detailed Submodule Comparison:")
    
    if not pytorch_data.get('success') or not mlx_data.get('success'):
        print("     ❌ Cannot compare: one or both data collection failed")
        return None
    
    # 对比最终输出
    pytorch_output = pytorch_data['output']
    mlx_output = mlx_data['output']
    
    # 转换为 numpy 进行对比
    pytorch_np = pytorch_output.detach().cpu().numpy()
    mlx_np = np.array(mlx_output)
    
    # 计算差异
    max_diff = np.max(np.abs(pytorch_np - mlx_np))
    mean_diff = np.mean(np.abs(pytorch_np - mlx_np))
    relative_diff = mean_diff / (np.mean(np.abs(pytorch_np)) + 1e-8)
    
    print(f"     Final Output Comparison:")
    print(f"       Max difference: {max_diff:.2e}")
    print(f"       Mean difference: {mean_diff:.2e}")
    print(f"       Relative difference: {relative_diff:.2e}")
    
    # 检查输出范围
    pytorch_range = [pytorch_np.min(), pytorch_np.max()]
    mlx_range = [mlx_np.min(), mlx_np.max()]
    
    print(f"       PyTorch range: [{pytorch_range[0]:.6f}, {pytorch_range[1]:.6f}]")
    print(f"       MLX range: [{mlx_range[0]:.6f}, {mlx_range[1]:.6f}]")
    
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
    if max_diff < 1e-5:
        severity = "微小"
        print(f"       ✅ Differences within acceptable range (< 1e-5)")
    elif max_diff < 1e-3:
        severity = "中等"
        print(f"       ⚠️ Differences moderate (< 1e-3)")
    else:
        severity = "严重"
        print(f"       ❌ Differences too large (> 1e-3)")
    
    # 分析子模块数据
    print(f"\n     Submodule Data Analysis:")
    
    # 检查是否有调试数据
    pytorch_debugger = pytorch_data.get('debugger')
    mlx_debugger = mlx_data.get('debugger')
    
    if pytorch_debugger and mlx_debugger:
        print(f"       ✅ Both PyTorch and MLX debuggers available")
        
        # 分析调试数据
        pytorch_debug_data = pytorch_debugger.debug_data
        mlx_debug_data = mlx_debugger.debug_data
        
        print(f"       PyTorch debug stages: {len(pytorch_debug_data)}")
        print(f"       MLX debug stages: {len(mlx_debug_data)}")
        
        # 对比各个阶段
        if pytorch_debug_data and mlx_debug_data:
            print(f"\n       Stage-by-stage comparison:")
            
            # 获取共同的阶段
            pytorch_stages = set(pytorch_debug_data.keys())
            mlx_stages = set(mlx_debug_data.keys())
            common_stages = pytorch_stages.intersection(mlx_stages)
            
            print(f"       Common stages: {len(common_stages)}")
            
            for stage in sorted(common_stages):
                pytorch_stage_data = pytorch_debug_data[stage]
                mlx_stage_data = mlx_debug_data[stage]
                
                print(f"         {stage}:")
                print(f"           PyTorch steps: {len(pytorch_stage_data)}")
                print(f"           MLX steps: {len(mlx_stage_data)}")
                
                # 对比每个步骤
                if pytorch_stage_data and mlx_stage_data:
                    pytorch_steps = set(pytorch_stage_data.keys())
                    mlx_steps = set(mlx_stage_data.keys())
                    common_steps = pytorch_steps.intersection(mlx_steps)
                    
                    if common_steps:
                        print(f"           Common steps: {len(common_steps)}")
                        
                        # 对比第一个共同步骤
                        first_step = min(common_steps)
                        pytorch_step_data = pytorch_stage_data[first_step]
                        mlx_step_data = mlx_stage_data[first_step]
                        
                        if 'tensor_comparisons' in pytorch_step_data and 'tensor_comparisons' in mlx_step_data:
                            pytorch_comparisons = pytorch_step_data['tensor_comparisons']
                            mlx_comparisons = mlx_step_data['tensor_comparisons']
                            
                            print(f"           Step {first_step} tensor comparisons:")
                            print(f"             PyTorch: {len(pytorch_comparisons)}")
                            print(f"             MLX: {len(mlx_comparisons)}")
                            
                            # 对比具体的张量
                            pytorch_tensors = set(pytorch_comparisons.keys())
                            mlx_tensors = set(mlx_comparisons.keys())
                            common_tensors = pytorch_tensors.intersection(mlx_tensors)
                            
                            if common_tensors:
                                print(f"             Common tensors: {len(common_tensors)}")
                                for tensor_name in sorted(common_tensors):
                                    pytorch_tensor_info = pytorch_comparisons[tensor_name]
                                    mlx_tensor_info = mlx_comparisons[tensor_name]
                                    
                                    print(f"               {tensor_name}:")
                                    print(f"                 PyTorch: {pytorch_tensor_info.get('shape', 'N/A')}")
                                    print(f"                 MLX: {mlx_tensor_info.get('shape', 'N/A')}")
                    else:
                        print(f"           No common steps found")
    else:
        print(f"       ❌ Debug data not available")
    
    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'relative_diff': relative_diff,
        'pytorch_range': pytorch_range,
        'mlx_range': mlx_range,
        'pytorch_clipping': pytorch_clipping,
        'mlx_clipping': mlx_clipping,
        'severity': severity,
        'pytorch_output': pytorch_np,
        'mlx_output': mlx_np,
        'pytorch_debugger': pytorch_debugger,
        'mlx_debugger': mlx_debugger
    }


def generate_detailed_report(cached_inputs, pytorch_data, mlx_data, comparison_results):
    """生成详细对比报告"""
    report_lines = []
    report_lines.append("# PyTorch vs MLX CFM 详细子模块对比分析报告")
    report_lines.append("="*60)
    report_lines.append("")
    
    report_lines.append("## 🎯 分析目标")
    report_lines.append("")
    report_lines.append("使用 CFM 前级缓存数据作为输入，详细对比 PyTorch 和 MLX 版本的 CFM 实现")
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
        pytorch_output = pytorch_data['output']
        report_lines.append(f"- 输出形状: {pytorch_output.shape}")
        report_lines.append(f"- 输出范围: [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]")
        
        pytorch_debugger = pytorch_data.get('debugger')
        if pytorch_debugger:
            report_lines.append(f"- 调试数据: {len(pytorch_debugger.debug_data)} 个阶段")
        else:
            report_lines.append("- 调试数据: 不可用")
    else:
        report_lines.append("❌ **数据收集失败**")
        report_lines.append(f"- 错误: {pytorch_data.get('error', 'Unknown error')}")
    
    report_lines.append("")
    
    report_lines.append("### MLX CFM")
    report_lines.append("")
    if mlx_data.get('success'):
        report_lines.append("✅ **数据收集成功**")
        mlx_output = mlx_data['output']
        report_lines.append(f"- 输出形状: {mlx_output.shape}")
        report_lines.append(f"- 输出范围: [{float(mlx_output.min()):.6f}, {float(mlx_output.max()):.6f}]")
        
        mlx_debugger = mlx_data.get('debugger')
        if mlx_debugger:
            report_lines.append(f"- 调试数据: {len(mlx_debugger.debug_data)} 个阶段")
        else:
            report_lines.append("- 调试数据: 不可用")
    else:
        report_lines.append("❌ **数据收集失败**")
        report_lines.append(f"- 错误: {mlx_data.get('error', 'Unknown error')}")
    
    report_lines.append("")
    
    # 对比结果分析
    if comparison_results:
        report_lines.append("## 📊 对比结果分析")
        report_lines.append("")
        
        report_lines.append("### 最终输出对比")
        report_lines.append("")
        report_lines.append(f"- **最大差异**: {comparison_results['max_diff']:.2e}")
        report_lines.append(f"- **平均差异**: {comparison_results['mean_diff']:.2e}")
        report_lines.append(f"- **相对差异**: {comparison_results['relative_diff']:.2e}")
        report_lines.append(f"- **差异严重程度**: {comparison_results['severity']}")
        report_lines.append("")
        
        report_lines.append("### 输出范围分析")
        report_lines.append("")
        report_lines.append(f"- **PyTorch 输出范围**: [{comparison_results['pytorch_range'][0]:.6f}, {comparison_results['pytorch_range'][1]:.6f}]")
        report_lines.append(f"- **MLX 输出范围**: [{comparison_results['mlx_range'][0]:.6f}, {comparison_results['mlx_range'][1]:.6f}]")
        report_lines.append("")
        
        if comparison_results['pytorch_clipping']:
            report_lines.append("- ⚠️ **PyTorch 输出超出音频范围 [-1, 1]**")
        if comparison_results['mlx_clipping']:
            report_lines.append("- ❌ **MLX 输出超出音频范围 [-1, 1]**")
        
        if not comparison_results['pytorch_clipping'] and not comparison_results['mlx_clipping']:
            report_lines.append("- ✅ **两个版本输出都在音频范围 [-1, 1] 内**")
        
        report_lines.append("")
        
        # 子模块分析
        report_lines.append("### 子模块数据分析")
        report_lines.append("")
        
        pytorch_debugger = comparison_results.get('pytorch_debugger')
        mlx_debugger = comparison_results.get('mlx_debugger')
        
        if pytorch_debugger and mlx_debugger:
            report_lines.append("✅ **子模块数据可用**")
            report_lines.append("")
            
            pytorch_debug_data = pytorch_debugger.debug_data
            mlx_debug_data = mlx_debugger.debug_data
            
            report_lines.append(f"- **PyTorch 调试阶段**: {len(pytorch_debug_data)}")
            report_lines.append(f"- **MLX 调试阶段**: {len(mlx_debug_data)}")
            report_lines.append("")
            
            # 分析各个阶段
            pytorch_stages = set(pytorch_debug_data.keys())
            mlx_stages = set(mlx_debug_data.keys())
            common_stages = pytorch_stages.intersection(mlx_stages)
            
            report_lines.append(f"- **共同阶段**: {len(common_stages)}")
            report_lines.append("")
            
            if common_stages:
                report_lines.append("#### 阶段详细分析")
                report_lines.append("")
                
                for stage in sorted(common_stages):
                    report_lines.append(f"**{stage}**:")
                    report_lines.append("")
                    
                    pytorch_stage_data = pytorch_debug_data[stage]
                    mlx_stage_data = mlx_debug_data[stage]
                    
                    report_lines.append(f"- PyTorch 步骤数: {len(pytorch_stage_data)}")
                    report_lines.append(f"- MLX 步骤数: {len(mlx_stage_data)}")
                    
                    # 对比每个步骤
                    if pytorch_stage_data and mlx_stage_data:
                        pytorch_steps = set(pytorch_stage_data.keys())
                        mlx_steps = set(mlx_stage_data.keys())
                        common_steps = pytorch_steps.intersection(mlx_steps)
                        
                        if common_steps:
                            report_lines.append(f"- 共同步骤数: {len(common_steps)}")
                            
                            # 分析第一个共同步骤
                            first_step = min(common_steps)
                            pytorch_step_data = pytorch_stage_data[first_step]
                            mlx_step_data = mlx_stage_data[first_step]
                            
                            if 'tensor_comparisons' in pytorch_step_data and 'tensor_comparisons' in mlx_step_data:
                                pytorch_comparisons = pytorch_step_data['tensor_comparisons']
                                mlx_comparisons = mlx_step_data['tensor_comparisons']
                                
                                report_lines.append(f"- 步骤 {first_step} 张量对比:")
                                report_lines.append(f"  - PyTorch 张量数: {len(pytorch_comparisons)}")
                                report_lines.append(f"  - MLX 张量数: {len(mlx_comparisons)}")
                                
                                # 对比具体的张量
                                pytorch_tensors = set(pytorch_comparisons.keys())
                                mlx_tensors = set(mlx_comparisons.keys())
                                common_tensors = pytorch_tensors.intersection(mlx_tensors)
                                
                                if common_tensors:
                                    report_lines.append(f"  - 共同张量数: {len(common_tensors)}")
                                    report_lines.append("")
                                    
                                    for tensor_name in sorted(common_tensors):
                                        pytorch_tensor_info = pytorch_comparisons[tensor_name]
                                        mlx_tensor_info = mlx_comparisons[tensor_name]
                                        
                                        report_lines.append(f"    **{tensor_name}**:")
                                        report_lines.append(f"    - PyTorch: {pytorch_tensor_info.get('shape', 'N/A')}")
                                        report_lines.append(f"    - MLX: {mlx_tensor_info.get('shape', 'N/A')}")
                                        report_lines.append("")
                        else:
                            report_lines.append("- 无共同步骤")
                    
                    report_lines.append("")
        else:
            report_lines.append("❌ **子模块数据不可用**")
            report_lines.append("")
    
    else:
        report_lines.append("## ❌ 对比失败")
        report_lines.append("")
        report_lines.append("无法进行子模块对比，请检查数据收集过程。")
        report_lines.append("")
    
    # 关键发现
    report_lines.append("## 🎯 关键发现")
    report_lines.append("")
    
    if comparison_results:
        if comparison_results['severity'] == "严重":
            report_lines.append("### ❌ 严重差异")
            report_lines.append("")
            report_lines.append(f"PyTorch 和 MLX 版本存在严重差异 (最大差异: {comparison_results['max_diff']:.2e})，")
            report_lines.append("这可能导致音频质量问题。")
            report_lines.append("")
            
            if comparison_results['mlx_clipping']:
                report_lines.append("**主要问题**: MLX 输出超出音频范围 [-1, 1]，这是爆音的根本原因。")
                report_lines.append("")
                report_lines.append("**解决方案**: 在 MLX CFM 的最终输出前添加范围限制:")
                report_lines.append("```python")
                report_lines.append("output = mx.clip(output, -1.0, 1.0)")
                report_lines.append("```")
                report_lines.append("")
        
        elif comparison_results['severity'] == "中等":
            report_lines.append("### ⚠️ 中等差异")
            report_lines.append("")
            report_lines.append(f"PyTorch 和 MLX 版本存在中等差异 (最大差异: {comparison_results['max_diff']:.2e})，")
            report_lines.append("建议进一步调查数值精度问题。")
            report_lines.append("")
        
        else:
            report_lines.append("### ✅ 微小差异")
            report_lines.append("")
            report_lines.append(f"PyTorch 和 MLX 版本差异微小 (最大差异: {comparison_results['max_diff']:.2e})，")
            report_lines.append("在可接受范围内。")
            report_lines.append("")
    
    # 建议
    report_lines.append("## 💡 修复建议")
    report_lines.append("")
    
    if comparison_results and comparison_results['mlx_clipping']:
        report_lines.append("### 1. 立即修复音频爆音问题")
        report_lines.append("")
        report_lines.append("**问题**: MLX 输出超出音频范围 [-1, 1]")
        report_lines.append("")
        report_lines.append("**解决方案**:")
        report_lines.append("1. 在 MLX CFM 的最终输出前添加范围限制")
        report_lines.append("2. 使用 `mx.clip(output, -1.0, 1.0)` 或 `mx.tanh(output)`")
        report_lines.append("3. 添加音频范围检查")
        report_lines.append("")
    
    if comparison_results and comparison_results['severity'] in ["严重", "中等"]:
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
    
    if comparison_results:
        if comparison_results['mlx_clipping']:
            report_lines.append("**主要发现**: MLX 输出超出音频范围 [-1, 1]，这是爆音的根本原因。")
            report_lines.append("")
            report_lines.append("**立即行动**: 在 MLX CFM 的最终输出前添加范围限制。")
            report_lines.append("")
            report_lines.append("**长期目标**: 确保 PyTorch 和 MLX 版本的数值一致性，")
            report_lines.append("使输出差异小于 e-5。")
        else:
            report_lines.append("**主要发现**: PyTorch 和 MLX 版本存在数值差异，")
            report_lines.append(f"差异严重程度为 {comparison_results['severity']}。")
            report_lines.append("")
            report_lines.append("**建议**: 根据差异严重程度采取相应的修复措施。")
    else:
        report_lines.append("**状态**: 数据收集失败，需要重新收集数据进行对比分析。")
    
    report_lines.append("")
    
    # 保存报告
    report_path = Path("cfm_debug_outputs/detailed_submodule_comparison_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"     ✅ Detailed comparison report saved to: {report_path}")
    
    return report_path


def main():
    """主函数"""
    print("🔍 Detailed CFM Submodule Data Collection")
    print("="*60)
    
    try:
        # 收集详细的子模块数据
        collect_detailed_submodule_data()
        
        print("\n🎉 Detailed submodule data collection completed!")
        print("\n📋 Generated files:")
        print("1. cfm_debug_outputs/detailed_submodule_comparison_report.md - 详细子模块对比分析报告")
        print("2. cfm_debug_outputs/pytorch_detailed_submodules.pkl - PyTorch 详细子模块数据")
        print("3. cfm_debug_outputs/mlx_detailed_submodules.pkl - MLX 详细子模块数据")
        
        print("\n📊 Key Features:")
        print("   🔍 使用 CFM 前级缓存数据作为输入")
        print("   📊 详细收集 PyTorch 和 MLX 各个子模块数据")
        print("   🎯 对比两个版本各个子模块的输入输出")
        print("   💡 找出具体的差异点")
        print("   🔧 提供详细的修复建议")
        
        print("\n🎯 Analysis Focus:")
        print("   📏 输出差异应该小于 e-5")
        print("   🔍 识别 PyTorch 和 MLX 的差异")
        print("   📊 按严重程度分类差异")
        print("   🛠️ 提供针对性修复建议")
        
    except Exception as e:
        print(f"\n❌ Error during detailed data collection: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


