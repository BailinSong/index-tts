#!/usr/bin/env python3
"""
使用缓存输入测试 CFM 内部各个子模块的详细对比

确保入口数据一致性，收集 PyTorch 和 MLX 的所有内部子模块数据
"""

import sys
import os
import pickle
import torch
import mlx.core as mx
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append('/Users/bailin/index-tts')

from indextts.s2mel.modules.flow_matching import CFM
from indextts.s2mel.modules.mlx_cfm import MLXCFM
from indextts.utils.cfm_debugger import CFMDebugger


class SimpleConfig:
    """简单的配置类，用于测试"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, SimpleConfig(value))
            else:
                setattr(self, key, value)
    
    def get(self, key, default=None):
        """模拟字典的 get 方法"""
        return getattr(self, key, default)


def load_cached_inputs():
    """加载缓存的输入数据"""
    cache_file = Path("cfm_debug_outputs/cached_inputs.pkl")
    
    if not cache_file.exists():
        print("❌ Cached inputs not found!")
        return None
    
    print("📂 Loading cached inputs...")
    with open(cache_file, 'rb') as f:
        cached_inputs = pickle.load(f)
    
    print(f"✅ Cached inputs loaded from: {cache_file}")
    return cached_inputs


def create_pytorch_cfm(cached_inputs):
    """创建 PyTorch CFM"""
    print("\n🔍 Creating PyTorch CFM...")
    
    config_dict = {
        'DiT': {
            'in_channels': cached_inputs['mel_bins'],
            'hidden_dim': cached_inputs['hidden_dim'],
            'num_heads': 8,
            'depth': 13,
            'time_as_token': False,
            'style_as_token': False,
            'long_skip_connection': True,
            'style_condition': True,
            'is_causal': True,
            'final_layer_type': 'mlp',
            'content_type': 'codebook',
            'content_codebook_size': 1024,
            'content_dim': cached_inputs['hidden_dim'],
            'class_dropout_prob': 0.1,
            'zero_prompt_speech_token': False
        },
        'style_encoder': {
            'dim': cached_inputs['style_dim']
        },
        'reg_loss_type': 'l2',
        'dit_type': 'DiT'
    }
    
    config = SimpleConfig(config_dict)
    pytorch_cfm = CFM(config)
    pytorch_cfm.eval()
    
    # 初始化缓存
    pytorch_cfm.estimator.setup_caches(
        max_batch_size=cached_inputs['batch_size'],
        max_seq_length=cached_inputs['seq_len']
    )
    
    print("✅ PyTorch CFM created")
    return pytorch_cfm


def create_mlx_cfm(cached_inputs):
    """创建 MLX CFM"""
    print("\n🔍 Creating MLX CFM...")
    
    config = {
        'DiT': {
            'in_channels': cached_inputs['mel_bins'],
            'hidden_dim': cached_inputs['hidden_dim'],
            'num_heads': 8,
            'depth': 13,
            'time_as_token': False,
            'style_as_token': False,
            'long_skip_connection': True,
            'style_condition': True,
            'is_causal': True,
            'final_layer_type': 'mlp',
            'content_type': 'codebook',
            'content_codebook_size': 1024,
            'content_dim': cached_inputs['hidden_dim'],
            'class_dropout_prob': 0.1,
            'zero_prompt_speech_token': False
        },
        'style_encoder': {
            'dim': cached_inputs['style_dim']
        }
    }
    
    mlx_cfm = MLXCFM(config)
    
    print("✅ MLX CFM created")
    return mlx_cfm


def run_cached_inputs_submodules_comparison():
    """使用缓存输入运行详细的 CFM 子模块对比"""
    print("🔍 Running Cached Inputs CFM Submodules Detailed Comparison")
    print("="*70)
    
    # 加载缓存输入
    cached_inputs = load_cached_inputs()
    if cached_inputs is None:
        return
    
    # 创建调试器
    debugger = CFMDebugger(debug_dir="cfm_debug_outputs")
    
    # 创建模型
    pytorch_cfm = create_pytorch_cfm(cached_inputs)
    mlx_cfm = create_mlx_cfm(cached_inputs)
    
    # 提取输入数据
    x = cached_inputs['x']
    prompt_x = cached_inputs['prompt_x']
    x_lens = cached_inputs['x_lens']
    style = cached_inputs['style']
    mu = cached_inputs['mu']
    
    # 转换到 MLX
    x_mlx = mx.array(x.numpy())
    prompt_x_mlx = mx.array(prompt_x.numpy())
    x_lens_mlx = mx.array(x_lens.numpy())
    style_mlx = mx.array(style.numpy())
    mu_mlx = mx.array(mu.numpy())
    
    print(f"\n📊 Input data shapes:")
    print(f"   x: PyTorch {x.shape} vs MLX {x_mlx.shape}")
    print(f"   prompt_x: PyTorch {prompt_x.shape} vs MLX {prompt_x_mlx.shape}")
    print(f"   x_lens: PyTorch {x_lens.shape} vs MLX {x_lens_mlx.shape}")
    print(f"   style: PyTorch {style.shape} vs MLX {style_mlx.shape}")
    print(f"   mu: PyTorch {mu.shape} vs MLX {mu_mlx.shape}")
    
    # 验证输入数据一致性
    print(f"\n🔍 Verifying input data consistency:")
    
    # 检查 x
    x_diff = torch.abs(x - torch.from_numpy(np.array(x_mlx))).max()
    print(f"   x max difference: {x_diff:.8f}")
    
    # 检查 prompt_x
    prompt_x_diff = torch.abs(prompt_x - torch.from_numpy(np.array(prompt_x_mlx))).max()
    print(f"   prompt_x max difference: {prompt_x_diff:.8f}")
    
    # 检查 style
    style_diff = torch.abs(style - torch.from_numpy(np.array(style_mlx))).max()
    print(f"   style max difference: {style_diff:.8f}")
    
    # 检查 mu
    mu_diff = torch.abs(mu - torch.from_numpy(np.array(mu_mlx))).max()
    print(f"   mu max difference: {mu_diff:.8f}")
    
    if x_diff < 1e-6 and prompt_x_diff < 1e-6 and style_diff < 1e-6 and mu_diff < 1e-6:
        print("   ✅ Input data is consistent between PyTorch and MLX")
    else:
        print("   ⚠️  Input data has differences - this may affect comparison")
    
    # 运行3步推理并记录调试数据
    print(f"\n🔍 Running detailed submodules comparison with cached inputs...")
    
    for step in range(1, 4):
        print(f"\n   Step {step}...")
        
        # 计算当前时间步
        t_scalar = step / 3.0
        t_tensor = torch.tensor([t_scalar] * cached_inputs['batch_size'])
        t_mlx = mx.array([t_scalar] * cached_inputs['batch_size'])
        
        print(f"     Time step: {t_scalar:.6f}")
        
        # PyTorch 推理 - 使用 solve_euler 启用 debug_layers
        print(f"     Running PyTorch CFM...")
        with torch.no_grad():
            # 创建时间跨度
            t_span = torch.linspace(0, t_scalar, 2)
            pytorch_dphi_dt = pytorch_cfm.solve_euler(
                x=x,
                x_lens=x_lens,
                prompt=prompt_x,
                mu=mu,
                style=style,
                f0=None,
                t_span=t_span,
                inference_cfg_rate=0.0,
                debug_layers=True
            )
        
        # MLX 推理 - 使用 solve_euler 启用 debug_layers
        print(f"     Running MLX CFM...")
        # 创建时间跨度
        t_span_mlx = mx.linspace(0, t_scalar, 2)
        mlx_dphi_dt = mlx_cfm.solve_euler(
            x=x_mlx,
            x_lens=x_lens_mlx,
            prompt=prompt_x_mlx,
            mu=mu_mlx,
            style=style_mlx,
            f0=None,
            t_span=t_span_mlx,
            inference_cfg_rate=0.0,
            debug_layers=True
        )
        
        # 记录最终输出调试数据
        debugger.log_stage(
            "cached_inputs_submodules_comparison",
            pytorch_data={'dphi_dt': pytorch_dphi_dt},
            mlx_data={'dphi_dt': mlx_dphi_dt},
            step=step,
            layer=0,
            additional_info={
                'cached_inputs': True, 
                'step': step,
                'time_step': t_scalar,
                'input_consistent': x_diff < 1e-6 and prompt_x_diff < 1e-6 and style_diff < 1e-6 and mu_diff < 1e-6
            }
        )
        
        print(f"       PyTorch output: {pytorch_dphi_dt.shape}, range: [{pytorch_dphi_dt.min():.6f}, {pytorch_dphi_dt.max():.6f}]")
        print(f"       MLX output: {mlx_dphi_dt.shape}, range: [{mlx_dphi_dt.min():.6f}, {mlx_dphi_dt.max():.6f}]")
        
        # 计算输出差异
        output_diff = torch.abs(pytorch_dphi_dt - torch.from_numpy(np.array(mlx_dphi_dt))).max()
        print(f"       Output max difference: {output_diff:.8f}")
    
    # 保存调试数据
    debugger.save_debug_data("cached_inputs_submodules_comparison.pkl")
    
    print("\n✅ Detailed submodules comparison with cached inputs completed!")
    print("\n📋 Generated files:")
    print("1. cfm_debug_outputs/cached_inputs_submodules_comparison.pkl - 详细子模块调试数据")
    
    return debugger


def analyze_cached_inputs_submodules_comparison():
    """分析缓存输入的详细子模块对比结果"""
    print("\n🔍 Analyzing cached inputs submodules comparison...")
    
    # 加载调试数据
    debug_file = Path("cfm_debug_outputs/cached_inputs_submodules_comparison.pkl")
    
    if not debug_file.exists():
        print("❌ Debug file not found!")
        return
    
    with open(debug_file, 'rb') as f:
        data = pickle.load(f)
    
    pytorch_data = data['pytorch']
    mlx_data = data['mlx']
    comparisons = data['comparisons']
    
    print(f"📊 Debug data loaded:")
    print(f"   PyTorch stages: {len(pytorch_data)}")
    print(f"   MLX stages: {len(mlx_data)}")
    print(f"   Comparisons: {len(comparisons)}")
    
    # 显示所有可用的阶段
    print(f"\n📋 Available PyTorch stages:")
    for key in sorted(pytorch_data.keys()):
        print(f"   {key}")
    
    print(f"\n📋 Available MLX stages:")
    for key in sorted(mlx_data.keys()):
        print(f"   {key}")
    
    # 分析内部子模块
    print(f"\n🔍 Internal Submodules Analysis:")
    
    # 按阶段类型分组
    pytorch_stage_types = {}
    mlx_stage_types = {}
    
    for stage_name in pytorch_data.keys():
        parts = stage_name.split('_')
        if len(parts) >= 3:
            stage_type = '_'.join(parts[2:])  # 去掉 step_X_layer_Y_
            if stage_type not in pytorch_stage_types:
                pytorch_stage_types[stage_type] = []
            pytorch_stage_types[stage_type].append(stage_name)
    
    for stage_name in mlx_data.keys():
        parts = stage_name.split('_')
        if len(parts) >= 3:
            stage_type = '_'.join(parts[2:])  # 去掉 step_X_layer_Y_
            if stage_type not in mlx_stage_types:
                mlx_stage_types[stage_type] = []
            mlx_stage_types[stage_type].append(stage_name)
    
    print(f"\n   PyTorch submodules:")
    for stage_type, stage_names in pytorch_stage_types.items():
        print(f"     {stage_type}: {len(stage_names)} stages")
    
    print(f"\n   MLX submodules:")
    for stage_type, stage_names in mlx_stage_types.items():
        print(f"     {stage_type}: {len(stage_names)} stages")
    
    # 分析对比结果
    print(f"\n🔍 Comparison Analysis:")
    
    total_tensors = 0
    close_tensors = 0
    significant_diffs = []
    
    for key in sorted(comparisons.keys()):
        print(f"\n📋 {key}:")
        
        comp_data = comparisons[key]
        
        if 'tensor_comparisons' in comp_data:
            for tensor_name, tensor_comp in comp_data['tensor_comparisons'].items():
                total_tensors += 1
                
                print(f"   {tensor_name}:")
                
                # 形状对比
                shape_match = tensor_comp.get('shape_match', False)
                print(f"     Shape match: {'✅' if shape_match else '❌'}")
                
                # 数值对比
                max_diff = tensor_comp.get('max_diff', 0)
                mean_diff = tensor_comp.get('mean_diff', 0)
                relative_diff = tensor_comp.get('relative_diff', 0)
                is_close = tensor_comp.get('is_close', False)
                
                print(f"     Max diff: {max_diff:.8f}")
                print(f"     Mean diff: {mean_diff:.8f}")
                print(f"     Relative diff: {relative_diff:.8f}")
                print(f"     Is close: {'✅' if is_close else '❌'}")
                
                if is_close:
                    close_tensors += 1
                else:
                    significant_diffs.append({
                        'stage': key,
                        'tensor': tensor_name,
                        'max_diff': max_diff,
                        'relative_diff': relative_diff
                    })
                
                # 统计信息
                pytorch_stats = tensor_comp.get('pytorch_stats', {})
                mlx_stats = tensor_comp.get('mlx_stats', {})
                
                if pytorch_stats and mlx_stats:
                    print(f"     PyTorch range: [{pytorch_stats.get('min', 0):.6f}, {pytorch_stats.get('max', 0):.6f}]")
                    print(f"     MLX range: [{mlx_stats.get('min', 0):.6f}, {mlx_stats.get('max', 0):.6f}]")
        
        # 显示额外信息
        if 'additional_info' in comp_data:
            print(f"   Additional info: {comp_data['additional_info']}")
    
    # 总结
    print(f"\n📊 Summary:")
    print(f"   Total tensors compared: {total_tensors}")
    print(f"   Close tensors: {close_tensors}")
    print(f"   Significant differences: {len(significant_diffs)}")
    
    if significant_diffs:
        print(f"\n⚠️  Significant Differences:")
        for diff in significant_diffs:
            print(f"   {diff['stage']} - {diff['tensor']}: "
                  f"max_diff={diff['max_diff']:.8f}, "
                  f"relative_diff={diff['relative_diff']:.8f}")
    else:
        print(f"\n✅ No significant differences found!")
    
    return {
        'total_tensors': total_tensors,
        'close_tensors': close_tensors,
        'significant_diffs': significant_diffs,
        'pytorch_stage_types': pytorch_stage_types,
        'mlx_stage_types': mlx_stage_types
    }


def main():
    """主函数"""
    print("🔍 Cached Inputs CFM Submodules Detailed Comparison")
    print("="*70)
    
    try:
        # 运行缓存输入的详细子模块对比
        debugger = run_cached_inputs_submodules_comparison()
        
        if debugger:
            # 分析对比结果
            analysis_results = analyze_cached_inputs_submodules_comparison()
            
            print("\n🎉 All tests completed successfully!")
            print("\n📋 Summary:")
            print(f"   Total tensors compared: {analysis_results['total_tensors']}")
            print(f"   Close tensors: {analysis_results['close_tensors']}")
            print(f"   Significant differences: {len(analysis_results['significant_diffs'])}")
            print(f"   PyTorch submodules: {len(analysis_results['pytorch_stage_types'])}")
            print(f"   MLX submodules: {len(analysis_results['mlx_stage_types'])}")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
