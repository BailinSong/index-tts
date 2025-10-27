#!/usr/bin/env python3
"""
使用缓存输入收集 CFM 子模块详细数据

使用 CFM 前级缓存作为输入，收集 PyTorch 和 MLX 版本 CFM 中各个子模块的详细输入输出数据
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

from indextts.utils.cfm_debugger import CFMDebugger


def collect_cached_submodule_data():
    """使用缓存输入收集子模块详细数据"""
    print("🔍 Collecting Cached Submodule Data")
    print("="*60)
    
    # 加载缓存输入数据
    cached_inputs = load_cached_inputs()
    if not cached_inputs:
        print("❌ No cached inputs available!")
        return
    
    print("\n📊 Cached Inputs Summary:")
    print_cached_inputs_summary(cached_inputs)
    
    # 初始化调试器
    debugger = CFMDebugger(debug_dir="cfm_debug_outputs")
    
    # 收集 PyTorch 子模块数据
    print("\n🔍 Collecting PyTorch Submodule Data:")
    pytorch_data = collect_pytorch_submodule_data(cached_inputs, debugger)
    
    # 收集 MLX 子模块数据
    print("\n🔍 Collecting MLX Submodule Data:")
    mlx_data = collect_mlx_submodule_data(cached_inputs, debugger)
    
    # 保存详细数据
    save_detailed_submodule_data(pytorch_data, mlx_data, debugger)
    
    print("\n🎉 Data collection completed successfully!")


def load_cached_inputs():
    """加载缓存输入数据"""
    cached_file = Path("cfm_debug_outputs/cached_inputs.pkl")
    
    if not cached_file.exists():
        print("❌ Cached inputs file not found!")
        return None
    
    with open(cached_file, 'rb') as f:
        cached_inputs = pickle.load(f)
    
    print(f"✅ Cached inputs loaded from: {cached_file}")
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


def collect_pytorch_submodule_data(cached_inputs, debugger):
    """收集 PyTorch 子模块数据"""
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
            'reg_loss_type': 'l2'  # 添加缺失的配置
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
        
        # 设置调试模式
        pytorch_cfm._debug_layers = True
        
        # 准备输入数据
        x = cached_inputs['x']
        prompt_x = cached_inputs['prompt_x']
        x_lens = cached_inputs['x_lens']
        style = cached_inputs['style']
        mu = cached_inputs['mu']
        
        print(f"     Input shapes: x={x.shape}, prompt_x={prompt_x.shape}, x_lens={x_lens.shape}")
        print(f"     Input shapes: style={style.shape}, mu={mu.shape}")
        
        # 运行 CFM 推理并收集子模块数据
        with torch.no_grad():
            # 设置调试标志
            pytorch_cfm._debug_layers = True
            
            # 创建时间跨度
            t_span = torch.linspace(0, 1, 4)  # 3 steps + 1
            
            # 运行 solve_euler 方法
            output = pytorch_cfm.solve_euler(
                x=x,
                x_lens=x_lens,
                prompt=prompt_x,  # 修正参数名
                mu=mu,
                style=style,
                f0=None,  # 添加 f0 参数
                t_span=t_span,
                debug_layers=True
            )
            
            print(f"     PyTorch output shape: {output.shape}")
            print(f"     PyTorch output range: [{output.min():.6f}, {output.max():.6f}]")
        
        return {
            'output': output,
            'config': config_dict,
            'success': True
        }
        
    except Exception as e:
        print(f"     ❌ PyTorch data collection failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'success': False
        }


def collect_mlx_submodule_data(cached_inputs, debugger):
    """收集 MLX 子模块数据"""
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
        
        # 运行 CFM 推理并收集子模块数据
        # 设置调试模式
        mlx_cfm._debug_layers = True
        
        # 创建时间跨度
        t_span_mlx = mx.array(np.linspace(0, 1, 4))  # 3 steps + 1
        
        # 运行 solve_euler 方法
        output = mlx_cfm.solve_euler(
            x=x_mlx,
            x_lens=x_lens_mlx,
            prompt=prompt_x_mlx,  # 修正参数名
            mu=mu_mlx,
            style=style_mlx,
            f0=None,  # 添加 f0 参数
            t_span=t_span_mlx,
            debug_layers=True
        )
        
        print(f"     MLX output shape: {output.shape}")
        print(f"     MLX output range: [{float(output.min()):.6f}, {float(output.max()):.6f}]")
        
        return {
            'output': output,
            'config': config_dict,
            'success': True
        }
        
    except Exception as e:
        print(f"     ❌ MLX data collection failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'success': False
        }


def save_detailed_submodule_data(pytorch_data, mlx_data, debugger):
    """保存详细子模块数据"""
    print("\n💾 Saving detailed submodule data...")
    
    # 保存调试数据
    debugger.save_debug_data("cached_detailed_submodules_comparison.pkl")
    
    # 创建额外的数据文件
    detailed_data = {
        'pytorch': pytorch_data,
        'mlx': mlx_data,
        'timestamp': str(Path().cwd()),
        'description': 'Detailed submodule data collected with cached inputs'
    }
    
    output_file = Path("cfm_debug_outputs/cached_detailed_submodules_data.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(detailed_data, f)
    
    print(f"     ✅ Detailed data saved to: {output_file}")
    
    # 如果两个版本都成功，进行对比分析
    if pytorch_data.get('success') and mlx_data.get('success'):
        print("\n📊 Comparing PyTorch vs MLX outputs:")
        
        pytorch_output = pytorch_data['output']
        mlx_output = mlx_data['output']
        
        # 转换为 numpy 进行对比
        pytorch_np = pytorch_output.detach().cpu().numpy()
        mlx_np = np.array(mlx_output)
        
        # 计算差异
        max_diff = np.max(np.abs(pytorch_np - mlx_np))
        mean_diff = np.mean(np.abs(pytorch_np - mlx_np))
        
        print(f"     Max difference: {max_diff:.2e}")
        print(f"     Mean difference: {mean_diff:.2e}")
        
        if max_diff < 1e-5:
            print(f"     ✅ Differences within acceptable range (< 1e-5)")
        elif max_diff < 1e-3:
            print(f"     ⚠️ Differences moderate (< 1e-3)")
        else:
            print(f"     ❌ Differences too large (> 1e-3)")
        
        # 检查输出范围
        pytorch_range = [pytorch_np.min(), pytorch_np.max()]
        mlx_range = [mlx_np.min(), mlx_np.max()]
        
        print(f"     PyTorch range: [{pytorch_range[0]:.6f}, {pytorch_range[1]:.6f}]")
        print(f"     MLX range: [{mlx_range[0]:.6f}, {mlx_range[1]:.6f}]")
        
        # 检查音频范围
        pytorch_clipping = pytorch_range[0] < -1.0 or pytorch_range[1] > 1.0
        mlx_clipping = mlx_range[0] < -1.0 or mlx_range[1] > 1.0
        
        if pytorch_clipping:
            print(f"     ⚠️ PyTorch output exceeds audio range [-1, 1]")
        if mlx_clipping:
            print(f"     ❌ MLX output exceeds audio range [-1, 1]")
        
        if not pytorch_clipping and not mlx_clipping:
            print(f"     ✅ Both outputs within audio range [-1, 1]")


def main():
    """主函数"""
    print("🔍 Cached Submodule Data Collection")
    print("="*60)
    
    try:
        # 收集缓存子模块数据
        collect_cached_submodule_data()
        
        print("\n🎉 Data collection completed successfully!")
        print("\n📋 Generated files:")
        print("1. cfm_debug_outputs/cached_detailed_submodules_comparison.pkl - 详细子模块对比数据")
        print("2. cfm_debug_outputs/cached_detailed_submodules_data.pkl - 详细子模块原始数据")
        
        print("\n📊 Key Features:")
        print("   🔍 使用 CFM 前级缓存作为输入")
        print("   📊 收集 PyTorch 和 MLX 的详细子模块数据")
        print("   🎯 确保输入一致性")
        print("   💡 提供详细的对比分析")
        
        print("\n🎯 Next Steps:")
        print("   📏 分析输出差异是否小于 e-5")
        print("   🔍 识别存在差异的子模块")
        print("   📊 按严重程度分类差异")
        print("   🛠️ 提供针对性修复建议")
        
    except Exception as e:
        print(f"\n❌ Error during data collection: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
