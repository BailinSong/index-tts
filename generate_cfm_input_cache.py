#!/usr/bin/env python3
"""
生成 CFM 前级缓存数据

先运行 PyTorch 版本来生成 CFM 前级缓存数据，
然后使用这个缓存数据来测试 PyTorch 和 MLX 的 CFM 实现
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


def generate_cfm_input_cache():
    """生成 CFM 前级缓存数据"""
    print("🔍 Generating CFM Input Cache")
    print("="*60)
    
    # 加载现有的缓存输入数据
    cached_inputs = load_existing_cached_inputs()
    if not cached_inputs:
        print("❌ No existing cached inputs available!")
        return
    
    print("\n📊 Existing Cached Inputs Summary:")
    print_cached_inputs_summary(cached_inputs)
    
    # 运行 PyTorch CFM 生成前级缓存数据
    print("\n🔍 Running PyTorch CFM to Generate Input Cache:")
    cfm_input_cache = run_pytorch_cfm_for_cache(cached_inputs)
    
    if cfm_input_cache:
        # 保存 CFM 前级缓存数据
        print("\n💾 Saving CFM Input Cache:")
        save_cfm_input_cache(cfm_input_cache)
        
        # 使用 CFM 前级缓存数据测试 PyTorch 和 MLX
        print("\n🔍 Testing PyTorch and MLX CFM with Cache:")
        test_pytorch_mlx_with_cache(cfm_input_cache)
    
    print("\n🎉 CFM Input Cache Generation completed successfully!")


def load_existing_cached_inputs():
    """加载现有的缓存输入数据"""
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


def run_pytorch_cfm_for_cache(cached_inputs):
    """运行 PyTorch CFM 生成前级缓存数据"""
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
        print(f"\n     🔍 Running PyTorch CFM with debug layers:")
        
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
        
        # 创建 CFM 前级缓存数据
        cfm_input_cache = {
            'original_inputs': cached_inputs,
            'pytorch_output': output,
            'config': config_dict,
            't_span': t_span,
            'inference_cfg_rate': 0.0,
            'description': 'CFM input cache generated from PyTorch CFM run'
        }
        
        print(f"     ✅ CFM input cache created successfully")
        
        return cfm_input_cache
        
    except Exception as e:
        print(f"     ❌ PyTorch CFM run failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_cfm_input_cache(cfm_input_cache):
    """保存 CFM 前级缓存数据"""
    cache_file = Path("cfm_debug_outputs/cfm_input_cache.pkl")
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cfm_input_cache, f)
    
    print(f"     ✅ CFM input cache saved to: {cache_file}")
    
    # 打印缓存数据摘要
    print(f"\n     📊 CFM Input Cache Summary:")
    print(f"       Original inputs: {len(cfm_input_cache['original_inputs'])} items")
    print(f"       PyTorch output shape: {cfm_input_cache['pytorch_output'].shape}")
    print(f"       PyTorch output range: [{cfm_input_cache['pytorch_output'].min():.6f}, {cfm_input_cache['pytorch_output'].max():.6f}]")
    print(f"       Config: {len(cfm_input_cache['config'])} sections")
    print(f"       T span: {cfm_input_cache['t_span'].shape}")
    print(f"       Inference CFG rate: {cfm_input_cache['inference_cfg_rate']}")


def test_pytorch_mlx_with_cache(cfm_input_cache):
    """使用 CFM 前级缓存数据测试 PyTorch 和 MLX"""
    print("\n   📋 Testing PyTorch CFM with Cache:")
    pytorch_result = test_pytorch_cfm_with_cache(cfm_input_cache)
    
    print("\n   📋 Testing MLX CFM with Cache:")
    mlx_result = test_mlx_cfm_with_cache(cfm_input_cache)
    
    # 对比结果
    print("\n   📊 Comparison Results:")
    if pytorch_result and mlx_result:
        compare_results(pytorch_result, mlx_result)
    else:
        print("     ❌ Cannot compare: one or both tests failed")


def test_pytorch_cfm_with_cache(cfm_input_cache):
    """使用缓存数据测试 PyTorch CFM"""
    try:
        # 导入 PyTorch CFM
        from indextts.s2mel.modules.flow_matching import CFM
        
        # 使用缓存中的配置
        config_dict = cfm_input_cache['config']
        
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
        original_inputs = cfm_input_cache['original_inputs']
        batch_size = original_inputs['batch_size']
        seq_len = original_inputs['seq_len']
        pytorch_cfm.estimator.setup_caches(max_batch_size=batch_size, max_seq_length=seq_len)
        
        # 使用缓存中的输入数据
        x = original_inputs['x']
        prompt_x = original_inputs['prompt_x']
        x_lens = original_inputs['x_lens']
        style = original_inputs['style']
        mu = original_inputs['mu']
        t_span = cfm_input_cache['t_span']
        inference_cfg_rate = cfm_input_cache['inference_cfg_rate']
        
        print(f"       Input shapes: x={x.shape}, prompt_x={prompt_x.shape}, x_lens={x_lens.shape}")
        print(f"       Input shapes: style={style.shape}, mu={mu.shape}")
        
        # 运行 CFM 推理
        with torch.no_grad():
            output = pytorch_cfm.solve_euler(
                x=x,
                x_lens=x_lens,
                prompt=prompt_x,
                mu=mu,
                style=style,
                f0=None,
                t_span=t_span,
                inference_cfg_rate=inference_cfg_rate,
                debug_layers=False
            )
            
            print(f"       PyTorch output shape: {output.shape}")
            print(f"       PyTorch output range: [{output.min():.6f}, {output.max():.6f}]")
        
        return {
            'output': output,
            'success': True
        }
        
    except Exception as e:
        print(f"       ❌ PyTorch CFM test failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'success': False
        }


def test_mlx_cfm_with_cache(cfm_input_cache):
    """使用缓存数据测试 MLX CFM"""
    try:
        # 导入 MLX CFM
        from indextts.s2mel.modules.mlx_cfm import MLXCFM
        
        # 使用缓存中的配置
        config_dict = cfm_input_cache['config']
        
        # 初始化 MLX CFM
        mlx_cfm = MLXCFM(config_dict)
        
        # 使用缓存中的输入数据（转换为 MLX）
        original_inputs = cfm_input_cache['original_inputs']
        x_mlx = mx.array(original_inputs['x'].numpy())
        prompt_x_mlx = mx.array(original_inputs['prompt_x'].numpy())
        x_lens_mlx = mx.array(original_inputs['x_lens'].numpy())
        style_mlx = mx.array(original_inputs['style'].numpy())
        mu_mlx = mx.array(original_inputs['mu'].numpy())
        t_span_mlx = mx.array(cfm_input_cache['t_span'].numpy())
        inference_cfg_rate = cfm_input_cache['inference_cfg_rate']
        
        print(f"       Input shapes: x={x_mlx.shape}, prompt_x={prompt_x_mlx.shape}, x_lens={x_lens_mlx.shape}")
        print(f"       Input shapes: style={style_mlx.shape}, mu={mu_mlx.shape}")
        
        # 运行 CFM 推理
        output = mlx_cfm.solve_euler(
            x=x_mlx,
            x_lens=x_lens_mlx,
            prompt=prompt_x_mlx,
            mu=mu_mlx,
            style=style_mlx,
            f0=None,
            t_span=t_span_mlx,
            inference_cfg_rate=inference_cfg_rate,
            debug_layers=False
        )
        
        print(f"       MLX output shape: {output.shape}")
        print(f"       MLX output range: [{float(output.min()):.6f}, {float(output.max()):.6f}]")
        
        return {
            'output': output,
            'success': True
        }
        
    except Exception as e:
        print(f"       ❌ MLX CFM test failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'success': False
        }


def compare_results(pytorch_result, mlx_result):
    """对比 PyTorch 和 MLX 的结果"""
    pytorch_output = pytorch_result['output']
    mlx_output = mlx_result['output']
    
    # 转换为 numpy 进行对比
    pytorch_np = pytorch_output.detach().cpu().numpy()
    mlx_np = np.array(mlx_output)
    
    # 计算差异
    max_diff = np.max(np.abs(pytorch_np - mlx_np))
    mean_diff = np.mean(np.abs(pytorch_np - mlx_np))
    relative_diff = mean_diff / (np.mean(np.abs(pytorch_np)) + 1e-8)
    
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
    
    # 保存对比结果
    comparison_data = {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'relative_diff': relative_diff,
        'pytorch_range': pytorch_range,
        'mlx_range': mlx_range,
        'pytorch_clipping': pytorch_clipping,
        'mlx_clipping': mlx_clipping,
        'severity': severity,
        'pytorch_output': pytorch_np,
        'mlx_output': mlx_np
    }
    
    comparison_file = Path("cfm_debug_outputs/cfm_comparison_results.pkl")
    with open(comparison_file, 'wb') as f:
        pickle.dump(comparison_data, f)
    
    print(f"       ✅ Comparison results saved to: {comparison_file}")


def main():
    """主函数"""
    print("🔍 CFM Input Cache Generation")
    print("="*60)
    
    try:
        # 生成 CFM 前级缓存数据
        generate_cfm_input_cache()
        
        print("\n🎉 CFM Input Cache Generation completed successfully!")
        print("\n📋 Generated files:")
        print("1. cfm_debug_outputs/cfm_input_cache.pkl - CFM 前级缓存数据")
        print("2. cfm_debug_outputs/cfm_comparison_results.pkl - PyTorch vs MLX 对比结果")
        
        print("\n📊 Key Features:")
        print("   🔍 先运行 PyTorch 版本生成 CFM 前级缓存数据")
        print("   📊 使用缓存数据测试 PyTorch 和 MLX CFM")
        print("   🎯 确保两个版本使用相同的输入数据")
        print("   💡 对比找出数据处理差异")
        
        print("\n🎯 Analysis Focus:")
        print("   📏 输出差异应该小于 e-5")
        print("   🔍 识别 PyTorch 和 MLX 的差异")
        print("   📊 按严重程度分类差异")
        print("   🛠️ 提供针对性修复建议")
        
    except Exception as e:
        print(f"\n❌ Error during cache generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


