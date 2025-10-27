#!/usr/bin/env python3
"""
使用正确的 CFM 测试数据对比 PyTorch 和 MLX

使用修正后的缓存输入数据，分别运行 PyTorch 和 MLX 版本的 CFM 实现，
收集两个版本各个子流程的输入输出，然后进行对比找出差异
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


def compare_pytorch_mlx_cfm_correct():
    """使用正确的数据对比 PyTorch 和 MLX CFM 实现"""
    print("🔍 PyTorch vs MLX CFM Comparison (Correct Data)")
    print("="*60)
    
    # 加载正确的缓存输入数据
    cached_inputs = load_correct_cached_inputs()
    if not cached_inputs:
        print("❌ No correct cached inputs available!")
        return
    
    print("\n📊 Correct Cached Inputs Summary:")
    print_cached_inputs_summary(cached_inputs)
    
    # 收集 PyTorch CFM 子流程数据
    print("\n🔍 Collecting PyTorch CFM Submodule Data:")
    pytorch_data = collect_pytorch_cfm_data_correct(cached_inputs)
    
    # 收集 MLX CFM 子流程数据
    print("\n🔍 Collecting MLX CFM Submodule Data:")
    mlx_data = collect_mlx_cfm_data_correct(cached_inputs)
    
    # 对比分析
    print("\n📊 Submodule Comparison Analysis:")
    comparison_results = compare_submodule_data_correct(pytorch_data, mlx_data)
    
    # 生成对比报告
    print("\n📄 Generating Comparison Report:")
    generate_comparison_report_correct(cached_inputs, pytorch_data, mlx_data, comparison_results)
    
    print("\n🎉 Comparison completed successfully!")


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


def collect_pytorch_cfm_data_correct(cached_inputs):
    """收集 PyTorch CFM 子流程数据 (正确数据)"""
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
            }
        }
        
    except Exception as e:
        print(f"     ❌ PyTorch data collection failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'success': False
        }


def collect_mlx_cfm_data_correct(cached_inputs):
    """收集 MLX CFM 子流程数据 (正确数据)"""
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
        print(f"\n     🔍 Running MLX CFM with debug layers:")
        
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
            }
        }
        
    except Exception as e:
        print(f"     ❌ MLX data collection failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'success': False
        }


def compare_submodule_data_correct(pytorch_data, mlx_data):
    """对比子模块数据 (正确数据)"""
    print("\n   📋 Submodule Data Comparison:")
    
    if not pytorch_data.get('success') or not mlx_data.get('success'):
        print("     ❌ Cannot compare: one or both data collection failed")
        return None
    
    pytorch_output = pytorch_data['output']
    mlx_output = mlx_data['output']
    
    # 转换为 numpy 进行对比
    pytorch_np = pytorch_output.detach().cpu().numpy()
    mlx_np = np.array(mlx_output)
    
    # 计算差异
    max_diff = np.max(np.abs(pytorch_np - mlx_np))
    mean_diff = np.mean(np.abs(pytorch_np - mlx_np))
    relative_diff = mean_diff / (np.mean(np.abs(pytorch_np)) + 1e-8)
    
    print(f"     Max difference: {max_diff:.2e}")
    print(f"     Mean difference: {mean_diff:.2e}")
    print(f"     Relative difference: {relative_diff:.2e}")
    
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
    
    # 判断差异严重程度
    if max_diff < 1e-5:
        severity = "微小"
        print(f"     ✅ Differences within acceptable range (< 1e-5)")
    elif max_diff < 1e-3:
        severity = "中等"
        print(f"     ⚠️ Differences moderate (< 1e-3)")
    else:
        severity = "严重"
        print(f"     ❌ Differences too large (> 1e-3)")
    
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
        'mlx_output': mlx_np
    }


def generate_comparison_report_correct(cached_inputs, pytorch_data, mlx_data, comparison_results):
    """生成对比报告 (正确数据)"""
    report_lines = []
    report_lines.append("# PyTorch vs MLX CFM 子流程对比分析报告 (正确数据)")
    report_lines.append("="*60)
    report_lines.append("")
    
    report_lines.append("## 🎯 分析目标")
    report_lines.append("")
    report_lines.append("使用修正后的 CFM 前级缓存数据，分别运行 PyTorch 和 MLX 版本的 CFM 实现，")
    report_lines.append("收集两个版本各个子流程的输入输出，然后进行对比找出差异。")
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
    
    # 输入数据验证
    report_lines.append("### 输入数据验证")
    report_lines.append("")
    
    x = cached_inputs['x']
    prompt_x = cached_inputs['prompt_x']
    prompt_len = cached_inputs['prompt_len']
    seq_len = cached_inputs['seq_len']
    
    if x.min() == 0.0 and x.max() == 0.0:
        report_lines.append("❌ **发现问题**: `x` (噪声) 全为零")
    else:
        report_lines.append("✅ **输入数据正常**: `x` 包含有意义的噪声值")
        report_lines.append(f"   - 噪声范围: [{x.min():.6f}, {x.max():.6f}]")
    
    report_lines.append("")
    
    if prompt_len >= seq_len:
        report_lines.append("❌ **发现问题**: `prompt_len` >= `seq_len`")
        report_lines.append(f"   - prompt_len: {prompt_len}")
        report_lines.append(f"   - seq_len: {seq_len}")
    else:
        report_lines.append("✅ **Prompt 长度正常**: `prompt_len` < `seq_len`")
        report_lines.append(f"   - prompt_len: {prompt_len}")
        report_lines.append(f"   - seq_len: {seq_len}")
        report_lines.append(f"   - 比例: {prompt_len / seq_len:.2f}")
    
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
        
        if pytorch_output.min() == 0.0 and pytorch_output.max() == 0.0:
            report_lines.append("- ❌ **输出全为零**")
        else:
            report_lines.append("- ✅ **输出包含有意义的数值**")
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
        
        if float(mlx_output.min()) == 0.0 and float(mlx_output.max()) == 0.0:
            report_lines.append("- ❌ **输出全为零**")
        else:
            report_lines.append("- ✅ **输出包含有意义的数值**")
    else:
        report_lines.append("❌ **数据收集失败**")
        report_lines.append(f"- 错误: {mlx_data.get('error', 'Unknown error')}")
    
    report_lines.append("")
    
    # 对比结果分析
    if comparison_results:
        report_lines.append("## 📊 对比结果分析")
        report_lines.append("")
        
        report_lines.append("### 数值差异分析")
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
        
        # 关键发现
        report_lines.append("## 🎯 关键发现")
        report_lines.append("")
        
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
    
    else:
        report_lines.append("## ❌ 对比失败")
        report_lines.append("")
        report_lines.append("无法进行子模块对比，请检查数据收集过程。")
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
    
    report_lines.append("### 3. 输入数据标准化")
    report_lines.append("")
    report_lines.append("**建议**:")
    report_lines.append("1. 确保输入数据包含有意义的噪声值")
    report_lines.append("2. 验证 prompt_len < seq_len")
    report_lines.append("3. 使用标准化的测试数据")
    report_lines.append("4. 建立输入数据验证机制")
    report_lines.append("")
    
    report_lines.append("### 4. 监控建议")
    report_lines.append("")
    report_lines.append("1. 建立数值范围监控")
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
    report_path = Path("cfm_debug_outputs/pytorch_mlx_cfm_comparison_report_correct.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"     ✅ Comparison report saved to: {report_path}")
    
    return report_path


def main():
    """主函数"""
    print("🔍 PyTorch vs MLX CFM Comparison (Correct Data)")
    print("="*60)
    
    try:
        # 对比 PyTorch 和 MLX CFM
        compare_pytorch_mlx_cfm_correct()
        
        print("\n🎉 Comparison completed successfully!")
        print("\n📋 Generated files:")
        print("1. cfm_debug_outputs/pytorch_mlx_cfm_comparison_report_correct.md - PyTorch vs MLX CFM 对比分析报告 (正确数据)")
        
        print("\n📊 Key Features:")
        print("   🔍 使用修正后的 CFM 前级缓存数据")
        print("   📊 分别运行 PyTorch 和 MLX 版本")
        print("   🎯 收集两个版本各个子流程的输入输出")
        print("   💡 对比找出数据处理差异")
        print("   🔧 使用有意义的噪声数据")
        
        print("\n🎯 Analysis Focus:")
        print("   📏 输出差异应该小于 e-5")
        print("   🔍 识别 PyTorch 和 MLX 的差异")
        print("   📊 按严重程度分类差异")
        print("   🛠️ 提供针对性修复建议")
        
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


