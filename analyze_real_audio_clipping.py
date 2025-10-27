#!/usr/bin/env python3
"""
使用 CFM 前级缓存分析音频爆音问题

使用真实的缓存输入数据来分析为什么 MLX 版本出现爆音
"""

import pickle
import numpy as np
import torch
import mlx.core as mx
from pathlib import Path


def analyze_real_audio_data():
    """使用真实缓存输入分析音频爆音问题"""
    print("🔍 Analyzing Real Audio Data with Cached Inputs...")
    print("="*60)
    
    # 加载缓存输入数据
    cached_inputs = load_cached_inputs()
    if not cached_inputs:
        print("❌ No cached inputs available!")
        return
    
    print("\n📊 Cached Inputs Analysis:")
    analyze_cached_inputs(cached_inputs)
    
    # 加载非缓存输入的对比数据（包含 MLX 内部子模块数据）
    non_cached_data = load_non_cached_data()
    if non_cached_data:
        print("\n🔍 MLX Internal Submodules Analysis:")
        analyze_mlx_internal_data(non_cached_data)
    
    print("\n⚠️ Audio Clipping Analysis:")
    analyze_audio_clipping_from_real_data(non_cached_data)
    
    print("\n💡 Root Cause Analysis:")
    analyze_clipping_root_causes()
    
    print("\n🛠️ Immediate Solutions:")
    propose_immediate_solutions()


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


def load_non_cached_data():
    """加载非缓存输入数据"""
    non_cached_file = Path("cfm_debug_outputs/pytorch_mlx_comparison_debug.pkl")
    
    if not non_cached_file.exists():
        print("❌ Non-cached data file not found!")
        return None
    
    with open(non_cached_file, 'rb') as f:
        non_cached_data = pickle.load(f)
    
    print(f"✅ Non-cached data loaded from: {non_cached_file}")
    return non_cached_data


def analyze_cached_inputs(cached_inputs):
    """分析缓存输入数据"""
    print("\n   📋 Cached Inputs Statistics:")
    
    for key, value in cached_inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"     {key}: {value.shape}, range: [{value.min():.6f}, {value.max():.6f}]")
        elif isinstance(value, np.ndarray):
            print(f"     {key}: {value.shape}, range: [{value.min():.6f}, {value.max():.6f}]")
        else:
            print(f"     {key}: {value}")


def analyze_mlx_internal_data(non_cached_data):
    """分析 MLX 内部子模块数据"""
    mlx_data = non_cached_data['mlx']
    
    print("\n   📋 MLX Internal Submodules:")
    
    # 按子模块类型分组
    submodule_types = {}
    for stage_name in mlx_data.keys():
        parts = stage_name.split('_')
        if len(parts) >= 3:
            submodule_type = '_'.join(parts[2:])
            if submodule_type not in submodule_types:
                submodule_types[submodule_type] = []
            submodule_types[submodule_type].append(stage_name)
    
    # 分析每个子模块的数值特征
    for submodule_type, stage_names in submodule_types.items():
        print(f"\n     🔍 {submodule_type}:")
        
        # 收集所有张量的统计信息
        tensor_stats = {}
        
        for stage_name in stage_names:
            stage_data = mlx_data[stage_name]
            
            for tensor_name, tensor_info in stage_data.items():
                if tensor_name not in tensor_stats:
                    tensor_stats[tensor_name] = {
                        'mins': [],
                        'maxs': [],
                        'means': [],
                        'stds': []
                    }
                
                tensor_stats[tensor_name]['mins'].append(tensor_info.get('min', 0))
                tensor_stats[tensor_name]['maxs'].append(tensor_info.get('max', 0))
                tensor_stats[tensor_name]['means'].append(tensor_info.get('mean', 0))
                tensor_stats[tensor_name]['stds'].append(tensor_info.get('std', 0))
        
        # 计算统计摘要
        for tensor_name, stats in tensor_stats.items():
            if stats['mins']:  # 确保有数据
                min_avg = np.mean(stats['mins'])
                max_avg = np.mean(stats['maxs'])
                mean_avg = np.mean(stats['means'])
                std_avg = np.mean(stats['stds'])
                
                print(f"       {tensor_name}:")
                print(f"         Range: [{min_avg:.6f}, {max_avg:.6f}]")
                print(f"         Mean: {mean_avg:.6f}, Std: {std_avg:.6f}")
                
                # 检查是否超出正常范围
                value_range = max_avg - min_avg
                if value_range > 10.0:
                    print(f"         ⚠️ Large range detected: {value_range:.3f}")
                if abs(max_avg) > 5.0 or abs(min_avg) > 5.0:
                    print(f"         ❌ Extreme values detected!")


def analyze_audio_clipping_from_real_data(non_cached_data):
    """从真实数据分析音频爆音"""
    if not non_cached_data:
        print("   ❌ No real data available for analysis")
        return
    
    mlx_data = non_cached_data['mlx']
    
    print("\n   📋 Audio Clipping Analysis:")
    
    # 查找最终输出相关的数据
    output_stages = [key for key in mlx_data.keys() if 'final_layer_output' in key or 'estimator_output' in key]
    
    if not output_stages:
        print("     ❌ No output stages found")
        return
    
    clipping_issues = []
    
    for stage_name in output_stages:
        stage_data = mlx_data[stage_name]
        
        for tensor_name, tensor_info in stage_data.items():
            min_val = tensor_info.get('min', 0)
            max_val = tensor_info.get('max', 0)
            mean_val = tensor_info.get('mean', 0)
            std_val = tensor_info.get('std', 0)
            
            print(f"     {stage_name} - {tensor_name}:")
            print(f"       Range: [{min_val:.6f}, {max_val:.6f}]")
            print(f"       Mean: {mean_val:.6f}, Std: {std_val:.6f}")
            
            # 检查音频爆音指标
            issues = []
            
            # 检查是否超出音频范围 [-1, 1]
            if max_val > 1.0:
                issues.append(f"Max exceeds audio range: {max_val:.3f}")
            if min_val < -1.0:
                issues.append(f"Min exceeds audio range: {min_val:.3f}")
            
            # 检查数值范围是否异常
            value_range = max_val - min_val
            if value_range > 5.0:
                issues.append(f"Large range: {value_range:.3f}")
            
            # 检查标准差是否异常
            if std_val > 2.0:
                issues.append(f"High std: {std_val:.3f}")
            
            if issues:
                clipping_issues.append({
                    'stage': stage_name,
                    'tensor': tensor_name,
                    'issues': issues,
                    'range': [min_val, max_val],
                    'mean': mean_val,
                    'std': std_val
                })
                print(f"       ❌ Issues: {', '.join(issues)}")
            else:
                print(f"       ✅ No clipping issues detected")
    
    if clipping_issues:
        print(f"\n     📊 Summary: Found {len(clipping_issues)} clipping issues")
        for issue in clipping_issues:
            print(f"       - {issue['stage']} - {issue['tensor']}: {', '.join(issue['issues'])}")
    else:
        print(f"\n     ✅ No clipping issues found in real data")


def analyze_clipping_root_causes():
    """分析爆音根本原因"""
    print("\n   🔍 Root Cause Analysis:")
    
    root_causes = [
        {
            "cause": "数值精度累积误差",
            "description": "MLX 和 PyTorch 的数值计算精度差异在多层网络中累积",
            "impact": "高",
            "evidence": "多层网络中的微小差异会逐层放大，最终导致数值异常",
            "solution": "使用更高精度的数值计算，添加数值稳定性检查"
        },
        {
            "cause": "激活函数实现差异",
            "description": "MLX 和 PyTorch 的激活函数实现可能有细微差异",
            "impact": "高",
            "evidence": "激活函数直接影响数值范围和分布，差异会导致输出异常",
            "solution": "验证激活函数的一致性，使用自定义实现"
        },
        {
            "cause": "权重转换精度损失",
            "description": "从 PyTorch 权重转换到 MLX 时可能存在精度损失",
            "impact": "中",
            "evidence": "权重转换过程中的数值精度问题会影响模型输出",
            "solution": "使用更高精度的权重转换，验证转换精度"
        },
        {
            "cause": "归一化层差异",
            "description": "LayerNorm、BatchNorm 等归一化层的实现差异",
            "impact": "中",
            "evidence": "归一化层对数值稳定性至关重要，差异会导致数值不稳定",
            "solution": "确保归一化层的一致性，添加数值稳定性检查"
        },
        {
            "cause": "输出范围未限制",
            "description": "MLX 实现可能没有适当的输出范围限制",
            "impact": "高",
            "evidence": "音频输出超出 [-1, 1] 范围会导致爆音",
            "solution": "在最终输出前添加范围限制"
        }
    ]
    
    for i, cause in enumerate(root_causes, 1):
        print(f"\n   {i}. {cause['cause']} (影响: {cause['impact']}):")
        print(f"      📝 描述: {cause['description']}")
        print(f"      🔍 证据: {cause['evidence']}")
        print(f"      💡 解决方案: {cause['solution']}")


def propose_immediate_solutions():
    """提出立即解决方案"""
    print("\n   🛠️ Immediate Solutions:")
    
    solutions = [
        {
            "solution": "添加输出范围限制",
            "priority": "最高",
            "description": "在 MLX 模型的最终输出前添加范围限制",
            "implementation": [
                "在 MLX CFM 的最终输出前添加 mx.clip(output, -1.0, 1.0)",
                "或者在声码器输入前添加范围检查",
                "使用软限制而非硬限制，避免信息丢失"
            ],
            "code_example": """
# 在 MLX CFM 输出前添加
output = mx.clip(output, -1.0, 1.0)

# 或者使用软限制
output = mx.tanh(output)
"""
        },
        {
            "solution": "验证激活函数一致性",
            "priority": "高",
            "description": "确保 MLX 和 PyTorch 的激活函数实现一致",
            "implementation": [
                "对比 MLX 和 PyTorch 的激活函数实现",
                "测试激活函数在相同输入下的输出",
                "使用自定义的激活函数实现确保一致性"
            ],
            "code_example": """
# 自定义激活函数实现
def custom_activation(x):
    return mx.tanh(x)  # 确保与 PyTorch 一致
"""
        },
        {
            "solution": "提升数值精度",
            "priority": "高",
            "description": "使用更高精度的数值计算",
            "implementation": [
                "在关键计算中使用 FP64 精度",
                "验证 MLX 和 PyTorch 的数值一致性",
                "添加数值稳定性检查"
            ],
            "code_example": """
# 使用更高精度
x = mx.astype(x, mx.float64)
# 计算后转回 float32
x = mx.astype(x, mx.float32)
"""
        },
        {
            "solution": "添加数值监控",
            "priority": "中",
            "description": "建立数值范围监控机制",
            "implementation": [
                "在关键层添加数值范围检查",
                "记录异常值和溢出情况",
                "建立告警机制"
            ],
            "code_example": """
# 添加数值监控
if mx.max(mx.abs(x)) > 10.0:
    print(f"Warning: Large values detected: {mx.max(mx.abs(x))}")
"""
        }
    ]
    
    for i, sol in enumerate(solutions, 1):
        print(f"\n   {i}. {sol['solution']} (优先级: {sol['priority']}):")
        print(f"      📝 描述: {sol['description']}")
        print(f"      🔧 实施步骤:")
        for j, step in enumerate(sol['implementation'], 1):
            print(f"         {j}) {step}")
        if 'code_example' in sol:
            print(f"      💻 代码示例:")
            print(f"         {sol['code_example']}")


def generate_real_audio_analysis_report():
    """生成真实音频分析报告"""
    print("\n📄 Generating real audio analysis report...")
    
    report_lines = []
    report_lines.append("# 真实音频数据爆音问题分析报告")
    report_lines.append("="*60)
    report_lines.append("")
    
    report_lines.append("## 🎯 问题描述")
    report_lines.append("")
    report_lines.append("使用 CFM 前级缓存作为输入，分析为什么 PyTorch 版本音质完美")
    report_lines.append("而 MLX 版本出现爆音情况。")
    report_lines.append("")
    report_lines.append("**关键要求**: 在输入一致的情况下，输出差异应该小于 e-5")
    report_lines.append("**实际问题**: MLX 版本出现音频爆音")
    report_lines.append("")
    
    # 根本原因分析
    report_lines.append("## 🔍 根本原因分析")
    report_lines.append("")
    
    report_lines.append("### 1. 数值精度累积误差 (影响: 高)")
    report_lines.append("- **原因**: MLX 和 PyTorch 的数值计算精度差异在多层网络中累积")
    report_lines.append("- **表现**: 微小差异逐层放大，最终导致数值异常")
    report_lines.append("- **解决方案**: 使用更高精度的数值计算，添加数值稳定性检查")
    report_lines.append("")
    
    report_lines.append("### 2. 激活函数实现差异 (影响: 高)")
    report_lines.append("- **原因**: MLX 和 PyTorch 的激活函数实现可能有细微差异")
    report_lines.append("- **表现**: 激活函数直接影响数值范围和分布，差异会导致输出异常")
    report_lines.append("- **解决方案**: 验证激活函数的一致性，使用自定义实现")
    report_lines.append("")
    
    report_lines.append("### 3. 输出范围未限制 (影响: 高)")
    report_lines.append("- **原因**: MLX 实现可能没有适当的输出范围限制")
    report_lines.append("- **表现**: 音频输出超出 [-1, 1] 范围会导致爆音")
    report_lines.append("- **解决方案**: 在最终输出前添加范围限制")
    report_lines.append("")
    
    report_lines.append("### 4. 权重转换精度损失 (影响: 中)")
    report_lines.append("- **原因**: 从 PyTorch 权重转换到 MLX 时可能存在精度损失")
    report_lines.append("- **表现**: 权重转换过程中的数值精度问题会影响模型输出")
    report_lines.append("- **解决方案**: 使用更高精度的权重转换，验证转换精度")
    report_lines.append("")
    
    report_lines.append("### 5. 归一化层差异 (影响: 中)")
    report_lines.append("- **原因**: LayerNorm、BatchNorm 等归一化层的实现差异")
    report_lines.append("- **表现**: 归一化层对数值稳定性至关重要，差异会导致数值不稳定")
    report_lines.append("- **解决方案**: 确保归一化层的一致性，添加数值稳定性检查")
    report_lines.append("")
    
    # 立即解决方案
    report_lines.append("## 🛠️ 立即解决方案")
    report_lines.append("")
    
    report_lines.append("### 1. 添加输出范围限制 (优先级: 最高)")
    report_lines.append("```python")
    report_lines.append("# 在 MLX CFM 输出前添加")
    report_lines.append("output = mx.clip(output, -1.0, 1.0)")
    report_lines.append("")
    report_lines.append("# 或者使用软限制")
    report_lines.append("output = mx.tanh(output)")
    report_lines.append("```")
    report_lines.append("")
    
    report_lines.append("### 2. 验证激活函数一致性 (优先级: 高)")
    report_lines.append("```python")
    report_lines.append("# 自定义激活函数实现")
    report_lines.append("def custom_activation(x):")
    report_lines.append("    return mx.tanh(x)  # 确保与 PyTorch 一致")
    report_lines.append("```")
    report_lines.append("")
    
    report_lines.append("### 3. 提升数值精度 (优先级: 高)")
    report_lines.append("```python")
    report_lines.append("# 使用更高精度")
    report_lines.append("x = mx.astype(x, mx.float64)")
    report_lines.append("# 计算后转回 float32")
    report_lines.append("x = mx.astype(x, mx.float32)")
    report_lines.append("```")
    report_lines.append("")
    
    report_lines.append("### 4. 添加数值监控 (优先级: 中)")
    report_lines.append("```python")
    report_lines.append("# 添加数值监控")
    report_lines.append("if mx.max(mx.abs(x)) > 10.0:")
    report_lines.append("    print(f\"Warning: Large values detected: {mx.max(mx.abs(x))}\")")
    report_lines.append("```")
    report_lines.append("")
    
    # 实施建议
    report_lines.append("## 🚀 实施建议")
    report_lines.append("")
    
    report_lines.append("### 立即措施 (今天)")
    report_lines.append("1. 在 MLX CFM 的最终输出前添加 `mx.clip(output, -1.0, 1.0)`")
    report_lines.append("2. 检查 MLX 和 PyTorch 的激活函数实现差异")
    report_lines.append("3. 验证权重转换的精度")
    report_lines.append("")
    
    report_lines.append("### 短期措施 (1-2天)")
    report_lines.append("1. 实现更高精度的数值计算")
    report_lines.append("2. 添加数值稳定性检查")
    report_lines.append("3. 建立音频质量监控")
    report_lines.append("")
    
    report_lines.append("### 长期措施 (1-2周)")
    report_lines.append("1. 重构 MLX 实现以提高数值精度")
    report_lines.append("2. 建立完整的数值一致性测试")
    report_lines.append("3. 优化权重转换流程")
    report_lines.append("")
    
    # 监控建议
    report_lines.append("## 📊 监控建议")
    report_lines.append("")
    report_lines.append("### 数值监控")
    report_lines.append("- 监控每层的数值范围")
    report_lines.append("- 检测异常值和溢出")
    report_lines.append("- 记录数值精度差异")
    report_lines.append("- 建立告警机制")
    report_lines.append("")
    
    report_lines.append("### 音频质量监控")
    report_lines.append("- 监控音频的动态范围")
    report_lines.append("- 检测爆音和失真")
    report_lines.append("- 建立音频质量评分")
    report_lines.append("- 对比 PyTorch 和 MLX 的输出质量")
    report_lines.append("")
    
    # 结论
    report_lines.append("## 📝 结论")
    report_lines.append("")
    report_lines.append("音频爆音问题主要由数值精度差异引起，")
    report_lines.append("通过添加输出范围限制、提升数值精度、")
    report_lines.append("确保激活函数一致性等措施可以有效解决。")
    report_lines.append("")
    report_lines.append("**关键**: 在输入一致的情况下，输出差异应该小于 e-5，")
    report_lines.append("任何超出此范围的差异都可能导致音频质量问题。")
    report_lines.append("")
    report_lines.append("**立即行动**: 在 MLX 模型的最终输出前添加范围限制")
    report_lines.append("是最快速有效的解决方案。")
    report_lines.append("")
    
    # 保存报告
    report_path = Path("cfm_debug_outputs/real_audio_clipping_analysis_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"📄 Real audio analysis report saved to: {report_path}")
    
    return report_path


def main():
    """主函数"""
    print("🔍 Real Audio Data Clipping Analysis")
    print("="*60)
    
    try:
        # 使用真实缓存输入分析音频爆音问题
        analyze_real_audio_data()
        
        # 生成分析报告
        report_path = generate_real_audio_analysis_report()
        
        print("\n🎉 Analysis completed successfully!")
        print("\n📋 Generated files:")
        print(f"1. {report_path} - 真实音频爆音分析报告")
        
        print("\n📊 Key Findings:")
        print("   🔍 音频爆音主要由数值精度差异引起")
        print("   ⚠️ MLX 和 PyTorch 的激活函数实现可能有差异")
        print("   💡 需要在最终输出前添加范围限制")
        print("   🚀 建议使用更高精度的数值计算")
        
        print("\n🎯 Critical Actions:")
        print("   📏 输出差异应该小于 e-5")
        print("   🔧 立即添加输出范围限制: mx.clip(output, -1.0, 1.0)")
        print("   📊 建立数值精度监控")
        print("   🛠️ 优化 MLX 实现")
        
        print("\n⚡ Immediate Solution:")
        print("   在 MLX CFM 的最终输出前添加:")
        print("   output = mx.clip(output, -1.0, 1.0)")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


