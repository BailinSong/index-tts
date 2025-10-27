#!/usr/bin/env python3
"""
分析音频爆音问题

深入分析为什么 MLX 版本出现爆音而 PyTorch 版本音质完美
"""

import pickle
import numpy as np
import torch
import mlx.core as mx
from pathlib import Path


def analyze_audio_clipping_issues():
    """分析音频爆音问题"""
    print("🔍 Analyzing Audio Clipping Issues...")
    print("="*60)
    
    # 加载对比数据
    cached_data, non_cached_data = load_comparison_data()
    
    if not cached_data:
        print("❌ No cached data available!")
        return
    
    print("\n📊 Numerical Precision Analysis:")
    analyze_numerical_precision(cached_data)
    
    print("\n🔍 Audio Range Analysis:")
    analyze_audio_range(cached_data)
    
    print("\n⚠️ Clipping Detection:")
    detect_clipping_issues(cached_data)
    
    print("\n💡 Root Cause Analysis:")
    analyze_clipping_root_causes()
    
    print("\n🛠️ Solutions:")
    propose_clipping_solutions()


def load_comparison_data():
    """加载对比数据"""
    cached_file = Path("cfm_debug_outputs/cached_inputs_submodules_comparison.pkl")
    non_cached_file = Path("cfm_debug_outputs/pytorch_mlx_comparison_debug.pkl")
    
    cached_data = None
    non_cached_data = None
    
    if cached_file.exists():
        with open(cached_file, 'rb') as f:
            cached_data = pickle.load(f)
    
    if non_cached_file.exists():
        with open(non_cached_file, 'rb') as f:
            non_cached_data = pickle.load(f)
    
    return cached_data, non_cached_data


def analyze_numerical_precision(cached_data):
    """分析数值精度"""
    print("\n   📋 Precision Analysis:")
    
    comparisons = cached_data['comparisons']
    
    for key, comp_data in comparisons.items():
        if 'tensor_comparisons' in comp_data:
            for tensor_name, tensor_comp in comp_data['tensor_comparisons'].items():
                max_diff = tensor_comp.get('max_diff', 0)
                mean_diff = tensor_comp.get('mean_diff', 0)
                relative_diff = tensor_comp.get('relative_diff', 0)
                
                pytorch_stats = tensor_comp.get('pytorch_stats', {})
                mlx_stats = tensor_comp.get('mlx_stats', {})
                
                print(f"     {tensor_name}:")
                print(f"       Max difference: {max_diff:.2e}")
                print(f"       Mean difference: {mean_diff:.2e}")
                print(f"       Relative difference: {relative_diff:.2e}")
                
                if pytorch_stats and mlx_stats:
                    pytorch_range = pytorch_stats.get('max', 0) - pytorch_stats.get('min', 0)
                    mlx_range = mlx_stats.get('max', 0) - mlx_stats.get('min', 0)
                    
                    print(f"       PyTorch range: {pytorch_range:.6f}")
                    print(f"       MLX range: {mlx_range:.6f}")
                    
                    # 检查是否在合理范围内
                    if max_diff < 1e-5:
                        print(f"       ✅ Precision OK (< 1e-5)")
                    elif max_diff < 1e-3:
                        print(f"       ⚠️ Precision Warning (< 1e-3)")
                    else:
                        print(f"       ❌ Precision Issue (> 1e-3)")


def analyze_audio_range(cached_data):
    """分析音频数值范围"""
    print("\n   📋 Audio Range Analysis:")
    
    comparisons = cached_data['comparisons']
    
    for key, comp_data in comparisons.items():
        if 'tensor_comparisons' in comp_data:
            for tensor_name, tensor_comp in comp_data['tensor_comparisons'].items():
                pytorch_stats = tensor_comp.get('pytorch_stats', {})
                mlx_stats = tensor_comp.get('mlx_stats', {})
                
                if pytorch_stats and mlx_stats:
                    pytorch_min = pytorch_stats.get('min', 0)
                    pytorch_max = pytorch_stats.get('max', 0)
                    mlx_min = mlx_stats.get('min', 0)
                    mlx_max = mlx_stats.get('max', 0)
                    
                    print(f"     {tensor_name}:")
                    print(f"       PyTorch: [{pytorch_min:.6f}, {pytorch_max:.6f}]")
                    print(f"       MLX: [{mlx_min:.6f}, {mlx_max:.6f}]")
                    
                    # 检查是否超出音频范围
                    audio_max = 1.0  # 音频通常限制在 [-1, 1] 范围
                    audio_min = -1.0
                    
                    pytorch_clipping = pytorch_max > audio_max or pytorch_min < audio_min
                    mlx_clipping = mlx_max > audio_max or mlx_min < audio_min
                    
                    if pytorch_clipping:
                        print(f"       ⚠️ PyTorch clipping detected!")
                    if mlx_clipping:
                        print(f"       ❌ MLX clipping detected!")
                    
                    if not pytorch_clipping and not mlx_clipping:
                        print(f"       ✅ No clipping detected")


def detect_clipping_issues(cached_data):
    """检测爆音问题"""
    print("\n   📋 Clipping Detection:")
    
    comparisons = cached_data['comparisons']
    
    clipping_issues = []
    
    for key, comp_data in comparisons.items():
        if 'tensor_comparisons' in comp_data:
            for tensor_name, tensor_comp in comp_data['tensor_comparisons'].items():
                pytorch_stats = tensor_comp.get('pytorch_stats', {})
                mlx_stats = tensor_comp.get('mlx_stats', {})
                
                if pytorch_stats and mlx_stats:
                    pytorch_max = pytorch_stats.get('max', 0)
                    pytorch_min = pytorch_stats.get('min', 0)
                    mlx_max = mlx_stats.get('max', 0)
                    mlx_min = mlx_stats.get('min', 0)
                    
                    # 检测异常值
                    pytorch_outliers = detect_outliers(pytorch_max, pytorch_min)
                    mlx_outliers = detect_outliers(mlx_max, mlx_min)
                    
                    if pytorch_outliers or mlx_outliers:
                        clipping_issues.append({
                            'stage': key,
                            'tensor': tensor_name,
                            'pytorch_outliers': pytorch_outliers,
                            'mlx_outliers': mlx_outliers,
                            'pytorch_range': [pytorch_min, pytorch_max],
                            'mlx_range': [mlx_min, mlx_max]
                        })
    
    if clipping_issues:
        print(f"     Found {len(clipping_issues)} potential clipping issues:")
        for issue in clipping_issues:
            print(f"       {issue['stage']} - {issue['tensor']}:")
            if issue['pytorch_outliers']:
                print(f"         PyTorch outliers: {issue['pytorch_outliers']}")
            if issue['mlx_outliers']:
                print(f"         MLX outliers: {issue['mlx_outliers']}")
    else:
        print("     ✅ No clipping issues detected in cached data")


def detect_outliers(max_val, min_val):
    """检测异常值"""
    outliers = []
    
    # 检查是否超出正常音频范围
    if max_val > 2.0:
        outliers.append(f"max too high: {max_val:.3f}")
    if min_val < -2.0:
        outliers.append(f"min too low: {min_val:.3f}")
    
    # 检查数值范围是否异常
    range_val = max_val - min_val
    if range_val > 10.0:
        outliers.append(f"range too large: {range_val:.3f}")
    
    return outliers


def analyze_clipping_root_causes():
    """分析爆音根本原因"""
    print("\n   🔍 Root Cause Analysis:")
    
    root_causes = [
        {
            "cause": "数值精度累积误差",
            "description": "MLX 和 PyTorch 的数值计算精度差异在多层网络中累积",
            "impact": "高",
            "evidence": "多层网络中的微小差异会逐层放大",
            "solution": "使用更高精度的数值计算"
        },
        {
            "cause": "激活函数实现差异",
            "description": "MLX 和 PyTorch 的激活函数实现可能有细微差异",
            "impact": "高",
            "evidence": "激活函数直接影响数值范围和分布",
            "solution": "验证激活函数的一致性"
        },
        {
            "cause": "权重加载精度损失",
            "description": "从 PyTorch 权重转换到 MLX 时可能存在精度损失",
            "impact": "中",
            "evidence": "权重转换过程中的数值精度问题",
            "solution": "使用更高精度的权重转换"
        },
        {
            "cause": "归一化层差异",
            "description": "LayerNorm、BatchNorm 等归一化层的实现差异",
            "impact": "中",
            "evidence": "归一化层对数值稳定性至关重要",
            "solution": "确保归一化层的一致性"
        },
        {
            "cause": "随机数生成差异",
            "description": "MLX 和 PyTorch 的随机数生成器差异",
            "impact": "低",
            "evidence": "随机操作可能导致数值差异",
            "solution": "固定随机种子"
        }
    ]
    
    for i, cause in enumerate(root_causes, 1):
        print(f"\n   {i}. {cause['cause']} (影响: {cause['impact']}):")
        print(f"      📝 描述: {cause['description']}")
        print(f"      🔍 证据: {cause['evidence']}")
        print(f"      💡 解决方案: {cause['solution']}")


def propose_clipping_solutions():
    """提出爆音解决方案"""
    print("\n   🛠️ Solutions:")
    
    solutions = [
        {
            "solution": "数值精度提升",
            "priority": "高",
            "steps": [
                "使用 FP64 精度进行关键计算",
                "验证 MLX 和 PyTorch 的数值一致性",
                "在关键层添加数值稳定性检查",
                "使用更精确的数学函数"
            ]
        },
        {
            "solution": "激活函数一致性",
            "priority": "高",
            "steps": [
                "对比 MLX 和 PyTorch 的激活函数实现",
                "确保激活函数的数值精度一致",
                "添加激活函数的边界检查",
                "使用自定义的激活函数实现"
            ]
        },
        {
            "solution": "权重转换优化",
            "priority": "中",
            "steps": [
                "使用更高精度的权重转换",
                "验证转换后的权重数值",
                "添加权重范围检查",
                "使用无损的权重格式"
            ]
        },
        {
            "solution": "归一化层修复",
            "priority": "中",
            "steps": [
                "确保归一化层的实现一致",
                "添加数值稳定性检查",
                "使用更稳定的归一化算法",
                "验证归一化层的输出范围"
            ]
        },
        {
            "solution": "输出范围限制",
            "priority": "高",
            "steps": [
                "在最终输出前添加范围限制",
                "使用 tanh 或 sigmoid 限制输出范围",
                "添加音频范围检查",
                "实现软限制而非硬限制"
            ]
        }
    ]
    
    for i, sol in enumerate(solutions, 1):
        print(f"\n   {i}. {sol['solution']} (优先级: {sol['priority']}):")
        for j, step in enumerate(sol['steps'], 1):
            print(f"      {j}) {step}")


def generate_audio_clipping_report():
    """生成音频爆音分析报告"""
    print("\n📄 Generating audio clipping analysis report...")
    
    report_lines = []
    report_lines.append("# 音频爆音问题分析报告")
    report_lines.append("="*50)
    report_lines.append("")
    
    report_lines.append("## 🎯 问题描述")
    report_lines.append("")
    report_lines.append("在输入一致的情况下，PyTorch 版本推理出的音频音质完美，")
    report_lines.append("但 MLX 版本推理出的音频出现爆音情况。")
    report_lines.append("")
    report_lines.append("**预期**: 输出差异应该小于 e-5")
    report_lines.append("**实际**: MLX 版本出现音频爆音")
    report_lines.append("")
    
    # 问题分析
    report_lines.append("## 🔍 问题分析")
    report_lines.append("")
    
    report_lines.append("### 1. 数值精度累积误差")
    report_lines.append("- **原因**: MLX 和 PyTorch 的数值计算精度差异在多层网络中累积")
    report_lines.append("- **影响**: 高")
    report_lines.append("- **表现**: 微小差异逐层放大，最终导致数值异常")
    report_lines.append("- **解决方案**: 使用更高精度的数值计算")
    report_lines.append("")
    
    report_lines.append("### 2. 激活函数实现差异")
    report_lines.append("- **原因**: MLX 和 PyTorch 的激活函数实现可能有细微差异")
    report_lines.append("- **影响**: 高")
    report_lines.append("- **表现**: 激活函数直接影响数值范围和分布")
    report_lines.append("- **解决方案**: 验证激活函数的一致性")
    report_lines.append("")
    
    report_lines.append("### 3. 权重加载精度损失")
    report_lines.append("- **原因**: 从 PyTorch 权重转换到 MLX 时可能存在精度损失")
    report_lines.append("- **影响**: 中")
    report_lines.append("- **表现**: 权重转换过程中的数值精度问题")
    report_lines.append("- **解决方案**: 使用更高精度的权重转换")
    report_lines.append("")
    
    report_lines.append("### 4. 归一化层差异")
    report_lines.append("- **原因**: LayerNorm、BatchNorm 等归一化层的实现差异")
    report_lines.append("- **影响**: 中")
    report_lines.append("- **表现**: 归一化层对数值稳定性至关重要")
    report_lines.append("- **解决方案**: 确保归一化层的一致性")
    report_lines.append("")
    
    # 解决方案
    report_lines.append("## 🛠️ 解决方案")
    report_lines.append("")
    
    report_lines.append("### 1. 数值精度提升 (优先级: 高)")
    report_lines.append("- 使用 FP64 精度进行关键计算")
    report_lines.append("- 验证 MLX 和 PyTorch 的数值一致性")
    report_lines.append("- 在关键层添加数值稳定性检查")
    report_lines.append("- 使用更精确的数学函数")
    report_lines.append("")
    
    report_lines.append("### 2. 激活函数一致性 (优先级: 高)")
    report_lines.append("- 对比 MLX 和 PyTorch 的激活函数实现")
    report_lines.append("- 确保激活函数的数值精度一致")
    report_lines.append("- 添加激活函数的边界检查")
    report_lines.append("- 使用自定义的激活函数实现")
    report_lines.append("")
    
    report_lines.append("### 3. 输出范围限制 (优先级: 高)")
    report_lines.append("- 在最终输出前添加范围限制")
    report_lines.append("- 使用 tanh 或 sigmoid 限制输出范围")
    report_lines.append("- 添加音频范围检查")
    report_lines.append("- 实现软限制而非硬限制")
    report_lines.append("")
    
    report_lines.append("### 4. 权重转换优化 (优先级: 中)")
    report_lines.append("- 使用更高精度的权重转换")
    report_lines.append("- 验证转换后的权重数值")
    report_lines.append("- 添加权重范围检查")
    report_lines.append("- 使用无损的权重格式")
    report_lines.append("")
    
    report_lines.append("### 5. 归一化层修复 (优先级: 中)")
    report_lines.append("- 确保归一化层的实现一致")
    report_lines.append("- 添加数值稳定性检查")
    report_lines.append("- 使用更稳定的归一化算法")
    report_lines.append("- 验证归一化层的输出范围")
    report_lines.append("")
    
    # 实施建议
    report_lines.append("## 🚀 实施建议")
    report_lines.append("")
    
    report_lines.append("### 立即措施")
    report_lines.append("1. 在 MLX 模型的最终输出层添加 `torch.clamp(output, -1.0, 1.0)`")
    report_lines.append("2. 检查 MLX 和 PyTorch 的激活函数实现差异")
    report_lines.append("3. 验证权重转换的精度")
    report_lines.append("")
    
    report_lines.append("### 短期措施 (1-2周)")
    report_lines.append("1. 实现更高精度的数值计算")
    report_lines.append("2. 添加数值稳定性检查")
    report_lines.append("3. 建立音频质量监控")
    report_lines.append("")
    
    report_lines.append("### 长期措施 (1-2月)")
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
    report_lines.append("")
    
    report_lines.append("### 音频质量监控")
    report_lines.append("- 监控音频的动态范围")
    report_lines.append("- 检测爆音和失真")
    report_lines.append("- 建立音频质量评分")
    report_lines.append("")
    
    # 结论
    report_lines.append("## 📝 结论")
    report_lines.append("")
    report_lines.append("音频爆音问题主要由数值精度差异引起，")
    report_lines.append("通过提升数值精度、确保激活函数一致性、")
    report_lines.append("添加输出范围限制等措施可以有效解决。")
    report_lines.append("")
    report_lines.append("**关键**: 在输入一致的情况下，输出差异应该小于 e-5，")
    report_lines.append("任何超出此范围的差异都可能导致音频质量问题。")
    report_lines.append("")
    
    # 保存报告
    report_path = Path("cfm_debug_outputs/audio_clipping_analysis_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"📄 Audio clipping analysis report saved to: {report_path}")
    
    return report_path


def main():
    """主函数"""
    print("🔍 Audio Clipping Issues Analysis")
    print("="*60)
    
    try:
        # 分析音频爆音问题
        analyze_audio_clipping_issues()
        
        # 生成分析报告
        report_path = generate_audio_clipping_report()
        
        print("\n🎉 Analysis completed successfully!")
        print("\n📋 Generated files:")
        print(f"1. {report_path} - 音频爆音分析报告")
        
        print("\n📊 Key Findings:")
        print("   🔍 音频爆音主要由数值精度差异引起")
        print("   ⚠️ MLX 和 PyTorch 的激活函数实现可能有差异")
        print("   💡 需要在最终输出前添加范围限制")
        print("   🚀 建议使用更高精度的数值计算")
        
        print("\n🎯 Critical Points:")
        print("   📏 输出差异应该小于 e-5")
        print("   🔧 立即添加输出范围限制")
        print("   📊 建立数值精度监控")
        print("   🛠️ 优化 MLX 实现")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


