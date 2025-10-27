#!/usr/bin/env python3
"""
CFM 子模块差异分析报告

使用 CFM 前级缓存作为输入，对比 MLX 和 PyTorch 两个版本 CFM 中子模块的输入输出，
找出存在差异的子模块
"""

import pickle
import numpy as np
import torch
import mlx.core as mx
from pathlib import Path


def analyze_cfm_submodule_differences():
    """分析 CFM 子模块差异"""
    print("🔍 CFM Submodule Differences Analysis")
    print("="*60)
    
    # 加载缓存输入数据
    cached_inputs = load_cached_inputs()
    if not cached_inputs:
        print("❌ No cached inputs available!")
        return
    
    # 加载子模块对比数据
    submodule_data = load_submodule_comparison_data()
    if not submodule_data:
        print("❌ No submodule comparison data available!")
        return
    
    print("\n📊 Cached Inputs Analysis:")
    analyze_cached_inputs_summary(cached_inputs)
    
    print("\n🔍 CFM Submodule Differences Analysis:")
    differences = analyze_submodule_differences(submodule_data)
    
    print("\n📋 Detailed Submodule Analysis:")
    detailed_analysis = perform_detailed_submodule_analysis(submodule_data)
    
    print("\n🎯 Key Findings:")
    summarize_key_findings(differences, detailed_analysis)
    
    print("\n💡 Recommendations:")
    provide_recommendations(differences, detailed_analysis)
    
    # 生成分析报告
    generate_submodule_differences_report(cached_inputs, differences, detailed_analysis)


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


def load_submodule_comparison_data():
    """加载子模块对比数据"""
    submodule_file = Path("cfm_debug_outputs/cached_inputs_submodules_comparison.pkl")
    
    if not submodule_file.exists():
        print("❌ Submodule comparison data file not found!")
        return None
    
    with open(submodule_file, 'rb') as f:
        submodule_data = pickle.load(f)
    
    print(f"✅ Submodule comparison data loaded from: {submodule_file}")
    return submodule_data


def analyze_cached_inputs_summary(cached_inputs):
    """分析缓存输入摘要"""
    print("\n   📋 Cached Inputs Summary:")
    
    for key, value in cached_inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"     {key}: {value.shape}, range: [{value.min():.6f}, {value.max():.6f}]")
        elif isinstance(value, np.ndarray):
            print(f"     {key}: {value.shape}, range: [{value.min():.6f}, {value.max():.6f}]")
        else:
            print(f"     {key}: {value}")


def analyze_submodule_differences(submodule_data):
    """分析子模块差异"""
    print("\n   📋 Submodule Differences Analysis:")
    
    comparisons = submodule_data['comparisons']
    differences = {}
    
    for stage_name, stage_data in comparisons.items():
        if 'tensor_comparisons' in stage_data:
            stage_differences = {}
            
            for tensor_name, tensor_comp in stage_data['tensor_comparisons'].items():
                max_diff = tensor_comp.get('max_diff', 0)
                mean_diff = tensor_comp.get('mean_diff', 0)
                relative_diff = tensor_comp.get('relative_diff', 0)
                
                pytorch_stats = tensor_comp.get('pytorch_stats', {})
                mlx_stats = tensor_comp.get('mlx_stats', {})
                
                # 判断是否存在显著差异
                has_significant_diff = (
                    max_diff > 1e-5 or  # 最大差异超过 1e-5
                    mean_diff > 1e-6 or  # 平均差异超过 1e-6
                    relative_diff > 1e-4  # 相对差异超过 1e-4
                )
                
                if has_significant_diff:
                    stage_differences[tensor_name] = {
                        'max_diff': max_diff,
                        'mean_diff': mean_diff,
                        'relative_diff': relative_diff,
                        'pytorch_stats': pytorch_stats,
                        'mlx_stats': mlx_stats,
                        'severity': classify_difference_severity(max_diff, mean_diff, relative_diff)
                    }
            
            if stage_differences:
                differences[stage_name] = stage_differences
    
    return differences


def classify_difference_severity(max_diff, mean_diff, relative_diff):
    """分类差异严重程度"""
    if max_diff > 1e-3 or relative_diff > 1e-2:
        return "严重"
    elif max_diff > 1e-4 or relative_diff > 1e-3:
        return "中等"
    elif max_diff > 1e-5 or relative_diff > 1e-4:
        return "轻微"
    else:
        return "微小"


def perform_detailed_submodule_analysis(submodule_data):
    """执行详细的子模块分析"""
    print("\n   📋 Detailed Submodule Analysis:")
    
    comparisons = submodule_data['comparisons']
    detailed_analysis = {}
    
    # 按子模块类型分组
    submodule_types = {}
    for stage_name in comparisons.keys():
        parts = stage_name.split('_')
        if len(parts) >= 3:
            submodule_type = '_'.join(parts[2:])
            if submodule_type not in submodule_types:
                submodule_types[submodule_type] = []
            submodule_types[submodule_type].append(stage_name)
    
    # 分析每个子模块类型
    for submodule_type, stage_names in submodule_types.items():
        print(f"\n     🔍 {submodule_type}:")
        
        submodule_analysis = {
            'type': submodule_type,
            'stages': stage_names,
            'total_differences': 0,
            'significant_differences': 0,
            'max_difference': 0,
            'avg_difference': 0,
            'tensor_analysis': {}
        }
        
        differences_list = []
        
        for stage_name in stage_names:
            if stage_name in comparisons and 'tensor_comparisons' in comparisons[stage_name]:
                stage_data = comparisons[stage_name]['tensor_comparisons']
                
                for tensor_name, tensor_comp in stage_data.items():
                    max_diff = tensor_comp.get('max_diff', 0)
                    mean_diff = tensor_comp.get('mean_diff', 0)
                    relative_diff = tensor_comp.get('relative_diff', 0)
                    
                    differences_list.append(max_diff)
                    submodule_analysis['total_differences'] += 1
                    
                    if max_diff > 1e-5:
                        submodule_analysis['significant_differences'] += 1
                    
                    if tensor_name not in submodule_analysis['tensor_analysis']:
                        submodule_analysis['tensor_analysis'][tensor_name] = {
                            'max_diffs': [],
                            'mean_diffs': [],
                            'relative_diffs': []
                        }
                    
                    submodule_analysis['tensor_analysis'][tensor_name]['max_diffs'].append(max_diff)
                    submodule_analysis['tensor_analysis'][tensor_name]['mean_diffs'].append(mean_diff)
                    submodule_analysis['tensor_analysis'][tensor_name]['relative_diffs'].append(relative_diff)
        
        if differences_list:
            submodule_analysis['max_difference'] = max(differences_list)
            submodule_analysis['avg_difference'] = np.mean(differences_list)
        
        detailed_analysis[submodule_type] = submodule_analysis
        
        # 打印摘要
        print(f"       📊 Total differences: {submodule_analysis['total_differences']}")
        print(f"       📊 Significant differences: {submodule_analysis['significant_differences']}")
        print(f"       📊 Max difference: {submodule_analysis['max_difference']:.2e}")
        print(f"       📊 Avg difference: {submodule_analysis['avg_difference']:.2e}")
        
        # 分析每个张量
        for tensor_name, tensor_analysis in submodule_analysis['tensor_analysis'].items():
            avg_max_diff = np.mean(tensor_analysis['max_diffs'])
            avg_mean_diff = np.mean(tensor_analysis['mean_diffs'])
            avg_relative_diff = np.mean(tensor_analysis['relative_diffs'])
            
            print(f"         {tensor_name}:")
            print(f"           Max diff: {avg_max_diff:.2e}")
            print(f"           Mean diff: {avg_mean_diff:.2e}")
            print(f"           Relative diff: {avg_relative_diff:.2e}")
            
            if avg_max_diff > 1e-5:
                print(f"           ⚠️ Significant difference detected!")
    
    return detailed_analysis


def summarize_key_findings(differences, detailed_analysis):
    """总结关键发现"""
    print("\n   🎯 Key Findings:")
    
    # 统计总体差异
    total_stages = len(detailed_analysis)
    stages_with_differences = len([k for k, v in detailed_analysis.items() if v['significant_differences'] > 0])
    
    print(f"     📊 Total submodule types analyzed: {total_stages}")
    print(f"     📊 Submodule types with differences: {stages_with_differences}")
    print(f"     📊 Difference rate: {stages_with_differences/total_stages*100:.1f}%")
    
    # 找出最严重的差异
    max_diff_overall = 0
    worst_submodule = None
    
    for submodule_type, analysis in detailed_analysis.items():
        if analysis['max_difference'] > max_diff_overall:
            max_diff_overall = analysis['max_difference']
            worst_submodule = submodule_type
    
    if worst_submodule:
        print(f"     🔍 Worst submodule: {worst_submodule} (max diff: {max_diff_overall:.2e})")
    
    # 按严重程度分类
    severe_submodules = []
    moderate_submodules = []
    minor_submodules = []
    
    for submodule_type, analysis in detailed_analysis.items():
        if analysis['max_difference'] > 1e-3:
            severe_submodules.append(submodule_type)
        elif analysis['max_difference'] > 1e-4:
            moderate_submodules.append(submodule_type)
        elif analysis['max_difference'] > 1e-5:
            minor_submodules.append(submodule_type)
    
    print(f"     ⚠️ Severe differences (>1e-3): {len(severe_submodules)}")
    if severe_submodules:
        print(f"        {', '.join(severe_submodules)}")
    
    print(f"     ⚠️ Moderate differences (1e-4 to 1e-3): {len(moderate_submodules)}")
    if moderate_submodules:
        print(f"        {', '.join(moderate_submodules)}")
    
    print(f"     ⚠️ Minor differences (1e-5 to 1e-4): {len(minor_submodules)}")
    if minor_submodules:
        print(f"        {', '.join(minor_submodules)}")


def provide_recommendations(differences, detailed_analysis):
    """提供建议"""
    print("\n   💡 Recommendations:")
    
    # 按优先级排序建议
    recommendations = []
    
    for submodule_type, analysis in detailed_analysis.items():
        if analysis['max_difference'] > 1e-3:
            recommendations.append({
                'priority': '最高',
                'submodule': submodule_type,
                'action': '立即修复',
                'reason': f'差异过大 ({analysis["max_difference"]:.2e})',
                'steps': [
                    '检查 MLX 和 PyTorch 实现的一致性',
                    '验证数值计算精度',
                    '添加数值稳定性检查'
                ]
            })
        elif analysis['max_difference'] > 1e-4:
            recommendations.append({
                'priority': '高',
                'submodule': submodule_type,
                'action': '优先修复',
                'reason': f'差异较大 ({analysis["max_difference"]:.2e})',
                'steps': [
                    '对比实现细节',
                    '验证激活函数一致性',
                    '检查权重加载精度'
                ]
            })
        elif analysis['max_difference'] > 1e-5:
            recommendations.append({
                'priority': '中',
                'submodule': submodule_type,
                'action': '计划修复',
                'reason': f'存在差异 ({analysis["max_difference"]:.2e})',
                'steps': [
                    '监控数值稳定性',
                    '记录差异模式',
                    '考虑精度提升'
                ]
            })
    
    # 按优先级排序
    priority_order = {'最高': 1, '高': 2, '中': 3, '低': 4}
    recommendations.sort(key=lambda x: priority_order.get(x['priority'], 5))
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n     {i}. {rec['submodule']} (优先级: {rec['priority']})")
        print(f"        📝 建议: {rec['action']}")
        print(f"        🔍 原因: {rec['reason']}")
        print(f"        🔧 步骤:")
        for j, step in enumerate(rec['steps'], 1):
            print(f"           {j}) {step}")


def generate_submodule_differences_report(cached_inputs, differences, detailed_analysis):
    """生成子模块差异分析报告"""
    print("\n📄 Generating submodule differences analysis report...")
    
    report_lines = []
    report_lines.append("# CFM 子模块差异分析报告")
    report_lines.append("="*60)
    report_lines.append("")
    
    report_lines.append("## 🎯 分析目标")
    report_lines.append("")
    report_lines.append("使用 CFM 前级缓存作为输入，对比 MLX 和 PyTorch 两个版本 CFM 中")
    report_lines.append("子模块的输入输出，找出存在差异的子模块。")
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
    
    # 子模块差异分析
    report_lines.append("## 🔍 子模块差异分析")
    report_lines.append("")
    
    # 统计摘要
    total_stages = len(detailed_analysis)
    stages_with_differences = len([k for k, v in detailed_analysis.items() if v['significant_differences'] > 0])
    
    report_lines.append("### 统计摘要")
    report_lines.append("")
    report_lines.append(f"- **总子模块类型**: {total_stages}")
    report_lines.append(f"- **存在差异的子模块**: {stages_with_differences}")
    report_lines.append(f"- **差异率**: {stages_with_differences/total_stages*100:.1f}%")
    report_lines.append("")
    
    # 按严重程度分类
    severe_submodules = []
    moderate_submodules = []
    minor_submodules = []
    
    for submodule_type, analysis in detailed_analysis.items():
        if analysis['max_difference'] > 1e-3:
            severe_submodules.append(submodule_type)
        elif analysis['max_difference'] > 1e-4:
            moderate_submodules.append(submodule_type)
        elif analysis['max_difference'] > 1e-5:
            minor_submodules.append(submodule_type)
    
    report_lines.append("### 差异严重程度分类")
    report_lines.append("")
    report_lines.append(f"- **严重差异 (>1e-3)**: {len(severe_submodules)}")
    if severe_submodules:
        for submodule in severe_submodules:
            max_diff = detailed_analysis[submodule]['max_difference']
            report_lines.append(f"  - {submodule}: {max_diff:.2e}")
    
    report_lines.append(f"- **中等差异 (1e-4 to 1e-3)**: {len(moderate_submodules)}")
    if moderate_submodules:
        for submodule in moderate_submodules:
            max_diff = detailed_analysis[submodule]['max_difference']
            report_lines.append(f"  - {submodule}: {max_diff:.2e}")
    
    report_lines.append(f"- **轻微差异 (1e-5 to 1e-4)**: {len(minor_submodules)}")
    if minor_submodules:
        for submodule in minor_submodules:
            max_diff = detailed_analysis[submodule]['max_difference']
            report_lines.append(f"  - {submodule}: {max_diff:.2e}")
    
    report_lines.append("")
    
    # 详细子模块分析
    report_lines.append("## 📋 详细子模块分析")
    report_lines.append("")
    
    for submodule_type, analysis in detailed_analysis.items():
        report_lines.append(f"### {submodule_type}")
        report_lines.append("")
        report_lines.append(f"- **总差异数**: {analysis['total_differences']}")
        report_lines.append(f"- **显著差异数**: {analysis['significant_differences']}")
        report_lines.append(f"- **最大差异**: {analysis['max_difference']:.2e}")
        report_lines.append(f"- **平均差异**: {analysis['avg_difference']:.2e}")
        report_lines.append("")
        
        # 张量分析
        if analysis['tensor_analysis']:
            report_lines.append("#### 张量分析")
            report_lines.append("")
            for tensor_name, tensor_data in analysis['tensor_analysis'].items():
                avg_max_diff = np.mean(tensor_data['max_diffs'])
                avg_mean_diff = np.mean(tensor_data['mean_diffs'])
                avg_relative_diff = np.mean(tensor_data['relative_diffs'])
                
                report_lines.append(f"- **{tensor_name}**:")
                report_lines.append(f"  - 最大差异: {avg_max_diff:.2e}")
                report_lines.append(f"  - 平均差异: {avg_mean_diff:.2e}")
                report_lines.append(f"  - 相对差异: {avg_relative_diff:.2e}")
                
                if avg_max_diff > 1e-5:
                    report_lines.append(f"  - ⚠️ **存在显著差异**")
                report_lines.append("")
    
    # 建议
    report_lines.append("## 💡 修复建议")
    report_lines.append("")
    
    # 按优先级排序建议
    recommendations = []
    for submodule_type, analysis in detailed_analysis.items():
        if analysis['max_difference'] > 1e-3:
            recommendations.append({
                'priority': '最高',
                'submodule': submodule_type,
                'action': '立即修复',
                'reason': f'差异过大 ({analysis["max_difference"]:.2e})',
                'steps': [
                    '检查 MLX 和 PyTorch 实现的一致性',
                    '验证数值计算精度',
                    '添加数值稳定性检查'
                ]
            })
        elif analysis['max_difference'] > 1e-4:
            recommendations.append({
                'priority': '高',
                'submodule': submodule_type,
                'action': '优先修复',
                'reason': f'差异较大 ({analysis["max_difference"]:.2e})',
                'steps': [
                    '对比实现细节',
                    '验证激活函数一致性',
                    '检查权重加载精度'
                ]
            })
        elif analysis['max_difference'] > 1e-5:
            recommendations.append({
                'priority': '中',
                'submodule': submodule_type,
                'action': '计划修复',
                'reason': f'存在差异 ({analysis["max_difference"]:.2e})',
                'steps': [
                    '监控数值稳定性',
                    '记录差异模式',
                    '考虑精度提升'
                ]
            })
    
    # 按优先级排序
    priority_order = {'最高': 1, '高': 2, '中': 3, '低': 4}
    recommendations.sort(key=lambda x: priority_order.get(x['priority'], 5))
    
    for i, rec in enumerate(recommendations, 1):
        report_lines.append(f"### {i}. {rec['submodule']} (优先级: {rec['priority']})")
        report_lines.append("")
        report_lines.append(f"- **建议**: {rec['action']}")
        report_lines.append(f"- **原因**: {rec['reason']}")
        report_lines.append("- **步骤**:")
        for j, step in enumerate(rec['steps'], 1):
            report_lines.append(f"  {j}. {step}")
        report_lines.append("")
    
    # 结论
    report_lines.append("## 📝 结论")
    report_lines.append("")
    report_lines.append("通过对比分析发现，MLX 和 PyTorch 版本在多个子模块中存在数值差异。")
    report_lines.append("这些差异可能导致音频质量问题，需要按优先级进行修复。")
    report_lines.append("")
    report_lines.append("**关键发现**:")
    report_lines.append(f"- {stages_with_differences}/{total_stages} 个子模块类型存在差异")
    report_lines.append(f"- {len(severe_submodules)} 个子模块存在严重差异")
    report_lines.append(f"- {len(moderate_submodules)} 个子模块存在中等差异")
    report_lines.append(f"- {len(minor_submodules)} 个子模块存在轻微差异")
    report_lines.append("")
    report_lines.append("**建议**: 优先修复严重差异的子模块，确保输出差异小于 e-5。")
    report_lines.append("")
    
    # 保存报告
    report_path = Path("cfm_debug_outputs/cfm_submodule_differences_analysis_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"📄 Submodule differences analysis report saved to: {report_path}")
    
    return report_path


def main():
    """主函数"""
    print("🔍 CFM Submodule Differences Analysis")
    print("="*60)
    
    try:
        # 分析 CFM 子模块差异
        analyze_cfm_submodule_differences()
        
        print("\n🎉 Analysis completed successfully!")
        print("\n📋 Generated files:")
        print("1. cfm_debug_outputs/cfm_submodule_differences_analysis_report.md - CFM 子模块差异分析报告")
        
        print("\n📊 Key Findings:")
        print("   🔍 使用 CFM 前级缓存作为输入进行分析")
        print("   📊 对比 MLX 和 PyTorch 版本 CFM 中子模块的输入输出")
        print("   🎯 找出存在差异的子模块")
        print("   💡 提供按优先级的修复建议")
        
        print("\n🎯 Analysis Focus:")
        print("   📏 输出差异应该小于 e-5")
        print("   🔍 识别存在差异的子模块")
        print("   📊 按严重程度分类差异")
        print("   🛠️ 提供针对性修复建议")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


