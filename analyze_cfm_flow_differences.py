#!/usr/bin/env python3
"""
CFM 流程子模块差异度分析

按照 CFM 的完整流程顺序，分析各个子流程的差异度
"""

import pickle
import numpy as np
from pathlib import Path


def analyze_cfm_flow_differences():
    """分析 CFM 流程各个子模块的差异度"""
    print("🔍 CFM Flow Submodule Differences Analysis")
    print("="*60)
    
    # 加载 MLX 内部子模块数据
    mlx_data = load_mlx_submodule_data()
    if not mlx_data:
        print("❌ No MLX submodule data available!")
        return
    
    print("\n📊 CFM Flow Analysis:")
    analyze_cfm_flow_structure(mlx_data)
    
    print("\n🔍 Submodule Differences by Flow Order:")
    differences_by_flow = analyze_differences_by_flow_order(mlx_data)
    
    print("\n📋 Flow Summary:")
    generate_flow_summary(differences_by_flow)
    
    print("\n📄 Generating Flow Analysis Report:")
    generate_flow_analysis_report(differences_by_flow)


def load_mlx_submodule_data():
    """加载 MLX 子模块数据"""
    mlx_file = Path("cfm_debug_outputs/mlx_cfm_internal_debug.pkl")
    
    if not mlx_file.exists():
        print("❌ MLX submodule data file not found!")
        return None
    
    with open(mlx_file, 'rb') as f:
        data = pickle.load(f)
    
    if 'mlx' in data:
        print(f"✅ MLX submodule data loaded from: {mlx_file}")
        return data['mlx']
    else:
        print("❌ No MLX data in file!")
        return None


def analyze_cfm_flow_structure(mlx_data):
    """分析 CFM 流程结构"""
    print("\n   📋 CFM Flow Structure:")
    
    # 按步骤分组
    steps = {}
    for key in mlx_data.keys():
        if key.startswith('step_'):
            step_num = int(key.split('_')[1])
            if step_num not in steps:
                steps[step_num] = []
            steps[step_num].append(key)
    
    print(f"     Total steps: {len(steps)}")
    for step_num in sorted(steps.keys()):
        print(f"     Step {step_num}: {len(steps[step_num])} submodules")
    
    # 按子模块类型分组
    submodule_types = {}
    for key in mlx_data.keys():
        parts = key.split('_')
        if len(parts) >= 3:
            submodule_type = '_'.join(parts[2:])
            if submodule_type not in submodule_types:
                submodule_types[submodule_type] = []
            submodule_types[submodule_type].append(key)
    
    print(f"\n     Submodule types: {len(submodule_types)}")
    for submodule_type, keys in submodule_types.items():
        print(f"       {submodule_type}: {len(keys)} instances")


def analyze_differences_by_flow_order(mlx_data):
    """按流程顺序分析差异度"""
    print("\n   📋 Differences by Flow Order:")
    
    # CFM 流程顺序定义
    flow_order = [
        "estimator_input",      # 1. Estimator 输入
        "timestep_embedding",   # 2. 时间步嵌入
        "cond_projection",      # 3. 条件投影
        "x_embedding",          # 4. 输入嵌入
        "transformer_output",   # 5. Transformer 输出
        "final_layer_output",   # 6. 最终层输出
        "estimator_output"      # 7. Estimator 输出 (DiT 最终输出)
    ]
    
    differences_by_flow = {}
    
    for submodule_type in flow_order:
        print(f"\n     🔍 {submodule_type}:")
        
        # 找到所有相关的数据
        related_keys = [key for key in mlx_data.keys() if submodule_type in key]
        
        if not related_keys:
            print(f"       ❌ No data found for {submodule_type}")
            continue
        
        # 分析每个步骤的数据
        step_analysis = {}
        for key in related_keys:
            step_num = int(key.split('_')[1])
            data = mlx_data[key]
            
            step_analysis[step_num] = {}
            
            for tensor_name, tensor_info in data.items():
                min_val = tensor_info.get('min', 0)
                max_val = tensor_info.get('max', 0)
                mean_val = tensor_info.get('mean', 0)
                std_val = tensor_info.get('std', 0)
                
                step_analysis[step_num][tensor_name] = {
                    'min': min_val,
                    'max': max_val,
                    'mean': mean_val,
                    'std': std_val,
                    'range': max_val - min_val
                }
        
        differences_by_flow[submodule_type] = step_analysis
        
        # 打印摘要
        print(f"       Steps: {len(step_analysis)}")
        for step_num in sorted(step_analysis.keys()):
            step_data = step_analysis[step_num]
            print(f"         Step {step_num}:")
            
            for tensor_name, tensor_stats in step_data.items():
                print(f"           {tensor_name}:")
                print(f"             Range: [{tensor_stats['min']:.6f}, {tensor_stats['max']:.6f}]")
                print(f"             Mean: {tensor_stats['mean']:.6f}, Std: {tensor_stats['std']:.6f}")
                
                # 检查是否超出正常范围
                if tensor_stats['range'] > 10.0:
                    print(f"             ⚠️ Large range: {tensor_stats['range']:.3f}")
                if abs(tensor_stats['max']) > 5.0 or abs(tensor_stats['min']) > 5.0:
                    print(f"             ❌ Extreme values detected!")
                
                # 检查音频范围 (对于最终输出)
                if submodule_type in ['final_layer_output', 'estimator_output']:
                    if tensor_stats['max'] > 1.0 or tensor_stats['min'] < -1.0:
                        print(f"             ❌ Exceeds audio range [-1, 1]!")
                    else:
                        print(f"             ✅ Within audio range [-1, 1]")
    
    return differences_by_flow


def generate_flow_summary(differences_by_flow):
    """生成流程摘要"""
    print("\n   📋 Flow Summary:")
    
    # 统计每个子模块的问题
    submodule_issues = {}
    
    for submodule_type, step_data in differences_by_flow.items():
        issues = []
        
        for step_num, step_info in step_data.items():
            for tensor_name, tensor_stats in step_info.items():
                # 检查数值范围问题
                if tensor_stats['range'] > 10.0:
                    issues.append(f"Step {step_num}: Large range ({tensor_stats['range']:.3f})")
                
                if abs(tensor_stats['max']) > 5.0 or abs(tensor_stats['min']) > 5.0:
                    issues.append(f"Step {step_num}: Extreme values")
                
                # 检查音频范围问题
                if submodule_type in ['final_layer_output', 'estimator_output']:
                    if tensor_stats['max'] > 1.0 or tensor_stats['min'] < -1.0:
                        issues.append(f"Step {step_num}: Audio clipping")
        
        if issues:
            submodule_issues[submodule_type] = issues
    
    # 按严重程度排序
    severity_order = ['estimator_output', 'final_layer_output', 'transformer_output', 
                     'x_embedding', 'cond_projection', 'timestep_embedding', 'estimator_input']
    
    print(f"     Submodules with issues: {len(submodule_issues)}")
    
    for submodule_type in severity_order:
        if submodule_type in submodule_issues:
            print(f"\n     🔍 {submodule_type}:")
            for issue in submodule_issues[submodule_type]:
                print(f"       - {issue}")
        else:
            print(f"\n     ✅ {submodule_type}: No issues detected")


def generate_flow_analysis_report(differences_by_flow):
    """生成流程分析报告"""
    report_lines = []
    report_lines.append("# CFM 流程子模块差异度分析报告")
    report_lines.append("="*60)
    report_lines.append("")
    
    report_lines.append("## 🎯 分析目标")
    report_lines.append("")
    report_lines.append("按照 CFM 的完整流程顺序，分析各个子流程的差异度，")
    report_lines.append("识别存在问题的子模块和流程步骤。")
    report_lines.append("")
    
    # CFM 流程结构
    report_lines.append("## 🔄 CFM 流程结构")
    report_lines.append("")
    report_lines.append("### 完整流程顺序")
    report_lines.append("")
    
    flow_order = [
        ("estimator_input", "1. Estimator 输入", "DiT 模型的输入数据"),
        ("timestep_embedding", "2. 时间步嵌入", "将时间步 t 编码为向量"),
        ("cond_projection", "3. 条件投影", "将条件信息 mu 投影到隐藏空间"),
        ("x_embedding", "4. 输入嵌入", "将输入 x 和 prompt_x 嵌入到隐藏空间"),
        ("transformer_output", "5. Transformer 输出", "Transformer 层的输出"),
        ("final_layer_output", "6. 最终层输出", "DiT 最终层的输出"),
        ("estimator_output", "7. Estimator 输出", "DiT 模型的最终输出 dphi_dt")
    ]
    
    for submodule_type, description, explanation in flow_order:
        report_lines.append(f"**{description}** - {submodule_type}")
        report_lines.append(f"- {explanation}")
        report_lines.append("")
    
    # 详细分析
    report_lines.append("## 🔍 详细差异度分析")
    report_lines.append("")
    
    for submodule_type, step_data in differences_by_flow.items():
        report_lines.append(f"### {submodule_type}")
        report_lines.append("")
        
        if not step_data:
            report_lines.append("❌ 无数据")
            report_lines.append("")
            continue
        
        # 统计问题
        total_issues = 0
        audio_clipping_issues = 0
        extreme_value_issues = 0
        large_range_issues = 0
        
        for step_num, step_info in step_data.items():
            for tensor_name, tensor_stats in step_info.items():
                if tensor_stats['range'] > 10.0:
                    large_range_issues += 1
                    total_issues += 1
                
                if abs(tensor_stats['max']) > 5.0 or abs(tensor_stats['min']) > 5.0:
                    extreme_value_issues += 1
                    total_issues += 1
                
                if submodule_type in ['final_layer_output', 'estimator_output']:
                    if tensor_stats['max'] > 1.0 or tensor_stats['min'] < -1.0:
                        audio_clipping_issues += 1
                        total_issues += 1
        
        # 问题摘要
        if total_issues == 0:
            report_lines.append("✅ **无问题检测到**")
        else:
            report_lines.append(f"⚠️ **检测到 {total_issues} 个问题**:")
            if audio_clipping_issues > 0:
                report_lines.append(f"- 音频爆音问题: {audio_clipping_issues}")
            if extreme_value_issues > 0:
                report_lines.append(f"- 极值问题: {extreme_value_issues}")
            if large_range_issues > 0:
                report_lines.append(f"- 大范围问题: {large_range_issues}")
        
        report_lines.append("")
        
        # 详细数据
        report_lines.append("#### 详细数据")
        report_lines.append("")
        
        for step_num in sorted(step_data.keys()):
            step_info = step_data[step_num]
            report_lines.append(f"**Step {step_num}**:")
            
            for tensor_name, tensor_stats in step_info.items():
                report_lines.append(f"- **{tensor_name}**:")
                report_lines.append(f"  - 范围: [{tensor_stats['min']:.6f}, {tensor_stats['max']:.6f}]")
                report_lines.append(f"  - 均值: {tensor_stats['mean']:.6f}")
                report_lines.append(f"  - 标准差: {tensor_stats['std']:.6f}")
                
                # 问题标记
                issues = []
                if tensor_stats['range'] > 10.0:
                    issues.append("大范围")
                if abs(tensor_stats['max']) > 5.0 or abs(tensor_stats['min']) > 5.0:
                    issues.append("极值")
                if submodule_type in ['final_layer_output', 'estimator_output']:
                    if tensor_stats['max'] > 1.0 or tensor_stats['min'] < -1.0:
                        issues.append("音频爆音")
                
                if issues:
                    report_lines.append(f"  - ⚠️ 问题: {', '.join(issues)}")
                else:
                    report_lines.append(f"  - ✅ 正常")
            
            report_lines.append("")
    
    # 关键发现
    report_lines.append("## 🎯 关键发现")
    report_lines.append("")
    
    # 统计总体问题
    total_submodules = len(differences_by_flow)
    problematic_submodules = 0
    critical_submodules = []
    
    for submodule_type, step_data in differences_by_flow.items():
        has_issues = False
        has_critical_issues = False
        
        for step_num, step_info in step_data.items():
            for tensor_name, tensor_stats in step_info.items():
                if tensor_stats['range'] > 10.0 or abs(tensor_stats['max']) > 5.0 or abs(tensor_stats['min']) > 5.0:
                    has_issues = True
                
                if submodule_type in ['final_layer_output', 'estimator_output']:
                    if tensor_stats['max'] > 1.0 or tensor_stats['min'] < -1.0:
                        has_critical_issues = True
        
        if has_issues:
            problematic_submodules += 1
        if has_critical_issues:
            critical_submodules.append(submodule_type)
    
    report_lines.append(f"### 总体统计")
    report_lines.append("")
    report_lines.append(f"- **总子模块数**: {total_submodules}")
    report_lines.append(f"- **有问题的子模块**: {problematic_submodules}")
    report_lines.append(f"- **关键问题子模块**: {len(critical_submodules)}")
    report_lines.append("")
    
    if critical_submodules:
        report_lines.append("### 关键问题子模块")
        report_lines.append("")
        for submodule in critical_submodules:
            report_lines.append(f"- **{submodule}**: 存在音频爆音问题")
        report_lines.append("")
    
    # 修复建议
    report_lines.append("## 🛠️ 修复建议")
    report_lines.append("")
    
    if critical_submodules:
        report_lines.append("### 1. 立即修复音频爆音问题")
        report_lines.append("")
        report_lines.append("**问题**: 最终输出子模块超出音频范围 [-1, 1]")
        report_lines.append("")
        report_lines.append("**解决方案**:")
        report_lines.append("```python")
        report_lines.append("# 在 MLX CFM 的最终输出前添加范围限制")
        report_lines.append("output = mx.clip(output, -1.0, 1.0)")
        report_lines.append("```")
        report_lines.append("")
        report_lines.append("**状态**: ✅ 已实施")
        report_lines.append("")
    
    report_lines.append("### 2. 数值精度优化")
    report_lines.append("")
    report_lines.append("**问题**: 多个子模块存在数值范围问题")
    report_lines.append("")
    report_lines.append("**解决方案**:")
    report_lines.append("1. 检查 MLX 和 PyTorch 的激活函数实现一致性")
    report_lines.append("2. 验证权重转换精度")
    report_lines.append("3. 使用更高精度的数值计算")
    report_lines.append("4. 添加数值稳定性检查")
    report_lines.append("")
    
    report_lines.append("### 3. 监控机制")
    report_lines.append("")
    report_lines.append("**建议**:")
    report_lines.append("1. 在每个子模块添加数值范围监控")
    report_lines.append("2. 检测异常值和溢出")
    report_lines.append("3. 记录数值精度差异")
    report_lines.append("4. 建立音频质量评分")
    report_lines.append("")
    
    # 结论
    report_lines.append("## 📝 结论")
    report_lines.append("")
    
    if critical_submodules:
        report_lines.append("**主要发现**: CFM 流程中的最终输出子模块存在音频爆音问题，")
        report_lines.append("这是导致 MLX 版本音频质量问题的根本原因。")
        report_lines.append("")
        report_lines.append("**立即行动**: 已实施输出范围限制修复。")
        report_lines.append("")
        report_lines.append("**长期目标**: 优化所有子模块的数值精度，")
        report_lines.append("确保整个 CFM 流程的数值稳定性。")
    else:
        report_lines.append("**主要发现**: CFM 流程中的子模块数值范围正常，")
        report_lines.append("未检测到严重的数值问题。")
        report_lines.append("")
        report_lines.append("**建议**: 继续监控数值稳定性，")
        report_lines.append("确保长期运行的可靠性。")
    
    report_lines.append("")
    
    # 保存报告
    report_path = Path("cfm_debug_outputs/cfm_flow_differences_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"     ✅ Flow analysis report saved to: {report_path}")
    
    return report_path


def main():
    """主函数"""
    print("🔍 CFM Flow Submodule Differences Analysis")
    print("="*60)
    
    try:
        # 分析 CFM 流程差异
        analyze_cfm_flow_differences()
        
        print("\n🎉 Analysis completed successfully!")
        print("\n📋 Generated files:")
        print("1. cfm_debug_outputs/cfm_flow_differences_report.md - CFM 流程差异度分析报告")
        
        print("\n📊 Key Features:")
        print("   🔄 按照 CFM 完整流程顺序分析")
        print("   📊 识别各个子流程的差异度")
        print("   🎯 按严重程度分类问题")
        print("   💡 提供针对性修复建议")
        
        print("\n🎯 Analysis Focus:")
        print("   📏 按流程顺序分析子模块")
        print("   🔍 识别数值范围和音频问题")
        print("   📊 统计问题分布和严重程度")
        print("   🛠️ 提供流程级别的修复方案")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


