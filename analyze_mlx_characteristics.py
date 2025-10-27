#!/usr/bin/env python3
"""
分析 MLX 内部子模块的数值特征

深入分析 MLX CFM 各个子模块的数值分布和特征
"""

import pickle
import numpy as np
from pathlib import Path


def analyze_mlx_submodule_characteristics():
    """分析 MLX 子模块的数值特征"""
    print("🔍 Analyzing MLX Submodule Characteristics...")
    
    # 加载非缓存输入数据（包含 MLX 内部子模块数据）
    non_cached_file = Path("cfm_debug_outputs/pytorch_mlx_comparison_debug.pkl")
    if not non_cached_file.exists():
        print("❌ Non-cached data file not found!")
        return
    
    with open(non_cached_file, 'rb') as f:
        data = pickle.load(f)
    
    mlx_data = data['mlx']
    
    print(f"📊 MLX Data Overview:")
    print(f"   Total stages: {len(mlx_data)}")
    
    # 按子模块类型分组
    submodule_types = {}
    for stage_name in mlx_data.keys():
        parts = stage_name.split('_')
        if len(parts) >= 3:
            submodule_type = '_'.join(parts[2:])  # 去掉 step_X_layer_Y_
            if submodule_type not in submodule_types:
                submodule_types[submodule_type] = []
            submodule_types[submodule_type].append(stage_name)
    
    print(f"   Submodule types: {len(submodule_types)}")
    
    # 分析每个子模块类型
    submodule_analysis = {}
    
    for submodule_type, stage_names in submodule_types.items():
        print(f"\n🔍 Analyzing {submodule_type}:")
        print(f"   Stages: {len(stage_names)}")
        
        # 收集所有张量的统计信息
        tensor_stats = {}
        
        for stage_name in stage_names:
            stage_data = mlx_data[stage_name]
            
            for tensor_name, tensor_info in stage_data.items():
                if tensor_name not in tensor_stats:
                    tensor_stats[tensor_name] = {
                        'shapes': [],
                        'mins': [],
                        'maxs': [],
                        'means': [],
                        'stds': [],
                        'dtypes': []
                    }
                
                tensor_stats[tensor_name]['shapes'].append(tensor_info.get('shape', []))
                tensor_stats[tensor_name]['mins'].append(tensor_info.get('min', 0))
                tensor_stats[tensor_name]['maxs'].append(tensor_info.get('max', 0))
                tensor_stats[tensor_name]['means'].append(tensor_info.get('mean', 0))
                tensor_stats[tensor_name]['stds'].append(tensor_info.get('std', 0))
                tensor_stats[tensor_name]['dtypes'].append(tensor_info.get('dtype', 'unknown'))
        
        # 计算统计摘要
        submodule_summary = {}
        
        for tensor_name, stats in tensor_stats.items():
            if stats['mins']:  # 确保有数据
                submodule_summary[tensor_name] = {
                    'shape': stats['shapes'][0],  # 假设所有阶段形状相同
                    'dtype': stats['dtypes'][0],  # 假设所有阶段类型相同
                    'min_range': [min(stats['mins']), max(stats['mins'])],
                    'max_range': [min(stats['maxs']), max(stats['maxs'])],
                    'mean_range': [min(stats['means']), max(stats['means'])],
                    'std_range': [min(stats['stds']), max(stats['stds'])],
                    'min_avg': np.mean(stats['mins']),
                    'max_avg': np.mean(stats['maxs']),
                    'mean_avg': np.mean(stats['means']),
                    'std_avg': np.mean(stats['stds']),
                    'min_std': np.std(stats['mins']),
                    'max_std': np.std(stats['maxs']),
                    'mean_std': np.std(stats['means']),
                    'std_std': np.std(stats['stds'])
                }
        
        submodule_analysis[submodule_type] = submodule_summary
        
        # 打印详细统计
        for tensor_name, summary in submodule_summary.items():
            print(f"   {tensor_name}:")
            print(f"     Shape: {summary['shape']}")
            print(f"     Dtype: {summary['dtype']}")
            print(f"     Min: {summary['min_range'][0]:.6f} ~ {summary['min_range'][1]:.6f} (avg: {summary['min_avg']:.6f})")
            print(f"     Max: {summary['max_range'][0]:.6f} ~ {summary['max_range'][1]:.6f} (avg: {summary['max_avg']:.6f})")
            print(f"     Mean: {summary['mean_range'][0]:.6f} ~ {summary['mean_range'][1]:.6f} (avg: {summary['mean_avg']:.6f})")
            print(f"     Std: {summary['std_range'][0]:.6f} ~ {summary['std_range'][1]:.6f} (avg: {summary['std_avg']:.6f})")
    
    return submodule_analysis


def analyze_data_flow_patterns(submodule_analysis):
    """分析数据流模式"""
    print("\n🔍 Analyzing Data Flow Patterns...")
    
    # 定义数据流顺序
    data_flow_order = [
        'estimator_input',
        'timestep_embedding', 
        'cond_projection',
        'x_embedding',
        'transformer_output',
        'final_layer_output',
        'estimator_output'
    ]
    
    print("\n📊 Data Flow Analysis:")
    
    for i, submodule_type in enumerate(data_flow_order):
        if submodule_type in submodule_analysis:
            print(f"\n   {i+1}. {submodule_type}:")
            
            for tensor_name, summary in submodule_analysis[submodule_type].items():
                # 分析数值范围
                value_range = summary['max_avg'] - summary['min_avg']
                mean_abs = abs(summary['mean_avg'])
                
                print(f"     {tensor_name}:")
                print(f"       Value range: {value_range:.6f}")
                print(f"       Mean magnitude: {mean_abs:.6f}")
                print(f"       Std magnitude: {summary['std_avg']:.6f}")
                
                # 判断数值特征
                if value_range < 0.01:
                    print(f"       → Small range (stable)")
                elif value_range < 0.1:
                    print(f"       → Medium range (moderate)")
                else:
                    print(f"       → Large range (dynamic)")
                
                if mean_abs < 0.01:
                    print(f"       → Near zero mean (centered)")
                else:
                    print(f"       → Non-zero mean (biased)")


def generate_mlx_characteristics_report(submodule_analysis):
    """生成 MLX 特征报告"""
    print("\n📄 Generating MLX characteristics report...")
    
    report_lines = []
    report_lines.append("# MLX CFM 子模块数值特征分析报告")
    report_lines.append("="*60)
    report_lines.append("")
    
    report_lines.append("## 🎯 分析目标")
    report_lines.append("")
    report_lines.append("深入分析 MLX CFM 各个子模块的数值特征，")
    report_lines.append("了解数据在模型中的流动和变换过程，")
    report_lines.append("为优化 MLX 实现提供数值分析支持。")
    report_lines.append("")
    
    # 数据概览
    report_lines.append("## 📊 数据概览")
    report_lines.append("")
    report_lines.append(f"- **分析子模块**: {len(submodule_analysis)} 个")
    report_lines.append(f"- **数据来源**: pytorch_mlx_comparison_debug.pkl")
    report_lines.append(f"- **分析维度**: 形状、数值范围、统计特征")
    report_lines.append("")
    
    # 子模块分析
    report_lines.append("## 🔍 子模块详细分析")
    report_lines.append("")
    
    for submodule_type, tensor_summaries in submodule_analysis.items():
        report_lines.append(f"### {submodule_type}")
        report_lines.append("")
        
        for tensor_name, summary in tensor_summaries.items():
            report_lines.append(f"#### {tensor_name}")
            report_lines.append("")
            report_lines.append(f"- **形状**: {summary['shape']}")
            report_lines.append(f"- **数据类型**: {summary['dtype']}")
            report_lines.append(f"- **最小值范围**: {summary['min_range'][0]:.6f} ~ {summary['min_range'][1]:.6f}")
            report_lines.append(f"- **最大值范围**: {summary['max_range'][0]:.6f} ~ {summary['max_range'][1]:.6f}")
            report_lines.append(f"- **均值范围**: {summary['mean_range'][0]:.6f} ~ {summary['mean_range'][1]:.6f}")
            report_lines.append(f"- **标准差范围**: {summary['std_range'][0]:.6f} ~ {summary['std_range'][1]:.6f}")
            report_lines.append("")
            
            # 数值特征分析
            value_range = summary['max_avg'] - summary['min_avg']
            mean_abs = abs(summary['mean_avg'])
            
            report_lines.append("**数值特征**:")
            report_lines.append(f"- 数值范围: {value_range:.6f}")
            report_lines.append(f"- 均值幅度: {mean_abs:.6f}")
            report_lines.append(f"- 标准差幅度: {summary['std_avg']:.6f}")
            report_lines.append("")
            
            # 特征判断
            if value_range < 0.01:
                range_desc = "小范围 (稳定)"
            elif value_range < 0.1:
                range_desc = "中等范围 (适中)"
            else:
                range_desc = "大范围 (动态)"
            
            if mean_abs < 0.01:
                mean_desc = "接近零均值 (居中)"
            else:
                mean_desc = "非零均值 (有偏)"
            
            report_lines.append(f"- 范围特征: {range_desc}")
            report_lines.append(f"- 均值特征: {mean_desc}")
            report_lines.append("")
    
    # 数据流分析
    report_lines.append("## 🌊 数据流分析")
    report_lines.append("")
    
    data_flow_order = [
        'estimator_input',
        'timestep_embedding', 
        'cond_projection',
        'x_embedding',
        'transformer_output',
        'final_layer_output',
        'estimator_output'
    ]
    
    for i, submodule_type in enumerate(data_flow_order):
        if submodule_type in submodule_analysis:
            report_lines.append(f"### {i+1}. {submodule_type}")
            report_lines.append("")
            
            tensor_summaries = submodule_analysis[submodule_type]
            for tensor_name, summary in tensor_summaries.items():
                value_range = summary['max_avg'] - summary['min_avg']
                mean_abs = abs(summary['mean_avg'])
                
                report_lines.append(f"- **{tensor_name}**:")
                report_lines.append(f"  - 数值范围: {value_range:.6f}")
                report_lines.append(f"  - 均值幅度: {mean_abs:.6f}")
                report_lines.append(f"  - 标准差幅度: {summary['std_avg']:.6f}")
            report_lines.append("")
    
    # 关键发现
    report_lines.append("## 🔍 关键发现")
    report_lines.append("")
    
    # 分析数值稳定性
    stable_modules = []
    dynamic_modules = []
    
    for submodule_type, tensor_summaries in submodule_analysis.items():
        for tensor_name, summary in tensor_summaries.items():
            value_range = summary['max_avg'] - summary['min_avg']
            if value_range < 0.01:
                stable_modules.append(f"{submodule_type}.{tensor_name}")
            else:
                dynamic_modules.append(f"{submodule_type}.{tensor_name}")
    
    report_lines.append("### 1. 数值稳定性")
    report_lines.append("")
    report_lines.append(f"- **稳定模块**: {len(stable_modules)} 个")
    report_lines.append(f"- **动态模块**: {len(dynamic_modules)} 个")
    report_lines.append("")
    
    if stable_modules:
        report_lines.append("**稳定模块列表**:")
        for module in stable_modules:
            report_lines.append(f"- {module}")
        report_lines.append("")
    
    if dynamic_modules:
        report_lines.append("**动态模块列表**:")
        for module in dynamic_modules[:10]:  # 只显示前10个
            report_lines.append(f"- {module}")
        if len(dynamic_modules) > 10:
            report_lines.append(f"- ... 还有 {len(dynamic_modules) - 10} 个")
        report_lines.append("")
    
    # 技术建议
    report_lines.append("## 💡 技术建议")
    report_lines.append("")
    
    report_lines.append("### 1. 数值优化")
    report_lines.append("")
    report_lines.append("- **稳定模块**: 可以优化为更紧凑的数值表示")
    report_lines.append("- **动态模块**: 需要保持足够的数值精度")
    report_lines.append("- **均值偏移**: 考虑添加归一化层减少偏移")
    report_lines.append("")
    
    report_lines.append("### 2. 性能优化")
    report_lines.append("")
    report_lines.append("- **内存优化**: 基于数值范围优化内存分配")
    report_lines.append("- **计算优化**: 针对数值特征优化计算精度")
    report_lines.append("- **并行优化**: 利用 MLX 的并行计算能力")
    report_lines.append("")
    
    # 保存报告
    report_path = Path("cfm_debug_outputs/mlx_submodule_characteristics_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"📄 MLX characteristics report saved to: {report_path}")
    
    return report_path


def main():
    """主函数"""
    print("🔍 MLX Submodule Characteristics Analysis")
    print("="*60)
    
    try:
        # 分析 MLX 子模块特征
        submodule_analysis = analyze_mlx_submodule_characteristics()
        
        if submodule_analysis:
            # 分析数据流模式
            analyze_data_flow_patterns(submodule_analysis)
            
            # 生成特征报告
            report_path = generate_mlx_characteristics_report(submodule_analysis)
            
            print("\n🎉 MLX characteristics analysis completed successfully!")
            print("\n📋 Generated files:")
            print(f"1. {report_path} - MLX 子模块特征分析报告")
            
            print("\n📊 Analysis Summary:")
            print(f"   Analyzed submodules: {len(submodule_analysis)}")
            
            total_tensors = sum(len(tensors) for tensors in submodule_analysis.values())
            print(f"   Total tensors analyzed: {total_tensors}")
            
            # 统计稳定和动态模块
            stable_count = 0
            dynamic_count = 0
            
            for submodule_type, tensor_summaries in submodule_analysis.items():
                for tensor_name, summary in tensor_summaries.items():
                    value_range = summary['max_avg'] - summary['min_avg']
                    if value_range < 0.01:
                        stable_count += 1
                    else:
                        dynamic_count += 1
            
            print(f"   Stable tensors: {stable_count}")
            print(f"   Dynamic tensors: {dynamic_count}")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


