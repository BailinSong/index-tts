#!/usr/bin/env python3
"""
深入分析使用前级缓存输入的 CFM 子模块对比数据

分析 PyTorch 和 MLX 实现之间的差异
"""

import pickle
import numpy as np
from pathlib import Path


def load_comparison_data():
    """加载对比数据"""
    print("📂 Loading comparison data...")
    
    # 加载缓存输入对比数据
    cached_file = Path("cfm_debug_outputs/cached_inputs_submodules_comparison.pkl")
    if not cached_file.exists():
        print("❌ Cached inputs comparison file not found!")
        return None, None
    
    with open(cached_file, 'rb') as f:
        cached_data = pickle.load(f)
    
    # 加载非缓存输入对比数据（用于对比）
    non_cached_file = Path("cfm_debug_outputs/pytorch_mlx_comparison_debug.pkl")
    non_cached_data = None
    if non_cached_file.exists():
        with open(non_cached_file, 'rb') as f:
            non_cached_data = pickle.load(f)
    
    print("✅ Data loaded successfully!")
    return cached_data, non_cached_data


def analyze_cached_vs_non_cached():
    """分析缓存输入 vs 非缓存输入的差异"""
    print("\n🔍 Analyzing cached vs non-cached input differences...")
    
    cached_data, non_cached_data = load_comparison_data()
    if cached_data is None:
        return
    
    print("\n📊 Data Overview:")
    print(f"   Cached inputs - PyTorch stages: {len(cached_data['pytorch'])}")
    print(f"   Cached inputs - MLX stages: {len(cached_data['mlx'])}")
    print(f"   Cached inputs - Comparisons: {len(cached_data['comparisons'])}")
    
    if non_cached_data:
        print(f"   Non-cached inputs - PyTorch stages: {len(non_cached_data['pytorch'])}")
        print(f"   Non-cached inputs - MLX stages: {len(non_cached_data['mlx'])}")
        print(f"   Non-cached inputs - Comparisons: {len(non_cached_data['comparisons'])}")
    
    # 分析缓存输入的对比结果
    print("\n🔍 Cached Inputs Analysis:")
    cached_analysis = analyze_comparison_data(cached_data, "Cached Inputs")
    
    # 分析非缓存输入的对比结果
    if non_cached_data:
        print("\n🔍 Non-Cached Inputs Analysis:")
        non_cached_analysis = analyze_comparison_data(non_cached_data, "Non-Cached Inputs")
        
        # 对比分析
        print("\n🔍 Comparison Analysis:")
        compare_cached_vs_non_cached(cached_analysis, non_cached_analysis)
    
    return cached_analysis


def analyze_comparison_data(data, data_type):
    """分析对比数据"""
    print(f"\n📋 {data_type} Detailed Analysis:")
    
    comparisons = data['comparisons']
    pytorch_data = data['pytorch']
    mlx_data = data['mlx']
    
    # 统计信息
    total_tensors = 0
    close_tensors = 0
    significant_diffs = []
    tensor_stats = []
    
    for key, comp_data in comparisons.items():
        print(f"\n   📊 {key}:")
        
        if 'tensor_comparisons' in comp_data:
            for tensor_name, tensor_comp in comp_data['tensor_comparisons'].items():
                total_tensors += 1
                
                # 提取统计信息
                max_diff = tensor_comp.get('max_diff', 0)
                mean_diff = tensor_comp.get('mean_diff', 0)
                relative_diff = tensor_comp.get('relative_diff', 0)
                is_close = tensor_comp.get('is_close', False)
                
                pytorch_stats = tensor_comp.get('pytorch_stats', {})
                mlx_stats = tensor_comp.get('mlx_stats', {})
                
                tensor_stats.append({
                    'stage': key,
                    'tensor': tensor_name,
                    'max_diff': max_diff,
                    'mean_diff': mean_diff,
                    'relative_diff': relative_diff,
                    'is_close': is_close,
                    'pytorch_min': pytorch_stats.get('min', 0),
                    'pytorch_max': pytorch_stats.get('max', 0),
                    'pytorch_mean': pytorch_stats.get('mean', 0),
                    'pytorch_std': pytorch_stats.get('std', 0),
                    'mlx_min': mlx_stats.get('min', 0),
                    'mlx_max': mlx_stats.get('max', 0),
                    'mlx_mean': mlx_stats.get('mean', 0),
                    'mlx_std': mlx_stats.get('std', 0),
                })
                
                print(f"     {tensor_name}:")
                print(f"       Max diff: {max_diff:.8f}")
                print(f"       Mean diff: {mean_diff:.8f}")
                print(f"       Relative diff: {relative_diff:.8f}")
                print(f"       Is close: {'✅' if is_close else '❌'}")
                
                if pytorch_stats and mlx_stats:
                    print(f"       PyTorch range: [{pytorch_stats.get('min', 0):.6f}, {pytorch_stats.get('max', 0):.6f}]")
                    print(f"       MLX range: [{mlx_stats.get('min', 0):.6f}, {mlx_stats.get('max', 0):.6f}]")
                
                if is_close:
                    close_tensors += 1
                else:
                    significant_diffs.append({
                        'stage': key,
                        'tensor': tensor_name,
                        'max_diff': max_diff,
                        'relative_diff': relative_diff
                    })
    
    # 总结
    print(f"\n📊 {data_type} Summary:")
    print(f"   Total tensors: {total_tensors}")
    print(f"   Close tensors: {close_tensors}")
    print(f"   Significant differences: {len(significant_diffs)}")
    
    if significant_diffs:
        print(f"\n⚠️  Significant Differences:")
        for diff in significant_diffs:
            print(f"   {diff['stage']} - {diff['tensor']}: "
                  f"max_diff={diff['max_diff']:.8f}, "
                  f"relative_diff={diff['relative_diff']:.8f}")
    
    return {
        'total_tensors': total_tensors,
        'close_tensors': close_tensors,
        'significant_diffs': significant_diffs,
        'tensor_stats': tensor_stats
    }


def compare_cached_vs_non_cached(cached_analysis, non_cached_analysis):
    """对比缓存输入和非缓存输入的分析结果"""
    print("\n📊 Cached vs Non-Cached Comparison:")
    
    print(f"   Cached inputs - Close tensors: {cached_analysis['close_tensors']}/{cached_analysis['total_tensors']}")
    print(f"   Non-cached inputs - Close tensors: {non_cached_analysis['close_tensors']}/{non_cached_analysis['total_tensors']}")
    
    cached_close_rate = cached_analysis['close_tensors'] / cached_analysis['total_tensors'] if cached_analysis['total_tensors'] > 0 else 0
    non_cached_close_rate = non_cached_analysis['close_tensors'] / non_cached_analysis['total_tensors'] if non_cached_analysis['total_tensors'] > 0 else 0
    
    print(f"   Cached inputs - Close rate: {cached_close_rate:.2%}")
    print(f"   Non-cached inputs - Close rate: {non_cached_close_rate:.2%}")
    
    improvement = cached_close_rate - non_cached_close_rate
    print(f"   Improvement with cached inputs: {improvement:.2%}")
    
    if improvement > 0:
        print("   ✅ Cached inputs significantly improve consistency!")
    else:
        print("   ⚠️  Cached inputs do not improve consistency")


def analyze_mlx_internal_submodules():
    """分析 MLX 内部子模块数据"""
    print("\n🔍 Analyzing MLX Internal Submodules...")
    
    _, non_cached_data = load_comparison_data()
    if non_cached_data is None:
        print("❌ Non-cached data not available for MLX internal analysis")
        return
    
    mlx_data = non_cached_data['mlx']
    
    # 按子模块类型分组
    submodule_types = {}
    for stage_name in mlx_data.keys():
        parts = stage_name.split('_')
        if len(parts) >= 3:
            submodule_type = '_'.join(parts[2:])  # 去掉 step_X_layer_Y_
            if submodule_type not in submodule_types:
                submodule_types[submodule_type] = []
            submodule_types[submodule_type].append(stage_name)
    
    print(f"\n📊 MLX Internal Submodules:")
    print(f"   Total submodule types: {len(submodule_types)}")
    
    for submodule_type, stage_names in submodule_types.items():
        print(f"\n   🔍 {submodule_type}:")
        print(f"     Stages: {len(stage_names)}")
        
        # 分析第一个阶段的详细信息
        if stage_names:
            first_stage = stage_names[0]
            stage_data = mlx_data[first_stage]
            
            print(f"     Sample stage: {first_stage}")
            for tensor_name, tensor_info in stage_data.items():
                print(f"       {tensor_name}:")
                print(f"         Shape: {tensor_info.get('shape', 'N/A')}")
                print(f"         Dtype: {tensor_info.get('dtype', 'N/A')}")
                print(f"         Min: {tensor_info.get('min', 'N/A'):.8f}")
                print(f"         Max: {tensor_info.get('max', 'N/A'):.8f}")
                print(f"         Mean: {tensor_info.get('mean', 'N/A'):.8f}")
                print(f"         Std: {tensor_info.get('std', 'N/A'):.8f}")


def generate_detailed_analysis_report():
    """生成详细分析报告"""
    print("\n📄 Generating detailed analysis report...")
    
    cached_data, non_cached_data = load_comparison_data()
    if cached_data is None:
        return
    
    report_lines = []
    report_lines.append("# CFM 子模块深度分析报告")
    report_lines.append("="*60)
    report_lines.append("")
    
    report_lines.append("## 🎯 分析目标")
    report_lines.append("")
    report_lines.append("深入分析使用前级缓存输入的 CFM 子模块对比数据，")
    report_lines.append("识别 PyTorch 和 MLX 实现之间的差异，")
    report_lines.append("为优化 MLX 实现提供数据支持。")
    report_lines.append("")
    
    # 数据概览
    report_lines.append("## 📊 数据概览")
    report_lines.append("")
    report_lines.append(f"- **缓存输入对比**: {len(cached_data['comparisons'])} 个对比")
    report_lines.append(f"- **PyTorch 阶段**: {len(cached_data['pytorch'])} 个")
    report_lines.append(f"- **MLX 阶段**: {len(cached_data['mlx'])} 个")
    
    if non_cached_data:
        report_lines.append(f"- **非缓存输入对比**: {len(non_cached_data['comparisons'])} 个对比")
        report_lines.append(f"- **MLX 内部子模块**: {len([k for k in non_cached_data['mlx'].keys() if any(sub in k for sub in ['timestep_embedding', 'cond_projection', 'x_embedding', 'transformer_output', 'final_layer_output'])])} 个")
    
    report_lines.append("")
    
    # 关键发现
    report_lines.append("## 🔍 关键发现")
    report_lines.append("")
    
    # 分析缓存输入结果
    cached_analysis = analyze_comparison_data(cached_data, "Cached Inputs")
    
    report_lines.append("### 1. 缓存输入效果")
    report_lines.append("")
    report_lines.append(f"- **总张量数**: {cached_analysis['total_tensors']}")
    report_lines.append(f"- **一致张量数**: {cached_analysis['close_tensors']}")
    report_lines.append(f"- **显著差异数**: {len(cached_analysis['significant_diffs'])}")
    report_lines.append(f"- **一致性率**: {cached_analysis['close_tensors']/cached_analysis['total_tensors']:.2%}")
    report_lines.append("")
    
    if len(cached_analysis['significant_diffs']) == 0:
        report_lines.append("✅ **结论**: 使用缓存输入后，PyTorch 和 MLX 的输出完全一致！")
        report_lines.append("")
        report_lines.append("这证明了：")
        report_lines.append("1. 入口数据一致性是确保输出一致性的关键")
        report_lines.append("2. PyTorch 和 MLX 的实现逻辑是等价的")
        report_lines.append("3. 差异主要来源于输入数据的不一致性")
    else:
        report_lines.append("⚠️ **结论**: 即使使用缓存输入，仍存在显著差异")
        report_lines.append("")
        report_lines.append("需要进一步分析差异来源：")
        for diff in cached_analysis['significant_diffs']:
            report_lines.append(f"- {diff['stage']} - {diff['tensor']}: 相对差异 {diff['relative_diff']:.2%}")
    
    report_lines.append("")
    
    # 技术建议
    report_lines.append("## 💡 技术建议")
    report_lines.append("")
    
    if len(cached_analysis['significant_diffs']) == 0:
        report_lines.append("### 1. 生产环境优化")
        report_lines.append("")
        report_lines.append("- **使用缓存输入**: 在生产环境中实现缓存输入机制")
        report_lines.append("- **统一随机种子**: 确保 PyTorch 和 MLX 使用相同的随机数生成")
        report_lines.append("- **输入验证**: 添加输入数据一致性检查")
        report_lines.append("")
        
        report_lines.append("### 2. 性能优化")
        report_lines.append("")
        report_lines.append("- **MLX 优势**: 利用 MLX 在 Apple Silicon 上的性能优势")
        report_lines.append("- **内存优化**: 使用 MLX 的内存管理优化")
        report_lines.append("- **并行计算**: 利用 MLX 的并行计算能力")
        report_lines.append("")
    else:
        report_lines.append("### 1. 差异分析")
        report_lines.append("")
        report_lines.append("- **数值精度**: 检查 PyTorch 和 MLX 的数值精度差异")
        report_lines.append("- **计算顺序**: 分析计算顺序对结果的影响")
        report_lines.append("- **优化算法**: 对比优化算法的实现差异")
        report_lines.append("")
        
        report_lines.append("### 2. 实现优化")
        report_lines.append("")
        report_lines.append("- **MLX 实现**: 优化 MLX 实现以提高数值精度")
        report_lines.append("- **算法对齐**: 确保 PyTorch 和 MLX 使用相同的算法")
        report_lines.append("- **测试验证**: 增加更多的测试用例验证一致性")
        report_lines.append("")
    
    # 下一步工作
    report_lines.append("## 🚀 下一步工作")
    report_lines.append("")
    report_lines.append("1. **深入分析**: 分析 MLX 内部子模块的具体数值特征")
    report_lines.append("2. **性能测试**: 对比 PyTorch 和 MLX 的性能表现")
    report_lines.append("3. **优化实现**: 基于分析结果优化 MLX 实现")
    report_lines.append("4. **生产部署**: 将优化后的实现部署到生产环境")
    report_lines.append("")
    
    # 保存报告
    report_path = Path("cfm_debug_outputs/cfm_deep_analysis_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"📄 Detailed analysis report saved to: {report_path}")
    
    return report_path


def main():
    """主函数"""
    print("🔍 CFM Submodules Deep Analysis")
    print("="*60)
    
    try:
        # 分析缓存输入 vs 非缓存输入
        cached_analysis = analyze_cached_vs_non_cached()
        
        # 分析 MLX 内部子模块
        analyze_mlx_internal_submodules()
        
        # 生成详细分析报告
        report_path = generate_detailed_analysis_report()
        
        print("\n🎉 Deep analysis completed successfully!")
        print("\n📋 Generated files:")
        print(f"1. {report_path} - 深度分析报告")
        
        print("\n📊 Analysis Summary:")
        if cached_analysis:
            print(f"   Total tensors analyzed: {cached_analysis['total_tensors']}")
            print(f"   Consistent tensors: {cached_analysis['close_tensors']}")
            print(f"   Significant differences: {len(cached_analysis['significant_diffs'])}")
            print(f"   Consistency rate: {cached_analysis['close_tensors']/cached_analysis['total_tensors']:.2%}")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
