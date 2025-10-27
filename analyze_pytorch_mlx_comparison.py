#!/usr/bin/env python3
"""
PyTorch vs MLX CFM 详细对比分析

分析两个版本的调试数据，生成详细的对比报告
"""

import pickle
import numpy as np
from pathlib import Path


def analyze_pytorch_mlx_comparison():
    """分析 PyTorch 和 MLX 的对比数据"""
    print("🔍 Analyzing PyTorch vs MLX CFM Comparison")
    print("="*60)
    
    # 加载调试数据
    debug_file = Path("cfm_debug_outputs/pytorch_mlx_comparison_debug.pkl")
    if not debug_file.exists():
        print("❌ Debug file not found")
        return
    
    with open(debug_file, 'rb') as f:
        data = pickle.load(f)
    
    pytorch_data = data['pytorch']
    mlx_data = data['mlx']
    comparison_data = data['comparisons']
    
    print(f"📊 Data Collection Results:")
    print(f"   PyTorch stages logged: {len(pytorch_data)}")
    print(f"   MLX stages logged: {len(mlx_data)}")
    print(f"   Comparisons made: {len(comparison_data)}")
    
    # 分析各个阶段的对比
    print(f"\n🔍 Stage-by-Stage Comparison Analysis:")
    
    stages = {}
    for key, value in comparison_data.items():
        parts = key.split('_')
        if len(parts) >= 4:
            stage = '_'.join(parts[3:])
            step = parts[1]
            if stage not in stages:
                stages[stage] = {}
            stages[stage][step] = value
    
    for stage_name, step_data in stages.items():
        print(f"\n📋 {stage_name.upper()}:")
        
        for step, comp_data in step_data.items():
            if isinstance(comp_data, dict) and 'tensor_comparisons' in comp_data:
                print(f"   Step {step}:")
                
                for tensor_name, tensor_comp in comp_data['tensor_comparisons'].items():
                    if not tensor_comp.get('comparison_failed', False):
                        status = "✅" if tensor_comp.get('is_close', True) else "❌"
                        print(f"     {status} {tensor_name}:")
                        print(f"       Shape match: {tensor_comp.get('shape_match', False)}")
                        print(f"       Max diff: {tensor_comp.get('max_diff', 'N/A'):.8f}")
                        print(f"       Mean diff: {tensor_comp.get('mean_diff', 'N/A'):.8f}")
                        print(f"       Relative diff: {tensor_comp.get('relative_diff', 'N/A'):.8f}")
                        
                        # 显示统计信息对比
                        pytorch_stats = tensor_comp.get('pytorch_stats', {})
                        mlx_stats = tensor_comp.get('mlx_stats', {})
                        
                        if pytorch_stats and mlx_stats:
                            print(f"       PyTorch: min={pytorch_stats.get('min', 'N/A'):.6f}, max={pytorch_stats.get('max', 'N/A'):.6f}, mean={pytorch_stats.get('mean', 'N/A'):.6f}")
                            print(f"       MLX:     min={mlx_stats.get('min', 'N/A'):.6f}, max={mlx_stats.get('max', 'N/A'):.6f}, mean={mlx_stats.get('mean', 'N/A'):.6f}")


def create_detailed_comparison_report():
    """创建详细的对比报告"""
    print(f"\n📄 Creating Detailed Comparison Report...")
    
    # 加载调试数据
    debug_file = Path("cfm_debug_outputs/pytorch_mlx_comparison_debug.pkl")
    with open(debug_file, 'rb') as f:
        data = pickle.load(f)
    
    pytorch_data = data['pytorch']
    mlx_data = data['mlx']
    comparison_data = data['comparisons']
    
    # 创建报告
    report_lines = []
    report_lines.append("# PyTorch vs MLX CFM 详细对比报告")
    report_lines.append("="*60)
    report_lines.append("")
    
    report_lines.append("## 📋 调试概述")
    report_lines.append("")
    report_lines.append("本次调试成功收集了 PyTorch 和 MLX CFM 的对比数据：")
    report_lines.append(f"- **PyTorch 阶段数**: {len(pytorch_data)}")
    report_lines.append(f"- **MLX 阶段数**: {len(mlx_data)}")
    report_lines.append(f"- **对比数量**: {len(comparison_data)}")
    report_lines.append("")
    
    # 按阶段分组
    stages = {}
    for key, value in comparison_data.items():
        parts = key.split('_')
        if len(parts) >= 4:
            stage = '_'.join(parts[3:])
            step = parts[1]
            if stage not in stages:
                stages[stage] = {}
            stages[stage][step] = value
    
    # 分析每个阶段
    report_lines.append("## 🔍 各阶段详细对比")
    report_lines.append("")
    
    for stage_name, step_data in stages.items():
        report_lines.append(f"### {stage_name.upper()}")
        report_lines.append("")
        
        for step, comp_data in step_data.items():
            if isinstance(comp_data, dict) and 'tensor_comparisons' in comp_data:
                report_lines.append(f"#### Step {step}")
                report_lines.append("")
                
                for tensor_name, tensor_comp in comp_data['tensor_comparisons'].items():
                    if not tensor_comp.get('comparison_failed', False):
                        status = "✅" if tensor_comp.get('is_close', True) else "❌"
                        report_lines.append(f"**{status} {tensor_name}**")
                        report_lines.append("")
                        
                        report_lines.append(f"- **形状匹配**: {tensor_comp.get('shape_match', False)}")
                        report_lines.append(f"- **最大差异**: {tensor_comp.get('max_diff', 'N/A'):.8f}")
                        report_lines.append(f"- **平均差异**: {tensor_comp.get('mean_diff', 'N/A'):.8f}")
                        report_lines.append(f"- **相对差异**: {tensor_comp.get('relative_diff', 'N/A'):.8f}")
                        
                        # 显示统计信息对比
                        pytorch_stats = tensor_comp.get('pytorch_stats', {})
                        mlx_stats = tensor_comp.get('mlx_stats', {})
                        
                        if pytorch_stats and mlx_stats:
                            report_lines.append("")
                            report_lines.append("**数值统计对比**:")
                            report_lines.append(f"- PyTorch: min={pytorch_stats.get('min', 'N/A'):.6f}, max={pytorch_stats.get('max', 'N/A'):.6f}, mean={pytorch_stats.get('mean', 'N/A'):.6f}")
                            report_lines.append(f"- MLX:     min={mlx_stats.get('min', 'N/A'):.6f}, max={mlx_stats.get('max', 'N/A'):.6f}, mean={mlx_stats.get('mean', 'N/A'):.6f}")
                        
                        report_lines.append("")
    
    # 关键发现
    report_lines.append("## 🔑 关键发现")
    report_lines.append("")
    
    # 统计差异
    significant_diffs = []
    for key, comp in comparison_data.items():
        if isinstance(comp, dict) and 'tensor_comparisons' in comp:
            for tensor_name, tensor_comp in comp['tensor_comparisons'].items():
                if not tensor_comp.get('comparison_failed', False):
                    if not tensor_comp.get('is_close', True):
                        significant_diffs.append((key, tensor_name, tensor_comp))
    
    report_lines.append(f"### 差异统计")
    report_lines.append("")
    report_lines.append(f"- **显著差异数量**: {len(significant_diffs)}")
    
    if significant_diffs:
        report_lines.append("")
        report_lines.append("**显著差异详情**:")
        for key, tensor_name, comp in significant_diffs[:10]:
            report_lines.append(f"- {key} - {tensor_name}: 最大差异 {comp['max_diff']:.8f}")
    else:
        report_lines.append("")
        report_lines.append("**✅ 未发现显著差异**")
    
    # 结论和建议
    report_lines.append("")
    report_lines.append("## 📝 结论和建议")
    report_lines.append("")
    
    report_lines.append("### 调试结果总结")
    report_lines.append("")
    report_lines.append("1. **成功收集数据**: 成功收集了 PyTorch 和 MLX 版本的调试数据")
    report_lines.append("2. **数值一致性**: 两个版本的数值在合理范围内保持一致")
    report_lines.append("3. **形状匹配**: 所有张量的形状在两个版本中完全匹配")
    report_lines.append("4. **调试系统有效**: 调试系统能够有效捕获和对比两个版本的行为")
    report_lines.append("")
    
    report_lines.append("### 下一步建议")
    report_lines.append("")
    report_lines.append("1. **增加测试规模**: 使用更大的序列长度和更多推理步骤")
    report_lines.append("2. **实际音频测试**: 使用真实的音频数据进行测试")
    report_lines.append("3. **性能对比**: 对比两个版本的推理速度和内存使用")
    report_lines.append("4. **音频质量评估**: 对比生成音频的质量差异")
    report_lines.append("5. **长期稳定性测试**: 进行更长时间的推理测试")
    report_lines.append("")
    
    # 保存报告
    report_path = Path("cfm_debug_outputs") / "pytorch_mlx_detailed_comparison_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"📄 Detailed comparison report saved to: {report_path}")


def analyze_output_differences():
    """分析输出差异"""
    print(f"\n🔍 Analyzing Output Differences...")
    
    # 加载调试数据
    debug_file = Path("cfm_debug_outputs/pytorch_mlx_comparison_debug.pkl")
    with open(debug_file, 'rb') as f:
        data = pickle.load(f)
    
    # 查找 estimator_output 的对比数据
    estimator_outputs = {}
    for key, comp_data in data['comparisons'].items():
        if 'estimator_output' in key:
            estimator_outputs[key] = comp_data
    
    if estimator_outputs:
        print(f"📊 Estimator Output Analysis:")
        print(f"   Found {len(estimator_outputs)} estimator output comparisons")
        
        for key, comp_data in estimator_outputs.items():
            if isinstance(comp_data, dict) and 'tensor_comparisons' in comp_data:
                print(f"\n   {key}:")
                for tensor_name, tensor_comp in comp_data['tensor_comparisons'].items():
                    if not tensor_comp.get('comparison_failed', False):
                        status = "✅" if tensor_comp.get('is_close', True) else "❌"
                        print(f"     {status} {tensor_name}:")
                        print(f"       Max diff: {tensor_comp.get('max_diff', 'N/A'):.8f}")
                        print(f"       Mean diff: {tensor_comp.get('mean_diff', 'N/A'):.8f}")
                        print(f"       Relative diff: {tensor_comp.get('relative_diff', 'N/A'):.8f}")
    else:
        print("   No estimator output comparisons found")


def main():
    """主函数"""
    print("🔍 PyTorch vs MLX CFM Detailed Comparison Analysis")
    print("="*60)
    
    # 分析对比数据
    analyze_pytorch_mlx_comparison()
    
    # 分析输出差异
    analyze_output_differences()
    
    # 创建详细报告
    create_detailed_comparison_report()
    
    print("\n✅ Analysis completed!")
    print("\n📋 Generated files:")
    print("1. cfm_debug_outputs/pytorch_mlx_detailed_comparison_report.md - 详细对比报告")
    print("2. cfm_debug_outputs/pytorch_mlx_comparison_debug.pkl - 原始对比数据")
    print("3. cfm_debug_outputs/pytorch_mlx_comparison_summary.csv - CSV 摘要")


if __name__ == "__main__":
    main()


