#!/usr/bin/env python3
"""
CFM 缓存数据验证测试
使用缓存的 PyTorch 和 MLX 数据来验证 CFM 的一致性
"""

import os
import pickle
import numpy as np
import torch
import mlx.core as mx
from typing import Dict, Any

def load_test_data():
    """加载测试数据"""
    print("📦 加载 CFM 测试数据...")
    
    # 加载 PyTorch 数据
    with open('cfm_test_data/pytorch_test_data.pkl', 'rb') as f:
        pytorch_data = pickle.load(f)
    
    # 加载 MLX 数据
    with open('cfm_test_data/mlx_test_data.pkl', 'rb') as f:
        mlx_data = pickle.load(f)
    
    print(f"✅ 加载完成:")
    print(f"   PyTorch 数据对: {len(pytorch_data)}")
    print(f"   MLX 数据对: {len(mlx_data)}")
    
    return pytorch_data, mlx_data

def analyze_data_consistency(pytorch_data: Dict, mlx_data: Dict):
    """分析数据一致性"""
    print(f"\n🔍 分析数据一致性...")
    
    # 分析输入数据
    print(f"\n📊 输入数据分析:")
    for i, (pt_key, mlx_key) in enumerate(zip(pytorch_data.keys(), mlx_data.keys())):
        print(f"\n数据对 {i+1}:")
        pt_inputs = pytorch_data[pt_key]
        mlx_inputs = mlx_data[mlx_key]
        
        # 比较关键输入
        for key in ['mu', 'prompt', 'style']:
            if key in pt_inputs and key in mlx_inputs:
                pt_val = pt_inputs[key]
                mlx_val = mlx_inputs[key]
                
                if isinstance(pt_val, torch.Tensor) and isinstance(mlx_val, mx.array):
                    pt_np = pt_val.detach().cpu().numpy()
                    mlx_np = np.array(mlx_val)
                    
                    print(f"   {key}:")
                    print(f"      PyTorch: {pt_val.shape} ({pt_val.dtype})")
                    print(f"      MLX:     {mlx_val.shape} ({mlx_val.dtype})")
                    
                    if pt_np.shape == mlx_np.shape:
                        diff = np.abs(pt_np - mlx_np)
                        max_diff = np.max(diff)
                        mean_diff = np.mean(diff)
                        print(f"      ✅ 形状匹配")
                        print(f"      最大差异: {max_diff:.8f}")
                        print(f"      平均差异: {mean_diff:.8f}")
                        
                        if max_diff < 1e-8:
                            print(f"      ✅ 数据完全一致")
                        else:
                            print(f"      ⚠️  数据存在差异")
                    else:
                        print(f"      ❌ 形状不匹配")
                        print(f"      🔧 这可能是由于不同的序列长度导致的")
        
        # 比较序列长度
        if 'x_lens' in pt_inputs and 'x_lens' in mlx_inputs:
            pt_len = pt_inputs['x_lens'].item() if isinstance(pt_inputs['x_lens'], torch.Tensor) else pt_inputs['x_lens']
            mlx_len = mlx_inputs['x_lens'].item() if isinstance(mlx_inputs['x_lens'], mx.array) else mlx_inputs['x_lens']
            print(f"   x_lens: PyTorch={pt_len}, MLX={mlx_len}, 差异={abs(pt_len - mlx_len)}")

def create_consistency_report(pytorch_data: Dict, mlx_data: Dict):
    """创建一致性报告"""
    print(f"\n📝 创建一致性报告...")
    
    report = {
        'summary': {
            'pytorch_pairs': len(pytorch_data),
            'mlx_pairs': len(mlx_data),
            'total_pairs': min(len(pytorch_data), len(mlx_data))
        },
        'findings': [],
        'recommendations': []
    }
    
    # 分析每个数据对
    for i, (pt_key, mlx_key) in enumerate(zip(pytorch_data.keys(), mlx_data.keys())):
        pt_inputs = pytorch_data[pt_key]
        mlx_inputs = mlx_data[mlx_key]
        
        pair_analysis = {
            'pair_id': i + 1,
            'pytorch_key': pt_key,
            'mlx_key': mlx_key,
            'issues': [],
            'consistent_fields': []
        }
        
        # 检查关键字段
        for key in ['mu', 'prompt', 'style']:
            if key in pt_inputs and key in mlx_inputs:
                pt_val = pt_inputs[key]
                mlx_val = mlx_inputs[key]
                
                if isinstance(pt_val, torch.Tensor) and isinstance(mlx_val, mx.array):
                    pt_np = pt_val.detach().cpu().numpy()
                    mlx_np = np.array(mlx_val)
                    
                    if pt_np.shape == mlx_np.shape:
                        diff = np.abs(pt_np - mlx_np)
                        max_diff = np.max(diff)
                        
                        if max_diff < 1e-8:
                            pair_analysis['consistent_fields'].append(key)
                        else:
                            pair_analysis['issues'].append(f"{key}: 数据存在差异 (max_diff={max_diff:.8f})")
                    else:
                        pair_analysis['issues'].append(f"{key}: 形状不匹配 (PyTorch: {pt_np.shape}, MLX: {mlx_np.shape})")
        
        # 检查序列长度
        if 'x_lens' in pt_inputs and 'x_lens' in mlx_inputs:
            pt_len = pt_inputs['x_lens'].item() if isinstance(pt_inputs['x_lens'], torch.Tensor) else pt_inputs['x_lens']
            mlx_len = mlx_inputs['x_lens'].item() if isinstance(mlx_inputs['x_lens'], mx.array) else mlx_inputs['x_lens']
            
            if pt_len != mlx_len:
                pair_analysis['issues'].append(f"x_lens: 序列长度不匹配 (PyTorch: {pt_len}, MLX: {mlx_len})")
            else:
                pair_analysis['consistent_fields'].append('x_lens')
        
        report['findings'].append(pair_analysis)
    
    # 生成建议
    all_issues = []
    for finding in report['findings']:
        all_issues.extend(finding['issues'])
    
    if any('形状不匹配' in issue for issue in all_issues):
        report['recommendations'].append("序列长度不匹配：检查 GPT 生成的长度调节器实现")
    
    if any('数据存在差异' in issue for issue in all_issues):
        report['recommendations'].append("数据差异：检查随机种子设置和权重加载")
    
    if not all_issues:
        report['recommendations'].append("所有数据对都完全一致！")
    
    # 保存报告
    with open('cfm_test_data/consistency_report.pkl', 'wb') as f:
        pickle.dump(report, f)
    
    print(f"✅ 一致性报告已保存: cfm_test_data/consistency_report.pkl")
    
    # 显示报告摘要
    print(f"\n📊 一致性报告摘要:")
    print(f"   总数据对: {report['summary']['total_pairs']}")
    print(f"   问题数量: {len(all_issues)}")
    print(f"   建议数量: {len(report['recommendations'])}")
    
    if report['recommendations']:
        print(f"\n🔧 建议:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")
    
    return report

def main():
    """主函数"""
    print("🔧 CFM 缓存数据验证测试")
    print("=" * 50)
    
    # 检查测试数据是否存在
    if not os.path.exists('cfm_test_data/pytorch_test_data.pkl'):
        print("❌ 测试数据不存在，请先运行 analyze_cfm_cache.py")
        return
    
    # 加载测试数据
    pytorch_data, mlx_data = load_test_data()
    
    # 分析数据一致性
    analyze_data_consistency(pytorch_data, mlx_data)
    
    # 创建一致性报告
    report = create_consistency_report(pytorch_data, mlx_data)
    
    print(f"\n🎉 CFM 缓存数据验证完成!")
    print(f"📁 测试数据目录: cfm_test_data/")
    print(f"📊 一致性报告: cfm_test_data/consistency_report.pkl")

if __name__ == "__main__":
    main()
