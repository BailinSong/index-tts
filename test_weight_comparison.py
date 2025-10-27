#!/usr/bin/env python3
"""
测试代码：对比 MLX 加载后的权重与 torch 加载权重的区别

这个脚本会：
1. 加载 PyTorch 版本的 s2mel 权重
2. 加载 MLX 版本的 s2mel 权重（从缓存）
3. 对比两者的数值差异、形状差异等
4. 生成详细的差异报告
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, Tuple, List

# 添加项目路径
sys.path.append('/Users/bailin/index-tts')

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("⚠️ MLX not available, will only test PyTorch weights")

from indextts.utils.mlx_cache import MLXModelCache
from indextts.utils.checkpoint import load_checkpoint
from indextts.s2mel.modules.commons import MyModel
from omegaconf import OmegaConf


class WeightComparisonTester:
    """权重对比测试器"""
    
    def __init__(self, config_path="checkpoints/config.yaml", model_dir="checkpoints"):
        self.config_path = config_path
        self.model_dir = model_dir
        self.cfg = OmegaConf.load(config_path)
        
        # 初始化缓存管理器
        if MLX_AVAILABLE:
            self.mlx_cache = MLXModelCache(cache_dir="checkpoints/mlx")
        
        # 权重存储
        self.pytorch_weights = {}
        self.mlx_weights = {}
        
        print(f"🔧 Weight Comparison Tester initialized")
        print(f"   Config: {config_path}")
        print(f"   Model dir: {model_dir}")
        print(f"   MLX available: {MLX_AVAILABLE}")
    
    def load_pytorch_weights(self, checkpoint_path: str) -> Dict[str, Any]:
        """加载 PyTorch 权重"""
        print(f"\n📦 Loading PyTorch weights from: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # 加载 checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 提取权重
        if 'model' in checkpoint:
            weights = checkpoint['model']
            print(f"   Format: Standard (model key)")
        elif 'net' in checkpoint:
            # S2MEL format - flatten nested structure
            weights = {}
            for key in checkpoint['net']:
                for param_name, param_value in checkpoint['net'][key].items():
                    weights[f"{key}.{param_name}"] = param_value
            print(f"   Format: S2MEL (net key)")
        else:
            weights = checkpoint
            print(f"   Format: Direct state dict")
        
        # 转换为 numpy 数组便于比较
        numpy_weights = {}
        for key, value in weights.items():
            if isinstance(value, torch.Tensor):
                numpy_weights[key] = value.cpu().numpy()
            else:
                numpy_weights[key] = value
        
        print(f"   Loaded {len(numpy_weights)} weight tensors")
        self.pytorch_weights = numpy_weights
        return numpy_weights
    
    def load_mlx_weights(self, model_name: str) -> Dict[str, Any]:
        """加载 MLX 权重"""
        if not MLX_AVAILABLE:
            print("⚠️ MLX not available, skipping MLX weight loading")
            return {}
        
        print(f"\n📦 Loading MLX weights for: {model_name}")
        
        # 从缓存加载
        mlx_state = self.mlx_cache.load_from_cache(model_name)
        if mlx_state is None:
            print(f"❌ No MLX cache found for {model_name}")
            return {}
        
        # 转换为 numpy 数组便于比较
        numpy_weights = {}
        for key, value in mlx_state.items():
            if isinstance(value, mx.array):
                numpy_weights[key] = np.array(value)
            else:
                numpy_weights[key] = value
        
        print(f"   Loaded {len(numpy_weights)} weight tensors")
        self.mlx_weights = numpy_weights
        return numpy_weights
    
    def compare_weights(self) -> Dict[str, Any]:
        """对比权重差异"""
        print(f"\n🔍 Comparing weights...")
        
        if not self.pytorch_weights:
            print("❌ No PyTorch weights loaded")
            return {}
        
        if not self.mlx_weights:
            print("❌ No MLX weights loaded")
            return {}
        
        comparison_results = {
            'total_pytorch_weights': len(self.pytorch_weights),
            'total_mlx_weights': len(self.mlx_weights),
            'common_keys': [],
            'pytorch_only_keys': [],
            'mlx_only_keys': [],
            'weight_differences': {},
            'shape_differences': {},
            'summary': {}
        }
        
        # 找出共同的键
        pytorch_keys = set(self.pytorch_weights.keys())
        mlx_keys = set(self.mlx_weights.keys())
        
        common_keys = pytorch_keys & mlx_keys
        pytorch_only = pytorch_keys - mlx_keys
        mlx_only = mlx_keys - pytorch_keys
        
        comparison_results['common_keys'] = sorted(list(common_keys))
        comparison_results['pytorch_only_keys'] = sorted(list(pytorch_only))
        comparison_results['mlx_only_keys'] = sorted(list(mlx_only))
        
        print(f"   Common keys: {len(common_keys)}")
        print(f"   PyTorch only: {len(pytorch_only)}")
        print(f"   MLX only: {len(mlx_only)}")
        
        # 对比共同键的权重
        max_diff = 0.0
        total_diff = 0.0
        diff_count = 0
        
        for key in common_keys:
            pytorch_weight = self.pytorch_weights[key]
            mlx_weight = self.mlx_weights[key]
            
            # 形状对比
            if pytorch_weight.shape != mlx_weight.shape:
                comparison_results['shape_differences'][key] = {
                    'pytorch_shape': pytorch_weight.shape,
                    'mlx_shape': mlx_weight.shape
                }
                print(f"   ⚠️ Shape difference for {key}: {pytorch_weight.shape} vs {mlx_weight.shape}")
                continue
            
            # 数值对比
            if isinstance(pytorch_weight, np.ndarray) and isinstance(mlx_weight, np.ndarray):
                diff = np.abs(pytorch_weight - mlx_weight)
                max_diff_key = np.max(diff)
                mean_diff_key = np.mean(diff)
                
                comparison_results['weight_differences'][key] = {
                    'max_difference': float(max_diff_key),
                    'mean_difference': float(mean_diff_key),
                    'pytorch_range': [float(np.min(pytorch_weight)), float(np.max(pytorch_weight))],
                    'mlx_range': [float(np.min(mlx_weight)), float(np.max(mlx_weight))],
                    'pytorch_mean': float(np.mean(pytorch_weight)),
                    'mlx_mean': float(np.mean(mlx_weight)),
                    'pytorch_std': float(np.std(pytorch_weight)),
                    'mlx_std': float(np.std(mlx_weight))
                }
                
                max_diff = max(max_diff, max_diff_key)
                total_diff += mean_diff_key
                diff_count += 1
                
                if max_diff_key > 1e-6:
                    print(f"   ⚠️ Large difference for {key}: max={max_diff_key:.8f}, mean={mean_diff_key:.8f}")
        
        # 生成摘要
        comparison_results['summary'] = {
            'max_difference_overall': float(max_diff),
            'mean_difference_overall': float(total_diff / diff_count) if diff_count > 0 else 0.0,
            'keys_with_differences': len([k for k, v in comparison_results['weight_differences'].items() 
                                        if v['max_difference'] > 1e-6]),
            'keys_with_shape_differences': len(comparison_results['shape_differences'])
        }
        
        print(f"\n📊 Comparison Summary:")
        print(f"   Max difference: {max_diff:.8f}")
        print(f"   Mean difference: {total_diff/diff_count:.8f}" if diff_count > 0 else "   Mean difference: N/A")
        print(f"   Keys with differences > 1e-6: {comparison_results['summary']['keys_with_differences']}")
        print(f"   Keys with shape differences: {comparison_results['summary']['keys_with_shape_differences']}")
        
        return comparison_results
    
    def analyze_specific_components(self) -> Dict[str, Any]:
        """分析特定组件的权重差异"""
        print(f"\n🔬 Analyzing specific components...")
        
        component_analysis = {}
        
        # 分析 CFM 相关权重
        cfm_keys = [k for k in self.pytorch_weights.keys() if 'cfm' in k.lower()]
        mlx_cfm_keys = [k for k in self.mlx_weights.keys() if 'cfm' in k.lower()]
        
        component_analysis['cfm'] = {
            'pytorch_count': len(cfm_keys),
            'mlx_count': len(mlx_cfm_keys),
            'pytorch_keys': cfm_keys[:10],  # 只显示前10个
            'mlx_keys': mlx_cfm_keys[:10]
        }
        
        # 分析 DiT 相关权重
        dit_keys = [k for k in self.pytorch_weights.keys() if any(x in k.lower() for x in ['dit', 'transformer', 'embedder'])]
        mlx_dit_keys = [k for k in self.mlx_weights.keys() if any(x in k.lower() for x in ['dit', 'transformer', 'embedder'])]
        
        component_analysis['dit'] = {
            'pytorch_count': len(dit_keys),
            'mlx_count': len(mlx_dit_keys),
            'pytorch_keys': dit_keys[:10],
            'mlx_keys': mlx_dit_keys[:10]
        }
        
        # 分析长度调节器权重
        lr_keys = [k for k in self.pytorch_weights.keys() if 'length_regulator' in k.lower()]
        mlx_lr_keys = [k for k in self.mlx_weights.keys() if 'length_regulator' in k.lower()]
        
        component_analysis['length_regulator'] = {
            'pytorch_count': len(lr_keys),
            'mlx_count': len(mlx_lr_keys),
            'pytorch_keys': lr_keys[:10],
            'mlx_keys': mlx_lr_keys[:10]
        }
        
        print(f"   CFM weights: PyTorch={len(cfm_keys)}, MLX={len(mlx_cfm_keys)}")
        print(f"   DiT weights: PyTorch={len(dit_keys)}, MLX={len(mlx_dit_keys)}")
        print(f"   Length regulator weights: PyTorch={len(lr_keys)}, MLX={len(mlx_lr_keys)}")
        
        return component_analysis
    
    def generate_report(self, comparison_results: Dict[str, Any], component_analysis: Dict[str, Any]) -> str:
        """生成详细的差异报告"""
        report = []
        report.append("# S2MEL 权重对比报告")
        report.append("=" * 50)
        report.append("")
        
        # 基本信息
        report.append("## 基本信息")
        report.append(f"- PyTorch 权重数量: {comparison_results['total_pytorch_weights']}")
        report.append(f"- MLX 权重数量: {comparison_results['total_mlx_weights']}")
        report.append(f"- 共同键数量: {len(comparison_results['common_keys'])}")
        report.append(f"- PyTorch 独有键数量: {len(comparison_results['pytorch_only_keys'])}")
        report.append(f"- MLX 独有键数量: {len(comparison_results['mlx_only_keys'])}")
        report.append("")
        
        # 数值差异摘要
        summary = comparison_results['summary']
        report.append("## 数值差异摘要")
        report.append(f"- 最大差异: {summary['max_difference_overall']:.8f}")
        report.append(f"- 平均差异: {summary['mean_difference_overall']:.8f}")
        report.append(f"- 有显著差异的键数量 (>1e-6): {summary['keys_with_differences']}")
        report.append(f"- 形状不同的键数量: {summary['keys_with_shape_differences']}")
        report.append("")
        
        # 组件分析
        report.append("## 组件分析")
        for component, data in component_analysis.items():
            report.append(f"### {component}")
            report.append(f"- PyTorch 权重数量: {data['pytorch_count']}")
            report.append(f"- MLX 权重数量: {data['mlx_count']}")
            report.append("")
        
        # 详细差异
        if comparison_results['weight_differences']:
            report.append("## 详细权重差异")
            report.append("| 键名 | 最大差异 | 平均差异 | PyTorch范围 | MLX范围 |")
            report.append("|------|----------|----------|-------------|---------|")
            
            for key, diff in sorted(comparison_results['weight_differences'].items()):
                if diff['max_difference'] > 1e-6:  # 只显示有显著差异的
                    pytorch_range = f"[{diff['pytorch_range'][0]:.3f}, {diff['pytorch_range'][1]:.3f}]"
                    mlx_range = f"[{diff['mlx_range'][0]:.3f}, {diff['mlx_range'][1]:.3f}]"
                    report.append(f"| {key} | {diff['max_difference']:.8f} | {diff['mean_difference']:.8f} | {pytorch_range} | {mlx_range} |")
        
        return "\n".join(report)
    
    def run_comparison(self, pytorch_checkpoint: str, mlx_model_name: str = "s2mel_cfm"):
        """运行完整的权重对比测试"""
        print(f"🚀 Starting weight comparison test")
        print(f"   PyTorch checkpoint: {pytorch_checkpoint}")
        print(f"   MLX model name: {mlx_model_name}")
        
        try:
            # 1. 加载 PyTorch 权重
            self.load_pytorch_weights(pytorch_checkpoint)
            
            # 2. 加载 MLX 权重
            self.load_mlx_weights(mlx_model_name)
            
            # 3. 对比权重
            comparison_results = self.compare_weights()
            
            # 4. 分析特定组件
            component_analysis = self.analyze_specific_components()
            
            # 5. 生成报告
            report = self.generate_report(comparison_results, component_analysis)
            
            # 6. 保存报告
            report_path = "weight_comparison_report.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"\n✅ Weight comparison completed!")
            print(f"   Report saved to: {report_path}")
            
            # 7. 保存详细结果到 JSON
            detailed_results = {
                'comparison_results': comparison_results,
                'component_analysis': component_analysis,
                'test_info': {
                    'pytorch_checkpoint': pytorch_checkpoint,
                    'mlx_model_name': mlx_model_name,
                    'mlx_available': MLX_AVAILABLE
                }
            }
            
            json_path = "weight_comparison_results.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
            
            print(f"   Detailed results saved to: {json_path}")
            
            return comparison_results, component_analysis
            
        except Exception as e:
            print(f"❌ Weight comparison failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None


def main():
    """主函数"""
    print("🧪 S2MEL Weight Comparison Test")
    print("=" * 50)
    
    # 检查必要的文件
    config_path = "checkpoints/config.yaml"
    pytorch_checkpoint = "checkpoints/s2mel.pth"  # 根据实际情况调整
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return
    
    if not os.path.exists(pytorch_checkpoint):
        print(f"❌ PyTorch checkpoint not found: {pytorch_checkpoint}")
        print("   Please check the checkpoint path")
        return
    
    # 创建测试器
    tester = WeightComparisonTester(config_path)
    
    # 运行对比测试
    comparison_results, component_analysis = tester.run_comparison(pytorch_checkpoint)
    
    if comparison_results:
        print(f"\n🎉 Test completed successfully!")
        
        # 显示关键结果
        summary = comparison_results['summary']
        print(f"\n📊 Key Results:")
        print(f"   Max weight difference: {summary['max_difference_overall']:.8f}")
        print(f"   Keys with significant differences: {summary['keys_with_differences']}")
        
        if summary['keys_with_differences'] == 0:
            print(f"   ✅ All weights match within tolerance!")
        else:
            print(f"   ⚠️ Found {summary['keys_with_differences']} weights with significant differences")
    else:
        print(f"❌ Test failed")


if __name__ == "__main__":
    main()
