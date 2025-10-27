#!/usr/bin/env python3
"""
CFM 调试器工具类

用于在生产代码中记录和对比 CFM 内部各个阶段的输入输出
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
import mlx.core as mx


class CFMDebugger:
    """CFM 调试器 - 对比 PyTorch 和 MLX 版本的内部状态"""
    
    def __init__(self, debug_dir: str = "cfm_debug_outputs"):
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(exist_ok=True)
        
        # 存储调试数据
        self.debug_data = {
            'pytorch': {},
            'mlx': {},
            'comparisons': {}
        }
        
        # 当前步骤信息
        self.current_step = 0
        self.current_layer = 0
        self.current_stage = ""
        
    def log_stage(self, 
                  stage_name: str, 
                  pytorch_data: Optional[Dict[str, Any]] = None, 
                  mlx_data: Optional[Dict[str, Any]] = None, 
                  step: Optional[int] = None, 
                  layer: Optional[int] = None,
                  additional_info: Optional[Dict[str, Any]] = None):
        """记录某个阶段的调试数据"""
        
        if step is not None:
            self.current_step = step
        if layer is not None:
            self.current_layer = layer
            
        self.current_stage = stage_name
        key = f"step_{self.current_step}_layer_{self.current_layer}_{stage_name}"
        
        if pytorch_data is not None:
            self.debug_data['pytorch'][key] = self._extract_tensor_info(pytorch_data)
            
        if mlx_data is not None:
            self.debug_data['mlx'][key] = self._extract_tensor_info(mlx_data)
            
        # 如果两个数据都有，进行对比
        if pytorch_data is not None and mlx_data is not None:
            self.debug_data['comparisons'][key] = self._compare_tensors(
                pytorch_data, mlx_data, stage_name
            )
            
        # 添加额外信息
        if additional_info:
            if key not in self.debug_data['comparisons']:
                self.debug_data['comparisons'][key] = {}
            self.debug_data['comparisons'][key]['additional_info'] = additional_info
    
    def _extract_tensor_info(self, data: Any) -> Dict[str, Any]:
        """提取张量信息"""
        if isinstance(data, (torch.Tensor, mx.array)):
            # 处理整数类型张量
            if isinstance(data, torch.Tensor) and data.dtype in [torch.int64, torch.int32, torch.long]:
                # 对于整数张量，只提取基本信息
                return {
                    'shape': list(data.shape),
                    'dtype': str(data.dtype),
                    'min': int(data.min()),
                    'max': int(data.max()),
                    'mean': 'N/A (integer tensor)',
                    'std': 'N/A (integer tensor)',
                    'data_sample': self._get_sample_data(data)
                }
            else:
                return {
                    'shape': list(data.shape),
                    'dtype': str(data.dtype),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'data_sample': self._get_sample_data(data)
                }
        elif isinstance(data, (list, tuple)):
            return [self._extract_tensor_info(item) for item in data]
        elif isinstance(data, dict):
            return {k: self._extract_tensor_info(v) for k, v in data.items()}
        else:
            return str(data)
    
    def _get_sample_data(self, tensor: Union[torch.Tensor, mx.array], max_elements: int = 10) -> list:
        """获取张量样本数据"""
        try:
            if isinstance(tensor, torch.Tensor):
                flat = tensor.flatten()
            else:
                flat = mx.reshape(tensor, (-1,))
            
            # 取前几个元素
            if len(flat) > max_elements:
                if isinstance(tensor, torch.Tensor):
                    return flat[:max_elements].cpu().numpy().tolist()
                else:
                    return mx.array(flat[:max_elements]).tolist()
            else:
                if isinstance(tensor, torch.Tensor):
                    return flat.cpu().numpy().tolist()
                else:
                    return flat.tolist()
        except Exception as e:
            return f"Error extracting sample data: {str(e)}"
    
    def _compare_tensors(self, pytorch_data: Dict[str, Any], mlx_data: Dict[str, Any], stage_name: str) -> Dict[str, Any]:
        """对比两个张量字典"""
        comparison = {
            'stage': stage_name,
            'step': self.current_step,
            'layer': self.current_layer,
            'tensor_comparisons': {}
        }
        
        # 对比每个张量
        for key in pytorch_data.keys():
            if key in mlx_data:
                try:
                    pytorch_tensor = pytorch_data[key]
                    mlx_tensor = mlx_data[key]
                    
                    comp_result = self._compare_single_tensor(pytorch_tensor, mlx_tensor, key)
                    comparison['tensor_comparisons'][key] = comp_result
                    
                except Exception as e:
                    comparison['tensor_comparisons'][key] = {
                        'error': str(e),
                        'comparison_failed': True
                    }
        
        return comparison
    
    def _compare_single_tensor(self, pytorch_tensor: Any, mlx_tensor: Any, tensor_name: str) -> Dict[str, Any]:
        """对比单个张量"""
        try:
            # 转换为 numpy 进行对比
            if isinstance(pytorch_tensor, torch.Tensor):
                pytorch_np = pytorch_tensor.detach().cpu().numpy()
            else:
                pytorch_np = pytorch_tensor
                
            if isinstance(mlx_tensor, mx.array):
                mlx_np = np.array(mlx_tensor)
            else:
                mlx_np = mlx_tensor
            
            # 形状对比
            shape_match = pytorch_np.shape == mlx_np.shape
            
            # 数值对比
            if shape_match:
                diff = np.abs(pytorch_np - mlx_np)
                max_diff = float(np.max(diff))
                mean_diff = float(np.mean(diff))
                rel_diff = max_diff / (np.max(np.abs(pytorch_np)) + 1e-8)
            else:
                max_diff = float('inf')
                mean_diff = float('inf')
                rel_diff = float('inf')
            
            return {
                'tensor_name': tensor_name,
                'shape_match': shape_match,
                'pytorch_shape': pytorch_np.shape,
                'mlx_shape': mlx_np.shape,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'relative_diff': rel_diff,
                'is_close': max_diff < 1e-5,
                'pytorch_stats': {
                    'min': float(np.min(pytorch_np)),
                    'max': float(np.max(pytorch_np)),
                    'mean': float(np.mean(pytorch_np)),
                    'std': float(np.std(pytorch_np))
                },
                'mlx_stats': {
                    'min': float(np.min(mlx_np)),
                    'max': float(np.max(mlx_np)),
                    'mean': float(np.mean(mlx_np)),
                    'std': float(np.std(mlx_np))
                }
            }
        except Exception as e:
            return {
                'tensor_name': tensor_name,
                'error': str(e),
                'comparison_failed': True
            }
    
    def save_debug_data(self, filename: str = "cfm_debug_analysis.pkl") -> Path:
        """保存调试数据"""
        filepath = self.debug_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(self.debug_data, f)
        print(f"💾 Debug data saved to: {filepath}")
        return filepath
    
    def load_debug_data(self, filename: str = "cfm_debug_analysis.pkl") -> Dict[str, Any]:
        """加载调试数据"""
        filepath = self.debug_dir / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                self.debug_data = pickle.load(f)
            print(f"📂 Debug data loaded from: {filepath}")
            return self.debug_data
        else:
            print(f"❌ Debug data file not found: {filepath}")
            return {}
    
    def print_summary(self):
        """打印调试摘要"""
        print("\n" + "="*80)
        print("🔍 CFM DEBUG SUMMARY")
        print("="*80)
        
        pytorch_keys = set(self.debug_data['pytorch'].keys())
        mlx_keys = set(self.debug_data['mlx'].keys())
        comparison_keys = set(self.debug_data['comparisons'].keys())
        
        print(f"📊 PyTorch stages logged: {len(pytorch_keys)}")
        print(f"📊 MLX stages logged: {len(mlx_keys)}")
        print(f"📊 Comparisons made: {len(comparison_keys)}")
        
        # 分析差异
        significant_diffs = []
        for key, comp in self.debug_data['comparisons'].items():
            if isinstance(comp, dict) and 'tensor_comparisons' in comp:
                for tensor_name, tensor_comp in comp['tensor_comparisons'].items():
                    if not tensor_comp.get('comparison_failed', False):
                        if not tensor_comp.get('is_close', True):
                            significant_diffs.append((key, tensor_name, tensor_comp))
        
        print(f"\n⚠️  Significant differences found: {len(significant_diffs)}")
        
        for key, tensor_name, comp in significant_diffs[:10]:  # 只显示前10个
            print(f"   {key} - {tensor_name}:")
            print(f"     Max diff: {comp['max_diff']:.6f}")
            print(f"     Mean diff: {comp['mean_diff']:.6f}")
            print(f"     Relative diff: {comp['relative_diff']:.6f}")
            print(f"     Shape match: {comp['shape_match']}")
        
        if len(significant_diffs) > 10:
            print(f"   ... and {len(significant_diffs) - 10} more")
    
    def get_stage_summary(self, stage_name: str) -> Dict[str, Any]:
        """获取特定阶段的摘要"""
        stage_data = {}
        
        for key, data in self.debug_data['comparisons'].items():
            if stage_name in key:
                stage_data[key] = data
        
        return stage_data
    
    def export_to_csv(self, filename: str = "cfm_debug_summary.csv"):
        """导出调试摘要到 CSV"""
        import csv
        
        filepath = self.debug_dir / filename
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Stage', 'Tensor', 'Shape_Match', 'Max_Diff', 'Mean_Diff', 
                'Relative_Diff', 'Is_Close', 'PyTorch_Min', 'PyTorch_Max', 
                'PyTorch_Mean', 'MLX_Min', 'MLX_Max', 'MLX_Mean'
            ])
            
            for key, comp in self.debug_data['comparisons'].items():
                if isinstance(comp, dict) and 'tensor_comparisons' in comp:
                    for tensor_name, tensor_comp in comp['tensor_comparisons'].items():
                        if not tensor_comp.get('comparison_failed', False):
                            writer.writerow([
                                key,
                                tensor_name,
                                tensor_comp.get('shape_match', False),
                                tensor_comp.get('max_diff', float('inf')),
                                tensor_comp.get('mean_diff', float('inf')),
                                tensor_comp.get('relative_diff', float('inf')),
                                tensor_comp.get('is_close', False),
                                tensor_comp.get('pytorch_stats', {}).get('min', 0),
                                tensor_comp.get('pytorch_stats', {}).get('max', 0),
                                tensor_comp.get('pytorch_stats', {}).get('mean', 0),
                                tensor_comp.get('mlx_stats', {}).get('min', 0),
                                tensor_comp.get('mlx_stats', {}).get('max', 0),
                                tensor_comp.get('mlx_stats', {}).get('mean', 0)
                            ])
        
        print(f"📊 CSV summary exported to: {filepath}")
        return filepath


# 全局调试器实例
_global_debugger = None

def get_debugger() -> CFMDebugger:
    """获取全局调试器实例"""
    global _global_debugger
    if _global_debugger is None:
        _global_debugger = CFMDebugger()
    return _global_debugger

def log_cfm_stage(stage_name: str, 
                  pytorch_data: Optional[Dict[str, Any]] = None, 
                  mlx_data: Optional[Dict[str, Any]] = None, 
                  step: Optional[int] = None, 
                  layer: Optional[int] = None,
                  additional_info: Optional[Dict[str, Any]] = None):
    """便捷函数：记录 CFM 阶段"""
    debugger = get_debugger()
    debugger.log_stage(stage_name, pytorch_data, mlx_data, step, layer, additional_info)

def save_cfm_debug(filename: str = "cfm_debug_analysis.pkl") -> Path:
    """便捷函数：保存调试数据"""
    debugger = get_debugger()
    return debugger.save_debug_data(filename)

def print_cfm_summary():
    """便捷函数：打印调试摘要"""
    debugger = get_debugger()
    debugger.print_summary()
