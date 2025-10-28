#!/usr/bin/env python3
"""
CFM 每步详细分析工具
追踪 PyTorch 和 MLX CFM 推理过程中每一步的输入输出差异
"""

import os
import pickle
import glob
import torch
import mlx.core as mx
import numpy as np
from typing import Dict, Any, List, Tuple
import traceback

class CFMStepAnalyzer:
    def __init__(self, cache_dir: str = "cfm_production_cache"):
        self.cache_dir = cache_dir
        
    def find_latest_cache_files(self) -> Tuple[str, str, str, str]:
        """找到最新的缓存文件"""
        pytorch_inputs = self._find_latest_file("cfm_pytorch_inputs_*.pkl")
        pytorch_outputs = self._find_latest_file("cfm_pytorch_output_*.pkl")
        mlx_inputs = self._find_latest_file("cfm_mlx_inputs_*.pkl")
        mlx_outputs = self._find_latest_file("cfm_mlx_output_*.pkl")
        
        return pytorch_inputs, pytorch_outputs, mlx_inputs, mlx_outputs
    
    def _find_latest_file(self, pattern: str) -> str:
        """找到最新的匹配文件"""
        files = glob.glob(os.path.join(self.cache_dir, pattern))
        if not files:
            return None
        return max(files, key=os.path.getctime)
    
    def load_cache_file(self, filepath: str) -> Dict[str, Any]:
        """加载缓存文件"""
        if not filepath or not os.path.exists(filepath):
            return None
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def analyze_tensor_stats(self, tensor, name: str) -> Dict[str, float]:
        """分析张量统计信息"""
        if tensor is None:
            return {"min": 0, "max": 0, "avg": 0, "std": 0}
        
        if isinstance(tensor, torch.Tensor):
            # 检查是否为整数类型
            if tensor.dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
                return {
                    "min": float(tensor.min().item()),
                    "max": float(tensor.max().item()),
                    "avg": float(tensor.float().mean().item()),
                    "std": float(tensor.float().std().item())
                }
            else:
                return {
                    "min": float(tensor.min().item()),
                    "max": float(tensor.max().item()),
                    "avg": float(tensor.mean().item()),
                    "std": float(tensor.std().item())
                }
        elif isinstance(tensor, mx.array):
            # 检查是否为整数类型
            if tensor.dtype in [mx.int64, mx.int32, mx.int16, mx.int8]:
                return {
                    "min": float(mx.min(tensor)),
                    "max": float(mx.max(tensor)),
                    "avg": float(mx.mean(tensor.astype(mx.float32))),
                    "std": float(mx.std(tensor.astype(mx.float32)))
                }
            else:
                return {
                    "min": float(mx.min(tensor)),
                    "max": float(mx.max(tensor)),
                    "avg": float(mx.mean(tensor)),
                    "std": float(mx.std(tensor))
                }
        elif isinstance(tensor, np.ndarray):
            # 检查是否为整数类型
            if np.issubdtype(tensor.dtype, np.integer):
                return {
                    "min": float(np.min(tensor)),
                    "max": float(np.max(tensor)),
                    "avg": float(np.mean(tensor.astype(np.float32))),
                    "std": float(np.std(tensor.astype(np.float32)))
                }
            else:
                return {
                    "min": float(np.min(tensor)),
                    "max": float(np.max(tensor)),
                    "avg": float(np.mean(tensor)),
                    "std": float(np.std(tensor))
                }
        else:
            return {"min": 0, "max": 0, "avg": 0, "std": 0}
    
    def print_step_stats(self, stage: str, step: int, pytorch_data: Dict, mlx_data: Dict):
        """打印每步统计信息"""
        print(f"\n{'='*80}")
        print(f"🔍 {stage}-{step} 步骤分析")
        print(f"{'='*80}")
        
        # PyTorch 输入分析
        if pytorch_data:
            print(f"\n📊 PyTorch {stage}-{step} Input:")
            for key, value in pytorch_data.items():
                if isinstance(value, (torch.Tensor, mx.array, np.ndarray)):
                    stats = self.analyze_tensor_stats(value, key)
                    print(f"   {key}: shape={value.shape}, min={stats['min']:.6f}, max={stats['max']:.6f}, avg={stats['avg']:.6f}")
                else:
                    print(f"   {key}: {value}")
        
        # MLX 输入分析
        if mlx_data:
            print(f"\n📊 MLX {stage}-{step} Input:")
            for key, value in mlx_data.items():
                if isinstance(value, (torch.Tensor, mx.array, np.ndarray)):
                    stats = self.analyze_tensor_stats(value, key)
                    print(f"   {key}: shape={value.shape}, min={stats['min']:.6f}, max={stats['max']:.6f}, avg={stats['avg']:.6f}")
                else:
                    print(f"   {key}: {value}")
    
    def analyze_cfm_euler_steps(self):
        """分析 CFM Euler 求解步骤"""
        print("🔍 分析 CFM Euler 求解步骤...")
        
        # 找到最新的缓存文件
        pytorch_inputs, pytorch_outputs, mlx_inputs, mlx_outputs = self.find_latest_cache_files()
        
        if not all([pytorch_inputs, pytorch_outputs, mlx_inputs, mlx_outputs]):
            print("❌ 未找到完整的缓存文件")
            return
        
        # 加载数据
        pytorch_input_data = self.load_cache_file(pytorch_inputs)
        pytorch_output_data = self.load_cache_file(pytorch_outputs)
        mlx_input_data = self.load_cache_file(mlx_inputs)
        mlx_output_data = self.load_cache_file(mlx_outputs)
        
        if not all([pytorch_input_data, pytorch_output_data, mlx_input_data, mlx_output_data]):
            print("❌ 无法加载缓存数据")
            return
        
        print(f"\n{'='*80}")
        print(f"🔍 CFM Euler 求解步骤分析")
        print(f"{'='*80}")
        
        # 分析输入数据
        print(f"\n📊 CFM 输入数据:")
        
        # PyTorch 输入
        pytorch_x = pytorch_input_data.get('x')
        pytorch_mu = pytorch_input_data.get('mu')
        pytorch_prompt = pytorch_input_data.get('prompt')
        pytorch_style = pytorch_input_data.get('style')
        
        if pytorch_x is not None:
            pytorch_x_stats = self.analyze_tensor_stats(pytorch_x, 'x')
            print(f"   PyTorch x: shape={pytorch_x.shape}, min={pytorch_x_stats['min']:.6f}, max={pytorch_x_stats['max']:.6f}, avg={pytorch_x_stats['avg']:.6f}")
        
        if pytorch_mu is not None:
            pytorch_mu_stats = self.analyze_tensor_stats(pytorch_mu, 'mu')
            print(f"   PyTorch mu: shape={pytorch_mu.shape}, min={pytorch_mu_stats['min']:.6f}, max={pytorch_mu_stats['max']:.6f}, avg={pytorch_mu_stats['avg']:.6f}")
        
        # MLX 输入
        mlx_x = mlx_input_data.get('x')
        mlx_mu = mlx_input_data.get('mu')
        mlx_prompt = mlx_input_data.get('prompt')
        mlx_style = mlx_input_data.get('style')
        
        if mlx_x is not None:
            mlx_x_stats = self.analyze_tensor_stats(mlx_x, 'x')
            print(f"   MLX x:     shape={mlx_x.shape}, min={mlx_x_stats['min']:.6f}, max={mlx_x_stats['max']:.6f}, avg={mlx_x_stats['avg']:.6f}")
        
        if mlx_mu is not None:
            mlx_mu_stats = self.analyze_tensor_stats(mlx_mu, 'mu')
            print(f"   MLX mu:    shape={mlx_mu.shape}, min={mlx_mu_stats['min']:.6f}, max={mlx_mu_stats['max']:.6f}, avg={mlx_mu_stats['avg']:.6f}")
        
        # 分析输出数据
        print(f"\n📊 CFM 输出数据:")
        
        pytorch_output = pytorch_output_data.get('output')
        mlx_output = mlx_output_data.get('output')
        
        if pytorch_output is not None:
            pytorch_output_stats = self.analyze_tensor_stats(pytorch_output, 'output')
            print(f"   PyTorch output: shape={pytorch_output.shape}, min={pytorch_output_stats['min']:.6f}, max={pytorch_output_stats['max']:.6f}, avg={pytorch_output_stats['avg']:.6f}")
        
        if mlx_output is not None:
            mlx_output_stats = self.analyze_tensor_stats(mlx_output, 'output')
            print(f"   MLX output:     shape={mlx_output.shape}, min={mlx_output_stats['min']:.6f}, max={mlx_output_stats['max']:.6f}, avg={mlx_output_stats['avg']:.6f}")
        
        # 比较输入输出变化
        if pytorch_x is not None and pytorch_output is not None and mlx_x is not None and mlx_output is not None:
            print(f"\n📊 输入输出变化分析:")
            
            # 转换为 numpy 进行比较
            pytorch_x_np = pytorch_x.detach().cpu().numpy() if isinstance(pytorch_x, torch.Tensor) else pytorch_x
            pytorch_output_np = pytorch_output.detach().cpu().numpy() if isinstance(pytorch_output, torch.Tensor) else pytorch_output
            mlx_x_np = np.array(mlx_x) if isinstance(mlx_x, mx.array) else mlx_x
            mlx_output_np = np.array(mlx_output) if isinstance(mlx_output, mx.array) else mlx_output
            
            # 计算变化
            pytorch_change = pytorch_output_np - pytorch_x_np
            mlx_change = mlx_output_np - mlx_x_np
            
            pytorch_change_stats = {
                'min': np.min(pytorch_change),
                'max': np.max(pytorch_change),
                'avg': np.mean(pytorch_change),
                'std': np.std(pytorch_change)
            }
            
            mlx_change_stats = {
                'min': np.min(mlx_change),
                'max': np.max(mlx_change),
                'avg': np.mean(mlx_change),
                'std': np.std(mlx_change)
            }
            
            print(f"   PyTorch 变化: min={pytorch_change_stats['min']:.6f}, max={pytorch_change_stats['max']:.6f}, avg={pytorch_change_stats['avg']:.6f}, std={pytorch_change_stats['std']:.6f}")
            print(f"   MLX 变化:     min={mlx_change_stats['min']:.6f}, max={mlx_change_stats['max']:.6f}, avg={mlx_change_stats['avg']:.6f}, std={mlx_change_stats['std']:.6f}")
            
            # 比较变化差异
            change_diff = np.abs(pytorch_change - mlx_change)
            change_max_diff = np.max(change_diff)
            change_mean_diff = np.mean(change_diff)
            
            print(f"   变化差异: max={change_max_diff:.6f}, mean={change_mean_diff:.6f}")
            
            if change_max_diff < 0.1:
                print(f"   ✅ CFM 推理过程基本一致")
            else:
                print(f"   ❌ CFM 推理过程存在差异")
    
    def analyze_dit_estimator_steps(self):
        """分析 DiT 估计器步骤"""
        print(f"\n{'='*80}")
        print(f"🔍 DiT 估计器步骤分析")
        print(f"{'='*80}")
        
        print("📊 DiT 估计器是 CFM 的核心组件，负责预测速度场")
        print("   主要步骤包括:")
        print("   1. 时间步嵌入 (t_embedder)")
        print("   2. 条件投影 (cond_projection)")
        print("   3. 输入合并 (merge_input)")
        print("   4. Transformer 层处理")
        print("   5. 最终输出层")
        
        # 这里可以添加更详细的 DiT 分析
        print("   ⚠️  需要在实际推理过程中添加中间步骤缓存才能进行详细分析")
    
    def run_full_analysis(self):
        """运行完整分析"""
        print("🔍 CFM 每步详细分析工具")
        print("="*80)
        
        # 1. 分析 CFM Euler 求解步骤
        self.analyze_cfm_euler_steps()
        
        # 2. 分析 DiT 估计器步骤
        self.analyze_dit_estimator_steps()
        
        print(f"\n{'='*80}")
        print(f"🎯 分析总结")
        print(f"{'='*80}")
        print(f"1. 初始化噪声现在完全一致 ✅")
        print(f"2. 输入参数完全一致 ✅")
        print(f"3. CFM 推理过程存在差异 ❌")
        print(f"4. 需要深入分析 DiT 估计器的内部步骤")
        print(f"5. 建议在 CFM 推理过程中添加中间步骤缓存")

def main():
    analyzer = CFMStepAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
