#!/usr/bin/env python3
"""
检查噪声初始化后条件编码的输入输出一致性
分析 CFM 推理过程中条件编码的处理
"""

import os
import pickle
import glob
import torch
import mlx.core as mx
import numpy as np
from typing import Dict, Any, List, Tuple

class ConditioningAnalyzer:
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
    
    def analyze_conditioning_consistency(self):
        """分析条件编码的一致性"""
        print("🔍 分析噪声初始化后条件编码的输入输出一致性")
        print("="*80)
        
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
        
        print(f"\n📊 条件编码输入分析:")
        
        # 分析条件编码输入
        conditioning_inputs = ['mu', 'prompt', 'style']
        
        for input_name in conditioning_inputs:
            pytorch_input = pytorch_input_data.get(input_name)
            mlx_input = mlx_input_data.get(input_name)
            
            if pytorch_input is not None and mlx_input is not None:
                pytorch_stats = self.analyze_tensor_stats(pytorch_input, input_name)
                mlx_stats = self.analyze_tensor_stats(mlx_input, input_name)
                
                print(f"\n   {input_name} 条件编码:")
                print(f"     PyTorch: shape={pytorch_input.shape}, min={pytorch_stats['min']:.6f}, max={pytorch_stats['max']:.6f}, avg={pytorch_stats['avg']:.6f}")
                print(f"     MLX:     shape={mlx_input.shape}, min={mlx_stats['min']:.6f}, max={mlx_stats['max']:.6f}, avg={mlx_stats['avg']:.6f}")
                
                # 比较差异
                if pytorch_input.shape == mlx_input.shape:
                    if isinstance(pytorch_input, torch.Tensor):
                        pytorch_np = pytorch_input.detach().cpu().numpy()
                    else:
                        pytorch_np = pytorch_input
                    
                    if isinstance(mlx_input, mx.array):
                        mlx_np = np.array(mlx_input)
                    else:
                        mlx_np = mlx_input
                    
                    diff = np.abs(pytorch_np - mlx_np)
                    max_diff = np.max(diff)
                    avg_diff = np.mean(diff)
                    
                    print(f"     差异: max={max_diff:.6f}, avg={avg_diff:.6f}")
                    
                    if max_diff < 1e-6:
                        print(f"     ✅ {input_name} 条件编码输入一致")
                    else:
                        print(f"     ❌ {input_name} 条件编码输入不一致")
                else:
                    print(f"     ❌ {input_name} 形状不匹配")
        
        print(f"\n📊 条件编码在 CFM 推理中的作用分析:")
        
        # 分析噪声初始化
        pytorch_x = pytorch_input_data.get('x')
        mlx_x = mlx_input_data.get('x')
        
        if pytorch_x is not None and mlx_x is not None:
            pytorch_x_stats = self.analyze_tensor_stats(pytorch_x, 'x')
            mlx_x_stats = self.analyze_tensor_stats(mlx_x, 'x')
            
            print(f"\n   噪声初始化 (x):")
            print(f"     PyTorch: shape={pytorch_x.shape}, min={pytorch_x_stats['min']:.6f}, max={pytorch_x_stats['max']:.6f}, avg={pytorch_x_stats['avg']:.6f}")
            print(f"     MLX:     shape={mlx_x.shape}, min={mlx_x_stats['min']:.6f}, max={mlx_x_stats['max']:.6f}, avg={mlx_x_stats['avg']:.6f}")
            
            # 比较噪声差异
            if pytorch_x.shape == mlx_x.shape:
                if isinstance(pytorch_x, torch.Tensor):
                    pytorch_x_np = pytorch_x.detach().cpu().numpy()
                else:
                    pytorch_x_np = pytorch_x
                
                if isinstance(mlx_x, mx.array):
                    mlx_x_np = np.array(mlx_x)
                else:
                    mlx_x_np = mlx_x
                
                noise_diff = np.abs(pytorch_x_np - mlx_x_np)
                noise_max_diff = np.max(noise_diff)
                noise_avg_diff = np.mean(noise_diff)
                
                print(f"     噪声差异: max={noise_max_diff:.6f}, avg={noise_avg_diff:.6f}")
                
                if noise_max_diff < 1e-6:
                    print(f"     ✅ 噪声初始化一致")
                else:
                    print(f"     ❌ 噪声初始化不一致")
        
        print(f"\n📊 CFM 推理过程分析:")
        print(f"   CFM 推理过程:")
        print(f"   1. 噪声初始化: x ~ N(0, 1)")
        print(f"   2. 条件编码处理: mu, prompt, style")
        print(f"   3. DiT 估计器: 预测速度场 dphi_dt")
        print(f"   4. Euler 求解: x = x + dt * dphi_dt")
        print(f"   5. 重复步骤 2-4 直到 t=1")
        
        print(f"\n📊 条件编码一致性总结:")
        
        # 检查所有条件编码是否一致
        all_consistent = True
        for input_name in conditioning_inputs:
            pytorch_input = pytorch_input_data.get(input_name)
            mlx_input = mlx_input_data.get(input_name)
            
            if pytorch_input is not None and mlx_input is not None:
                if pytorch_input.shape == mlx_input.shape:
                    if isinstance(pytorch_input, torch.Tensor):
                        pytorch_np = pytorch_input.detach().cpu().numpy()
                    else:
                        pytorch_np = pytorch_input
                    
                    if isinstance(mlx_input, mx.array):
                        mlx_np = np.array(mlx_input)
                    else:
                        mlx_np = mlx_input
                    
                    diff = np.abs(pytorch_np - mlx_np)
                    max_diff = np.max(diff)
                    
                    if max_diff > 1e-6:
                        all_consistent = False
                        break
        
        if all_consistent:
            print(f"   ✅ 所有条件编码输入一致")
            print(f"   ✅ 噪声初始化一致")
            print(f"   ❌ CFM 推理过程存在差异")
            print(f"   🔍 问题可能在于:")
            print(f"      - DiT 估计器的实现差异")
            print(f"      - Euler 求解的数值精度差异")
            print(f"      - 条件编码在 DiT 中的处理差异")
        else:
            print(f"   ❌ 条件编码输入不一致")
            print(f"   🔍 需要检查条件编码的生成过程")

def main():
    analyzer = ConditioningAnalyzer()
    analyzer.analyze_conditioning_consistency()

if __name__ == "__main__":
    main()
