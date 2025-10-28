#!/usr/bin/env python3
"""
分析 DiT 估计器内部差异
比较 PyTorch 和 MLX DiT 的详细计算过程
"""

import os
import pickle
import glob
import torch
import mlx.core as mx
import numpy as np
from typing import Dict, Any, List, Tuple

class DiTDifferenceAnalyzer:
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
    
    def analyze_dit_step_differences(self):
        """分析 DiT 每步差异"""
        print("🔍 分析 DiT 估计器每步差异")
        print("="*80)
        
        # 找到最新的缓存文件
        pytorch_inputs, pytorch_outputs, mlx_inputs, mlx_outputs = self.find_latest_cache_files()
        
        if not all([pytorch_inputs, pytorch_outputs, mlx_inputs, mlx_outputs]):
            print("❌ 未找到完整的缓存文件")
            return
        
        # 分析前几步的 DiT 差异
        steps_to_analyze = [1, 2, 3, 5, 10, 15, 20, 25]
        
        for step in steps_to_analyze:
            print(f"\n📊 Step {step:03d} DiT 差异分析:")
            print("-" * 60)
            
            # 查找对应的 DiT 输入输出文件
            pytorch_dit_input = self._find_latest_file(f"cfm_pytorch_step_{step:03d}_dit_input_*.pkl")
            pytorch_dit_output = self._find_latest_file(f"cfm_pytorch_step_{step:03d}_dit_output_*.pkl")
            mlx_dit_input = self._find_latest_file(f"cfm_mlx_step_{step:03d}_dit_input_*.pkl")
            mlx_dit_output = self._find_latest_file(f"cfm_mlx_step_{step:03d}_dit_output_*.pkl")
            
            if not all([pytorch_dit_input, pytorch_dit_output, mlx_dit_input, mlx_dit_output]):
                print(f"   ❌ Step {step} 缺少 DiT 缓存文件")
                continue
            
            # 加载 DiT 数据
            pytorch_dit_input_data = self.load_cache_file(pytorch_dit_input)
            pytorch_dit_output_data = self.load_cache_file(pytorch_dit_output)
            mlx_dit_input_data = self.load_cache_file(mlx_dit_input)
            mlx_dit_output_data = self.load_cache_file(mlx_dit_output)
            
            if not all([pytorch_dit_input_data, pytorch_dit_output_data, mlx_dit_input_data, mlx_dit_output_data]):
                print(f"   ❌ Step {step} 无法加载 DiT 数据")
                continue
            
            # 分析 DiT 输入
            print(f"   📥 DiT 输入:")
            pytorch_x = pytorch_dit_input_data.get('x')
            mlx_x = mlx_dit_input_data.get('x')
            
            if pytorch_x is not None and mlx_x is not None:
                pytorch_x_stats = self.analyze_tensor_stats(pytorch_x, 'x')
                mlx_x_stats = self.analyze_tensor_stats(mlx_x, 'x')
                
                print(f"     x: PyTorch min={pytorch_x_stats['min']:.6f}, max={pytorch_x_stats['max']:.6f}, avg={pytorch_x_stats['avg']:.6f}")
                print(f"     x: MLX     min={mlx_x_stats['min']:.6f}, max={mlx_x_stats['max']:.6f}, avg={mlx_x_stats['avg']:.6f}")
                
                # 检查输入是否一致
                if (abs(pytorch_x_stats['min'] - mlx_x_stats['min']) < 1e-6 and 
                    abs(pytorch_x_stats['max'] - mlx_x_stats['max']) < 1e-6 and
                    abs(pytorch_x_stats['avg'] - mlx_x_stats['avg']) < 1e-6):
                    print(f"     ✅ DiT 输入一致")
                else:
                    print(f"     ❌ DiT 输入不一致")
            
            # 分析 DiT 输出
            print(f"   📤 DiT 输出:")
            pytorch_dphi_dt = pytorch_dit_output_data.get('dphi_dt')
            mlx_dphi_dt = mlx_dit_output_data.get('dphi_dt')
            
            if pytorch_dphi_dt is not None and mlx_dphi_dt is not None:
                pytorch_dphi_stats = self.analyze_tensor_stats(pytorch_dphi_dt, 'dphi_dt')
                mlx_dphi_stats = self.analyze_tensor_stats(mlx_dphi_dt, 'dphi_dt')
                
                print(f"     dphi_dt: PyTorch min={pytorch_dphi_stats['min']:.6f}, max={pytorch_dphi_stats['max']:.6f}, avg={pytorch_dphi_stats['avg']:.6f}")
                print(f"     dphi_dt: MLX     min={mlx_dphi_stats['min']:.6f}, max={mlx_dphi_stats['max']:.6f}, avg={mlx_dphi_stats['avg']:.6f}")
                
                # 计算输出差异
                max_diff = abs(pytorch_dphi_stats['max'] - mlx_dphi_stats['max'])
                min_diff = abs(pytorch_dphi_stats['min'] - mlx_dphi_stats['min'])
                avg_diff = abs(pytorch_dphi_stats['avg'] - mlx_dphi_stats['avg'])
                
                print(f"     📊 差异: max={max_diff:.6f}, min={min_diff:.6f}, avg={avg_diff:.6f}")
                
                if max_diff < 0.1:
                    print(f"     ✅ DiT 输出基本一致")
                else:
                    print(f"     ❌ DiT 输出存在显著差异")
                    
                    # 分析差异趋势
                    if pytorch_dphi_stats['avg'] > mlx_dphi_stats['avg']:
                        print(f"     📈 PyTorch 输出偏大")
                    else:
                        print(f"     📉 PyTorch 输出偏小")
    
    def analyze_dit_weight_differences(self):
        """分析 DiT 权重差异"""
        print(f"\n🔍 分析 DiT 权重差异")
        print("="*80)
        
        # 这里需要加载实际的 DiT 模型权重进行比较
        # 由于我们没有直接的权重访问，先提供分析框架
        print("📊 DiT 权重差异分析框架:")
        print("   1. cond_projection 权重")
        print("   2. t_embedder 权重")
        print("   3. cond_embedder 权重")
        print("   4. transformer layers 权重")
        print("   5. final layer 权重")
        print("\n💡 建议:")
        print("   - 检查权重加载是否完全一致")
        print("   - 验证权重转换精度")
        print("   - 比较关键层的权重范围")

def main():
    analyzer = DiTDifferenceAnalyzer()
    analyzer.analyze_dit_step_differences()
    analyzer.analyze_dit_weight_differences()

if __name__ == "__main__":
    main()
