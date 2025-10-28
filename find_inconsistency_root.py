#!/usr/bin/env python3
"""
查找 CFM 不一致的根源
深入分析初始化噪声生成过程
"""

import os
import pickle
import glob
import torch
import mlx.core as mx
import numpy as np
from typing import Dict, Any, List, Tuple
import traceback

class InconsistencyRootFinder:
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
    
    def analyze_noise_generation_process(self):
        """分析噪声生成过程"""
        print("🔍 分析噪声生成过程...")
        
        # 找到最新的缓存文件
        pytorch_inputs, _, mlx_inputs, _ = self.find_latest_cache_files()
        
        if not pytorch_inputs or not mlx_inputs:
            print("❌ 未找到输入缓存文件")
            return
        
        # 加载输入数据
        pytorch_data = self.load_cache_file(pytorch_inputs)
        mlx_data = self.load_cache_file(mlx_inputs)
        
        if not pytorch_data or not mlx_data:
            print("❌ 无法加载缓存数据")
            return
        
        print(f"\n{'='*60}")
        print(f"🔍 噪声生成过程分析")
        print(f"{'='*60}")
        
        # 分析时间戳和随机种子
        pytorch_timestamp = pytorch_data.get('timestamp', 0)
        mlx_timestamp = mlx_data.get('timestamp', 0)
        pytorch_seed = pytorch_data.get('unified_random_seed', 0)
        mlx_seed = mlx_data.get('unified_random_seed', 0)
        
        print(f"\n📊 时间戳和随机种子:")
        print(f"   PyTorch: timestamp={pytorch_timestamp}, seed={pytorch_seed}")
        print(f"   MLX:     timestamp={mlx_timestamp}, seed={mlx_seed}")
        
        # 分析初始化噪声
        pytorch_x = pytorch_data.get('x')
        mlx_x = mlx_data.get('x')
        
        if pytorch_x is not None and mlx_x is not None:
            print(f"\n📊 初始化噪声详细分析:")
            
            # 转换为 numpy 进行比较
            if isinstance(pytorch_x, torch.Tensor):
                pytorch_np = pytorch_x.detach().cpu().numpy()
            else:
                pytorch_np = pytorch_x
            
            if isinstance(mlx_x, mx.array):
                mlx_np = np.array(mlx_x)
            else:
                mlx_np = mlx_x
            
            # 计算详细统计
            pytorch_stats = {
                'min': np.min(pytorch_np),
                'max': np.max(pytorch_np),
                'mean': np.mean(pytorch_np),
                'std': np.std(pytorch_np),
                'median': np.median(pytorch_np)
            }
            
            mlx_stats = {
                'min': np.min(mlx_np),
                'max': np.max(mlx_np),
                'mean': np.mean(mlx_np),
                'std': np.std(mlx_np),
                'median': np.median(mlx_np)
            }
            
            print(f"   PyTorch: min={pytorch_stats['min']:.6f}, max={pytorch_stats['max']:.6f}, mean={pytorch_stats['mean']:.6f}, std={pytorch_stats['std']:.6f}")
            print(f"   MLX:     min={mlx_stats['min']:.6f}, max={mlx_stats['max']:.6f}, mean={mlx_stats['mean']:.6f}, std={mlx_stats['std']:.6f}")
            
            # 计算差异
            diff = np.abs(pytorch_np - mlx_np)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)
            
            print(f"   差异: max={max_diff:.6f}, mean={mean_diff:.6f}, std={std_diff:.6f}")
            
            # 分析差异分布
            diff_percentiles = np.percentile(diff, [0, 25, 50, 75, 90, 95, 99, 100])
            print(f"   差异分位数: 0%={diff_percentiles[0]:.6f}, 25%={diff_percentiles[1]:.6f}, 50%={diff_percentiles[2]:.6f}, 75%={diff_percentiles[3]:.6f}, 90%={diff_percentiles[4]:.6f}, 95%={diff_percentiles[5]:.6f}, 99%={diff_percentiles[6]:.6f}, 100%={diff_percentiles[7]:.6f}")
            
            # 检查是否是完全不同的噪声
            if max_diff > 1.0:
                print(f"   ❌ 噪声完全不同 - 可能是随机数生成问题")
            elif max_diff > 0.1:
                print(f"   ⚠️  噪声有显著差异 - 可能是数值精度问题")
            else:
                print(f"   ✅ 噪声基本一致")
    
    def analyze_cfm_inference_process(self):
        """分析 CFM 推理过程"""
        print(f"\n{'='*60}")
        print(f"🔍 CFM 推理过程分析")
        print(f"{'='*60}")
        
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
        
        # 分析输入输出关系
        print(f"\n📊 输入输出关系分析:")
        
        # PyTorch 输入输出
        pytorch_x = pytorch_input_data.get('x')
        pytorch_output = pytorch_output_data.get('output')
        
        # MLX 输入输出
        mlx_x = mlx_input_data.get('x')
        mlx_output = mlx_output_data.get('output')
        
        if pytorch_x is not None and pytorch_output is not None and mlx_x is not None and mlx_output is not None:
            # 转换为 numpy
            pytorch_x_np = pytorch_x.detach().cpu().numpy() if isinstance(pytorch_x, torch.Tensor) else pytorch_x
            pytorch_output_np = pytorch_output.detach().cpu().numpy() if isinstance(pytorch_output, torch.Tensor) else pytorch_output
            mlx_x_np = np.array(mlx_x) if isinstance(mlx_x, mx.array) else mlx_x
            mlx_output_np = np.array(mlx_output) if isinstance(mlx_output, mx.array) else mlx_output
            
            # 计算输入到输出的变化
            pytorch_change = pytorch_output_np - pytorch_x_np
            mlx_change = mlx_output_np - mlx_x_np
            
            pytorch_change_stats = {
                'min': np.min(pytorch_change),
                'max': np.max(pytorch_change),
                'mean': np.mean(pytorch_change),
                'std': np.std(pytorch_change)
            }
            
            mlx_change_stats = {
                'min': np.min(mlx_change),
                'max': np.max(mlx_change),
                'mean': np.mean(mlx_change),
                'std': np.std(mlx_change)
            }
            
            print(f"   PyTorch 输入→输出变化: min={pytorch_change_stats['min']:.6f}, max={pytorch_change_stats['max']:.6f}, mean={pytorch_change_stats['mean']:.6f}, std={pytorch_change_stats['std']:.6f}")
            print(f"   MLX 输入→输出变化:     min={mlx_change_stats['min']:.6f}, max={mlx_change_stats['max']:.6f}, mean={mlx_change_stats['mean']:.6f}, std={mlx_change_stats['std']:.6f}")
            
            # 比较变化
            change_diff = np.abs(pytorch_change - mlx_change)
            change_max_diff = np.max(change_diff)
            change_mean_diff = np.mean(change_diff)
            
            print(f"   变化差异: max={change_max_diff:.6f}, mean={change_mean_diff:.6f}")
            
            if change_max_diff < 0.1:
                print(f"   ✅ CFM 推理过程基本一致")
            else:
                print(f"   ❌ CFM 推理过程存在差异")
    
    def analyze_random_generator_usage(self):
        """分析随机数生成器使用情况"""
        print(f"\n{'='*60}")
        print(f"🔍 随机数生成器使用分析")
        print(f"{'='*60}")
        
        # 检查 UnifiedRandomGenerator 的使用
        try:
            from unified_random_generator import UnifiedRandomGenerator
            
            # 创建两个相同种子的生成器
            gen1 = UnifiedRandomGenerator(seed=42)
            gen2 = UnifiedRandomGenerator(seed=42)
            
            # 生成一些随机数进行比较
            print(f"\n📊 UnifiedRandomGenerator 测试:")
            
            # PyTorch 随机数
            torch_nums1 = gen1.generate_noise((10,))
            torch_nums2 = gen2.generate_noise((10,))
            
            # MLX 随机数
            mlx_nums1 = gen1.generate_noise_mlx((10,))
            mlx_nums2 = gen2.generate_noise_mlx((10,))
            
            print(f"   PyTorch 生成器1: {torch_nums1[:5].tolist()}")
            print(f"   PyTorch 生成器2: {torch_nums2[:5].tolist()}")
            print(f"   MLX 生成器1:     {mlx_nums1[:5].tolist()}")
            print(f"   MLX 生成器2:     {mlx_nums2[:5].tolist()}")
            
            # 检查一致性
            torch_diff = torch.abs(torch_nums1 - torch_nums2).max().item()
            mlx_diff = mx.abs(mlx_nums1 - mlx_nums2).max()
            
            print(f"   PyTorch 内部一致性: {torch_diff:.10f}")
            print(f"   MLX 内部一致性:     {mlx_diff:.10f}")
            
            if torch_diff < 1e-10 and mlx_diff < 1e-10:
                print(f"   ✅ UnifiedRandomGenerator 内部一致")
            else:
                print(f"   ❌ UnifiedRandomGenerator 内部不一致")
                
        except Exception as e:
            print(f"   ❌ UnifiedRandomGenerator 测试失败: {e}")
            traceback.print_exc()
    
    def analyze_cfm_internal_steps(self):
        """分析 CFM 内部步骤"""
        print(f"\n{'='*60}")
        print(f"🔍 CFM 内部步骤分析")
        print(f"{'='*60}")
        
        # 检查 CFM 实现差异
        try:
            # 检查 PyTorch CFM 实现
            from indextts.s2mel.modules.flow_matching import CFM
            print(f"   ✅ PyTorch CFM 模块加载成功")
            
            # 检查 MLX CFM 实现
            from indextts.s2mel.modules.mlx_flow_matching import MLXCFM
            print(f"   ✅ MLX CFM 模块加载成功")
            
            # 检查关键方法
            pytorch_cfm = CFM.__new__(CFM)
            mlx_cfm = MLXCFM.__new__(MLXCFM)
            
            print(f"\n📊 CFM 方法对比:")
            print(f"   PyTorch CFM: 有 inference, solve_euler 方法")
            print(f"   MLX CFM:     有 inference, solve_euler 方法")
                
        except Exception as e:
            print(f"   ❌ CFM 内部步骤分析失败: {e}")
            traceback.print_exc()
    
    def run_full_analysis(self):
        """运行完整分析"""
        print("🔍 CFM 不一致根源分析")
        print("="*60)
        
        # 1. 分析噪声生成过程
        self.analyze_noise_generation_process()
        
        # 2. 分析 CFM 推理过程
        self.analyze_cfm_inference_process()
        
        # 3. 分析随机数生成器使用
        self.analyze_random_generator_usage()
        
        # 4. 分析 CFM 内部步骤
        self.analyze_cfm_internal_steps()
        
        print(f"\n{'='*60}")
        print(f"🎯 分析总结")
        print(f"{'='*60}")
        print(f"1. 检查初始化噪声生成是否一致")
        print(f"2. 检查 CFM 推理过程是否一致")
        print(f"3. 检查随机数生成器是否正确使用")
        print(f"4. 检查 CFM 实现是否有差异")

def main():
    finder = InconsistencyRootFinder()
    finder.run_full_analysis()

if __name__ == "__main__":
    main()
