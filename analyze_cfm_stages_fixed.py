#!/usr/bin/env python3
"""
CFM 阶段分析工具 (修复版)
逐个阶段分析 PyTorch 和 MLX CFM 的输入输出数据
"""

import os
import pickle
import glob
import torch
import mlx.core as mx
import numpy as np
from typing import Dict, Any, List, Tuple

class CFMStageAnalyzer:
    def __init__(self, cache_dir: str = "cfm_production_cache", consistent_test_file: str = "cfm_consistent_test/consistent_test_data.pkl"):
        self.cache_dir = cache_dir
        self.consistent_test_file = consistent_test_file
        
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
    
    def load_consistent_test_data(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """加载一致的测试数据"""
        if not os.path.exists(self.consistent_test_file):
            print(f"❌ 一致测试数据文件不存在: {self.consistent_test_file}")
            print("💡 请先运行 analyze_cfm_consistency.py 生成一致的测试数据")
            return None, None
        
        with open(self.consistent_test_file, 'rb') as f:
            data = pickle.load(f)
        
        pytorch_inputs = data['pytorch_inputs']
        mlx_inputs = data['mlx_inputs']
        
        print(f"✅ 加载一致的测试数据")
        print(f"   PyTorch 输入形状: {[f'{k}: {v.shape if hasattr(v, \"shape\") else v}' for k, v in pytorch_inputs.items()]}")
        print(f"   MLX 输入形状: {[f'{k}: {v.shape if hasattr(v, \"shape\") else v}' for k, v in mlx_inputs.items()]}")
        
        return pytorch_inputs, mlx_inputs
    
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
                    "avg": 0.0,  # 整数类型不计算平均值
                    "std": 0.0   # 整数类型不计算标准差
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
                    "avg": 0.0,  # 整数类型不计算平均值
                    "std": 0.0   # 整数类型不计算标准差
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
                    "avg": 0.0,  # 整数类型不计算平均值
                    "std": 0.0   # 整数类型不计算标准差
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
    
    def print_stage_stats(self, stage_name: str, pytorch_data: Dict[str, Any], mlx_data: Dict[str, Any]):
        """打印阶段统计信息"""
        print(f"\n{'='*60}")
        print(f"🔍 {stage_name} 阶段分析")
        print(f"{'='*60}")
        
        # PyTorch 输入分析
        if pytorch_data:
            print(f"\n📊 PyTorch {stage_name} Input:")
            for key, value in pytorch_data.items():
                if isinstance(value, (torch.Tensor, mx.array, np.ndarray)):
                    stats = self.analyze_tensor_stats(value, key)
                    print(f"   {key}: shape={value.shape}, min={stats['min']:.6f}, max={stats['max']:.6f}, avg={stats['avg']:.6f}")
                else:
                    print(f"   {key}: {value}")
        
        # MLX 输入分析
        if mlx_data:
            print(f"\n📊 MLX {stage_name} Input:")
            for key, value in mlx_data.items():
                if isinstance(value, (torch.Tensor, mx.array, np.ndarray)):
                    stats = self.analyze_tensor_stats(value, key)
                    print(f"   {key}: shape={value.shape}, min={stats['min']:.6f}, max={stats['max']:.6f}, avg={stats['avg']:.6f}")
                else:
                    print(f"   {key}: {value}")
    
    def analyze_initialization_noise(self):
        """分析初始化噪声阶段"""
        print("🔍 分析初始化噪声阶段...")
        
        # 加载一致的测试数据
        pytorch_data, mlx_data = self.load_consistent_test_data()
        
        if pytorch_data is None or mlx_data is None:
            print("❌ 无法加载一致的测试数据")
            return
        
        # 分析输入参数一致性
        print(f"\n{'='*60}")
        print(f"🔍 输入参数一致性分析")
        print(f"{'='*60}")
        
        # 分析各个输入参数
        params_to_analyze = ['mu', 'prompt', 'style', 'x_lens']
        
        for param in params_to_analyze:
            pytorch_param = pytorch_data.get(param)
            mlx_param = mlx_data.get(param)
            
            if pytorch_param is not None and mlx_param is not None:
                print(f"\n📊 {param} 参数:")
                
                pytorch_stats = self.analyze_tensor_stats(pytorch_param, param)
                mlx_stats = self.analyze_tensor_stats(mlx_param, param)
                
                print(f"   PyTorch: shape={pytorch_param.shape}, min={pytorch_stats['min']:.6f}, max={pytorch_stats['max']:.6f}, avg={pytorch_stats['avg']:.6f}")
                print(f"   MLX:     shape={mlx_param.shape}, min={mlx_stats['min']:.6f}, max={mlx_stats['max']:.6f}, avg={mlx_stats['avg']:.6f}")
                
                # 比较差异
                if pytorch_param.shape == mlx_param.shape:
                    # 转换为相同类型进行比较
                    if isinstance(pytorch_param, torch.Tensor):
                        pytorch_np = pytorch_param.detach().cpu().numpy()
                    else:
                        pytorch_np = pytorch_param
                    
                    if isinstance(mlx_param, mx.array):
                        mlx_np = np.array(mlx_param)
                    else:
                        mlx_np = mlx_param
                    
                    diff = np.abs(pytorch_np - mlx_np)
                    max_diff = np.max(diff)
                    avg_diff = np.mean(diff)
                    
                    print(f"   差异: max={max_diff:.6f}, avg={avg_diff:.6f}")
                    
                    if max_diff < 1e-6:
                        print(f"   ✅ {param} 参数完全一致")
                    else:
                        print(f"   ⚠️  {param} 参数存在差异")
                else:
                    print(f"   ❌ {param} 形状不匹配")
                    print(f"   💡 这不应该发生，因为使用了同一个 PyTorch CFM 前级缓存")
    
    def analyze_input_parameters(self):
        """分析输入参数"""
        print(f"\n{'='*60}")
        print(f"🔍 输入参数分析 (使用一致测试数据)")
        print(f"{'='*60}")
        
        # 加载一致的测试数据
        pytorch_data, mlx_data = self.load_consistent_test_data()
        
        if pytorch_data is None or mlx_data is None:
            print("❌ 无法加载一致的测试数据")
            return
        
        # 分析各个输入参数
        params_to_analyze = ['mu', 'prompt', 'style', 'x_lens']
        
        for param in params_to_analyze:
            pytorch_param = pytorch_data.get(param)
            mlx_param = mlx_data.get(param)
            
            if pytorch_param is not None and mlx_param is not None:
                print(f"\n📊 {param} 参数:")
                
                pytorch_stats = self.analyze_tensor_stats(pytorch_param, param)
                mlx_stats = self.analyze_tensor_stats(mlx_param, param)
                
                print(f"   PyTorch: shape={pytorch_param.shape}, min={pytorch_stats['min']:.6f}, max={pytorch_stats['max']:.6f}, avg={pytorch_stats['avg']:.6f}")
                print(f"   MLX:     shape={mlx_param.shape}, min={mlx_stats['min']:.6f}, max={mlx_stats['max']:.6f}, avg={mlx_stats['avg']:.6f}")
                
                # 比较差异
                if pytorch_param.shape == mlx_param.shape:
                    # 转换为相同类型进行比较
                    if isinstance(pytorch_param, torch.Tensor):
                        pytorch_np = pytorch_param.detach().cpu().numpy()
                    else:
                        pytorch_np = pytorch_param
                    
                    if isinstance(mlx_param, mx.array):
                        mlx_np = np.array(mlx_param)
                    else:
                        mlx_np = mlx_param
                    
                    diff = np.abs(pytorch_np - mlx_np)
                    max_diff = np.max(diff)
                    avg_diff = np.mean(diff)
                    
                    print(f"   差异: max={max_diff:.6f}, avg={avg_diff:.6f}")
                    
                    if max_diff < 1e-6:
                        print(f"   ✅ {param} 参数一致")
                    else:
                        print(f"   ❌ {param} 参数不一致")
                else:
                    print(f"   ❌ {param} 形状不匹配")
    
    def analyze_output_results(self):
        """分析输出结果"""
        print(f"\n{'='*60}")
        print(f"🔍 输出结果分析")
        print(f"{'='*60}")
        
        # 找到最新的缓存文件
        _, pytorch_outputs, _, mlx_outputs = self.find_latest_cache_files()
        
        if not pytorch_outputs or not mlx_outputs:
            print("❌ 未找到输出缓存文件")
            return
        
        # 加载输出数据
        pytorch_data = self.load_cache_file(pytorch_outputs)
        mlx_data = self.load_cache_file(mlx_outputs)
        
        if not pytorch_data or not mlx_data:
            print("❌ 无法加载输出数据")
            return
        
        # 分析输出结果
        pytorch_output = pytorch_data.get('output')
        mlx_output = mlx_data.get('output')
        
        if pytorch_output is not None and mlx_output is not None:
            print(f"\n📊 CFM 输出结果:")
            
            pytorch_stats = self.analyze_tensor_stats(pytorch_output, 'output')
            mlx_stats = self.analyze_tensor_stats(mlx_output, 'output')
            
            print(f"   PyTorch: shape={pytorch_output.shape}, min={pytorch_stats['min']:.6f}, max={pytorch_stats['max']:.6f}, avg={pytorch_stats['avg']:.6f}")
            print(f"   MLX:     shape={mlx_output.shape}, min={mlx_stats['min']:.6f}, max={mlx_stats['max']:.6f}, avg={mlx_stats['avg']:.6f}")
            
            # 比较差异
            if pytorch_output.shape == mlx_output.shape:
                # 转换为相同类型进行比较
                if isinstance(pytorch_output, torch.Tensor):
                    pytorch_np = pytorch_output.detach().cpu().numpy()
                else:
                    pytorch_np = pytorch_output
                
                if isinstance(mlx_output, mx.array):
                    mlx_np = np.array(mlx_output)
                else:
                    mlx_np = mlx_output
                
                diff = np.abs(pytorch_np - mlx_np)
                max_diff = np.max(diff)
                avg_diff = np.mean(diff)
                
                print(f"   差异: max={max_diff:.6f}, avg={avg_diff:.6f}")
                
                if max_diff < 1e-6:
                    print(f"   ✅ 输出结果完全一致")
                else:
                    print(f"   ❌ 输出结果不一致")
            else:
                print(f"   ❌ 输出形状不匹配")
    
    def analyze_cfm_euler_steps(self):
        """分析 CFM Euler 求解步骤"""
        print(f"\n{'='*60}")
        print(f"🔍 CFM Euler 求解步骤分析")
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
        
        # 分析输入数据
        pytorch_x = pytorch_input_data.get('x')
        pytorch_mu = pytorch_input_data.get('mu')
        mlx_x = mlx_input_data.get('x')
        mlx_mu = mlx_input_data.get('mu')
        
        print(f"\n📊 CFM 输入数据:")
        if pytorch_x is not None:
            pytorch_x_stats = self.analyze_tensor_stats(pytorch_x, 'x')
            print(f"   PyTorch x: shape={pytorch_x.shape}, min={pytorch_x_stats['min']:.6f}, max={pytorch_x_stats['max']:.6f}, avg={pytorch_x_stats['avg']:.6f}")
        if pytorch_mu is not None:
            pytorch_mu_stats = self.analyze_tensor_stats(pytorch_mu, 'mu')
            print(f"   PyTorch mu: shape={pytorch_mu.shape}, min={pytorch_mu_stats['min']:.6f}, max={pytorch_mu_stats['max']:.6f}, avg={pytorch_mu_stats['avg']:.6f}")
        if mlx_x is not None:
            mlx_x_stats = self.analyze_tensor_stats(mlx_x, 'x')
            print(f"   MLX x:     shape={mlx_x.shape}, min={mlx_x_stats['min']:.6f}, max={mlx_x_stats['max']:.6f}, avg={mlx_x_stats['avg']:.6f}")
        if mlx_mu is not None:
            mlx_mu_stats = self.analyze_tensor_stats(mlx_mu, 'mu')
            print(f"   MLX mu:    shape={mlx_mu.shape}, min={mlx_mu_stats['min']:.6f}, max={mlx_mu_stats['max']:.6f}, avg={mlx_mu_stats['avg']:.6f}")
        
        # 分析输出数据
        pytorch_output = pytorch_output_data.get('output')
        mlx_output = mlx_output_data.get('output')
        
        print(f"\n📊 CFM 输出数据:")
        if pytorch_output is not None:
            pytorch_output_stats = self.analyze_tensor_stats(pytorch_output, 'output')
            print(f"   PyTorch output: shape={pytorch_output.shape}, min={pytorch_output_stats['min']:.6f}, max={pytorch_output_stats['max']:.6f}, avg={pytorch_output_stats['avg']:.6f}")
        if mlx_output is not None:
            mlx_output_stats = self.analyze_tensor_stats(mlx_output, 'output')
            print(f"   MLX output:     shape={mlx_output.shape}, min={mlx_output_stats['min']:.6f}, max={mlx_output_stats['max']:.6f}, avg={mlx_output_stats['avg']:.6f}")
        
        # 计算输入输出变化（检查形状匹配）
        if pytorch_input_data and pytorch_output and mlx_input_data and mlx_output:
            pytorch_change = pytorch_output - pytorch_input_data['x']
            mlx_change = mlx_output - mlx_input_data['x']
            
            print(f"\n📊 输入输出变化分析:")
            pytorch_change_stats = self.analyze_tensor_stats(pytorch_change, 'change')
            mlx_change_stats = self.analyze_tensor_stats(mlx_change, 'change')
            
            print(f"   PyTorch 变化: min={pytorch_change_stats['min']:.6f}, max={pytorch_change_stats['max']:.6f}, avg={pytorch_change_stats['avg']:.6f}, std={pytorch_change_stats['std']:.6f}")
            print(f"   MLX 变化:     min={mlx_change_stats['min']:.6f}, max={mlx_change_stats['max']:.6f}, avg={mlx_change_stats['avg']:.6f}, std={mlx_change_stats['std']:.6f}")
            
            # 比较变化差异（检查形状匹配）
            if pytorch_change.shape == mlx_change.shape:
                change_diff = np.abs(pytorch_change - mlx_change)
                max_change_diff = np.max(change_diff)
                avg_change_diff = np.mean(change_diff)
                
                print(f"\n🔍 变化差异分析:")
                print(f"   最大变化差异: {max_change_diff:.6f}")
                print(f"   平均变化差异: {avg_change_diff:.6f}")
                
                if max_change_diff < 1e-6:
                    print(f"   ✅ CFM 变化完全一致")
                else:
                    print(f"   ⚠️  CFM 变化存在差异")
            else:
                print(f"\n🔍 变化差异分析:")
                print(f"   ❌ 形状不匹配，无法比较变化差异")
                print(f"   PyTorch 变化形状: {pytorch_change.shape}")
                print(f"   MLX 变化形状:     {mlx_change.shape}")
                print(f"   💡 建议使用一致的输入数据重新生成缓存")
    
    def analyze_dit_weights(self):
        """分析 DiT 权重"""
        print(f"\n{'='*60}")
        print(f"🔍 DiT 权重分析")
        print(f"{'='*60}")
        
        print("   ⚠️  DiT 权重分析需要加载实际模型，当前跳过")
        print("   💡 建议使用专门的权重分析工具")
    
    def analyze_all_stages(self):
        """分析所有阶段"""
        print("🔍 CFM 阶段分析工具 (修复版)")
        print("="*60)
        
        # 分析初始化噪声
        self.analyze_initialization_noise()
        
        # 分析其他输入参数
        self.analyze_input_parameters()
        
        # 分析输出结果
        self.analyze_output_results()
        
        # 分析 CFM Euler 求解步骤
        self.analyze_cfm_euler_steps()
        
        # 分析 DiT 权重
        self.analyze_dit_weights()

def main():
    analyzer = CFMStageAnalyzer()
    analyzer.analyze_all_stages()

if __name__ == "__main__":
    main()
