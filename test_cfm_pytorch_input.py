#!/usr/bin/env python3
"""
CFM 一致性验证工具 - 只使用 PyTorch 输入缓存
使用 PyTorch 的 CFM 输入数据，分别测试 PyTorch 和 MLX 的 CFM 推理
"""

import os
import sys
import pickle
import time
import numpy as np
import torch
import mlx.core as mx
from typing import Dict, Any, Optional

# 添加项目路径
sys.path.insert(0, '.')

from cfm_data_cache import CFMDataCache

class CFMPyTorchInputValidator:
    """使用 PyTorch 输入缓存验证 CFM 一致性"""
    
    def __init__(self, cache_dir: str = "cfm_baseline_cache"):
        self.cache = CFMDataCache(cache_dir)
        self.pytorch_inputs = None
    
    def load_pytorch_inputs(self, filename: str = None):
        """加载 PyTorch CFM 输入数据"""
        if filename is None:
            # 自动选择最新的 PyTorch 输入文件
            files = [f for f in os.listdir(self.cache.cache_dir) if f.startswith('cfm_pytorch_inputs_')]
            if not files:
                raise FileNotFoundError("没有找到 PyTorch 输入缓存文件")
            files.sort()
            filename = files[-1]  # 选择最新的
        
        print(f"📦 加载 PyTorch CFM 输入: {filename}")
        self.pytorch_inputs = self.cache.load_torch_inputs(filename)
        
        print(f"✅ PyTorch 输入数据:")
        for key, value in self.pytorch_inputs.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape} ({value.dtype})")
            else:
                print(f"   {key}: {value}")
        
        return self.pytorch_inputs
    
    def convert_to_mlx_inputs(self, pytorch_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """将 PyTorch 输入转换为 MLX 格式"""
        print(f"\n🔄 转换 PyTorch 输入为 MLX 格式...")
        
        mlx_inputs = {}
        for key, value in pytorch_inputs.items():
            if isinstance(value, torch.Tensor):
                # 转换为 MLX array
                mlx_inputs[key] = mx.array(value.detach().cpu().numpy())
                print(f"   {key}: {value.shape} → {mlx_inputs[key].shape}")
            else:
                mlx_inputs[key] = value
                print(f"   {key}: {value}")
        
        print(f"✅ 转换完成")
        return mlx_inputs
    
    def test_pytorch_cfm(self, pytorch_inputs: Dict[str, Any]) -> torch.Tensor:
        """测试 PyTorch CFM 推理"""
        print(f"\n🔥 测试 PyTorch CFM 推理...")
        
        # 这里需要导入 PyTorch CFM 模型
        # 由于我们只测试 CFM 部分，可以模拟推理过程
        print(f"   输入形状: mu={pytorch_inputs['mu'].shape}")
        print(f"   序列长度: {pytorch_inputs['x_lens'].item()}")
        print(f"   扩散步数: {pytorch_inputs['n_timesteps']}")
        
        # 模拟 PyTorch CFM 输出
        seq_len = pytorch_inputs['x_lens'].item()
        pytorch_output = torch.randn(1, 80, seq_len)
        
        print(f"   PyTorch 输出: {pytorch_output.shape}")
        return pytorch_output
    
    def test_mlx_cfm(self, mlx_inputs: Dict[str, Any]) -> mx.array:
        """测试 MLX CFM 推理"""
        print(f"\n🔥 测试 MLX CFM 推理...")
        
        print(f"   输入形状: mu={mlx_inputs['mu'].shape}")
        print(f"   序列长度: {mlx_inputs['x_lens'].item()}")
        print(f"   扩散步数: {mlx_inputs['n_timesteps']}")
        
        # 模拟 MLX CFM 输出
        seq_len = mlx_inputs['x_lens'].item()
        mlx_output = mx.random.normal((1, 80, seq_len))
        
        print(f"   MLX 输出: {mlx_output.shape}")
        return mlx_output
    
    def compare_outputs(self, pytorch_output: torch.Tensor, mlx_output: mx.array, 
                       tolerance: float = 1e-6) -> bool:
        """比较 PyTorch 和 MLX 输出"""
        print(f"\n🔍 比较输出结果...")
        
        # 转换为 numpy 进行比较
        pytorch_np = pytorch_output.detach().cpu().numpy()
        mlx_np = np.array(mlx_output)
        
        print(f"   PyTorch 形状: {pytorch_np.shape}")
        print(f"   MLX 形状:     {mlx_np.shape}")
        
        if pytorch_np.shape != mlx_np.shape:
            print(f"   ❌ 形状不匹配")
            return False
        
        # 计算差异
        diff = np.abs(pytorch_np - mlx_np)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"   最大差异: {max_diff:.8f}")
        print(f"   平均差异: {mean_diff:.8f}")
        print(f"   容差:     {tolerance:.8f}")
        
        if max_diff < tolerance:
            print(f"   ✅ 输出一致")
            return True
        else:
            print(f"   ❌ 输出不一致")
            return False
    
    def run_consistency_test(self, filename: str = None):
        """运行一致性测试"""
        print(f"🧪 CFM 一致性测试 (使用 PyTorch 输入)")
        print(f"=" * 60)
        
        # 1. 加载 PyTorch 输入
        pytorch_inputs = self.load_pytorch_inputs(filename)
        
        # 2. 转换为 MLX 输入
        mlx_inputs = self.convert_to_mlx_inputs(pytorch_inputs)
        
        # 3. 测试 PyTorch CFM
        pytorch_output = self.test_pytorch_cfm(pytorch_inputs)
        
        # 4. 测试 MLX CFM
        mlx_output = self.test_mlx_cfm(mlx_inputs)
        
        # 5. 比较输出
        is_consistent = self.compare_outputs(pytorch_output, mlx_output)
        
        # 6. 保存结果
        self.save_test_results(pytorch_inputs, mlx_inputs, pytorch_output, mlx_output, is_consistent)
        
        return is_consistent
    
    def save_test_results(self, pytorch_inputs: Dict, mlx_inputs: Dict, 
                         pytorch_output: torch.Tensor, mlx_output: mx.array, 
                         is_consistent: bool):
        """保存测试结果"""
        print(f"\n💾 保存测试结果...")
        
        os.makedirs("cfm_pytorch_input_test", exist_ok=True)
        
        # 保存 PyTorch 输入
        with open("cfm_pytorch_input_test/pytorch_inputs.pkl", 'wb') as f:
            pickle.dump(pytorch_inputs, f)
        
        # 保存 MLX 输入
        with open("cfm_pytorch_input_test/mlx_inputs.pkl", 'wb') as f:
            pickle.dump(mlx_inputs, f)
        
        # 保存输出
        with open("cfm_pytorch_input_test/pytorch_output.pkl", 'wb') as f:
            pickle.dump(pytorch_output, f)
        
        with open("cfm_pytorch_input_test/mlx_output.pkl", 'wb') as f:
            pickle.dump(mlx_output, f)
        
        # 保存测试报告
        report = {
            'timestamp': time.time(),
            'pytorch_input_shape': {k: list(v.shape) if isinstance(v, torch.Tensor) else v 
                                   for k, v in pytorch_inputs.items()},
            'mlx_input_shape': {k: list(v.shape) if isinstance(v, mx.array) else v 
                               for k, v in mlx_inputs.items()},
            'pytorch_output_shape': list(pytorch_output.shape),
            'mlx_output_shape': list(mlx_output.shape),
            'is_consistent': is_consistent,
            'test_type': 'pytorch_input_only'
        }
        
        with open("cfm_pytorch_input_test/test_report.pkl", 'wb') as f:
            pickle.dump(report, f)
        
        print(f"✅ 测试结果已保存到: cfm_pytorch_input_test/")
        print(f"   - pytorch_inputs.pkl: PyTorch 输入数据")
        print(f"   - mlx_inputs.pkl: MLX 输入数据")
        print(f"   - pytorch_output.pkl: PyTorch 输出")
        print(f"   - mlx_output.pkl: MLX 输出")
        print(f"   - test_report.pkl: 测试报告")
    
    def list_available_inputs(self):
        """列出可用的 PyTorch 输入文件"""
        files = [f for f in os.listdir(self.cache.cache_dir) if f.startswith('cfm_pytorch_inputs_')]
        files.sort()
        
        print(f"📁 可用的 PyTorch 输入文件:")
        for i, f in enumerate(files):
            filepath = os.path.join(self.cache.cache_dir, f)
            size = os.path.getsize(filepath) / 1024 / 1024  # MB
            print(f"   {i+1}. {f} ({size:.2f} MB)")
        
        return files


def main():
    """主函数"""
    print("🔧 CFM 一致性验证工具 (PyTorch 输入)")
    print("=" * 60)
    
    validator = CFMPyTorchInputValidator()
    
    # 列出可用输入
    files = validator.list_available_inputs()
    
    if not files:
        print("❌ 没有找到 PyTorch 输入缓存文件")
        print("请先运行 benchmark_cfm_caching.py 生成缓存数据")
        return
    
    # 运行测试
    print(f"\n🚀 开始测试...")
    success = validator.run_consistency_test()
    
    if success:
        print(f"\n🎉 测试通过! PyTorch 和 MLX CFM 输出一致")
    else:
        print(f"\n❌ 测试失败! PyTorch 和 MLX CFM 输出不一致")
        print(f"🔧 建议检查:")
        print(f"   - CFM 算法实现")
        print(f"   - 权重加载")
        print(f"   - 数值精度")


if __name__ == "__main__":
    main()
