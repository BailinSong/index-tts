#!/usr/bin/env python3
"""
CFM 输入数据缓存和验证工具
用于验证 PyTorch 和 MLX CFM 推理的一致性
"""

import os
import pickle
import numpy as np
import torch
import mlx.core as mx
from typing import Dict, Any, Tuple
import time

class CFMDataCache:
    """CFM 输入数据缓存管理器"""
    
    def __init__(self, cache_dir: str = "cfm_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def save_torch_inputs(self, inputs: Dict[str, Any], filename: str = None):
        """
        保存 PyTorch CFM 输入数据
        
        Args:
            inputs: CFM 输入字典，包含 mu, x_lens, prompt, style, f0 等
            filename: 保存文件名，默认使用时间戳
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"cfm_torch_inputs_{timestamp}.pkl"
        
        filepath = os.path.join(self.cache_dir, filename)
        
        # 转换 PyTorch tensors 为 numpy arrays 以便跨框架使用
        processed_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                processed_inputs[key] = {
                    'data': value.detach().cpu().numpy(),
                    'dtype': str(value.dtype),
                    'device': str(value.device),
                    'shape': value.shape
                }
            else:
                processed_inputs[key] = value
        
        with open(filepath, 'wb') as f:
            pickle.dump(processed_inputs, f)
        
        print(f"✅ PyTorch CFM inputs saved to: {filepath}")
        print(f"📊 Input shapes:")
        for key, value in processed_inputs.items():
            if isinstance(value, dict) and 'shape' in value:
                print(f"   {key}: {value['shape']}")
            else:
                print(f"   {key}: {type(value)}")
        
        return filepath
    
    def load_torch_inputs(self, filename: str) -> Dict[str, Any]:
        """
        加载 PyTorch CFM 输入数据
        
        Args:
            filename: 文件名
            
        Returns:
            恢复的 PyTorch tensors 字典
        """
        filepath = os.path.join(self.cache_dir, filename)
        
        with open(filepath, 'rb') as f:
            processed_inputs = pickle.load(f)
        
        # 恢复 PyTorch tensors
        restored_inputs = {}
        for key, value in processed_inputs.items():
            if isinstance(value, dict) and 'data' in value:
                # 恢复为 PyTorch tensor
                tensor = torch.from_numpy(value['data'])
                if value['dtype'] != 'torch.float32':
                    tensor = tensor.to(getattr(torch, value['dtype'].split('.')[-1]))
                restored_inputs[key] = tensor
            else:
                restored_inputs[key] = value
        
        print(f"✅ PyTorch CFM inputs loaded from: {filepath}")
        return restored_inputs
    
    def convert_to_mlx(self, torch_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 PyTorch inputs 转换为 MLX 格式
        
        Args:
            torch_inputs: PyTorch 输入字典
            
        Returns:
            MLX 格式的输入字典
        """
        mlx_inputs = {}
        
        for key, value in torch_inputs.items():
            if isinstance(value, torch.Tensor):
                # 转换为 MLX array
                mlx_inputs[key] = mx.array(value.detach().cpu().numpy())
            else:
                mlx_inputs[key] = value
        
        print("✅ Converted PyTorch inputs to MLX format")
        return mlx_inputs
    
    def save_mlx_inputs(self, inputs: Dict[str, Any], filename: str = None):
        """
        保存 MLX CFM 输入数据
        
        Args:
            inputs: MLX 输入字典
            filename: 保存文件名
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"cfm_mlx_inputs_{timestamp}.pkl"
        
        filepath = os.path.join(self.cache_dir, filename)
        
        # 转换 MLX arrays 为 numpy arrays
        processed_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, mx.array):
                processed_inputs[key] = {
                    'data': np.array(value),
                    'dtype': str(value.dtype),
                    'shape': value.shape
                }
            else:
                processed_inputs[key] = value
        
        with open(filepath, 'wb') as f:
            pickle.dump(processed_inputs, f)
        
        print(f"✅ MLX CFM inputs saved to: {filepath}")
        return filepath
    
    def load_mlx_inputs(self, filename: str) -> Dict[str, Any]:
        """
        加载 MLX CFM 输入数据
        
        Args:
            filename: 文件名
            
        Returns:
            恢复的 MLX arrays 字典
        """
        filepath = os.path.join(self.cache_dir, filename)
        
        with open(filepath, 'rb') as f:
            processed_inputs = pickle.load(f)
        
        # 恢复 MLX arrays
        restored_inputs = {}
        for key, value in processed_inputs.items():
            if isinstance(value, dict) and 'data' in value:
                restored_inputs[key] = mx.array(value['data'])
            else:
                restored_inputs[key] = value
        
        print(f"✅ MLX CFM inputs loaded from: {filepath}")
        return restored_inputs
    
    def compare_outputs(self, torch_output: torch.Tensor, mlx_output: mx.array, 
                       tolerance: float = 1e-6, name: str = "output") -> bool:
        """
        比较 PyTorch 和 MLX 输出的一致性
        
        Args:
            torch_output: PyTorch 输出
            mlx_output: MLX 输出
            tolerance: 容差
            name: 输出名称
            
        Returns:
            是否一致
        """
        # 转换为 numpy 进行比较
        torch_np = torch_output.detach().cpu().numpy()
        mlx_np = np.array(mlx_output)
        
        # 检查形状
        if torch_np.shape != mlx_np.shape:
            print(f"❌ {name} shape mismatch:")
            print(f"   PyTorch: {torch_np.shape}")
            print(f"   MLX:     {mlx_np.shape}")
            return False
        
        # 检查数值
        diff = np.abs(torch_np - mlx_np)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"🔍 {name} comparison:")
        print(f"   Max difference:  {max_diff:.8f}")
        print(f"   Mean difference: {mean_diff:.8f}")
        print(f"   Tolerance:       {tolerance:.8f}")
        
        if max_diff < tolerance:
            print(f"✅ {name} outputs are consistent!")
            return True
        else:
            print(f"❌ {name} outputs are inconsistent!")
            return False
    
    def list_cache_files(self):
        """列出所有缓存文件"""
        files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        print(f"📁 Cache files in {self.cache_dir}:")
        for f in sorted(files):
            filepath = os.path.join(self.cache_dir, f)
            size = os.path.getsize(filepath) / 1024 / 1024  # MB
            print(f"   {f} ({size:.2f} MB)")
        return files


def create_cfm_test_data():
    """创建测试用的 CFM 输入数据"""
    # 模拟典型的 CFM 输入
    batch_size = 2
    seq_len = 100
    in_channels = 80
    content_dim = 512
    style_dim = 192
    
    # 创建 PyTorch 数据
    torch_inputs = {
        'mu': torch.randn(batch_size, seq_len, content_dim),
        'x_lens': torch.tensor([seq_len, seq_len-10]),  # 不同的序列长度
        'prompt': torch.randn(batch_size, in_channels, 50),  # 参考音频
        'style': torch.randn(batch_size, style_dim),  # 风格向量
        'f0': None,  # F0 通常为 None
        'n_timesteps': 25,
        'temperature': 1.0,
        'inference_cfg_rate': 0.5
    }
    
    print("🧪 Created test CFM input data:")
    for key, value in torch_inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape} ({value.dtype})")
        else:
            print(f"   {key}: {value}")
    
    return torch_inputs


def main():
    """主函数 - 演示缓存工具的使用"""
    print("🔧 CFM Data Cache Tool Demo")
    print("=" * 50)
    
    # 创建缓存管理器
    cache = CFMDataCache()
    
    # 创建测试数据
    print("\n1. Creating test data...")
    torch_inputs = create_cfm_test_data()
    
    # 保存 PyTorch 输入
    print("\n2. Saving PyTorch inputs...")
    torch_file = cache.save_torch_inputs(torch_inputs)
    
    # 转换为 MLX 格式
    print("\n3. Converting to MLX format...")
    mlx_inputs = cache.convert_to_mlx(torch_inputs)
    
    # 保存 MLX 输入
    print("\n4. Saving MLX inputs...")
    mlx_file = cache.save_mlx_inputs(mlx_inputs)
    
    # 列出缓存文件
    print("\n5. Cache files:")
    cache.list_cache_files()
    
    # 验证加载
    print("\n6. Testing load functionality...")
    loaded_torch = cache.load_torch_inputs(os.path.basename(torch_file))
    loaded_mlx = cache.load_mlx_inputs(os.path.basename(mlx_file))
    
    print("\n✅ Cache tool demo completed!")
    print("\n📝 Usage:")
    print("   - Use cache.save_torch_inputs() to save PyTorch CFM inputs")
    print("   - Use cache.convert_to_mlx() to convert to MLX format")
    print("   - Use cache.compare_outputs() to verify consistency")


if __name__ == "__main__":
    main()
