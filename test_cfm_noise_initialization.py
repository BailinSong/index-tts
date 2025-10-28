#!/usr/bin/env python3
"""
CFM 初始化噪声一致性测试
测试 PyTorch 和 MLX 版本在初始化噪声阶段的一致性
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
from unified_random_generator import UnifiedRandomGenerator

class CFMNoiseInitializationTest:
    """CFM 初始化噪声一致性测试"""
    
    def __init__(self, cache_dir: str = "cfm_baseline_cache"):
        self.cache = CFMDataCache(cache_dir)
        self.pytorch_inputs = None
        self.unified_random = None
    
    def load_pytorch_inputs(self, filename: str = None):
        """加载 PyTorch CFM 输入数据"""
        if filename is None:
            files = [f for f in os.listdir(self.cache.cache_dir) if f.startswith('cfm_pytorch_inputs_')]
            if not files:
                raise FileNotFoundError("没有找到 PyTorch 输入缓存文件")
            files.sort()
            filename = files[-1]
        
        print(f"📦 加载 PyTorch CFM 输入: {filename}")
        self.pytorch_inputs = self.cache.load_torch_inputs(filename)
        
        print(f"✅ PyTorch 输入数据:")
        for key, value in self.pytorch_inputs.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape} ({value.dtype})")
            else:
                print(f"   {key}: {value}")
        
        return self.pytorch_inputs
    
    def setup_unified_random(self, seed: int = 42):
        """设置统一随机数生成器"""
        print(f"\n🎲 设置统一随机数生成器 (seed={seed})...")
        
        self.unified_random = UnifiedRandomGenerator(seed)
        print(f"✅ 统一随机数生成器已设置")
        
        return self.unified_random
    
    def test_pytorch_noise_initialization(self, pytorch_inputs: Dict[str, Any]) -> torch.Tensor:
        """测试 PyTorch 噪声初始化"""
        print(f"\n🔥 测试 PyTorch 噪声初始化...")
        
        # 提取参数
        mu = pytorch_inputs['mu']
        x_lens = pytorch_inputs['x_lens']
        temperature = pytorch_inputs['temperature']
        
        print(f"   输入参数:")
        print(f"     mu: {mu.shape}")
        print(f"     x_lens: {x_lens.item()}")
        print(f"     temperature: {temperature}")
        
        # 获取序列长度
        seq_len = x_lens.item()
        batch_size = mu.shape[0]
        in_channels = 80  # mel频谱通道数
        
        print(f"   噪声参数:")
        print(f"     batch_size: {batch_size}")
        print(f"     in_channels: {in_channels}")
        print(f"     seq_len: {seq_len}")
        print(f"     noise_shape: ({batch_size}, {in_channels}, {seq_len})")
        
        # 设置随机种子
        if self.unified_random:
            self.unified_random.reset_seed(self.unified_random.seed)
            torch.manual_seed(self.unified_random.seed)
            print(f"     使用统一随机种子: {self.unified_random.seed}")
        else:
            torch.manual_seed(42)
            print(f"     使用固定随机种子: 42")
        
        # 生成噪声
        print(f"   🔄 生成 PyTorch 噪声...")
        noise_shape = (batch_size, in_channels, seq_len)
        
        if self.unified_random:
            # 使用统一随机数生成器
            z = self.unified_random.generate_noise(noise_shape, device=mu.device) * temperature
        else:
            # 使用标准 PyTorch 随机数
            z = torch.randn(noise_shape, device=mu.device) * temperature
        
        print(f"   ✅ PyTorch 噪声生成完成:")
        print(f"     形状: {z.shape}")
        print(f"     数据类型: {z.dtype}")
        print(f"     设备: {z.device}")
        print(f"     范围: [{z.min().item():.6f}, {z.max().item():.6f}]")
        print(f"     均值: {z.mean().item():.6f}")
        print(f"     标准差: {z.std().item():.6f}")
        
        return z
    
    def test_mlx_noise_initialization(self, pytorch_inputs: Dict[str, Any]) -> mx.array:
        """测试 MLX 噪声初始化"""
        print(f"\n🔥 测试 MLX 噪声初始化...")
        
        # 提取参数
        mu = pytorch_inputs['mu']
        x_lens = pytorch_inputs['x_lens']
        temperature = pytorch_inputs['temperature']
        
        print(f"   输入参数:")
        print(f"     mu: {mu.shape}")
        print(f"     x_lens: {x_lens.item()}")
        print(f"     temperature: {temperature}")
        
        # 获取序列长度
        seq_len = x_lens.item()
        batch_size = mu.shape[0]
        in_channels = 80  # mel频谱通道数
        
        print(f"   噪声参数:")
        print(f"     batch_size: {batch_size}")
        print(f"     in_channels: {in_channels}")
        print(f"     seq_len: {seq_len}")
        print(f"     noise_shape: ({batch_size}, {in_channels}, {seq_len})")
        
        # 设置随机种子
        if self.unified_random:
            self.unified_random.reset_seed(self.unified_random.seed)
            mx.random.seed(self.unified_random.seed)
            print(f"     使用统一随机种子: {self.unified_random.seed}")
        else:
            mx.random.seed(42)
            print(f"     使用固定随机种子: 42")
        
        # 生成噪声
        print(f"   🔄 生成 MLX 噪声...")
        noise_shape = (batch_size, in_channels, seq_len)
        
        if self.unified_random:
            # 使用统一随机数生成器
            z = self.unified_random.generate_noise_mlx(noise_shape) * temperature
        else:
            # 使用标准 MLX 随机数
            z = mx.random.normal(noise_shape) * temperature
        
        print(f"   ✅ MLX 噪声生成完成:")
        print(f"     形状: {z.shape}")
        print(f"     数据类型: {z.dtype}")
        print(f"     范围: [{float(mx.min(z)):.6f}, {float(mx.max(z)):.6f}]")
        print(f"     均值: {float(mx.mean(z)):.6f}")
        print(f"     标准差: {float(mx.std(z)):.6f}")
        
        return z
    
    def compare_noise_outputs(self, pytorch_noise: torch.Tensor, mlx_noise: mx.array, 
                            tolerance: float = 1e-6) -> bool:
        """比较 PyTorch 和 MLX 噪声输出"""
        print(f"\n🔍 比较噪声输出...")
        
        # 转换为 numpy 进行比较
        pytorch_np = pytorch_noise.detach().cpu().numpy()
        mlx_np = np.array(mlx_noise)
        
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
        
        # 计算统计信息
        pytorch_stats = {
            'min': float(np.min(pytorch_np)),
            'max': float(np.max(pytorch_np)),
            'mean': float(np.mean(pytorch_np)),
            'std': float(np.std(pytorch_np))
        }
        
        mlx_stats = {
            'min': float(np.min(mlx_np)),
            'max': float(np.max(mlx_np)),
            'mean': float(np.mean(mlx_np)),
            'std': float(np.std(mlx_np))
        }
        
        print(f"\n   📊 统计信息比较:")
        print(f"     最小值: PyTorch={pytorch_stats['min']:.6f}, MLX={mlx_stats['min']:.6f}")
        print(f"     最大值: PyTorch={pytorch_stats['max']:.6f}, MLX={mlx_stats['max']:.6f}")
        print(f"     均值:   PyTorch={pytorch_stats['mean']:.6f}, MLX={mlx_stats['mean']:.6f}")
        print(f"     标准差: PyTorch={pytorch_stats['std']:.6f}, MLX={mlx_stats['std']:.6f}")
        
        if max_diff < tolerance:
            print(f"   ✅ 噪声完全一致")
            return True
        else:
            print(f"   ❌ 噪声不一致")
            return False
    
    def test_multiple_seeds(self, pytorch_inputs: Dict[str, Any], seeds: list = [42, 123, 456]):
        """测试多个随机种子的噪声一致性"""
        print(f"\n🎲 测试多个随机种子的噪声一致性...")
        
        results = []
        
        for i, seed in enumerate(seeds):
            print(f"\n--- 测试种子 {i+1}/{len(seeds)}: {seed} ---")
            
            # 设置随机种子
            self.unified_random = UnifiedRandomGenerator(seed)
            
            # 测试 PyTorch
            pytorch_noise = self.test_pytorch_noise_initialization(pytorch_inputs)
            
            # 测试 MLX
            mlx_noise = self.test_mlx_noise_initialization(pytorch_inputs)
            
            # 比较结果
            is_consistent = self.compare_noise_outputs(pytorch_noise, mlx_noise)
            
            results.append({
                'seed': seed,
                'pytorch_noise': pytorch_noise,
                'mlx_noise': mlx_noise,
                'is_consistent': is_consistent
            })
        
        # 总结结果
        consistent_count = sum(1 for r in results if r['is_consistent'])
        print(f"\n📊 多种子测试结果:")
        print(f"   总测试数: {len(seeds)}")
        print(f"   一致数量: {consistent_count}")
        print(f"   一致率: {consistent_count/len(seeds)*100:.1f}%")
        
        return results
    
    def save_test_results(self, pytorch_inputs: Dict, pytorch_noise: torch.Tensor, 
                         mlx_noise: mx.array, is_consistent: bool, test_type: str = "single"):
        """保存测试结果"""
        print(f"\n💾 保存测试结果...")
        
        os.makedirs("cfm_noise_test", exist_ok=True)
        
        # 保存测试数据
        test_data = {
            'pytorch_inputs': pytorch_inputs,
            'pytorch_noise': pytorch_noise,
            'mlx_noise': mlx_noise,
            'is_consistent': is_consistent,
            'test_type': test_type,
            'timestamp': time.time()
        }
        
        filename = f"cfm_noise_test/noise_test_{test_type}_{int(time.time())}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(test_data, f)
        
        print(f"✅ 测试结果已保存到: {filename}")
    
    def run_noise_initialization_test(self, filename: str = None, test_multiple_seeds: bool = False):
        """运行噪声初始化测试"""
        print(f"🧪 CFM 噪声初始化一致性测试")
        print(f"=" * 60)
        
        # 1. 加载 PyTorch 输入
        pytorch_inputs = self.load_pytorch_inputs(filename)
        
        # 2. 设置统一随机数生成器
        self.setup_unified_random(42)
        
        if test_multiple_seeds:
            # 3. 测试多个随机种子
            results = self.test_multiple_seeds(pytorch_inputs)
            
            # 4. 保存结果
            self.save_test_results(pytorch_inputs, None, None, 
                                 all(r['is_consistent'] for r in results), "multiple")
            
            return all(r['is_consistent'] for r in results)
        else:
            # 3. 测试 PyTorch 噪声初始化
            pytorch_noise = self.test_pytorch_noise_initialization(pytorch_inputs)
            
            # 4. 测试 MLX 噪声初始化
            mlx_noise = self.test_mlx_noise_initialization(pytorch_inputs)
            
            # 5. 比较输出
            is_consistent = self.compare_noise_outputs(pytorch_noise, mlx_noise)
            
            # 6. 保存结果
            self.save_test_results(pytorch_inputs, pytorch_noise, mlx_noise, is_consistent, "single")
            
            return is_consistent
    
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
    print("🔧 CFM 噪声初始化一致性测试")
    print("=" * 60)
    
    tester = CFMNoiseInitializationTest()
    
    # 列出可用输入
    files = tester.list_available_inputs()
    
    if not files:
        print("❌ 没有找到 PyTorch 输入缓存文件")
        print("请先运行 benchmark_cfm_caching.py 生成缓存数据")
        return
    
    # 运行单种子测试
    print(f"\n🚀 开始单种子测试...")
    success = tester.run_noise_initialization_test(test_multiple_seeds=False)
    
    if success:
        print(f"\n🎉 单种子测试通过! PyTorch 和 MLX 噪声初始化一致")
    else:
        print(f"\n❌ 单种子测试失败! PyTorch 和 MLX 噪声初始化不一致")
    
    # 运行多种子测试
    print(f"\n🚀 开始多种子测试...")
    success = tester.run_noise_initialization_test(test_multiple_seeds=True)
    
    if success:
        print(f"\n🎉 多种子测试通过! PyTorch 和 MLX 噪声初始化一致")
    else:
        print(f"\n❌ 多种子测试失败! PyTorch 和 MLX 噪声初始化不一致")
    
    print(f"\n🔧 建议检查:")
    print(f"   - 随机数生成器实现")
    print(f"   - 种子设置时机")
    print(f"   - 数值精度差异")


if __name__ == "__main__":
    main()
