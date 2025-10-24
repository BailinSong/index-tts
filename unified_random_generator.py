#!/usr/bin/env python3
"""
一致性随机数生成器
确保相同seed生成的随机数序列固定，且多实例相同seed生成序列相同
"""

import numpy as np
import torch
import mlx.core as mx
from typing import Union, Tuple, Optional

class UnifiedRandomGenerator:
    """一致性随机数生成器"""
    
    def __init__(self, seed: int = 42):
        """
        初始化随机数生成器
        
        Args:
            seed: 随机种子
        """
        self.seed = seed
        self.np_generator = np.random.RandomState(seed)
        self.torch_generator = torch.Generator()
        self.torch_generator.manual_seed(seed)
        
        # 设置全局种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        mx.random.seed(seed)
        
        print(f"UnifiedRandomGenerator initialized with seed: {seed}")
    
    def reset_seed(self, seed: int):
        """重置种子"""
        self.seed = seed
        self.np_generator = np.random.RandomState(seed)
        self.torch_generator = torch.Generator()
        self.torch_generator.manual_seed(seed)
        
        # 设置全局种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        mx.random.seed(seed)
        
        print(f"UnifiedRandomGenerator reset to seed: {seed}")
    
    def generate_noise(self, shape: Tuple[int, ...], device: str = 'cpu') -> torch.Tensor:
        """
        生成一致性噪声
        
        Args:
            shape: 噪声形状
            device: 设备
            
        Returns:
            噪声张量
        """
        # 使用numpy生成基础随机数
        noise_np = self.np_generator.randn(*shape).astype(np.float32)
        
        # 转换为PyTorch张量
        noise_torch = torch.from_numpy(noise_np).to(device)
        
        return noise_torch
    
    def generate_noise_mlx(self, shape: Tuple[int, ...]) -> mx.array:
        """
        生成一致性MLX噪声
        
        Args:
            shape: 噪声形状
            
        Returns:
            MLX噪声数组
        """
        # 使用numpy生成基础随机数
        noise_np = self.np_generator.randn(*shape).astype(np.float32)
        
        # 转换为MLX数组
        noise_mlx = mx.array(noise_np)
        
        return noise_mlx
    
    def generate_noise_like(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        生成与给定张量形状相同的一致性噪声
        
        Args:
            tensor: 参考张量
            
        Returns:
            噪声张量
        """
        # 使用numpy生成基础随机数
        noise_np = self.np_generator.randn(*tensor.shape).astype(np.float32)
        
        # 转换为PyTorch张量
        noise_torch = torch.from_numpy(noise_np).to(tensor.device, dtype=tensor.dtype)
        
        return noise_torch
    
    def generate_uniform(self, shape: Tuple[int, ...], low: float = 0.0, high: float = 1.0, device: str = 'cpu') -> torch.Tensor:
        """
        生成一致性均匀分布随机数
        
        Args:
            shape: 形状
            low: 最小值
            high: 最大值
            device: 设备
            
        Returns:
            均匀分布张量
        """
        # 使用numpy生成基础随机数
        uniform_np = self.np_generator.uniform(low, high, shape).astype(np.float32)
        
        # 转换为PyTorch张量
        uniform_torch = torch.from_numpy(uniform_np).to(device)
        
        return uniform_torch
    
    def generate_uniform_mlx(self, shape: Tuple[int, ...], low: float = 0.0, high: float = 1.0) -> mx.array:
        """
        生成一致性MLX均匀分布随机数
        
        Args:
            shape: 形状
            low: 最小值
            high: 最大值
            
        Returns:
            MLX均匀分布数组
        """
        # 使用numpy生成基础随机数
        uniform_np = self.np_generator.uniform(low, high, shape).astype(np.float32)
        
        # 转换为MLX数组
        uniform_mlx = mx.array(uniform_np)
        
        return uniform_mlx
    
    def generate_integers(self, low: int, high: int, size: int) -> np.ndarray:
        """
        生成一致性整数随机数
        
        Args:
            low: 最小值
            high: 最大值
            size: 数量
            
        Returns:
            整数数组
        """
        return self.np_generator.randint(low, high, size)
    
    def generate_choice(self, a: Union[int, np.ndarray], size: int, replace: bool = True) -> np.ndarray:
        """
        生成一致性选择随机数
        
        Args:
            a: 选择范围
            size: 数量
            replace: 是否替换
            
        Returns:
            选择数组
        """
        return self.np_generator.choice(a, size, replace=replace)
    
    def generate_permutation(self, n: int) -> np.ndarray:
        """
        生成一致性排列
        
        Args:
            n: 排列长度
            
        Returns:
            排列数组
        """
        return self.np_generator.permutation(n)
    
    def generate_shuffle(self, arr: np.ndarray) -> np.ndarray:
        """
        生成一致性打乱
        
        Args:
            arr: 输入数组
            
        Returns:
            打乱后的数组
        """
        shuffled = arr.copy()
        self.np_generator.shuffle(shuffled)
        return shuffled
    
    def get_state(self) -> dict:
        """获取随机数生成器状态"""
        return {
            'seed': self.seed,
            'np_state': self.np_generator.get_state(),
            'torch_state': self.torch_generator.get_state()
        }
    
    def set_state(self, state: dict):
        """设置随机数生成器状态"""
        self.seed = state['seed']
        self.np_generator.set_state(state['np_state'])
        self.torch_generator.set_state(state['torch_state'])
    
    def __repr__(self):
        return f"UnifiedRandomGenerator(seed={self.seed})"


def test_unified_random_generator():
    """测试一致性随机数生成器"""
    
    print("=== 测试一致性随机数生成器 ===")
    
    # 测试1: 相同seed生成相同序列
    print("\\n1. 测试相同seed生成相同序列")
    
    gen1 = UnifiedRandomGenerator(seed=42)
    gen2 = UnifiedRandomGenerator(seed=42)
    
    # 生成噪声
    noise1 = gen1.generate_noise((2, 3, 4))
    noise2 = gen2.generate_noise((2, 3, 4))
    
    print(f"噪声1: {noise1}")
    print(f"噪声2: {noise2}")
    print(f"是否相同: {torch.allclose(noise1, noise2)}")
    
    # 测试2: 不同seed生成不同序列
    print("\\n2. 测试不同seed生成不同序列")
    
    gen3 = UnifiedRandomGenerator(seed=123)
    noise3 = gen3.generate_noise((2, 3, 4))
    
    print(f"噪声1 (seed=42): {noise1}")
    print(f"噪声3 (seed=123): {noise3}")
    print(f"是否不同: {not torch.allclose(noise1, noise3)}")
    
    # 测试3: 多实例相同seed
    print("\\n3. 测试多实例相同seed")
    
    gen4 = UnifiedRandomGenerator(seed=42)
    gen5 = UnifiedRandomGenerator(seed=42)
    
    noise4 = gen4.generate_noise((2, 3, 4))
    noise5 = gen5.generate_noise((2, 3, 4))
    
    print(f"噪声4: {noise4}")
    print(f"噪声5: {noise5}")
    print(f"是否相同: {torch.allclose(noise4, noise5)}")
    
    # 测试4: 序列一致性
    print("\\n4. 测试序列一致性")
    
    gen6 = UnifiedRandomGenerator(seed=42)
    gen7 = UnifiedRandomGenerator(seed=42)
    
    # 生成多个随机数
    seq1 = []
    seq2 = []
    
    for i in range(5):
        seq1.append(gen6.generate_noise((1, 1, 1)).item())
        seq2.append(gen7.generate_noise((1, 1, 1)).item())
    
    print(f"序列1: {seq1}")
    print(f"序列2: {seq2}")
    print(f"序列是否相同: {seq1 == seq2}")
    
    # 测试5: MLX兼容性
    print("\\n5. 测试MLX兼容性")
    
    gen8 = UnifiedRandomGenerator(seed=42)
    gen9 = UnifiedRandomGenerator(seed=42)
    
    noise_mlx1 = gen8.generate_noise_mlx((2, 3, 4))
    noise_mlx2 = gen9.generate_noise_mlx((2, 3, 4))
    
    print(f"MLX噪声1: {noise_mlx1}")
    print(f"MLX噪声2: {noise_mlx2}")
    print(f"是否相同: {mx.allclose(noise_mlx1, noise_mlx2)}")
    
    print("\\n✅ 一致性随机数生成器测试完成")


if __name__ == "__main__":
    test_unified_random_generator()
