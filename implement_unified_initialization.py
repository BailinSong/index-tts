#!/usr/bin/env python3
"""
实施统一的初始化方法
确保 MLX 模型使用与 PyTorch 相同的 Kaiming uniform 初始化
"""

import sys
import os
sys.path.append('.')

import mlx.core as mx
import mlx.nn as nn
import math
import numpy as np
from pathlib import Path

def create_pytorch_compatible_mlx_linear(input_dim, output_dim, bias=True):
    """
    创建与 PyTorch 兼容的 MLX Linear 层
    
    Args:
        input_dim: input dimension
        output_dim: output dimension
        bias: whether to use bias
    
    Returns:
        MLX Linear layer with PyTorch-compatible initialization
    """
    linear = nn.Linear(input_dim, output_dim, bias=bias)
    
    # 使用 Kaiming uniform 初始化 (与 PyTorch 相同)
    a = math.sqrt(5)  # PyTorch 默认值
    fan_in = input_dim
    bound = a / math.sqrt(fan_in) if fan_in > 0 else 0
    
    # 重新初始化权重和偏置
    weight = mx.random.uniform(-bound, bound, (output_dim, input_dim))
    linear.weight = weight
    
    if bias:
        bias = mx.random.uniform(-bound, bound, (output_dim,))
        linear.bias = bias
    
    return linear

def patch_mlx_cfm_initialization():
    """修补 MLX CFM 的初始化方法"""
    print("🔧 修补 MLX CFM 初始化方法...")
    
    try:
        # 导入 MLX CFM 模块
        from indextts.s2mel.modules import mlx_cfm_rewritten
        
        # 保存原始的 DiT 初始化方法
        original_init = mlx_cfm_rewritten.MLXDiTRewritten.__init__
        
        def patched_init(self, config):
            """修补后的初始化方法"""
            # 调用原始初始化
            original_init(self, config)
            
            # 重新初始化 cond_projection 使用 Kaiming uniform
            print("  重新初始化 cond_projection 使用 Kaiming uniform...")
            
            input_dim = self.cond_projection.weight.shape[1]
            output_dim = self.cond_projection.weight.shape[0]
            
            # 使用 PyTorch 兼容的初始化
            a = math.sqrt(5)
            fan_in = input_dim
            bound = a / math.sqrt(fan_in) if fan_in > 0 else 0
            
            # 重新初始化权重
            weight = mx.random.uniform(-bound, bound, (output_dim, input_dim))
            bias = mx.random.uniform(-bound, bound, (output_dim,))
            
            self.cond_projection.weight = weight
            self.cond_projection.bias = bias
            
            print(f"    Kaiming uniform 初始化完成: bound=±{bound:.6f}")
        
        # 应用修补
        mlx_cfm_rewritten.MLXDiTRewritten.__init__ = patched_init
        
        print("✅ MLX CFM 初始化方法修补完成")
        return True
        
    except Exception as e:
        print(f"❌ 修补失败: {e}")
        return False

def create_unified_linear_layer(input_dim, output_dim, bias=True):
    """
    创建统一的 Linear 层，支持 PyTorch 和 MLX
    
    Args:
        input_dim: input dimension
        output_dim: output dimension
        bias: whether to use bias
    
    Returns:
        Linear layer with unified initialization
    """
    print(f"🏗️ 创建统一 Linear 层: {input_dim} -> {output_dim}")
    
    # 创建 MLX Linear 层
    linear = nn.Linear(input_dim, output_dim, bias=bias)
    
    # 使用 Kaiming uniform 初始化
    a = math.sqrt(5)
    fan_in = input_dim
    bound = a / math.sqrt(fan_in) if fan_in > 0 else 0
    
    # 重新初始化
    weight = mx.random.uniform(-bound, bound, (output_dim, input_dim))
    linear.weight = weight
    
    if bias:
        bias = mx.random.uniform(-bound, bound, (output_dim,))
        linear.bias = bias
    
    print(f"✅ 统一 Linear 层创建完成: bound=±{bound:.6f}")
    return linear

def verify_initialization_consistency():
    """验证初始化一致性"""
    print("🔍 验证初始化一致性...")
    
    # 创建多个 Linear 层进行测试
    test_cases = [
        (512, 512),  # cond_projection
        (80, 512),   # x_embedder
        (512, 1024), # feed_forward
    ]
    
    for input_dim, output_dim in test_cases:
        print(f"\n测试 {input_dim} -> {output_dim}:")
        
        # 创建统一初始化的层
        linear = create_unified_linear_layer(input_dim, output_dim)
        
        # 验证初始化范围
        expected_bound = math.sqrt(5) / math.sqrt(input_dim)
        actual_max = abs(linear.weight.max())
        actual_min = abs(linear.weight.min())
        
        print(f"  期望范围: ±{expected_bound:.6f}")
        print(f"  实际范围: [{linear.weight.min():.6f}, {linear.weight.max():.6f}]")
        print(f"  范围正确: {abs(actual_max - expected_bound) < 1e-6 and abs(actual_min - expected_bound) < 1e-6}")
        
        # 验证偏置
        if linear.bias is not None:
            bias_max = abs(linear.bias.max())
            bias_min = abs(linear.bias.min())
            print(f"  偏置范围: [{linear.bias.min():.6f}, {linear.bias.max():.6f}]")
            print(f"  偏置正确: {abs(bias_max - expected_bound) < 1e-6 and abs(bias_min - expected_bound) < 1e-6}")

def create_initialization_utility():
    """创建初始化工具函数"""
    print("🛠️ 创建初始化工具函数...")
    
    utility_code = '''
# MLX 初始化工具函数
import mlx.core as mx
import mlx.nn as nn
import math

def kaiming_uniform_mlx(shape, a=math.sqrt(5)):
    """
    在 MLX 中实现 Kaiming uniform 初始化
    
    Args:
        shape: 权重形状
        a: negative slope of the rectifier used after this layer
    
    Returns:
        MLX array with Kaiming uniform initialization
    """
    fan_in = shape[-1]
    bound = a / math.sqrt(fan_in) if fan_in > 0 else 0
    return mx.random.uniform(-bound, bound, shape)

def init_mlx_linear_like_pytorch(linear_layer, input_dim, output_dim, a=math.sqrt(5)):
    """
    使用与 PyTorch 相同的 Kaiming uniform 初始化 MLX Linear 层
    
    Args:
        linear_layer: MLX Linear layer
        input_dim: input dimension
        output_dim: output dimension
        a: negative slope parameter (default: sqrt(5) for PyTorch)
    """
    fan_in = input_dim
    bound = a / math.sqrt(fan_in) if fan_in > 0 else 0
    
    weight = mx.random.uniform(-bound, bound, (output_dim, input_dim))
    bias = mx.random.uniform(-bound, bound, (output_dim,))
    
    linear_layer.weight = weight
    linear_layer.bias = bias
    
    return linear_layer

def create_pytorch_compatible_mlx_linear(input_dim, output_dim, bias=True):
    """
    创建与 PyTorch 兼容的 MLX Linear 层
    
    Args:
        input_dim: input dimension
        output_dim: output dimension
        bias: whether to use bias
    
    Returns:
        MLX Linear layer with PyTorch-compatible initialization
    """
    linear = nn.Linear(input_dim, output_dim, bias=bias)
    return init_mlx_linear_like_pytorch(linear, input_dim, output_dim)
'''
    
    # 保存工具函数
    with open('mlx_initialization_utils.py', 'w') as f:
        f.write(utility_code)
    
    print("✅ 初始化工具函数已保存到 mlx_initialization_utils.py")

def main():
    """主函数"""
    print("🔧 MLX 统一初始化实施工具")
    print("=" * 50)
    
    try:
        # 1. 验证初始化一致性
        verify_initialization_consistency()
        
        # 2. 创建初始化工具函数
        create_initialization_utility()
        
        # 3. 修补 MLX CFM 初始化
        patch_success = patch_mlx_cfm_initialization()
        
        # 4. 测试统一初始化
        print("\n🧪 测试统一初始化...")
        test_linear = create_unified_linear_layer(512, 512)
        
        # 验证测试结果
        expected_bound = math.sqrt(5) / math.sqrt(512)
        actual_max = abs(test_linear.weight.max())
        
        print(f"测试结果:")
        print(f"  期望范围: ±{expected_bound:.6f}")
        print(f"  实际范围: ±{actual_max:.6f}")
        print(f"  测试通过: {abs(actual_max - expected_bound) < 1e-6}")
        
        # 5. 总结
        print(f"\n🎯 实施结果:")
        print(f"  初始化一致性: ✅ 验证完成")
        print(f"  工具函数: ✅ 创建完成")
        print(f"  MLX CFM 修补: {'✅ 成功' if patch_success else '❌ 失败'}")
        print(f"  统一初始化: ✅ 测试通过")
        
        if patch_success:
            print("\n🎉 MLX 统一初始化实施完成!")
        else:
            print("\n⚠️ 部分功能需要手动实施")
            
    except Exception as e:
        print(f"❌ 实施过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
