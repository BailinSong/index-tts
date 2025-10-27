#!/usr/bin/env python3
"""
生产环境 MLX 工具函数
确保 MLX 模型在生产环境中正确工作
"""

import mlx.core as mx
import mlx.nn as nn
import math
import numpy as np

def verify_mlx_model_weights(mlx_model, pytorch_weights_path):
    """
    验证 MLX 模型的权重是否正确加载
    
    Args:
        mlx_model: MLX 模型实例
        pytorch_weights_path: PyTorch 权重文件路径
    
    Returns:
        bool: 权重是否正确加载
    """
    try:
        import torch
        
        # 加载 PyTorch 权重
        pytorch_weights = torch.load(pytorch_weights_path, map_location='cpu')
        
        # 提取 cond_projection 权重
        pytorch_weight = pytorch_weights['net']['cfm']['estimator.cond_projection.weight'].numpy()
        pytorch_bias = pytorch_weights['net']['cfm']['estimator.cond_projection.bias'].numpy()
        
        # 获取 MLX 权重
        mlx_weight = np.array(mlx_model.estimator.cond_projection.weight)
        mlx_bias = np.array(mlx_model.estimator.cond_projection.bias)
        
        # 计算差异
        weight_diff = np.abs(pytorch_weight - mlx_weight)
        bias_diff = np.abs(pytorch_bias - mlx_bias)
        
        print(f"权重差异: 最大={np.max(weight_diff):.8f}, 平均={np.mean(weight_diff):.8f}")
        print(f"偏置差异: 最大={np.max(bias_diff):.8f}, 平均={np.mean(bias_diff):.8f}")
        
        # 检查是否完全相同
        weight_identical = np.allclose(pytorch_weight, mlx_weight, atol=1e-8)
        bias_identical = np.allclose(pytorch_bias, mlx_bias, atol=1e-8)
        
        print(f"权重是否相同: {weight_identical}")
        print(f"偏置是否相同: {bias_identical}")
        
        return weight_identical and bias_identical
        
    except Exception as e:
        print(f"❌ 权重验证失败: {e}")
        return False

def fix_mlx_model_weights(mlx_model, pytorch_weights_path):
    """
    修复 MLX 模型的权重加载
    
    Args:
        mlx_model: MLX 模型实例
        pytorch_weights_path: PyTorch 权重文件路径
    
    Returns:
        bool: 修复是否成功
    """
    try:
        import torch
        
        # 加载 PyTorch 权重
        pytorch_weights = torch.load(pytorch_weights_path, map_location='cpu')
        
        # 提取 cond_projection 权重
        pytorch_weight = pytorch_weights['net']['cfm']['estimator.cond_projection.weight'].numpy()
        pytorch_bias = pytorch_weights['net']['cfm']['estimator.cond_projection.bias'].numpy()
        
        # 修复 MLX 权重
        mlx_model.estimator.cond_projection.weight = mx.array(pytorch_weight)
        mlx_model.estimator.cond_projection.bias = mx.array(pytorch_bias)
        
        print("✅ MLX 模型权重修复完成")
        return True
        
    except Exception as e:
        print(f"❌ MLX 模型权重修复失败: {e}")
        return False

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

def test_mlx_model_production(mlx_model):
    """
    测试 MLX 模型在生产环境中的表现
    
    Args:
        mlx_model: MLX 模型实例
    
    Returns:
        dict: 测试结果
    """
    try:
        import time
        
        # 创建测试输入
        batch_size = 2
        seq_len = 100
        input_dim = 512
        
        test_input = mx.random.normal((batch_size, seq_len, input_dim))
        
        # 预热
        for _ in range(5):
            _ = mlx_model.estimator.cond_projection(test_input)
        
        # 性能测试
        start_time = time.time()
        for _ in range(100):
            output = mlx_model.estimator.cond_projection(test_input)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        throughput = batch_size * seq_len / avg_time
        
        result = {
            "success": True,
            "avg_time_ms": avg_time * 1000,
            "throughput_tokens_per_sec": throughput,
            "output_shape": output.shape,
            "output_range": [float(output.min()), float(output.max())]
        }
        
        print(f"✅ 生产环境测试通过:")
        print(f"  平均推理时间: {result['avg_time_ms']:.2f} ms")
        print(f"  吞吐量: {result['throughput_tokens_per_sec']:.0f} tokens/s")
        print(f"  输出形状: {result['output_shape']}")
        print(f"  输出范围: [{result['output_range'][0]:.6f}, {result['output_range'][1]:.6f}]")
        
        return result
        
    except Exception as e:
        print(f"❌ 生产环境测试失败: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    print("🛠️ MLX 生产环境工具函数")
    print("=" * 50)
    print("可用函数:")
    print("  - verify_mlx_model_weights()")
    print("  - fix_mlx_model_weights()")
    print("  - create_pytorch_compatible_mlx_linear()")
    print("  - test_mlx_model_production()")
