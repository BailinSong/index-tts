#!/usr/bin/env python3
"""
cond_projection 实现差异分析
详细对比 PyTorch 和 MLX 中 cond_projection 层的实现差异
"""

import torch
import mlx.core as mx
import numpy as np
import pickle
import os
from pathlib import Path

def load_cached_inputs():
    """加载缓存的输入数据"""
    cache_file = "cfm_debug_outputs/correct_cached_inputs.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ 加载缓存输入数据: {cache_file}")
        return data
    else:
        raise FileNotFoundError(f"缓存文件不存在: {cache_file}")

def analyze_cond_projection_weights():
    """分析 cond_projection 权重差异"""
    print("🔍 cond_projection 权重分析")
    print("=" * 50)
    
    # 加载模型
    from indextts.s2mel.modules.flow_matching import CFM
    from indextts.s2mel.modules.mlx_cfm import MLXCFM
    import yaml
    
    # 加载配置
    with open("checkpoints/config.yaml", 'r') as f:
        config_dict = yaml.safe_load(f)
    
    class SimpleConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    setattr(self, key, SimpleConfig(value))
                else:
                    setattr(self, key, value)
        
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    config = SimpleConfig(config_dict['s2mel'])
    
    # PyTorch 模型
    pytorch_cfm = CFM(config)
    pytorch_cfm.eval()
    
    # MLX 模型
    mlx_cfm = MLXCFM(config)
    
    # 获取 cond_projection 层
    pytorch_cond_proj = pytorch_cfm.estimator.cond_projection
    mlx_cond_proj = mlx_cfm.estimator.cond_projection
    
    print(f"📊 PyTorch cond_projection:")
    print(f"   类型: {type(pytorch_cond_proj)}")
    print(f"   权重形状: {pytorch_cond_proj.weight.shape}")
    print(f"   偏置形状: {pytorch_cond_proj.bias.shape}")
    print(f"   权重范围: [{pytorch_cond_proj.weight.min():.6f}, {pytorch_cond_proj.weight.max():.6f}]")
    print(f"   偏置范围: [{pytorch_cond_proj.bias.min():.6f}, {pytorch_cond_proj.bias.max():.6f}]")
    
    print(f"\n📊 MLX cond_projection:")
    print(f"   类型: {type(mlx_cond_proj)}")
    print(f"   权重形状: {mlx_cond_proj.weight.shape}")
    print(f"   偏置形状: {mlx_cond_proj.bias.shape}")
    print(f"   权重范围: [{float(mlx_cond_proj.weight.min()):.6f}, {float(mlx_cond_proj.weight.max()):.6f}]")
    print(f"   偏置范围: [{float(mlx_cond_proj.bias.min()):.6f}, {float(mlx_cond_proj.bias.max()):.6f}]")
    
    # 对比权重
    pytorch_weight = pytorch_cond_proj.weight.detach().cpu().numpy()
    mlx_weight = np.array(mlx_cond_proj.weight)
    
    weight_diff = np.abs(pytorch_weight - mlx_weight)
    print(f"\n🔍 权重差异分析:")
    print(f"   最大差异: {np.max(weight_diff):.8f}")
    print(f"   平均差异: {np.mean(weight_diff):.8f}")
    print(f"   相对差异: {np.mean(weight_diff) / np.mean(np.abs(pytorch_weight)):.8f}")
    
    # 对比偏置
    pytorch_bias = pytorch_cond_proj.bias.detach().cpu().numpy()
    mlx_bias = np.array(mlx_cond_proj.bias)
    
    bias_diff = np.abs(pytorch_bias - mlx_bias)
    print(f"\n🔍 偏置差异分析:")
    print(f"   最大差异: {np.max(bias_diff):.8f}")
    print(f"   平均差异: {np.mean(bias_diff):.8f}")
    print(f"   相对差异: {np.mean(bias_diff) / np.mean(np.abs(pytorch_bias)):.8f}")
    
    return pytorch_cond_proj, mlx_cond_proj

def test_cond_projection_forward():
    """测试 cond_projection 前向传播"""
    print("\n🔍 cond_projection 前向传播测试")
    print("=" * 50)
    
    # 加载缓存数据
    cached_data = load_cached_inputs()
    mu = cached_data['mu']
    mu_mlx = mx.array(mu.numpy())
    
    # 获取模型
    from indextts.s2mel.modules.flow_matching import CFM
    from indextts.s2mel.modules.mlx_cfm import MLXCFM
    import yaml
    
    with open("checkpoints/config.yaml", 'r') as f:
        config_dict = yaml.safe_load(f)
    
    class SimpleConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    setattr(self, key, SimpleConfig(value))
                else:
                    setattr(self, key, value)
        
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    config = SimpleConfig(config_dict['s2mel'])
    
    pytorch_cfm = CFM(config)
    pytorch_cfm.eval()
    mlx_cfm = MLXCFM(config)
    
    # 测试前向传播
    print(f"📊 输入数据:")
    print(f"   mu: {mu.shape}, range: [{mu.min():.6f}, {mu.max():.6f}]")
    print(f"   mu_mlx: {mu_mlx.shape}, range: [{float(mu_mlx.min()):.6f}, {float(mu_mlx.max()):.6f}]")
    
    # PyTorch 前向传播
    with torch.no_grad():
        pytorch_output = pytorch_cfm.estimator.cond_projection(mu)
    
    # MLX 前向传播
    mlx_output = mlx_cfm.estimator.cond_projection(mu_mlx)
    
    print(f"\n📊 输出对比:")
    print(f"   PyTorch: {pytorch_output.shape}, range: [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]")
    print(f"   MLX: {mlx_output.shape}, range: [{float(mlx_output.min()):.6f}, {float(mlx_output.max()):.6f}]")
    
    # 计算差异
    pytorch_np = pytorch_output.detach().cpu().numpy()
    mlx_np = np.array(mlx_output)
    
    diff = np.abs(pytorch_np - mlx_np)
    print(f"\n🔍 输出差异分析:")
    print(f"   最大差异: {np.max(diff):.8f}")
    print(f"   平均差异: {np.mean(diff):.8f}")
    print(f"   相对差异: {np.mean(diff) / np.mean(np.abs(pytorch_np)):.8f}")
    
    # 逐元素分析
    print(f"\n🔍 逐元素差异分析:")
    print(f"   差异 > 1.0 的元素数: {np.sum(diff > 1.0)}")
    print(f"   差异 > 0.1 的元素数: {np.sum(diff > 0.1)}")
    print(f"   差异 > 0.01 的元素数: {np.sum(diff > 0.01)}")
    
    # 找出最大差异的位置
    max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"\n🔍 最大差异位置:")
    print(f"   位置: {max_diff_idx}")
    print(f"   PyTorch 值: {pytorch_np[max_diff_idx]:.8f}")
    print(f"   MLX 值: {mlx_np[max_diff_idx]:.8f}")
    print(f"   差异: {diff[max_diff_idx]:.8f}")
    
    return pytorch_output, mlx_output

def analyze_linear_layer_implementation():
    """分析 Linear 层实现差异"""
    print("\n🔍 Linear 层实现分析")
    print("=" * 50)
    
    # 创建简单的测试数据
    batch_size, seq_len, input_dim = 2, 50, 512
    output_dim = 512
    
    # 创建测试输入
    x_pytorch = torch.randn(batch_size, seq_len, input_dim)
    x_mlx = mx.array(x_pytorch.numpy())
    
    print(f"📊 测试输入:")
    print(f"   PyTorch: {x_pytorch.shape}, range: [{x_pytorch.min():.6f}, {x_pytorch.max():.6f}]")
    print(f"   MLX: {x_mlx.shape}, range: [{float(x_mlx.min()):.6f}, {float(x_mlx.max()):.6f}]")
    
    # 创建相同的权重和偏置
    weight = torch.randn(output_dim, input_dim)
    bias = torch.randn(output_dim)
    
    # PyTorch Linear
    pytorch_linear = torch.nn.Linear(input_dim, output_dim, bias=True)
    pytorch_linear.weight.data = weight
    pytorch_linear.bias.data = bias
    
    # MLX Linear
    from mlx.nn import Linear as MLXLinear
    mlx_linear = MLXLinear(input_dim, output_dim, bias=True)
    mlx_linear.weight = mx.array(weight.numpy())
    mlx_linear.bias = mx.array(bias.numpy())
    
    print(f"\n📊 权重对比:")
    print(f"   PyTorch weight: {pytorch_linear.weight.shape}, range: [{pytorch_linear.weight.min():.6f}, {pytorch_linear.weight.max():.6f}]")
    print(f"   MLX weight: {mlx_linear.weight.shape}, range: [{float(mlx_linear.weight.min()):.6f}, {float(mlx_linear.weight.max()):.6f}]")
    
    # 前向传播
    with torch.no_grad():
        pytorch_output = pytorch_linear(x_pytorch)
    
    mlx_output = mlx_linear(x_mlx)
    
    print(f"\n📊 输出对比:")
    print(f"   PyTorch: {pytorch_output.shape}, range: [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]")
    print(f"   MLX: {mlx_output.shape}, range: [{float(mlx_output.min()):.6f}, {float(mlx_output.max()):.6f}]")
    
    # 计算差异
    pytorch_np = pytorch_output.detach().cpu().numpy()
    mlx_np = np.array(mlx_output)
    
    diff = np.abs(pytorch_np - mlx_np)
    print(f"\n🔍 输出差异分析:")
    print(f"   最大差异: {np.max(diff):.8f}")
    print(f"   平均差异: {np.mean(diff):.8f}")
    print(f"   相对差异: {np.mean(diff) / np.mean(np.abs(pytorch_np)):.8f}")
    
    return pytorch_output, mlx_output

def check_weight_loading():
    """检查权重加载过程"""
    print("\n🔍 权重加载过程检查")
    print("=" * 50)
    
    # 检查 MLX 模型是否从 PyTorch 权重加载
    from indextts.s2mel.modules.mlx_cfm import MLXCFM
    import yaml
    
    with open("checkpoints/config.yaml", 'r') as f:
        config_dict = yaml.safe_load(f)
    
    class SimpleConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    setattr(self, key, SimpleConfig(value))
                else:
                    setattr(self, key, value)
        
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    config = SimpleConfig(config_dict['s2mel'])
    
    # 创建 MLX 模型
    mlx_cfm = MLXCFM(config)
    
    # 检查是否有权重加载
    print(f"📊 MLX 模型状态:")
    print(f"   是否从缓存加载: {hasattr(mlx_cfm, '_loaded_from_cache')}")
    
    # 检查 cond_projection 权重来源
    cond_proj = mlx_cfm.estimator.cond_projection
    print(f"\n📊 cond_projection 权重状态:")
    print(f"   权重是否已初始化: {hasattr(cond_proj, 'weight')}")
    if hasattr(cond_proj, 'weight'):
        print(f"   权重形状: {cond_proj.weight.shape}")
        print(f"   权重范围: [{float(cond_proj.weight.min()):.6f}, {float(cond_proj.weight.max()):.6f}]")
    
    # 尝试从 PyTorch 权重加载
    print(f"\n🔍 尝试从 PyTorch 权重加载:")
    try:
        from indextts.s2mel.modules.flow_matching import CFM
        pytorch_cfm = CFM(config)
        pytorch_cfm.eval()
        
        # 获取 PyTorch 权重
        pytorch_weight = pytorch_cfm.estimator.cond_projection.weight.detach().cpu().numpy()
        pytorch_bias = pytorch_cfm.estimator.cond_projection.bias.detach().cpu().numpy()
        
        # 加载到 MLX
        cond_proj.weight = mx.array(pytorch_weight)
        cond_proj.bias = mx.array(pytorch_bias)
        
        print(f"   ✅ 权重加载成功")
        print(f"   新权重范围: [{float(cond_proj.weight.min()):.6f}, {float(cond_proj.weight.max()):.6f}]")
        print(f"   新偏置范围: [{float(cond_proj.bias.min()):.6f}, {float(cond_proj.bias.max()):.6f}]")
        
        # 重新测试
        cached_data = load_cached_inputs()
        mu = cached_data['mu']
        mu_mlx = mx.array(mu.numpy())
        
        mlx_output = cond_proj(mu_mlx)
        pytorch_output = pytorch_cfm.estimator.cond_projection(mu)
        
        pytorch_np = pytorch_output.detach().cpu().numpy()
        mlx_np = np.array(mlx_output)
        
        diff = np.abs(pytorch_np - mlx_np)
        print(f"\n📊 加载权重后的输出差异:")
        print(f"   最大差异: {np.max(diff):.8f}")
        print(f"   平均差异: {np.mean(diff):.8f}")
        print(f"   相对差异: {np.mean(diff) / np.mean(np.abs(pytorch_np)):.8f}")
        
    except Exception as e:
        print(f"   ❌ 权重加载失败: {e}")

def main():
    """主函数"""
    print("🔍 cond_projection 实现差异分析")
    print("=" * 60)
    
    try:
        # 1. 分析权重差异
        pytorch_cond_proj, mlx_cond_proj = analyze_cond_projection_weights()
        
        # 2. 测试前向传播
        pytorch_output, mlx_output = test_cond_projection_forward()
        
        # 3. 分析 Linear 层实现
        analyze_linear_layer_implementation()
        
        # 4. 检查权重加载
        check_weight_loading()
        
        print(f"\n🎉 cond_projection 差异分析完成!")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
