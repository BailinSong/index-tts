#!/usr/bin/env python3
"""
timestep_embedding 实现差异分析
详细对比 PyTorch 和 MLX 中 timestep_embedding 层的实现差异
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

def analyze_timestep_embedding_weights():
    """分析 timestep_embedding 权重差异"""
    print("🔍 timestep_embedding 权重分析")
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
    
    # 获取 timestep_embedding 层
    pytorch_t_embedder = pytorch_cfm.estimator.t_embedder
    mlx_t_embedder = mlx_cfm.estimator.t_embedder
    
    print(f"📊 PyTorch timestep_embedding:")
    print(f"   类型: {type(pytorch_t_embedder)}")
    print(f"   MLP 层数: {len(pytorch_t_embedder.mlp)}")
    
    # 分析 MLP 权重
    for i, layer in enumerate(pytorch_t_embedder.mlp):
        if hasattr(layer, 'weight'):
            print(f"   层 {i} 权重形状: {layer.weight.shape}")
            print(f"   层 {i} 权重范围: [{layer.weight.min():.6f}, {layer.weight.max():.6f}]")
        if hasattr(layer, 'bias') and layer.bias is not None:
            print(f"   层 {i} 偏置形状: {layer.bias.shape}")
            print(f"   层 {i} 偏置范围: [{layer.bias.min():.6f}, {layer.bias.max():.6f}]")
    
    print(f"\n📊 MLX timestep_embedding:")
    print(f"   类型: {type(mlx_t_embedder)}")
    
    # 分析 MLX MLP 权重
    print(f"   mlp_0 权重形状: {mlx_t_embedder.mlp_0.weight.shape}")
    print(f"   mlp_0 权重范围: [{float(mlx_t_embedder.mlp_0.weight.min()):.6f}, {float(mlx_t_embedder.mlp_0.weight.max()):.6f}]")
    print(f"   mlp_0 偏置形状: {mlx_t_embedder.mlp_0.bias.shape}")
    print(f"   mlp_0 偏置范围: [{float(mlx_t_embedder.mlp_0.bias.min()):.6f}, {float(mlx_t_embedder.mlp_0.bias.max()):.6f}]")
    
    print(f"   mlp_2 权重形状: {mlx_t_embedder.mlp_2.weight.shape}")
    print(f"   mlp_2 权重范围: [{float(mlx_t_embedder.mlp_2.weight.min()):.6f}, {float(mlx_t_embedder.mlp_2.weight.max()):.6f}]")
    print(f"   mlp_2 偏置形状: {mlx_t_embedder.mlp_2.bias.shape}")
    print(f"   mlp_2 偏置范围: [{float(mlx_t_embedder.mlp_2.bias.min()):.6f}, {float(mlx_t_embedder.mlp_2.bias.max()):.6f}]")
    
    # 对比权重
    pytorch_weights = []
    mlx_weights = []
    
    # 对比 mlp_0 (对应 PyTorch 的 mlp[0])
    pytorch_weights.append(pytorch_t_embedder.mlp[0].weight.detach().cpu().numpy())
    mlx_weights.append(np.array(mlx_t_embedder.mlp_0.weight))
    
    # 对比 mlp_2 (对应 PyTorch 的 mlp[2])
    pytorch_weights.append(pytorch_t_embedder.mlp[2].weight.detach().cpu().numpy())
    mlx_weights.append(np.array(mlx_t_embedder.mlp_2.weight))
    
    print(f"\n🔍 权重差异分析:")
    layer_names = ["mlp_0", "mlp_2"]
    for i, (pytorch_w, mlx_w) in enumerate(zip(pytorch_weights, mlx_weights)):
        weight_diff = np.abs(pytorch_w - mlx_w)
        print(f"   {layer_names[i]}:")
        print(f"     最大差异: {np.max(weight_diff):.8f}")
        print(f"     平均差异: {np.mean(weight_diff):.8f}")
        print(f"     相对差异: {np.mean(weight_diff) / np.mean(np.abs(pytorch_w)):.8f}")
    
    return pytorch_t_embedder, mlx_t_embedder

def test_timestep_embedding_forward():
    """测试 timestep_embedding 前向传播"""
    print("\n🔍 timestep_embedding 前向传播测试")
    print("=" * 50)
    
    # 创建测试时间步
    t_values = [0.0, 0.333333, 0.666667, 1.0]
    
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
    
    print(f"📊 测试时间步: {t_values}")
    
    for t_val in t_values:
        print(f"\n🔍 时间步 t = {t_val:.6f}:")
        
        # PyTorch 前向传播
        t_pytorch = torch.tensor([t_val])
        with torch.no_grad():
            pytorch_output = pytorch_cfm.estimator.t_embedder(t_pytorch)
        
        # MLX 前向传播
        t_mlx = mx.array([t_val])
        mlx_output = mlx_cfm.estimator.t_embedder(t_mlx)
        
        print(f"   PyTorch: {pytorch_output.shape}, range: [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]")
        print(f"   MLX: {mlx_output.shape}, range: [{float(mlx_output.min()):.6f}, {float(mlx_output.max()):.6f}]")
        
        # 计算差异
        pytorch_np = pytorch_output.detach().cpu().numpy()
        mlx_np = np.array(mlx_output)
        
        diff = np.abs(pytorch_np - mlx_np)
        print(f"   最大差异: {np.max(diff):.8f}")
        print(f"   平均差异: {np.mean(diff):.8f}")
        print(f"   相对差异: {np.mean(diff) / np.mean(np.abs(pytorch_np)):.8f}")

def analyze_timestep_embedding_implementation():
    """分析 timestep_embedding 实现差异"""
    print("\n🔍 timestep_embedding 实现分析")
    print("=" * 50)
    
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
    
    pytorch_t_embedder = pytorch_cfm.estimator.t_embedder
    mlx_t_embedder = mlx_cfm.estimator.t_embedder
    
    print(f"📊 PyTorch TimestepEmbedder 属性:")
    print(f"   frequency_embedding_size: {pytorch_t_embedder.frequency_embedding_size}")
    print(f"   max_period: {pytorch_t_embedder.max_period}")
    print(f"   scale: {pytorch_t_embedder.scale}")
    print(f"   freqs 形状: {pytorch_t_embedder.freqs.shape}")
    print(f"   freqs 范围: [{pytorch_t_embedder.freqs.min():.6f}, {pytorch_t_embedder.freqs.max():.6f}]")
    
    print(f"\n📊 MLX TimestepEmbedder 属性:")
    print(f"   frequency_embedding_size: {mlx_t_embedder.frequency_embedding_size}")
    print(f"   max_period: {mlx_t_embedder.max_period}")
    print(f"   scale: {mlx_t_embedder.scale}")
    print(f"   freqs 形状: {mlx_t_embedder.freqs.shape}")
    print(f"   freqs 范围: [{float(mlx_t_embedder.freqs.min()):.6f}, {float(mlx_t_embedder.freqs.max()):.6f}]")
    
    # 对比 freqs
    pytorch_freqs = pytorch_t_embedder.freqs.detach().cpu().numpy()
    mlx_freqs = np.array(mlx_t_embedder.freqs)
    
    freqs_diff = np.abs(pytorch_freqs - mlx_freqs)
    print(f"\n🔍 freqs 差异分析:")
    print(f"   最大差异: {np.max(freqs_diff):.8f}")
    print(f"   平均差异: {np.mean(freqs_diff):.8f}")
    print(f"   相对差异: {np.mean(freqs_diff) / np.mean(np.abs(pytorch_freqs)):.8f}")

def test_timestep_embedding_step_by_step():
    """逐步测试 timestep_embedding"""
    print("\n🔍 timestep_embedding 逐步测试")
    print("=" * 50)
    
    # 测试时间步
    t_val = 0.5
    t_pytorch = torch.tensor([t_val])
    t_mlx = mx.array([t_val])
    
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
    
    pytorch_t_embedder = pytorch_cfm.estimator.t_embedder
    mlx_t_embedder = mlx_cfm.estimator.t_embedder
    
    print(f"📊 测试时间步 t = {t_val}")
    
    # 1. 测试 timestep_embedding 方法
    print(f"\n🔍 1. timestep_embedding 方法:")
    
    # PyTorch
    with torch.no_grad():
        pytorch_freq_emb = pytorch_t_embedder.timestep_embedding(t_pytorch)
    
    # MLX
    mlx_freq_emb = mlx_t_embedder.timestep_embedding(t_mlx)
    
    print(f"   PyTorch: {pytorch_freq_emb.shape}, range: [{pytorch_freq_emb.min():.6f}, {pytorch_freq_emb.max():.6f}]")
    print(f"   MLX: {mlx_freq_emb.shape}, range: [{float(mlx_freq_emb.min()):.6f}, {float(mlx_freq_emb.max()):.6f}]")
    
    # 计算差异
    pytorch_np = pytorch_freq_emb.detach().cpu().numpy()
    mlx_np = np.array(mlx_freq_emb)
    
    diff = np.abs(pytorch_np - mlx_np)
    print(f"   最大差异: {np.max(diff):.8f}")
    print(f"   平均差异: {np.mean(diff):.8f}")
    print(f"   相对差异: {np.mean(diff) / np.mean(np.abs(pytorch_np)):.8f}")
    
    # 2. 测试完整前向传播
    print(f"\n🔍 2. 完整前向传播:")
    
    # PyTorch
    with torch.no_grad():
        pytorch_output = pytorch_t_embedder(t_pytorch)
    
    # MLX
    mlx_output = mlx_t_embedder(t_mlx)
    
    print(f"   PyTorch: {pytorch_output.shape}, range: [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]")
    print(f"   MLX: {mlx_output.shape}, range: [{float(mlx_output.min()):.6f}, {float(mlx_output.max()):.6f}]")
    
    # 计算差异
    pytorch_np = pytorch_output.detach().cpu().numpy()
    mlx_np = np.array(mlx_output)
    
    diff = np.abs(pytorch_np - mlx_np)
    print(f"   最大差异: {np.max(diff):.8f}")
    print(f"   平均差异: {np.mean(diff):.8f}")
    print(f"   相对差异: {np.mean(diff) / np.mean(np.abs(pytorch_np)):.8f}")

def check_timestep_embedding_weight_loading():
    """检查 timestep_embedding 权重加载"""
    print("\n🔍 timestep_embedding 权重加载检查")
    print("=" * 50)
    
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
    
    pytorch_t_embedder = pytorch_cfm.estimator.t_embedder
    mlx_t_embedder = mlx_cfm.estimator.t_embedder
    
    print(f"📊 尝试从 PyTorch 权重加载:")
    try:
        # 加载 MLP 权重
        # mlp_0 (对应 PyTorch 的 mlp[0])
        pytorch_weight = pytorch_t_embedder.mlp[0].weight.detach().cpu().numpy()
        mlx_t_embedder.mlp_0.weight = mx.array(pytorch_weight)
        print(f"   mlp_0 权重加载成功")
        
        pytorch_bias = pytorch_t_embedder.mlp[0].bias.detach().cpu().numpy()
        mlx_t_embedder.mlp_0.bias = mx.array(pytorch_bias)
        print(f"   mlp_0 偏置加载成功")
        
        # mlp_2 (对应 PyTorch 的 mlp[2])
        pytorch_weight = pytorch_t_embedder.mlp[2].weight.detach().cpu().numpy()
        mlx_t_embedder.mlp_2.weight = mx.array(pytorch_weight)
        print(f"   mlp_2 权重加载成功")
        
        pytorch_bias = pytorch_t_embedder.mlp[2].bias.detach().cpu().numpy()
        mlx_t_embedder.mlp_2.bias = mx.array(pytorch_bias)
        print(f"   mlp_2 偏置加载成功")
        
        # 加载 freqs
        pytorch_freqs = pytorch_t_embedder.freqs.detach().cpu().numpy()
        mlx_t_embedder.freqs = mx.array(pytorch_freqs)
        print(f"   freqs 加载成功")
        
        print(f"   ✅ 所有权重加载成功")
        
        # 重新测试
        t_val = 0.5
        t_pytorch = torch.tensor([t_val])
        t_mlx = mx.array([t_val])
        
        with torch.no_grad():
            pytorch_output = pytorch_t_embedder(t_pytorch)
        
        mlx_output = mlx_t_embedder(t_mlx)
        
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
    print("🔍 timestep_embedding 实现差异分析")
    print("=" * 60)
    
    try:
        # 1. 分析权重差异
        pytorch_t_embedder, mlx_t_embedder = analyze_timestep_embedding_weights()
        
        # 2. 测试前向传播
        test_timestep_embedding_forward()
        
        # 3. 分析实现差异
        analyze_timestep_embedding_implementation()
        
        # 4. 逐步测试
        test_timestep_embedding_step_by_step()
        
        # 5. 检查权重加载
        check_timestep_embedding_weight_loading()
        
        print(f"\n🎉 timestep_embedding 差异分析完成!")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
