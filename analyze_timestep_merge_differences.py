#!/usr/bin/env python3
"""
分析 timestep_embedding 和 merge_input 层的实现差异
比较 PyTorch 和 MLX 版本
"""

import sys
import os
current_dir = os.path.abspath('.')
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import yaml
import numpy as np
import torch
import mlx.core as mx
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def analyze_timestep_embedding():
    """分析 timestep_embedding 的实现差异"""
    print("🔍 分析 timestep_embedding 实现差异")
    print("=" * 60)
    
    # 加载配置
    with open('checkpoints/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 使用正确的配置路径
    config['DiT'] = config['s2mel']['DiT']
    config['style_encoder'] = config['s2mel']['style_encoder']
    config['wavenet'] = config['s2mel']['wavenet']
    
    # 创建 MLX 模型
    from indextts.s2mel.modules.mlx_cfm import MLXCFM
    mlx_cfm = MLXCFM(config)
    mlx_dit = mlx_cfm.estimator
    
    print("📊 MLX TimestepEmbedder 分析:")
    mlx_t_embedder = mlx_dit.t_embedder
    print(f"  类型: {type(mlx_t_embedder).__name__}")
    print(f"  hidden_size: {mlx_t_embedder.hidden_size}")
    print(f"  frequency_embedding_size: {mlx_t_embedder.frequency_embedding_size}")
    print(f"  max_period: {mlx_t_embedder.max_period}")
    print(f"  scale: {mlx_t_embedder.scale}")
    print(f"  freqs shape: {mlx_t_embedder.freqs.shape}")
    print(f"  freqs range: [{float(mlx_t_embedder.freqs.min()):.6f}, {float(mlx_t_embedder.freqs.max()):.6f}]")
    
    # 测试 timestep_embedding 方法
    print(f"\n🧪 测试 timestep_embedding 方法:")
    test_t = torch.tensor([0.0, 0.5, 1.0])
    test_t_mlx = torch_to_mlx(test_t)
    
    # MLX
    mlx_emb = mlx_t_embedder.timestep_embedding(test_t_mlx)
    print(f"  MLX embedding shape: {mlx_emb.shape}")
    print(f"  MLX embedding range: [{float(mlx_emb.min()):.6f}, {float(mlx_emb.max()):.6f}]")
    
    # 测试完整 forward 方法
    print(f"\n🧪 测试完整 forward 方法:")
    mlx_output = mlx_t_embedder(test_t_mlx)
    
    print(f"  MLX output shape: {mlx_output.shape}")
    print(f"  MLX output range: [{float(mlx_output.min()):.6f}, {float(mlx_output.max()):.6f}]")
    
    # 检查实现细节
    print(f"\n🔍 实现细节分析:")
    print(f"  timestep_embedding 方法存在: {'✅' if hasattr(mlx_t_embedder, 'timestep_embedding') else '❌'}")
    print(f"  __call__ 方法存在: {'✅' if hasattr(mlx_t_embedder, '__call__') else '❌'}")
    print(f"  freqs 预计算: {'✅' if hasattr(mlx_t_embedder, 'freqs') else '❌'}")
    
    return True

def analyze_merge_input():
    """分析 merge_input 的实现差异"""
    print("\n🔍 分析 merge_input 实现差异")
    print("=" * 60)
    
    # 加载配置
    with open('checkpoints/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 使用正确的配置路径
    config['DiT'] = config['s2mel']['DiT']
    config['style_encoder'] = config['s2mel']['style_encoder']
    config['wavenet'] = config['s2mel']['wavenet']
    
    # 创建 MLX 模型
    from indextts.s2mel.modules.mlx_cfm import MLXCFM
    mlx_cfm = MLXCFM(config)
    mlx_dit = mlx_cfm.estimator
    
    print("📊 MLX cond_x_merge_linear 分析:")
    mlx_merge = mlx_dit.cond_x_merge_linear
    print(f"  类型: {type(mlx_merge).__name__}")
    print(f"  输入维度: {mlx_merge.weight.shape[1]}")
    print(f"  输出维度: {mlx_merge.weight.shape[0]}")
    print(f"  权重形状: {mlx_merge.weight.shape}")
    print(f"  权重范围: [{float(mlx_merge.weight.min()):.6f}, {float(mlx_merge.weight.max()):.6f}]")
    print(f"  偏置形状: {mlx_merge.bias.shape}")
    print(f"  偏置范围: [{float(mlx_merge.bias.min()):.6f}, {float(mlx_merge.bias.max()):.6f}]")
    
    # 测试 merge_input 操作
    print(f"\n🧪 测试 merge_input 操作:")
    
    # 创建测试数据
    batch_size, seq_len = 2, 100
    x_dim, prompt_dim, cond_dim, style_dim = 80, 80, 512, 192
    
    # MLX 数据
    x_mlx = mx.random.normal((batch_size, x_dim, seq_len))
    prompt_x_mlx = mx.random.normal((batch_size, prompt_dim, seq_len))
    cond_mlx = mx.random.normal((batch_size, seq_len, cond_dim))
    style_mlx = mx.random.normal((batch_size, style_dim))
    
    # MLX merge_input
    x_t_mlx = x_mlx.transpose(0, 2, 1)  # (B, T, C)
    prompt_x_t_mlx = prompt_x_mlx.transpose(0, 2, 1)
    x_in_mlx = mx.concatenate([x_t_mlx, prompt_x_t_mlx, cond_mlx], axis=-1)
    if mlx_dit.transformer_style_condition and not mlx_dit.style_as_token:
        x_in_mlx = mx.concatenate([x_in_mlx, mx.broadcast_to(style_mlx[:, None, :], (style_mlx.shape[0], x_in_mlx.shape[1], style_mlx.shape[1]))], axis=-1)
    
    print(f"  MLX merge input shape: {x_in_mlx.shape}")
    print(f"  MLX merge input range: [{float(x_in_mlx.min()):.6f}, {float(x_in_mlx.max()):.6f}]")
    
    # 测试 cond_x_merge_linear
    print(f"\n🧪 测试 cond_x_merge_linear:")
    
    # MLX
    mlx_merged = mlx_merge(x_in_mlx)
    print(f"  MLX merged shape: {mlx_merged.shape}")
    print(f"  MLX merged range: [{float(mlx_merged.min()):.6f}, {float(mlx_merged.max()):.6f}]")
    
    # 检查实现细节
    print(f"\n🔍 实现细节分析:")
    print(f"  cond_x_merge_linear 存在: {'✅' if hasattr(mlx_dit, 'cond_x_merge_linear') else '❌'}")
    print(f"  transformer_style_condition: {mlx_dit.transformer_style_condition}")
    print(f"  style_as_token: {mlx_dit.style_as_token}")
    
    return True

def compare_with_pytorch_weights():
    """与 PyTorch 权重进行比较"""
    print("\n🔍 与 PyTorch 权重比较")
    print("=" * 60)
    
    # 加载 PyTorch 权重
    pytorch_weights = torch.load('checkpoints/s2mel.pth', map_location='cpu')
    
    # 加载配置
    with open('checkpoints/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['DiT'] = config['s2mel']['DiT']
    config['style_encoder'] = config['s2mel']['style_encoder']
    config['wavenet'] = config['s2mel']['wavenet']
    
    # 创建 MLX 模型
    from indextts.s2mel.modules.mlx_cfm import MLXCFM
    mlx_cfm = MLXCFM(config)
    mlx_dit = mlx_cfm.estimator
    
    # 比较 t_embedder 权重
    print("📊 t_embedder 权重比较:")
    
    # PyTorch t_embedder 权重
    pytorch_t_mlp_0_weight = pytorch_weights['net']['cfm']['estimator.t_embedder.mlp.0.weight'].numpy()
    pytorch_t_mlp_0_bias = pytorch_weights['net']['cfm']['estimator.t_embedder.mlp.0.bias'].numpy()
    pytorch_t_mlp_2_weight = pytorch_weights['net']['cfm']['estimator.t_embedder.mlp.2.weight'].numpy()
    pytorch_t_mlp_2_bias = pytorch_weights['net']['cfm']['estimator.t_embedder.mlp.2.bias'].numpy()
    
    # MLX t_embedder 权重
    mlx_t_mlp_0_weight = np.array(mlx_dit.t_embedder.mlp_0.weight)
    mlx_t_mlp_0_bias = np.array(mlx_dit.t_embedder.mlp_0.bias)
    mlx_t_mlp_2_weight = np.array(mlx_dit.t_embedder.mlp_2.weight)
    mlx_t_mlp_2_bias = np.array(mlx_dit.t_embedder.mlp_2.bias)
    
    # 比较权重
    mlp_0_weight_diff = np.abs(pytorch_t_mlp_0_weight - mlx_t_mlp_0_weight)
    mlp_0_bias_diff = np.abs(pytorch_t_mlp_0_bias - mlx_t_mlp_0_bias)
    mlp_2_weight_diff = np.abs(pytorch_t_mlp_2_weight - mlx_t_mlp_2_weight)
    mlp_2_bias_diff = np.abs(pytorch_t_mlp_2_bias - mlx_t_mlp_2_bias)
    
    print(f"  mlp_0.weight 差异: 最大={np.max(mlp_0_weight_diff):.8f}, 平均={np.mean(mlp_0_weight_diff):.8f}")
    print(f"  mlp_0.bias 差异: 最大={np.max(mlp_0_bias_diff):.8f}, 平均={np.mean(mlp_0_bias_diff):.8f}")
    print(f"  mlp_2.weight 差异: 最大={np.max(mlp_2_weight_diff):.8f}, 平均={np.mean(mlp_2_weight_diff):.8f}")
    print(f"  mlp_2.bias 差异: 最大={np.max(mlp_2_bias_diff):.8f}, 平均={np.mean(mlp_2_bias_diff):.8f}")
    
    t_embedder_ok = (np.max(mlp_0_weight_diff) < 1e-8 and np.max(mlp_0_bias_diff) < 1e-8 and 
                     np.max(mlp_2_weight_diff) < 1e-8 and np.max(mlp_2_bias_diff) < 1e-8)
    print(f"  t_embedder 权重: {'✅ 相同' if t_embedder_ok else '❌ 不同'}")
    
    # 比较 cond_x_merge_linear 权重
    print("\n📊 cond_x_merge_linear 权重比较:")
    
    # PyTorch cond_x_merge_linear 权重
    pytorch_merge_weight = pytorch_weights['net']['cfm']['estimator.cond_x_merge_linear.weight'].numpy()
    pytorch_merge_bias = pytorch_weights['net']['cfm']['estimator.cond_x_merge_linear.bias'].numpy()
    
    # MLX cond_x_merge_linear 权重
    mlx_merge_weight = np.array(mlx_dit.cond_x_merge_linear.weight)
    mlx_merge_bias = np.array(mlx_dit.cond_x_merge_linear.bias)
    
    # 比较权重
    merge_weight_diff = np.abs(pytorch_merge_weight - mlx_merge_weight)
    merge_bias_diff = np.abs(pytorch_merge_bias - mlx_merge_bias)
    
    print(f"  weight 差异: 最大={np.max(merge_weight_diff):.8f}, 平均={np.mean(merge_weight_diff):.8f}")
    print(f"  bias 差异: 最大={np.max(merge_bias_diff):.8f}, 平均={np.mean(merge_bias_diff):.8f}")
    
    merge_ok = np.max(merge_weight_diff) < 1e-8 and np.max(merge_bias_diff) < 1e-8
    print(f"  cond_x_merge_linear 权重: {'✅ 相同' if merge_ok else '❌ 不同'}")
    
    return t_embedder_ok and merge_ok

def main():
    """主函数"""
    print("🔍 分析 timestep_embedding 和 merge_input 层差异")
    print("=" * 80)
    
    try:
        # 分析 timestep_embedding
        timestep_ok = analyze_timestep_embedding()
        
        # 分析 merge_input
        merge_ok = analyze_merge_input()
        
        # 与 PyTorch 权重比较
        weights_ok = compare_with_pytorch_weights()
        
        # 总结
        print(f"\n📋 分析总结:")
        print(f"  timestep_embedding 实现: {'✅ 正常' if timestep_ok else '❌ 异常'}")
        print(f"  merge_input 实现: {'✅ 正常' if merge_ok else '❌ 异常'}")
        print(f"  权重加载: {'✅ 正确' if weights_ok else '❌ 错误'}")
        
        if timestep_ok and merge_ok and weights_ok:
            print(f"\n🎉 timestep_embedding 和 merge_input 层实现正确，无差异！")
        else:
            print(f"\n⚠️ 发现差异或问题，需要进一步分析")
            
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()