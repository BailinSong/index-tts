#!/usr/bin/env python3
"""
修复 MLX 模型权重加载问题
确保 MLX 模型正确加载预训练的 cond_projection 权重
"""

import sys
import os
sys.path.append('.')

import torch
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import yaml
from pathlib import Path

def load_pytorch_weights():
    """加载 PyTorch 权重"""
    print("📥 加载 PyTorch 权重...")
    
    pytorch_weights = torch.load('checkpoints/s2mel.pth', map_location='cpu')
    
    # 提取 cond_projection 权重
    cond_proj_weight = pytorch_weights['net']['cfm']['estimator.cond_projection.weight'].numpy()
    cond_proj_bias = pytorch_weights['net']['cfm']['estimator.cond_projection.bias'].numpy()
    
    print(f"✅ PyTorch 权重加载完成:")
    print(f"   weight: {cond_proj_weight.shape}, range: [{cond_proj_weight.min():.6f}, {cond_proj_weight.max():.6f}]")
    print(f"   bias: {cond_proj_bias.shape}, range: [{cond_proj_bias.min():.6f}, {cond_proj_bias.max():.6f}]")
    
    return cond_proj_weight, cond_proj_bias

def load_mlx_weights():
    """加载 MLX 权重"""
    print("📥 加载 MLX 权重...")
    
    mlx_weights = np.load('checkpoints/mlx/s2mel.npz')
    
    # 提取 cond_projection 权重
    cond_proj_weight = mlx_weights['models.cfm.estimator.cond_projection.weight']
    cond_proj_bias = mlx_weights['models.cfm.estimator.cond_projection.bias']
    
    print(f"✅ MLX 权重加载完成:")
    print(f"   weight: {cond_proj_weight.shape}, range: [{cond_proj_weight.min():.6f}, {cond_proj_weight.max():.6f}]")
    print(f"   bias: {cond_proj_bias.shape}, range: [{cond_proj_bias.min():.6f}, {cond_proj_bias.max():.6f}]")
    
    return cond_proj_weight, cond_proj_bias

def create_mlx_model():
    """创建 MLX 模型"""
    print("🏗️ 创建 MLX 模型...")
    
    try:
        from indextts.s2mel.modules.mlx_cfm import MLXCFM
        
        # 加载配置
        with open('checkpoints/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # 使用正确的配置路径
        config['DiT'] = config['s2mel']['DiT']
        config['style_encoder'] = config['s2mel']['style_encoder']
        config['wavenet'] = config['s2mel']['wavenet']
        
        # 创建 MLX 模型
        mlx_model = MLXCFM(config)
        
        print("✅ MLX 模型创建完成")
        return mlx_model
        
    except Exception as e:
        print(f"❌ MLX 模型创建失败: {e}")
        return None

def fix_cond_projection_weights(mlx_model, pytorch_weight, pytorch_bias):
    """修复 cond_projection 权重"""
    print("🔧 修复 cond_projection 权重...")
    
    # 获取 cond_projection 层
    cond_proj = mlx_model.estimator.cond_projection
    
    print(f"修复前:")
    print(f"  权重范围: [{cond_proj.weight.min():.6f}, {cond_proj.weight.max():.6f}]")
    print(f"  偏置范围: [{cond_proj.bias.min():.6f}, {cond_proj.bias.max():.6f}]")
    
    # 加载正确的权重
    cond_proj.weight = mx.array(pytorch_weight)
    cond_proj.bias = mx.array(pytorch_bias)
    
    print(f"修复后:")
    print(f"  权重范围: [{cond_proj.weight.min():.6f}, {cond_proj.weight.max():.6f}]")
    print(f"  偏置范围: [{cond_proj.bias.min():.6f}, {cond_proj.bias.max():.6f}]")
    
    return mlx_model

def verify_weight_loading(mlx_model, pytorch_weight, pytorch_bias):
    """验证权重加载是否正确"""
    print("🔍 验证权重加载...")
    
    cond_proj = mlx_model.estimator.cond_projection
    
    # 转换为 numpy 进行比较
    mlx_weight_np = np.array(cond_proj.weight)
    mlx_bias_np = np.array(cond_proj.bias)
    
    # 计算差异
    weight_diff = np.abs(pytorch_weight - mlx_weight_np)
    bias_diff = np.abs(pytorch_bias - mlx_bias_np)
    
    print(f"权重差异:")
    print(f"  最大差异: {np.max(weight_diff):.8f}")
    print(f"  平均差异: {np.mean(weight_diff):.8f}")
    print(f"  相对差异: {np.mean(weight_diff) / np.mean(np.abs(pytorch_weight)):.8f}")
    
    print(f"偏置差异:")
    print(f"  最大差异: {np.max(bias_diff):.8f}")
    print(f"  平均差异: {np.mean(bias_diff):.8f}")
    print(f"  相对差异: {np.mean(bias_diff) / np.mean(np.abs(pytorch_bias)):.8f}")
    
    # 检查是否完全相同
    weight_identical = np.allclose(pytorch_weight, mlx_weight_np, atol=1e-8)
    bias_identical = np.allclose(pytorch_bias, mlx_bias_np, atol=1e-8)
    
    print(f"权重是否相同 (atol=1e-8):")
    print(f"  weight: {weight_identical}")
    print(f"  bias: {bias_identical}")
    
    return weight_identical and bias_identical

def test_forward_pass(mlx_model):
    """测试前向传播"""
    print("🧪 测试前向传播...")
    
    try:
        # 创建测试输入
        batch_size = 2
        seq_len = 100
        input_dim = 512
        
        test_input = mx.random.normal((batch_size, seq_len, input_dim))
        
        # 前向传播
        output = mlx_model.estimator.cond_projection(test_input)
        
        print(f"✅ 前向传播测试成功:")
        print(f"  输入形状: {test_input.shape}")
        print(f"  输出形状: {output.shape}")
        print(f"  输出范围: [{output.min():.6f}, {output.max():.6f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ 前向传播测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🔧 MLX 权重加载修复工具")
    print("=" * 50)
    
    try:
        # 1. 加载权重
        pytorch_weight, pytorch_bias = load_pytorch_weights()
        mlx_weight, mlx_bias = load_mlx_weights()
        
        # 2. 验证权重是否相同
        weight_diff = np.abs(pytorch_weight - mlx_weight)
        bias_diff = np.abs(pytorch_bias - mlx_bias)
        
        print(f"\n📊 权重文件对比:")
        print(f"  权重最大差异: {np.max(weight_diff):.8f}")
        print(f"  偏置最大差异: {np.max(bias_diff):.8f}")
        
        if np.max(weight_diff) < 1e-8 and np.max(bias_diff) < 1e-8:
            print("✅ 权重文件完全相同")
        else:
            print("⚠️ 权重文件存在差异")
        
        # 3. 创建 MLX 模型
        mlx_model = create_mlx_model()
        if mlx_model is None:
            return
        
        # 4. 修复权重
        mlx_model = fix_cond_projection_weights(mlx_model, pytorch_weight, pytorch_bias)
        
        # 5. 验证修复
        success = verify_weight_loading(mlx_model, pytorch_weight, pytorch_bias)
        
        # 6. 测试前向传播
        forward_success = test_forward_pass(mlx_model)
        
        # 7. 总结
        print(f"\n🎯 修复结果:")
        print(f"  权重加载: {'✅ 成功' if success else '❌ 失败'}")
        print(f"  前向传播: {'✅ 成功' if forward_success else '❌ 失败'}")
        
        if success and forward_success:
            print("\n🎉 MLX 权重加载修复完成!")
        else:
            print("\n⚠️ 修复过程中出现问题，请检查错误信息")
            
    except Exception as e:
        print(f"❌ 修复过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()