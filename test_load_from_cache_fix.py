#!/usr/bin/env python3
"""
直接测试 load_from_cache 方法的 cond_projection 权重加载修复
"""

import sys
import os
current_dir = os.path.abspath('.')
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import yaml
import numpy as np
import mlx.core as mx
import torch

def test_load_from_cache_fix():
    """测试 load_from_cache 方法的修复"""
    print("🔍 测试 load_from_cache 方法的 cond_projection 权重加载修复")
    print("=" * 60)
    
    try:
        # 加载配置
        with open('checkpoints/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # 使用正确的配置路径
        config['DiT'] = config['s2mel']['DiT']
        config['style_encoder'] = config['s2mel']['style_encoder']
        config['wavenet'] = config['s2mel']['wavenet']
        
        # 创建 MLX 模型
        print("🏗️ 创建 MLX CFM 模型...")
        from indextts.s2mel.modules.mlx_cfm import MLXCFM
        mlx_model = MLXCFM(config)
        
        # 获取初始权重
        initial_weight = np.array(mlx_model.estimator.cond_projection.weight)
        initial_bias = np.array(mlx_model.estimator.cond_projection.bias)
        
        print(f"初始权重范围: [{initial_weight.min():.6f}, {initial_weight.max():.6f}]")
        print(f"初始偏置范围: [{initial_bias.min():.6f}, {initial_bias.max():.6f}]")
        
        # 加载 PyTorch 权重
        print("\n📥 加载 PyTorch 权重...")
        pytorch_weights = torch.load('checkpoints/s2mel.pth', map_location='cpu')
        pytorch_weight = pytorch_weights['net']['cfm']['estimator.cond_projection.weight'].numpy()
        pytorch_bias = pytorch_weights['net']['cfm']['estimator.cond_projection.bias'].numpy()
        
        print(f"PyTorch 权重范围: [{pytorch_weight.min():.6f}, {pytorch_weight.max():.6f}]")
        print(f"PyTorch 偏置范围: [{pytorch_bias.min():.6f}, {pytorch_bias.max():.6f}]")
        
        # 创建模拟的缓存数据
        cache_dict = {
            'models.cfm.estimator.cond_projection.weight': mx.array(pytorch_weight),
            'models.cfm.estimator.cond_projection.bias': mx.array(pytorch_bias)
        }
        
        # 测试 load_from_cache 方法
        print("\n🔧 测试 load_from_cache 方法...")
        loaded = mlx_model.load_from_cache(cache_dict)
        
        # 检查加载后的权重
        final_weight = np.array(mlx_model.estimator.cond_projection.weight)
        final_bias = np.array(mlx_model.estimator.cond_projection.bias)
        
        print(f"加载后权重范围: [{final_weight.min():.6f}, {final_weight.max():.6f}]")
        print(f"加载后偏置范围: [{final_bias.min():.6f}, {final_bias.max():.6f}]")
        
        # 计算差异
        weight_diff = np.abs(pytorch_weight - final_weight)
        bias_diff = np.abs(pytorch_bias - final_bias)
        
        print(f"\n📊 权重差异分析:")
        print(f"权重差异: 最大={np.max(weight_diff):.8f}, 平均={np.mean(weight_diff):.8f}")
        print(f"偏置差异: 最大={np.max(bias_diff):.8f}, 平均={np.mean(bias_diff):.8f}")
        
        # 检查是否成功加载
        if np.max(weight_diff) < 1e-8 and np.max(bias_diff) < 1e-8:
            print("✅ cond_projection 权重加载成功!")
            success = True
        else:
            print("❌ cond_projection 权重加载失败")
            success = False
        
        print(f"\n📋 测试结果:")
        print(f"  加载的权重数量: {loaded}")
        print(f"  权重加载状态: {'✅ 成功' if success else '❌ 失败'}")
        
        return success
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_load_from_cache_fix()
    
    if success:
        print("\n🎉 load_from_cache 修复测试通过!")
        print("✅ benchmark_v1_baseline.py 现在应该能正确加载 cond_projection 权重")
    else:
        print("\n❌ load_from_cache 修复测试失败")
        print("⚠️ 需要进一步调试修复问题")
