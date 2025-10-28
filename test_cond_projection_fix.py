#!/usr/bin/env python3
"""
测试 MLX cond_projection 权重加载修复
在 conda indextts2 环境中运行
"""

import sys
import os
# 强制使用当前目录的代码
current_dir = os.path.abspath('.')
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 重新加载模块以确保使用最新代码
import importlib
if 'indextts.s2mel.modules.mlx_cfm' in sys.modules:
    importlib.reload(sys.modules['indextts.s2mel.modules.mlx_cfm'])
if 'indextts.s2mel.modules.mlx_cfm_rewritten' in sys.modules:
    importlib.reload(sys.modules['indextts.s2mel.modules.mlx_cfm_rewritten'])

import yaml
import numpy as np
import mlx.core as mx

def test_cond_projection_fix():
    """测试 cond_projection 权重加载修复"""
    print("🔍 测试 MLX cond_projection 权重加载修复")
    print("=" * 50)
    
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
        
        # 检查初始化后的权重
        print("\n🔍 检查初始化后的 cond_projection...")
        cond_proj = mlx_model.estimator.cond_projection
        
        print(f"权重形状: {cond_proj.weight.shape}")
        print(f"权重范围: [{cond_proj.weight.min():.6f}, {cond_proj.weight.max():.6f}]")
        print(f"偏置范围: [{cond_proj.bias.min():.6f}, {cond_proj.bias.max():.6f}]")
        
        # 检查是否使用了 Kaiming uniform 初始化
        import math
        input_dim = cond_proj.weight.shape[1]
        expected_bound = math.sqrt(5) / math.sqrt(input_dim)
        actual_max = abs(cond_proj.weight.max())
        
        print(f"\n📊 初始化分析:")
        print(f"期望 Kaiming uniform 范围: ±{expected_bound:.6f}")
        print(f"实际权重范围: ±{actual_max:.6f}")
        
        if abs(actual_max - expected_bound) < 1e-6:
            print("✅ 使用了 Kaiming uniform 初始化")
            kaiming_init = True
        else:
            print("❌ 未使用 Kaiming uniform 初始化")
            kaiming_init = False
        
        # 测试权重加载
        print("\n🔍 测试权重加载...")
        
        # 加载 PyTorch 权重
        import torch
        pytorch_weights = torch.load('checkpoints/s2mel.pth', map_location='cpu')
        pytorch_weight = pytorch_weights['net']['cfm']['estimator.cond_projection.weight'].numpy()
        pytorch_bias = pytorch_weights['net']['cfm']['estimator.cond_projection.bias'].numpy()
        
        # 创建模拟的缓存数据
        cache_dict = {
            'models.cfm.estimator.cond_projection.weight': mx.array(pytorch_weight),
            'models.cfm.estimator.cond_projection.bias': mx.array(pytorch_bias)
        }
        
        # 测试 load_from_cache 方法
        print("🔧 测试 load_from_cache 方法...")
        loaded = mlx_model.load_from_cache(cache_dict)
        
        # 检查加载后的权重
        mlx_weight = np.array(mlx_model.estimator.cond_projection.weight)
        mlx_bias = np.array(mlx_model.estimator.cond_projection.bias)
        
        # 计算差异
        weight_diff = np.abs(pytorch_weight - mlx_weight)
        bias_diff = np.abs(pytorch_bias - mlx_bias)
        
        print(f"加载后权重差异: 最大={np.max(weight_diff):.8f}, 平均={np.mean(weight_diff):.8f}")
        print(f"加载后偏置差异: 最大={np.max(bias_diff):.8f}, 平均={np.mean(bias_diff):.8f}")
        
        if np.max(weight_diff) < 1e-8 and np.max(bias_diff) < 1e-8:
            print("✅ 权重加载成功")
            weight_loading = True
        else:
            print("❌ 权重加载失败")
            weight_loading = False
        
        # 总结
        print(f"\n🎯 修复状态总结:")
        print(f"  Kaiming uniform 初始化: {'✅ 已应用' if kaiming_init else '❌ 未应用'}")
        print(f"  权重加载修复: {'✅ 成功' if weight_loading else '❌ 失败'}")
        print(f"  加载的权重数量: {loaded}")
        
        return kaiming_init and weight_loading
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cond_projection_fix()
    
    if success:
        print("\n🎉 MLX cond_projection 修复测试通过!")
        print("✅ benchmark_v1_baseline.py 现在应该能正确使用修复后的 MLX 模型")
    else:
        print("\n❌ MLX cond_projection 修复测试失败")
        print("⚠️ 需要进一步调试修复问题")
