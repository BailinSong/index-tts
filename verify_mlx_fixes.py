#!/usr/bin/env python3
"""
验证 MLX 模型是否应用了权重和随机算法修复
"""

import sys
import os
# 确保当前工作目录在 Python 路径中
current_dir = os.path.abspath('.')
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import yaml
import numpy as np
import mlx.core as mx

def test_mlx_model_fixes():
    """测试 MLX 模型是否应用了修复"""
    print("🔍 验证 MLX 模型修复状态")
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
        
        # 检查 cond_projection 初始化
        print("\n🔍 检查 cond_projection 初始化...")
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
        else:
            print("❌ 未使用 Kaiming uniform 初始化")
        
        # 检查权重加载
        print("\n🔍 检查权重加载...")
        
        # 加载 PyTorch 权重
        import torch
        pytorch_weights = torch.load('checkpoints/s2mel.pth', map_location='cpu')
        pytorch_weight = pytorch_weights['net']['cfm']['estimator.cond_projection.weight'].numpy()
        pytorch_bias = pytorch_weights['net']['cfm']['estimator.cond_projection.bias'].numpy()
        
        # 获取 MLX 权重
        mlx_weight = np.array(cond_proj.weight)
        mlx_bias = np.array(cond_proj.bias)
        
        # 计算差异
        weight_diff = np.abs(pytorch_weight - mlx_weight)
        bias_diff = np.abs(pytorch_bias - mlx_bias)
        
        print(f"权重差异: 最大={np.max(weight_diff):.8f}, 平均={np.mean(weight_diff):.8f}")
        print(f"偏置差异: 最大={np.max(bias_diff):.8f}, 平均={np.mean(bias_diff):.8f}")
        
        if np.max(weight_diff) < 1e-6 and np.max(bias_diff) < 1e-6:
            print("✅ 权重已正确加载")
        else:
            print("❌ 权重未正确加载")
        
        # 测试权重修复
        print("\n🔧 测试权重修复...")
        from indextts.utils.mlx_production_utils import fix_mlx_model_weights
        
        fix_success = fix_mlx_model_weights(mlx_model, 'checkpoints/s2mel.pth')
        
        if fix_success:
            # 重新检查权重
            mlx_weight_fixed = np.array(mlx_model.estimator.cond_projection.weight)
            mlx_bias_fixed = np.array(mlx_model.estimator.cond_projection.bias)
            
            weight_diff_fixed = np.abs(pytorch_weight - mlx_weight_fixed)
            bias_diff_fixed = np.abs(pytorch_bias - mlx_bias_fixed)
            
            print(f"修复后权重差异: 最大={np.max(weight_diff_fixed):.8f}, 平均={np.mean(weight_diff_fixed):.8f}")
            print(f"修复后偏置差异: 最大={np.max(bias_diff_fixed):.8f}, 平均={np.mean(bias_diff_fixed):.8f}")
            
            if np.max(weight_diff_fixed) < 1e-8 and np.max(bias_diff_fixed) < 1e-8:
                print("✅ 权重修复成功")
            else:
                print("❌ 权重修复失败")
        
        # 总结
        print(f"\n🎯 修复状态总结:")
        print(f"  Kaiming uniform 初始化: {'✅ 已应用' if abs(actual_max - expected_bound) < 1e-6 else '❌ 未应用'}")
        print(f"  权重加载: {'✅ 正确' if np.max(weight_diff) < 1e-6 else '❌ 需要修复'}")
        print(f"  权重修复: {'✅ 可用' if fix_success else '❌ 不可用'}")
        
        return {
            "kaiming_init": abs(actual_max - expected_bound) < 1e-6,
            "weight_loading": np.max(weight_diff) < 1e-6,
            "weight_fix_available": fix_success
        }
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_mlx_model_fixes()
    
    if result:
        print(f"\n📋 最终结果:")
        if result["kaiming_init"] and result["weight_loading"]:
            print("🎉 MLX 模型已正确应用所有修复!")
        elif result["weight_fix_available"]:
            print("⚠️ MLX 模型需要运行时修复权重")
        else:
            print("❌ MLX 模型需要手动修复")
    else:
        print("❌ 无法验证 MLX 模型状态")
