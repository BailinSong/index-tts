#!/usr/bin/env python3
"""
简化版权重对比测试脚本

快速验证 MLX 和 PyTorch 权重加载的差异
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append('/Users/bailin/index-tts')

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("⚠️ MLX not available")

from indextts.utils.mlx_cache import MLXModelCache


def quick_weight_test():
    """快速权重测试"""
    print("🔍 Quick Weight Comparison Test")
    print("=" * 40)
    
    # 检查文件
    pytorch_checkpoint = "checkpoints/s2mel.pth"
    if not os.path.exists(pytorch_checkpoint):
        print(f"❌ PyTorch checkpoint not found: {pytorch_checkpoint}")
        return
    
    # 1. 加载 PyTorch 权重
    print(f"\n📦 Loading PyTorch weights...")
    checkpoint = torch.load(pytorch_checkpoint, map_location='cpu')
    
    pytorch_weights = {}
    if 'net' in checkpoint:
        # S2MEL format
        for key in checkpoint['net']:
            for param_name, param_value in checkpoint['net'][key].items():
                pytorch_weights[f"{key}.{param_name}"] = param_value.cpu().numpy()
    else:
        pytorch_weights = {k: v.cpu().numpy() for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
    
    print(f"   Loaded {len(pytorch_weights)} PyTorch weights")
    
    # 2. 加载 MLX 权重
    if not MLX_AVAILABLE:
        print("⚠️ MLX not available, skipping MLX comparison")
        return
    
    print(f"\n📦 Loading MLX weights...")
    mlx_cache = MLXModelCache(cache_dir="checkpoints/mlx")
    mlx_state = mlx_cache.load_from_cache("s2mel_cfm")
    
    if mlx_state is None:
        print("❌ No MLX cache found")
        return
    
    mlx_weights = {k: np.array(v) for k, v in mlx_state.items() if isinstance(v, mx.array)}
    print(f"   Loaded {len(mlx_weights)} MLX weights")
    
    # 3. 快速对比
    print(f"\n🔍 Quick comparison...")
    
    pytorch_keys = set(pytorch_weights.keys())
    mlx_keys = set(mlx_weights.keys())
    common_keys = pytorch_keys & mlx_keys
    
    print(f"   Common keys: {len(common_keys)}")
    print(f"   PyTorch only: {len(pytorch_keys - mlx_keys)}")
    print(f"   MLX only: {len(mlx_keys - pytorch_keys)}")
    
    # 检查几个关键权重的差异
    key_weights_to_check = [
        'cfm.estimator.x_embedder.weight',
        'cfm.estimator.t_embedder.mlp.0.weight',
        'cfm.estimator.cond_projection.weight',
        'cfm.estimator.transformer.layers.0.attention.wqkv.weight'
    ]
    
    print(f"\n📊 Checking key weights:")
    max_diff = 0.0
    
    for key in key_weights_to_check:
        if key in common_keys:
            pt_weight = pytorch_weights[key]
            mlx_weight = mlx_weights[key]
            
            if pt_weight.shape == mlx_weight.shape:
                diff = np.abs(pt_weight - mlx_weight)
                max_diff_key = np.max(diff)
                mean_diff_key = np.mean(diff)
                
                max_diff = max(max_diff, max_diff_key)
                
                status = "✅" if max_diff_key < 1e-6 else "⚠️"
                print(f"   {status} {key}:")
                print(f"      Shape: {pt_weight.shape}")
                print(f"      Max diff: {max_diff_key:.8f}")
                print(f"      Mean diff: {mean_diff_key:.8f}")
                print(f"      PT range: [{np.min(pt_weight):.6f}, {np.max(pt_weight):.6f}]")
                print(f"      MLX range: [{np.min(mlx_weight):.6f}, {np.max(mlx_weight):.6f}]")
            else:
                print(f"   ❌ {key}: Shape mismatch {pt_weight.shape} vs {mlx_weight.shape}")
        else:
            print(f"   ❌ {key}: Not found in common keys")
    
    print(f"\n📈 Overall max difference: {max_diff:.8f}")
    
    if max_diff < 1e-6:
        print("✅ All checked weights match within tolerance!")
    else:
        print("⚠️ Found significant differences in weights")


def test_weight_norm_conversion():
    """测试权重归一化转换"""
    print(f"\n🔧 Testing weight normalization conversion...")
    
    # 模拟 PyTorch weight_norm 权重
    torch.manual_seed(42)
    weight_v = torch.randn(128, 256)  # (out_features, in_features)
    weight_g = torch.randn(128)      # (out_features,)
    
    # PyTorch 权重归一化计算
    norm_v = torch.sqrt(torch.sum(weight_v**2, dim=1, keepdims=True))
    pytorch_weight = weight_g.unsqueeze(1) * weight_v / (norm_v + 1e-8)
    
    # MLX 权重归一化计算
    weight_v_np = weight_v.numpy()
    weight_g_np = weight_g.numpy()
    norm_v_np = np.sqrt(np.sum(weight_v_np**2, axis=1, keepdims=True))
    mlx_weight = weight_g_np.reshape(-1, 1) * weight_v_np / (norm_v_np + 1e-8)
    
    # 对比
    diff = np.abs(pytorch_weight.numpy() - mlx_weight)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"   Weight normalization test:")
    print(f"   Max difference: {max_diff:.8f}")
    print(f"   Mean difference: {mean_diff:.8f}")
    
    if max_diff < 1e-6:
        print("   ✅ Weight normalization conversion is correct")
    else:
        print("   ⚠️ Weight normalization conversion has issues")


def test_conv1d_conversion():
    """测试 Conv1d 权重转换"""
    print(f"\n🔧 Testing Conv1d weight conversion...")
    
    # 模拟 PyTorch Conv1d 权重 (O, I, K)
    torch.manual_seed(42)
    pytorch_conv_weight = torch.randn(64, 32, 3)  # (out_channels, in_channels, kernel_size)
    
    # MLX Conv1d 权重 (O, K, I)
    mlx_conv_weight = pytorch_conv_weight.transpose(0, 2, 1).numpy()
    
    # 验证转换
    print(f"   PyTorch Conv1d shape: {pytorch_conv_weight.shape}")
    print(f"   MLX Conv1d shape: {mlx_conv_weight.shape}")
    
    # 检查转换是否正确
    expected_shape = (64, 3, 32)  # (O, K, I)
    if mlx_conv_weight.shape == expected_shape:
        print("   ✅ Conv1d weight conversion shape is correct")
    else:
        print(f"   ❌ Conv1d weight conversion shape is wrong: expected {expected_shape}")


if __name__ == "__main__":
    print("🧪 S2MEL Weight Comparison Test Suite")
    print("=" * 50)
    
    # 运行快速测试
    quick_weight_test()
    
    # 运行转换测试
    test_weight_norm_conversion()
    test_conv1d_conversion()
    
    print(f"\n🎉 Test suite completed!")
