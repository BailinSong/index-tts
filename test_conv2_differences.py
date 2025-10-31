#!/usr/bin/env python3
"""
专门测试conv2_output差异的脚本
分析conv2层输入输出差异的产生过程
"""

import torch
import mlx.core as mx
import numpy as np
from pathlib import Path
import pickle
import sys
import os

# 添加项目路径
sys.path.append('/Users/bailin/index-tts')

def load_debug_data():
    """加载调试数据"""
    debug_file = Path('/Users/bailin/index-tts/cfm_debug_outputs/complete_dit_debug_analysis.pkl')
    if not debug_file.exists():
        print(f"❌ 调试数据文件不存在: {debug_file}")
        return None
    
    with open(debug_file, 'rb') as f:
        data = pickle.load(f)
    
    return data

def analyze_conv2_inputs(data):
    """分析conv2的输入数据"""
    print("🔍 分析conv2输入数据...")
    
    # 获取final_layer输出（conv2的输入）
    pytorch_final_layer = data.get('pytorch_data', {}).get('step_0_layer_0_final_layer', {})
    mlx_final_layer = data.get('mlx_data', {}).get('step_0_layer_0_final_layer', {})
    
    if not pytorch_final_layer or not mlx_final_layer:
        print("❌ 找不到final_layer数据")
        return
    
    pytorch_final_output = pytorch_final_layer.get('final_layer_output', {})
    mlx_final_output = mlx_final_layer.get('final_layer_output', {})
    
    if not pytorch_final_output or not mlx_final_output:
        print("❌ 找不到final_layer_output数据")
        return
    
    print(f"📊 Final Layer输出分析:")
    print(f"   PyTorch: shape={pytorch_final_output.get('shape')}, min={pytorch_final_output.get('min'):.6f}, max={pytorch_final_output.get('max'):.6f}")
    print(f"   MLX:     shape={mlx_final_output.get('shape')}, min={mlx_final_output.get('min'):.6f}, max={mlx_final_output.get('max'):.6f}")
    
    # 计算差异
    if 'data_sample' in pytorch_final_output and 'data_sample' in mlx_final_output:
        pytorch_sample = pytorch_final_output['data_sample']
        mlx_sample = mlx_final_output['data_sample']
        
        if pytorch_sample is not None and mlx_sample is not None:
            # 转换为numpy数组进行比较
            if isinstance(pytorch_sample, torch.Tensor):
                pytorch_np = pytorch_sample.detach().cpu().numpy()
            else:
                pytorch_np = np.array(pytorch_sample)
            
            if isinstance(mlx_sample, mx.array):
                mlx_np = np.array(mlx_sample)
            else:
                mlx_np = np.array(mlx_sample)
            
            diff = np.abs(pytorch_np - mlx_np)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"   差异: max={max_diff:.6f}, mean={mean_diff:.6f}")
            
            if max_diff > 1e-5:
                print(f"   ❌ Final Layer输出存在显著差异")
            else:
                print(f"   ✅ Final Layer输出基本一致")

def analyze_conv2_transpose_operations(data):
    """分析conv2前后的转置操作"""
    print("\n🔍 分析conv2转置操作...")
    
    # 获取conv2输入和输出数据
    pytorch_conv2 = data.get('pytorch_data', {}).get('step_0_layer_0_conv2_output', {})
    mlx_conv2 = data.get('mlx_data', {}).get('step_0_layer_0_conv2_output', {})
    
    if not pytorch_conv2 or not mlx_conv2:
        print("❌ 找不到conv2数据")
        return
    
    pytorch_conv2_out = pytorch_conv2.get('conv2_out', {})
    mlx_conv2_out = mlx_conv2.get('conv2_out', {})
    
    if not pytorch_conv2_out or not mlx_conv2_out:
        print("❌ 找不到conv2输出数据")
        return
    
    print(f"📊 Conv2输出分析:")
    print(f"   PyTorch: shape={pytorch_conv2_out.get('shape')}, min={pytorch_conv2_out.get('min'):.6f}, max={pytorch_conv2_out.get('max'):.6f}")
    print(f"   MLX:     shape={mlx_conv2_out.get('shape')}, min={mlx_conv2_out.get('min'):.6f}, max={mlx_conv2_out.get('max'):.6f}")
    
    # 分析形状差异
    pytorch_shape = pytorch_conv2_out.get('shape')
    mlx_shape = mlx_conv2_out.get('shape')
    
    if pytorch_shape and mlx_shape:
        print(f"   形状差异: PyTorch={pytorch_shape} vs MLX={mlx_shape}")
        
        # 检查形状是否匹配
        if pytorch_shape == mlx_shape:
            print("   ✅ 形状一致")
        else:
            print("   ❌ 形状不匹配")
            
            # 分析形状差异的具体原因
            if len(pytorch_shape) == len(mlx_shape):
                for i, (p, m) in enumerate(zip(pytorch_shape, mlx_shape)):
                    if p != m:
                        print(f"     维度{i}: PyTorch={p}, MLX={m}, 差异={abs(p-m)}")
    
    # 计算数值差异
    if 'data_sample' in pytorch_conv2_out and 'data_sample' in mlx_conv2_out:
        pytorch_sample = pytorch_conv2_out['data_sample']
        mlx_sample = mlx_conv2_out['data_sample']
        
        if pytorch_sample is not None and mlx_sample is not None:
            # 转换为numpy数组进行比较
            if isinstance(pytorch_sample, torch.Tensor):
                pytorch_np = pytorch_sample.detach().cpu().numpy()
            else:
                pytorch_np = np.array(pytorch_sample)
            
            if isinstance(mlx_sample, mx.array):
                mlx_np = np.array(mlx_sample)
            else:
                mlx_np = np.array(mlx_sample)
            
            diff = np.abs(pytorch_np - mlx_np)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"   数值差异: max={max_diff:.6f}, mean={mean_diff:.6f}")
            
            if max_diff > 1e-5:
                print(f"   ❌ Conv2输出存在显著差异")
                
                # 分析差异分布
                print(f"   差异统计:")
                print(f"     最大差异: {max_diff:.6f}")
                print(f"     平均差异: {mean_diff:.6f}")
                print(f"     差异标准差: {np.std(diff):.6f}")
                
                # 找出差异最大的位置
                max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
                print(f"     最大差异位置: {max_diff_idx}")
                print(f"     PyTorch值: {pytorch_np[max_diff_idx]:.6f}")
                print(f"     MLX值: {mlx_np[max_diff_idx]:.6f}")
                
            else:
                print(f"   ✅ Conv2输出基本一致")

def test_conv2_weight_loading():
    """测试conv2权重加载"""
    print("\n🔍 测试conv2权重加载...")
    
    try:
        from indextts.s2mel.modules.mlx_diffusion_transformer_weights import convert_conv1d_weight
        
        # 模拟conv2权重 (80, 512, 1)
        conv2_weight = np.random.randn(80, 512, 1).astype(np.float32)
        print(f"   原始权重形状: {conv2_weight.shape}")
        
        # 转换权重
        converted_weight = convert_conv1d_weight(conv2_weight)
        print(f"   转换后形状: {converted_weight.shape}")
        
        # 检查转换是否正确
        expected_shape = (80, 1, 512)
        if converted_weight.shape == expected_shape:
            print(f"   ✅ 权重转换正确: {conv2_weight.shape} -> {converted_weight.shape}")
        else:
            print(f"   ❌ 权重转换错误: 期望{expected_shape}, 实际{converted_weight.shape}")
            
    except Exception as e:
        print(f"   ❌ 权重转换测试失败: {e}")

def test_conv2_forward_pass():
    """测试conv2前向传播"""
    print("\n🔍 测试conv2前向传播...")
    
    try:
        import mlx.nn as nn
        
        # 创建MLX Conv1d层
        conv2 = nn.Conv1d(512, 80, kernel_size=1, bias=True)
        
        # 创建测试输入 (batch=2, seq_len=415, in_channels=512)
        batch_size = 2
        seq_len = 415
        in_channels = 512
        
        # 模拟final_layer输出
        x = mx.random.normal((batch_size, seq_len, in_channels))
        print(f"   输入形状: {x.shape}")
        
        # 转置为MLX Conv1d期望的格式 (batch, in_channels, seq_len)
        x_transposed = x.transpose(0, 2, 1)
        print(f"   转置后形状: {x_transposed.shape}")
        
        # 执行conv2
        output = conv2(x_transposed)
        print(f"   输出形状: {output.shape}")
        
        # 检查输出形状是否正确
        expected_shape = (batch_size, 80, seq_len)
        if output.shape == expected_shape:
            print(f"   ✅ Conv2前向传播正确: 输入{x_transposed.shape} -> 输出{output.shape}")
        else:
            print(f"   ❌ Conv2前向传播错误: 期望输出{expected_shape}, 实际{output.shape}")
            
    except Exception as e:
        print(f"   ❌ Conv2前向传播测试失败: {e}")

def main():
    """主函数"""
    print("🔍 Conv2差异分析工具")
    print("=" * 60)
    
    # 加载调试数据
    data = load_debug_data()
    if data is None:
        return
    
    # 分析conv2输入
    analyze_conv2_inputs(data)
    
    # 分析conv2转置操作
    analyze_conv2_transpose_operations(data)
    
    # 测试conv2权重加载
    test_conv2_weight_loading()
    
    # 测试conv2前向传播
    test_conv2_forward_pass()
    
    print("\n✅ 分析完成")

if __name__ == "__main__":
    main()

