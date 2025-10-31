#!/usr/bin/env python3
"""
测试实际权重转换过程中的数值精度问题
"""

import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append('/Users/bailin/index-tts')

def test_actual_weight_conversion():
    """测试实际权重转换过程中的数值精度问题"""
    
    print("🔍 测试实际权重转换过程")
    
    # 加载 s2mel 模型
    try:
        s2mel_path = "/Users/bailin/index-tts/checkpoints/s2mel.pth"
        s2mel_state = torch.load(s2mel_path, map_location='cpu')
        
        print("✅ 成功加载 s2mel.pth")
        
        # 查找 CFM 相关权重
        cfm_keys = [k for k in s2mel_state.keys() if 'cfm' in k.lower()]
        print(f"找到 {len(cfm_keys)} 个 CFM 相关权重")
        
        # 查找 final_layer 权重
        final_layer_keys = [k for k in s2mel_state.keys() if 'final_layer' in k]
        print(f"找到 {len(final_layer_keys)} 个 final_layer 权重:")
        for key in final_layer_keys:
            if hasattr(s2mel_state[key], 'shape'):
                print(f"  {key}: {s2mel_state[key].shape}")
        
        # 查找 WaveNet 相关权重
        wavenet_keys = [k for k in s2mel_state.keys() if 'wavenet' in k.lower()]
        print(f"找到 {len(wavenet_keys)} 个 WaveNet 相关权重:")
        for key in wavenet_keys[:5]:  # 只显示前5个
            if hasattr(s2mel_state[key], 'shape'):
                print(f"  {key}: {s2mel_state[key].shape}")
        
        # 查找 in_layers 权重
        in_layers_keys = [k for k in s2mel_state.keys() if 'in_layers' in k]
        print(f"找到 {len(in_layers_keys)} 个 in_layers 权重:")
        for key in in_layers_keys[:10]:  # 只显示前10个
            if hasattr(s2mel_state[key], 'shape'):
                print(f"  {key}: {s2mel_state[key].shape}")
        
        # 测试第0层卷积权重的转换
        layer0_weight_key = None
        layer0_bias_key = None
        
        for key in in_layers_keys:
            if 'in_layers.0' in key and 'weight' in key:
                layer0_weight_key = key
            elif 'in_layers.0' in key and 'bias' in key:
                layer0_bias_key = key
        
        if layer0_weight_key and layer0_bias_key:
            print(f"\n🔍 测试第0层卷积权重转换:")
            
            # 获取原始权重
            original_weight = s2mel_state[layer0_weight_key]
            original_bias = s2mel_state[layer0_bias_key]
            
            print(f"  原始权重形状: {original_weight.shape}")
            print(f"  原始偏置形状: {original_bias.shape}")
            print(f"  权重范围: [{original_weight.min().item():.6f}, {original_weight.max().item():.6f}]")
            print(f"  权重均值: {original_weight.mean().item():.6f}")
            print(f"  权重标准差: {original_weight.std().item():.6f}")
            
            # 转换为 MLX 格式
            def convert_conv1d_weight(w):
                """PyTorch Conv1d (O, I, K) -> MLX (O, K, I)"""
                if len(w.shape) == 3:
                    return np.transpose(w.numpy(), (0, 2, 1))  # (O, I, K) -> (O, K, I)
                return w.numpy()
            
            converted_weight = convert_conv1d_weight(original_weight)
            converted_bias = original_bias.numpy()
            
            print(f"  转换后权重形状: {converted_weight.shape}")
            print(f"  转换后偏置形状: {converted_bias.shape}")
            
            # 检查转换精度
            reconverted = np.transpose(converted_weight, (0, 2, 1))  # 转回原始格式
            weight_diff = np.abs(original_weight.numpy() - reconverted)
            print(f"  转换精度检查:")
            print(f"    最大差异: {weight_diff.max():.10f}")
            print(f"    平均差异: {weight_diff.mean():.10f}")
            
            # 测试实际卷积操作
            print(f"\n🧪 测试实际卷积操作:")
            
            # 创建测试输入
            batch_size, seq_len = 2, 415
            in_channels = original_weight.shape[1]
            out_channels = original_weight.shape[0]
            
            # PyTorch 输入: (batch, channels, seq_len)
            x_pytorch = torch.randn(batch_size, in_channels, seq_len)
            
            # MLX 输入: (batch, seq_len, channels)
            x_mlx = mx.array(x_pytorch.numpy().transpose(0, 2, 1))
            
            print(f"  输入形状 - PyTorch: {x_pytorch.shape}, MLX: {x_mlx.shape}")
            
            # PyTorch 卷积
            torch_conv = torch.nn.Conv1d(in_channels, out_channels, 
                                       kernel_size=original_weight.shape[2], 
                                       dilation=1, bias=True, padding=2)
            torch_conv.weight.data = original_weight
            torch_conv.bias.data = original_bias
            
            torch_output = torch_conv(x_pytorch)
            
            # MLX 卷积
            mlx_conv = nn.Conv1d(in_channels, out_channels, 
                               kernel_size=original_weight.shape[2], 
                               dilation=1, bias=True, padding=2)
            mlx_conv.weight = mx.array(converted_weight)
            mlx_conv.bias = mx.array(converted_bias)
            
            mlx_output = mlx_conv(x_mlx)
            
            # 比较结果
            mlx_output_torch_format = torch.from_numpy(np.array(mlx_output).transpose(0, 2, 1))
            diff = torch.abs(torch_output - mlx_output_torch_format)
            
            print(f"  输出形状 - PyTorch: {torch_output.shape}, MLX: {mlx_output_torch_format.shape}")
            print(f"  数值差异:")
            print(f"    最大差异: {diff.max().item():.6f}")
            print(f"    平均差异: {diff.mean().item():.6f}")
            print(f"    标准差: {diff.std().item():.6f}")
            
            # 分析差异分布
            large_diff = diff > 1e-5
            print(f"    大差异(>1e-5)数量: {large_diff.sum().item()}/{diff.numel()} ({100*large_diff.sum().item()/diff.numel():.1f}%)")
            
            if large_diff.sum().item() > 0:
                print(f"    最大差异位置: {torch.unravel_index(diff.argmax(), diff.shape)}")
                print(f"    最大差异值: {diff.max().item():.6f}")
                
                # 显示前几个差异最大的位置
                top_diff_indices = torch.topk(diff.flatten(), 5).indices
                print(f"    前5个最大差异:")
                for i, idx in enumerate(top_diff_indices):
                    pos = torch.unravel_index(idx, diff.shape)
                    print(f"      位置 {pos}: PyTorch={torch_output[pos].item():.6f}, MLX={mlx_output_torch_format[pos].item():.6f}, 差异={diff[pos].item():.6f}")
            
            # 检查是否是由于数值精度问题
            if diff.max().item() < 1e-4:
                print(f"  ✅ 差异在可接受范围内 (< 1e-4)")
            else:
                print(f"  ❌ 存在显著差异 (>= 1e-4)")
                
                # 进一步分析
                print(f"\n🔍 进一步分析:")
                
                # 检查输入是否一致
                x_mlx_torch_format = torch.from_numpy(np.array(x_mlx).transpose(0, 2, 1))
                input_diff = torch.abs(x_pytorch - x_mlx_torch_format)
                print(f"  输入差异: {input_diff.max().item():.10f}")
                
                # 检查权重是否一致
                weight_torch_format = torch.from_numpy(np.transpose(converted_weight, (0, 2, 1)))
                weight_diff = torch.abs(original_weight - weight_torch_format)
                print(f"  权重差异: {weight_diff.max().item():.10f}")
                
                # 检查偏置是否一致
                bias_diff = torch.abs(original_bias - torch.from_numpy(converted_bias))
                print(f"  偏置差异: {bias_diff.max().item():.10f}")
        
        else:
            print("❌ 未找到第0层卷积权重")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_actual_weight_conversion()