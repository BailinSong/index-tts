#!/usr/bin/env python3
"""
测试 WaveNet 第0层的实际权重转换
"""

import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append('/Users/bailin/index-tts')

def test_wavenet_layer0_actual():
    """测试 WaveNet 第0层的实际权重转换"""
    
    print("🔍 测试 WaveNet 第0层实际权重转换")
    
    # 加载 s2mel 模型
    try:
        s2mel_path = "/Users/bailin/index-tts/checkpoints/s2mel.pth"
        s2mel_state = torch.load(s2mel_path, map_location='cpu')
        cfm = s2mel_state['net']['cfm']
        
        print("✅ 成功加载 s2mel.pth")
        
        # 获取第0层卷积权重
        layer0_weight_key = 'estimator.wavenet.in_layers.0.conv.conv.weight_v'
        layer0_bias_key = 'estimator.wavenet.in_layers.0.conv.conv.bias'
        layer0_weight_g_key = 'estimator.wavenet.in_layers.0.conv.conv.weight_g'
        
        if layer0_weight_key in cfm and layer0_bias_key in cfm:
            print(f"\n🔍 分析第0层卷积权重:")
            
            # 获取原始权重
            original_weight_v = cfm[layer0_weight_key]  # (1024, 512, 5)
            original_bias = cfm[layer0_bias_key]        # (1024,)
            original_weight_g = cfm[layer0_weight_g_key] # (1024, 1, 1)
            
            print(f"  原始权重形状:")
            print(f"    weight_v: {original_weight_v.shape}")
            print(f"    bias: {original_bias.shape}")
            print(f"    weight_g: {original_weight_g.shape}")
            
            print(f"  权重统计:")
            print(f"    weight_v 范围: [{original_weight_v.min().item():.6f}, {original_weight_v.max().item():.6f}]")
            print(f"    weight_v 均值: {original_weight_v.mean().item():.6f}")
            print(f"    weight_v 标准差: {original_weight_v.std().item():.6f}")
            print(f"    bias 范围: [{original_bias.min().item():.6f}, {original_bias.max().item():.6f}]")
            print(f"    weight_g 范围: [{original_weight_g.min().item():.6f}, {original_weight_g.max().item():.6f}]")
            
            # 重构权重 (weight_norm 格式)
            # 在 PyTorch 中，weight_norm 的权重是 weight_g * weight_v / ||weight_v||
            def reconstruct_weight_norm(weight_g, weight_v):
                """重构 weight_norm 权重"""
                # weight_g: (out_channels, 1, 1)
                # weight_v: (out_channels, in_channels, kernel_size)
                weight_g = weight_g.squeeze()  # (out_channels,)
                weight_v_norm = torch.norm(weight_v, dim=(1, 2), keepdim=True)  # (out_channels, 1, 1)
                weight_v_norm = weight_v_norm.squeeze()  # (out_channels,)
                
                # 避免除零
                weight_v_norm = torch.clamp(weight_v_norm, min=1e-8)
                
                # 重构权重
                reconstructed = weight_g.unsqueeze(1).unsqueeze(2) * weight_v / weight_v_norm.unsqueeze(1).unsqueeze(2)
                return reconstructed
            
            reconstructed_weight = reconstruct_weight_norm(original_weight_g, original_weight_v)
            print(f"  重构后权重形状: {reconstructed_weight.shape}")
            print(f"  重构后权重范围: [{reconstructed_weight.min().item():.6f}, {reconstructed_weight.max().item():.6f}]")
            
            # 转换为 MLX 格式
            def convert_conv1d_weight(w):
                """PyTorch Conv1d (O, I, K) -> MLX (O, K, I)"""
                if len(w.shape) == 3:
                    return np.transpose(w.numpy(), (0, 2, 1))  # (O, I, K) -> (O, K, I)
                return w.numpy()
            
            converted_weight = convert_conv1d_weight(reconstructed_weight)
            converted_bias = original_bias.numpy()
            
            print(f"  转换后权重形状: {converted_weight.shape}")
            print(f"  转换后偏置形状: {converted_bias.shape}")
            
            # 检查转换精度
            reconverted = np.transpose(converted_weight, (0, 2, 1))  # 转回原始格式
            weight_diff = np.abs(reconstructed_weight.numpy() - reconverted)
            print(f"  转换精度检查:")
            print(f"    最大差异: {weight_diff.max():.10f}")
            print(f"    平均差异: {weight_diff.mean():.10f}")
            
            # 测试实际卷积操作
            print(f"\n🧪 测试实际卷积操作:")
            
            # 创建测试输入 (使用实际的数据形状)
            batch_size, seq_len = 2, 415
            in_channels = reconstructed_weight.shape[1]  # 512
            out_channels = reconstructed_weight.shape[0]  # 1024
            kernel_size = reconstructed_weight.shape[2]   # 5
            
            print(f"  测试参数:")
            print(f"    batch_size: {batch_size}")
            print(f"    seq_len: {seq_len}")
            print(f"    in_channels: {in_channels}")
            print(f"    out_channels: {out_channels}")
            print(f"    kernel_size: {kernel_size}")
            
            # PyTorch 输入: (batch, channels, seq_len)
            x_pytorch = torch.randn(batch_size, in_channels, seq_len)
            
            # MLX 输入: (batch, seq_len, channels)
            x_mlx = mx.array(x_pytorch.numpy().transpose(0, 2, 1))
            
            print(f"  输入形状 - PyTorch: {x_pytorch.shape}, MLX: {x_mlx.shape}")
            
            # PyTorch 卷积 (使用 weight_norm)
            torch_conv = torch.nn.Conv1d(in_channels, out_channels, 
                                       kernel_size=kernel_size, 
                                       dilation=1, bias=True, padding=2)
            
            # 手动设置 weight_norm 权重
            with torch.no_grad():
                torch_conv.weight.data = reconstructed_weight
                torch_conv.bias.data = original_bias
            
            torch_output = torch_conv(x_pytorch)
            
            # MLX 卷积
            mlx_conv = nn.Conv1d(in_channels, out_channels, 
                               kernel_size=kernel_size, 
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
                weight_diff = torch.abs(reconstructed_weight - weight_torch_format)
                print(f"  权重差异: {weight_diff.max().item():.10f}")
                
                # 检查偏置是否一致
                bias_diff = torch.abs(original_bias - torch.from_numpy(converted_bias))
                print(f"  偏置差异: {bias_diff.max().item():.10f}")
                
                # 检查权重重构是否正确
                print(f"\n🔍 检查权重重构:")
                print(f"  原始 weight_g: {original_weight_g.squeeze()[:5]}")
                print(f"  原始 weight_v norm: {torch.norm(original_weight_v, dim=(1, 2))[:5]}")
                print(f"  重构后权重范围: [{reconstructed_weight.min().item():.6f}, {reconstructed_weight.max().item():.6f}]")
                
                # 检查是否有异常值
                weight_abs = torch.abs(reconstructed_weight)
                large_weights = weight_abs > 10.0
                if large_weights.sum().item() > 0:
                    print(f"  发现 {large_weights.sum().item()} 个大权重值 (>10.0)")
                    print(f"  最大权重值: {weight_abs.max().item():.6f}")
        
        else:
            print("❌ 未找到第0层卷积权重")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_wavenet_layer0_actual()
