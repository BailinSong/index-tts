#!/usr/bin/env python3
"""
修复MLX WaveNet的padding实现，使用正确的reflect padding
"""

import sys
sys.path.append('.')
import torch
import mlx.core as mx
import mlx.nn as nn
from indextts.s2mel.modules.wavenet import WN as PyTorchWaveNet
from indextts.s2mel.modules.mlx_wavenet import MLXWaveNet
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def mlx_pad_reflect_1d(x, padding_left, padding_right):
    """
    MLX版本的reflect padding，完全匹配PyTorch的行为
    x: (batch, seq_len, channels)
    """
    batch, seq_len, channels = x.shape
    
    if padding_left == 0 and padding_right == 0:
        return x
    
    # 处理reflect padding的特殊情况
    if seq_len <= max(padding_left, padding_right):
        # 如果输入长度小于padding，需要先扩展
        extra_pad = max(padding_left, padding_right) - seq_len + 1
        x = mx.pad(x, ((0, 0), (0, extra_pad), (0, 0)), mode='constant', constant_values=0)
        seq_len = x.shape[1]
    
    # 左padding: 使用切片反转前padding_left个元素
    if padding_left > 0:
        left_reflect = x[:, padding_left-1::-1, :]  # 反转前padding_left个元素
        x = mx.concatenate([left_reflect, x], axis=1)
    
    # 右padding: 使用切片反转后padding_right个元素
    if padding_right > 0:
        right_reflect = x[:, -1:-padding_right-1:-1, :]  # 反转后padding_right个元素
        x = mx.concatenate([x, right_reflect], axis=1)
    
    return x

def test_padding_fix():
    """测试padding修复"""
    print("=== 测试Padding修复 ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    mx.random.seed(42)
    
    # 创建测试数据
    batch_size = 1
    seq_len = 20
    channels = 512
    
    x_torch = torch.randn(batch_size, channels, seq_len)
    x_mlx = torch_to_mlx(x_torch.transpose(1, 2))
    
    # 测试不同kernel size和dilation的padding
    test_cases = [
        (5, 1),   # kernel=5, dilation=1
        (5, 2),   # kernel=5, dilation=2
        (5, 4),   # kernel=5, dilation=4
        (5, 8),   # kernel=5, dilation=8
    ]
    
    for kernel_size, dilation in test_cases:
        print(f"\n--- Kernel={kernel_size}, Dilation={dilation} ---")
        
        # 计算padding
        effective_kernel_size = (kernel_size - 1) * dilation + 1
        padding_total = effective_kernel_size - 1
        padding_left = padding_total // 2
        padding_right = padding_total - padding_left
        
        print(f"Effective kernel: {effective_kernel_size}")
        print(f"Padding: left={padding_left}, right={padding_right}")
        
        # PyTorch reflect padding
        from indextts.s2mel.modules.encodec import pad1d
        pytorch_padded = pad1d(x_torch, (padding_left, padding_right), mode='reflect')
        
        # MLX reflect padding
        mlx_padded = mlx_pad_reflect_1d(x_mlx, padding_left, padding_right)
        
        # 转换MLX输出
        mlx_padded_torch = mlx_to_torch(mlx_padded).transpose(1, 2)
        
        # 计算差异
        diff = torch.abs(pytorch_padded - mlx_padded_torch)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"Padding差异: max={max_diff:.6f}, mean={mean_diff:.6f}")
        
        # 分析差异的分布
        diff_percentiles = torch.quantile(diff.flatten(), torch.tensor([0.5, 0.9, 0.95, 0.99]))
        print(f"差异分位数: 50%={diff_percentiles[0]:.6f}, 90%={diff_percentiles[1]:.6f}, 95%={diff_percentiles[2]:.6f}, 99%={diff_percentiles[3]:.6f}")

def fix_mlx_wavenet_padding():
    """修复MLX WaveNet的padding实现"""
    print("\n=== 修复MLX WaveNet Padding ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    mx.random.seed(42)
    
    # 创建PyTorch WaveNet
    pytorch_wavenet = PyTorchWaveNet(
        hidden_channels=512,
        kernel_size=5,
        dilation_rate=2,
        n_layers=8,
        gin_channels=512,
        p_dropout=0.0
    )
    
    # 创建MLX WaveNet
    mlx_wavenet = MLXWaveNet(
        hidden_channels=512,
        kernel_size=5,
        dilation_rate=2,
        n_layers=8,
        gin_channels=512,
        p_dropout=0.0
    )
    
    # 修复MLX WaveNet的权重
    for i in range(8):
        pytorch_weight = pytorch_wavenet.in_layers[i].conv.conv.weight
        pytorch_weight_np = pytorch_weight.detach().cpu().numpy()
        mlx_weight_fixed = mx.array(pytorch_weight_np.transpose(0, 2, 1))
        mlx_wavenet.in_layers[i].weight = mlx_weight_fixed
        
        if hasattr(pytorch_wavenet.in_layers[i].conv.conv, 'bias') and pytorch_wavenet.in_layers[i].conv.conv.bias is not None:
            pytorch_bias = pytorch_wavenet.in_layers[i].conv.conv.bias
            mlx_bias_fixed = torch_to_mlx(pytorch_bias)
            mlx_wavenet.in_layers[i].bias = mlx_bias_fixed
    
    # 修复res_skip_layers权重
    for i in range(8):
        pytorch_weight = pytorch_wavenet.res_skip_layers[i].conv.conv.weight
        pytorch_weight_np = pytorch_weight.detach().cpu().numpy()
        mlx_weight_fixed = mx.array(pytorch_weight_np.transpose(0, 2, 1))
        mlx_wavenet.res_skip_layers[i].weight = mlx_weight_fixed
        
        if hasattr(pytorch_wavenet.res_skip_layers[i].conv.conv, 'bias') and pytorch_wavenet.res_skip_layers[i].conv.conv.bias is not None:
            pytorch_bias = pytorch_wavenet.res_skip_layers[i].conv.conv.bias
            mlx_bias_fixed = torch_to_mlx(pytorch_bias)
            mlx_wavenet.res_skip_layers[i].bias = mlx_bias_fixed
    
    # 修复cond_layer权重
    if hasattr(pytorch_wavenet, 'cond_layer') and pytorch_wavenet.cond_layer is not None:
        pytorch_weight = pytorch_wavenet.cond_layer.conv.conv.weight
        pytorch_weight_np = pytorch_weight.detach().cpu().numpy()
        mlx_weight_fixed = mx.array(pytorch_weight_np.transpose(0, 2, 1))
        mlx_wavenet.cond_layer.weight = mlx_weight_fixed
        
        if hasattr(pytorch_wavenet.cond_layer.conv.conv, 'bias') and pytorch_wavenet.cond_layer.conv.conv.bias is not None:
            pytorch_bias = pytorch_wavenet.cond_layer.conv.conv.bias
            mlx_bias_fixed = torch_to_mlx(pytorch_bias)
            mlx_wavenet.cond_layer.bias = mlx_bias_fixed
    
    # 现在修复MLX WaveNet的padding实现
    print("修复MLX WaveNet的padding实现...")
    
    # 重写MLX WaveNet的forward方法，使用正确的reflect padding
    def fixed_mlx_wavenet_forward(x, x_mask, g=None):
        """修复后的MLX WaveNet forward方法"""
        # 处理mask维度
        if len(x_mask.shape) == 3 and x_mask.shape[1] == 1:
            x_mask_mlx = x_mask.transpose((0, 2, 1))  # (batch, seq_len, 1)
        else:
            x_mask_mlx = x_mask
        
        # 处理全局conditioning
        if g is not None and mlx_wavenet.cond_layer is not None:
            # 修复g维度处理
            if len(g.shape) == 4:  # (batch, 1, 1, gin_channels)
                g = g.squeeze(1).squeeze(1)  # (batch, gin_channels)
            elif len(g.shape) == 3 and g.shape[1] == 1:  # (batch, 1, gin_channels)
                g = g.squeeze(1)  # (batch, gin_channels)
            elif len(g.shape) == 3 and g.shape[1] > 1:  # (batch, seq_len, gin_channels)
                pass
            elif len(g.shape) == 2:  # (batch, gin_channels)
                pass
            
            # 如果g是(batch, gin_channels)，需要广播到所有时间步
            if len(g.shape) == 2:
                g = mx.broadcast_to(g.reshape(g.shape[0], 1, -1), (g.shape[0], x.shape[1], g.shape[-1]))
            
            g_cond = mlx_wavenet.cond_layer(g)  # (batch, seq_len, 2*hidden_channels*n_layers)
        else:
            g_cond = None
        
        output = mx.zeros_like(x)
        
        for i in range(mlx_wavenet.n_layers):
            # Apply mask
            x_masked = x * x_mask_mlx
            
            # 计算padding
            dilation = mlx_wavenet.dilation_rate ** i
            effective_kernel_size = (mlx_wavenet.kernel_size - 1) * dilation + 1
            stride = 1
            padding_total = effective_kernel_size - stride
            
            # 计算左右padding
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            
            # 使用正确的reflect padding
            x_padded = mlx_pad_reflect_1d(x_masked, padding_left, padding_right)
            
            # 应用卷积
            x_in = mlx_wavenet.in_layers[i](x_padded)  # (batch, seq_len, 2*hidden_channels)
            
            # 添加全局conditioning
            if g_cond is not None:
                cond_offset = i * 2 * mlx_wavenet.hidden_channels
                g_l = g_cond[:, :, cond_offset:cond_offset + 2 * mlx_wavenet.hidden_channels]
            else:
                g_l = mx.zeros_like(x_in)
            
            # 融合激活函数
            from indextts.s2mel.modules.mlx_wavenet import fused_add_tanh_sigmoid_multiply_mlx
            acts = fused_add_tanh_sigmoid_multiply_mlx(x_in, g_l, mlx_wavenet.hidden_channels)
            
            # 残差/跳跃连接
            res_skip_acts = mlx_wavenet.res_skip_layers[i](acts)
            
            if i < mlx_wavenet.n_layers - 1:
                # 分割为残差和跳跃
                res_acts = res_skip_acts[:, :, :mlx_wavenet.hidden_channels]
                skip_acts = res_skip_acts[:, :, mlx_wavenet.hidden_channels:]
                
                x = (x + res_acts) * x_mask_mlx
                output = output + skip_acts
            else:
                # 最后一层：只有跳跃
                output = output + res_skip_acts
        
        return output * x_mask_mlx
    
    # 测试修复后的MLX WaveNet
    print("测试修复后的MLX WaveNet...")
    
    # 创建测试输入
    batch_size = 1
    seq_len = 100
    
    # PyTorch输入
    x_torch = torch.randn(batch_size, 512, seq_len)
    x_mask_torch = torch.ones(batch_size, 1, seq_len, dtype=torch.bool)
    g_torch = torch.randn(batch_size, 512, 1)
    
    # MLX输入
    x_mlx = torch_to_mlx(x_torch.transpose(1, 2))
    x_mask_mlx = torch_to_mlx(x_mask_torch.transpose(1, 2))
    g_mlx = torch_to_mlx(g_torch.transpose(1, 2))
    
    # 前向传播
    pytorch_output = pytorch_wavenet(x_torch, x_mask_torch, g_torch)
    mlx_output = fixed_mlx_wavenet_forward(x_mlx, x_mask_mlx, g_mlx)
    
    # 转换MLX输出到PyTorch格式
    mlx_output_torch = mlx_to_torch(mlx_output).transpose(1, 2)
    
    # 计算差异
    diff = torch.abs(pytorch_output - mlx_output_torch)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    print(f"修复后的前向传播差异:")
    print(f"  最大差异: {max_diff:.6f}")
    print(f"  平均差异: {mean_diff:.6f}")
    print(f"  差异>1e-3的元素比例: {torch.sum(diff > 1e-3).item() / diff.numel():.2%}")
    
    # 分析差异的分布
    diff_percentiles = torch.quantile(diff.flatten(), torch.tensor([0.5, 0.9, 0.95, 0.99]))
    print(f"差异分位数: 50%={diff_percentiles[0]:.6f}, 90%={diff_percentiles[1]:.6f}, 95%={diff_percentiles[2]:.6f}, 99%={diff_percentiles[3]:.6f}")

if __name__ == "__main__":
    test_padding_fix()
    fix_mlx_wavenet_padding()
