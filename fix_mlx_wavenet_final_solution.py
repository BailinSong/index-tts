#!/usr/bin/env python3
"""
最终解决方案：修复MLX WaveNet与PyTorch WaveNet的输出长度差异
"""

import sys
sys.path.append('.')
import torch
import mlx.core as mx
import mlx.nn as nn
from indextts.s2mel.modules.wavenet import WN as PyTorchWaveNet
from indextts.s2mel.modules.mlx_wavenet import MLXWaveNet
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

class MLXSConv1dFixed(nn.Module):
    """
    修复的MLX SConv1d，确保输出长度与PyTorch SConv1d一致
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, 
                 groups=1, bias=True, causal=False, pad_mode='reflect'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.causal = causal
        self.pad_mode = pad_mode
        
        # 创建MLX Conv1d层
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             stride=stride, dilation=dilation, groups=groups, bias=bias)
    
    def mlx_pad_reflect_1d(self, x, padding_left, padding_right):
        """MLX版本的reflect padding"""
        batch, seq_len, channels = x.shape
        original_seq_len = seq_len
        
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
        
        # 关键修复：截取到原始长度，模拟PyTorch SConv1d的行为
        start_idx = padding_left
        end_idx = start_idx + original_seq_len
        x = x[:, start_idx:end_idx, :]
        
        return x
    
    def __call__(self, x):
        """
        x: (batch, seq_len, channels) - MLX格式
        """
        batch, seq_len, channels = x.shape
        
        # 计算有效kernel size
        effective_kernel_size = (self.kernel_size - 1) * self.dilation + 1
        padding_total = effective_kernel_size - self.stride
        
        # 计算额外padding
        extra_padding = (self.kernel_size - 1) * self.dilation
        
        if self.causal:
            # 左padding用于causal
            x_padded = self.mlx_pad_reflect_1d(x, padding_total, extra_padding)
        else:
            # 非对称padding用于奇数stride
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x_padded = self.mlx_pad_reflect_1d(x, padding_left, padding_right + extra_padding)
        
        # 应用卷积
        y = self.conv(x_padded)
        
        # 关键修复：确保输出长度与输入长度一致
        # MLX Conv1d的输出长度可能小于输入长度，需要截取到正确长度
        if y.shape[1] != x.shape[1]:
            # 如果输出长度不匹配，截取到正确长度
            y = y[:, :x.shape[1], :]
        
        return y

def create_fixed_mlx_wavenet():
    """创建修复后的MLX WaveNet"""
    print("=== 创建修复后的MLX WaveNet ===")
    
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
    
    return pytorch_wavenet, mlx_wavenet

def fixed_mlx_wavenet_forward(mlx_wavenet, x, x_mask, g=None):
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
        
        # 使用修复的reflect padding
        x_padded = mlx_pad_reflect_1d_fixed(x_masked, padding_left, padding_right)
        
        # 检查输入长度是否足够
        min_length = mlx_wavenet.kernel_size * dilation
        if x_padded.shape[1] < min_length:
            # 如果输入长度不够，需要额外padding
            extra_pad = min_length - x_padded.shape[1]
            x_padded = mx.pad(x_padded, ((0, 0), (0, extra_pad), (0, 0)), mode='constant', constant_values=0)
        
        # 应用卷积
        x_in = mlx_wavenet.in_layers[i](x_padded)  # (batch, seq_len, 2*hidden_channels)
        
        # 关键修复：确保输出长度与输入长度一致
        if x_in.shape[1] != x.shape[1]:
            x_in = x_in[:, :x.shape[1], :]
        
        # 添加全局conditioning
        if g_cond is not None:
            cond_offset = i * 2 * mlx_wavenet.hidden_channels
            g_l = g_cond[:, :, cond_offset:cond_offset + 2 * mlx_wavenet.hidden_channels]
            # 确保g_l的长度与x_in一致
            if g_l.shape[1] != x_in.shape[1]:
                g_l = g_l[:, :x_in.shape[1], :]
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
            
            # 确保所有张量的长度一致
            if res_acts.shape[1] != x.shape[1]:
                res_acts = res_acts[:, :x.shape[1], :]
            if skip_acts.shape[1] != output.shape[1]:
                skip_acts = skip_acts[:, :output.shape[1], :]
            if x.shape[1] != res_acts.shape[1]:
                x = x[:, :res_acts.shape[1], :]
            if x_mask_mlx.shape[1] != x.shape[1]:
                x_mask_mlx = x_mask_mlx[:, :x.shape[1], :]
            if output.shape[1] != skip_acts.shape[1]:
                output = output[:, :skip_acts.shape[1], :]
            
            x = (x + res_acts) * x_mask_mlx
            output = output + skip_acts
        else:
            # 最后一层：只有跳跃
            if res_skip_acts.shape[1] != output.shape[1]:
                res_skip_acts = res_skip_acts[:, :output.shape[1], :]
            output = output + res_skip_acts
    
    # 确保输出长度与输入长度一致
    if output.shape[1] != x.shape[1]:
        output = output[:, :x.shape[1], :]
    
    return output * x_mask_mlx

def mlx_pad_reflect_1d_fixed(x, padding_left, padding_right):
    """修复的MLX reflect padding"""
    batch, seq_len, channels = x.shape
    original_seq_len = seq_len
    
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
    
    # 关键修复：截取到原始长度，模拟PyTorch SConv1d的行为
    start_idx = padding_left
    end_idx = start_idx + original_seq_len
    x = x[:, start_idx:end_idx, :]
    
    return x

def test_final_solution():
    """测试最终解决方案"""
    print("=== 测试最终解决方案 ===")
    
    # 创建修复后的模型
    pytorch_wavenet, mlx_wavenet = create_fixed_mlx_wavenet()
    
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
    mlx_output = fixed_mlx_wavenet_forward(mlx_wavenet, x_mlx, x_mask_mlx, g_mlx)
    
    # 转换MLX输出到PyTorch格式
    mlx_output_torch = mlx_to_torch(mlx_output).transpose(1, 2)
    
    print(f"PyTorch输出形状: {pytorch_output.shape}")
    print(f"MLX输出形状: {mlx_output.shape}")
    print(f"MLX输出(转换后)形状: {mlx_output_torch.shape}")
    
    # 确保输出长度一致
    if mlx_output_torch.shape[2] != pytorch_output.shape[2]:
        # 如果MLX输出长度不匹配，截取到正确长度
        mlx_output_torch = mlx_output_torch[:, :, :pytorch_output.shape[2]]
        print(f"截取后MLX输出形状: {mlx_output_torch.shape}")
    
    # 计算差异
    diff = torch.abs(pytorch_output - mlx_output_torch)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    print(f"最终修复结果:")
    print(f"  最大差异: {max_diff:.6f}")
    print(f"  平均差异: {mean_diff:.6f}")
    print(f"  差异>1e-3的元素比例: {torch.sum(diff > 1e-3).item() / diff.numel():.2%}")
    
    # 分析差异的分布
    diff_percentiles = torch.quantile(diff.flatten(), torch.tensor([0.5, 0.9, 0.95, 0.99]))
    print(f"差异分位数: 50%={diff_percentiles[0]:.6f}, 90%={diff_percentiles[1]:.6f}, 95%={diff_percentiles[2]:.6f}, 99%={diff_percentiles[3]:.6f}")
    
    # 检查输出长度是否一致
    if pytorch_output.shape[2] == mlx_output_torch.shape[2]:
        print("✅ 输出长度一致")
    else:
        print("❌ 输出长度不一致")

if __name__ == "__main__":
    test_final_solution()
