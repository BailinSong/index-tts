#!/usr/bin/env python3
"""
深入对比PyTorch和MLX WaveNet的逐层实现
分析padding、激活函数、残差连接的差异
"""

import sys
sys.path.append('.')
import torch
import mlx.core as mx
import numpy as np
import pickle
from indextts.s2mel.modules.wavenet import WN as PyTorchWaveNet
from indextts.s2mel.modules.mlx_wavenet import MLXWaveNet
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

class WaveNetDebugger:
    def __init__(self):
        self.pytorch_results = {}
        self.mlx_results = {}
        
    def debug_pytorch_wavenet(self, wavenet, x, x_mask, g=None):
        """调试PyTorch WaveNet的逐层输出"""
        print("=== 调试PyTorch WaveNet ===")
        
        # 保存原始输入
        self.pytorch_results['input'] = x
        self.pytorch_results['mask'] = x_mask
        self.pytorch_results['g'] = g
        
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([wavenet.hidden_channels])
        
        # 处理全局conditioning
        if g is not None and wavenet.cond_layer is not None:
            g_cond = wavenet.cond_layer(g)
            self.pytorch_results['g_cond'] = g_cond
            print(f"g_cond: {g_cond.shape}, range=[{torch.min(g_cond):.6f}, {torch.max(g_cond):.6f}]")
        else:
            g_cond = None
            self.pytorch_results['g_cond'] = None
        
        for i in range(wavenet.n_layers):
            print(f"\n--- Layer {i} ---")
            
            # 输入层处理
            x_in = wavenet.in_layers[i](x)
            self.pytorch_results[f'layer_{i}_x_in'] = x_in
            print(f"x_in: {x_in.shape}, range=[{torch.min(x_in):.6f}, {torch.max(x_in):.6f}]")
            
            # 全局conditioning处理
            if g is not None:
                cond_offset = i * 2 * wavenet.hidden_channels
                g_l = g_cond[:, cond_offset:cond_offset + 2 * wavenet.hidden_channels, :]
                self.pytorch_results[f'layer_{i}_g_l'] = g_l
                print(f"g_l: {g_l.shape}, range=[{torch.min(g_l):.6f}, {torch.max(g_l):.6f}]")
            else:
                g_l = torch.zeros_like(x_in)
                self.pytorch_results[f'layer_{i}_g_l'] = g_l
            
            # 融合激活函数
            from indextts.s2mel.modules.commons import fused_add_tanh_sigmoid_multiply
            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = wavenet.drop(acts)
            self.pytorch_results[f'layer_{i}_acts'] = acts
            print(f"acts: {acts.shape}, range=[{torch.min(acts):.6f}, {torch.max(acts):.6f}]")
            
            # 残差/跳跃连接
            res_skip_acts = wavenet.res_skip_layers[i](acts)
            self.pytorch_results[f'layer_{i}_res_skip_acts'] = res_skip_acts
            print(f"res_skip_acts: {res_skip_acts.shape}, range=[{torch.min(res_skip_acts):.6f}, {torch.max(res_skip_acts):.6f}]")
            
            if i < wavenet.n_layers - 1:
                res_acts = res_skip_acts[:, :wavenet.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, wavenet.hidden_channels:, :]
                self.pytorch_results[f'layer_{i}_res_acts'] = res_acts
                self.pytorch_results[f'layer_{i}_x_updated'] = x
                print(f"res_acts: {res_acts.shape}, range=[{torch.min(res_acts):.6f}, {torch.max(res_acts):.6f}]")
                print(f"x_updated: {x.shape}, range=[{torch.min(x):.6f}, {torch.max(x):.6f}]")
            else:
                output = output + res_skip_acts
                self.pytorch_results[f'layer_{i}_final_output'] = res_skip_acts
            
            self.pytorch_results[f'layer_{i}_output'] = output
            print(f"output: {output.shape}, range=[{torch.min(output):.6f}, {torch.max(output):.6f}]")
        
        final_output = output * x_mask
        self.pytorch_results['final_output'] = final_output
        print(f"\nFinal output: {final_output.shape}, range=[{torch.min(final_output):.6f}, {torch.max(final_output):.6f}]")
        
        return final_output
    
    def debug_mlx_wavenet(self, wavenet, x, x_mask, g=None):
        """调试MLX WaveNet的逐层输出"""
        print("=== 调试MLX WaveNet ===")
        
        # 保存原始输入
        self.mlx_results['input'] = x
        self.mlx_results['mask'] = x_mask
        self.mlx_results['g'] = g
        
        # 处理mask维度
        if len(x_mask.shape) == 3 and x_mask.shape[1] == 1:
            x_mask_mlx = x_mask.transpose((0, 2, 1))  # (batch, seq_len, 1)
        else:
            x_mask_mlx = x_mask
        
        # 处理全局conditioning
        if g is not None and wavenet.cond_layer is not None:
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
            
            g_cond = wavenet.cond_layer(g)
            self.mlx_results['g_cond'] = g_cond
            print(f"g_cond: {g_cond.shape}, range=[{mx.min(g_cond):.6f}, {mx.max(g_cond):.6f}]")
        else:
            g_cond = None
            self.mlx_results['g_cond'] = None
        
        output = mx.zeros_like(x)
        
        for i in range(wavenet.n_layers):
            print(f"\n--- Layer {i} ---")
            
            # 应用mask
            x_masked = x * x_mask_mlx
            self.mlx_results[f'layer_{i}_x_masked'] = x_masked
            print(f"x_masked: {x_masked.shape}, range=[{mx.min(x_masked):.6f}, {mx.max(x_masked):.6f}]")
            
            # 计算padding
            dilation = wavenet.dilation_rate ** i
            effective_kernel_size = (wavenet.kernel_size - 1) * dilation + 1
            stride = 1
            padding_total = effective_kernel_size - stride
            
            # 应用padding (模拟PyTorch SConv1d的行为)
            if padding_total > 0:
                # 计算左右padding
                padding_right = padding_total // 2
                padding_left = padding_total - padding_right
                
                # 模拟PyTorch pad1d的reflect模式行为
                # 1. 如果输入长度小于padding，先添加额外的0 padding
                length = x_masked.shape[1]
                max_pad = max(padding_left, padding_right)
                extra_pad = 0
                if length <= max_pad:
                    extra_pad = max_pad - length + 1
                    x_masked = mx.pad(x_masked, ((0, 0), (0, extra_pad), (0, 0)), mode='constant', constant_values=0)
                
                # 2. 进行edge padding (MLX不支持reflect)
                x_padded = mx.pad(x_masked, ((0, 0), (padding_left, padding_right), (0, 0)), mode='edge')
                
                # 3. 截取到原始长度 (模拟PyTorch的行为)
                end = x_padded.shape[1] - extra_pad
                x_padded = x_padded[:, :end, :]
                
                self.mlx_results[f'layer_{i}_x_padded'] = x_padded
                print(f"x_padded: {x_padded.shape}, range=[{mx.min(x_padded):.6f}, {mx.max(x_padded):.6f}]")
            else:
                x_padded = x_masked
                self.mlx_results[f'layer_{i}_x_padded'] = x_padded
            
            # 输入层处理
            x_in = wavenet.in_layers[i](x_padded)
            self.mlx_results[f'layer_{i}_x_in'] = x_in
            print(f"x_in: {x_in.shape}, range=[{mx.min(x_in):.6f}, {mx.max(x_in):.6f}]")
            
            # 全局conditioning处理
            if g is not None:
                cond_offset = i * 2 * wavenet.hidden_channels
                g_l = g_cond[:, :, cond_offset:cond_offset + 2 * wavenet.hidden_channels]
                self.mlx_results[f'layer_{i}_g_l'] = g_l
                print(f"g_l: {g_l.shape}, range=[{mx.min(g_l):.6f}, {mx.max(g_l):.6f}]")
            else:
                g_l = mx.zeros_like(x_in)
                self.mlx_results[f'layer_{i}_g_l'] = g_l
            
            # 融合激活函数
            from indextts.s2mel.modules.mlx_wavenet import fused_add_tanh_sigmoid_multiply_mlx
            acts = fused_add_tanh_sigmoid_multiply_mlx(x_in, g_l, wavenet.hidden_channels)
            acts = wavenet.dropout(acts)
            self.mlx_results[f'layer_{i}_acts'] = acts
            print(f"acts: {acts.shape}, range=[{mx.min(acts):.6f}, {mx.max(acts):.6f}]")
            
            # 残差/跳跃连接
            res_skip_acts = wavenet.res_skip_layers[i](acts)
            self.mlx_results[f'layer_{i}_res_skip_acts'] = res_skip_acts
            print(f"res_skip_acts: {res_skip_acts.shape}, range=[{mx.min(res_skip_acts):.6f}, {mx.max(res_skip_acts):.6f}]")
            
            if i < wavenet.n_layers - 1:
                res_acts = res_skip_acts[:, :, :wavenet.hidden_channels]
                x = (x + res_acts) * x_mask_mlx
                output = output + res_skip_acts[:, :, wavenet.hidden_channels:]
                self.mlx_results[f'layer_{i}_res_acts'] = res_acts
                self.mlx_results[f'layer_{i}_x_updated'] = x
                print(f"res_acts: {res_acts.shape}, range=[{mx.min(res_acts):.6f}, {mx.max(res_acts):.6f}]")
                print(f"x_updated: {x.shape}, range=[{mx.min(x):.6f}, {mx.max(x):.6f}]")
            else:
                output = output + res_skip_acts
                self.mlx_results[f'layer_{i}_final_output'] = res_skip_acts
            
            self.mlx_results[f'layer_{i}_output'] = output
            print(f"output: {output.shape}, range=[{mx.min(output):.6f}, {mx.max(output):.6f}]")
        
        final_output = output * x_mask_mlx
        self.mlx_results['final_output'] = final_output
        print(f"\nFinal output: {final_output.shape}, range=[{mx.min(final_output):.6f}, {mx.max(final_output):.6f}]")
        
        return final_output
    
    def compare_results(self):
        """对比PyTorch和MLX的结果"""
        print("\n=== 逐层差异分析 ===")
        
        # 对比最终输出
        pytorch_final = self.pytorch_results['final_output']
        mlx_final = self.mlx_results['final_output']
        
        # 转换MLX输出到PyTorch格式进行对比
        mlx_final_torch = mlx_to_torch(mlx_final).transpose(1, 2)
        
        diff = torch.abs(pytorch_final - mlx_final_torch)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"最终输出差异:")
        print(f"  最大差异: {max_diff:.6f}")
        print(f"  平均差异: {mean_diff:.6f}")
        print(f"  差异>1e-3的元素比例: {torch.sum(diff > 1e-3).item() / diff.numel():.2%}")
        
        # 逐层对比
        for i in range(8):  # 假设8层
            if f'layer_{i}_acts' in self.pytorch_results and f'layer_{i}_acts' in self.mlx_results:
                pytorch_acts = self.pytorch_results[f'layer_{i}_acts']
                mlx_acts = self.mlx_results[f'layer_{i}_acts']
                
                # 转换MLX输出到PyTorch格式
                mlx_acts_torch = mlx_to_torch(mlx_acts).transpose(1, 2)
                
                diff = torch.abs(pytorch_acts - mlx_acts_torch)
                max_diff = torch.max(diff).item()
                mean_diff = torch.mean(diff).item()
                
                print(f"\n--- Layer {i} acts ---")
                print(f"最大差异: {max_diff:.6f}")
                print(f"平均差异: {mean_diff:.6f}")
                print(f"差异>1e-3的元素比例: {torch.sum(diff > 1e-3).item() / diff.numel():.2%}")

def main():
    """主测试函数"""
    print("=== WaveNet逐层对比调试工具 ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    mx.random.seed(42)
    
    # 创建WaveNet实例
    hidden_channels = 512
    kernel_size = 5
    dilation_rate = 2
    n_layers = 8
    gin_channels = 512
    
    pytorch_wavenet = PyTorchWaveNet(
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        n_layers=n_layers,
        gin_channels=gin_channels,
        p_dropout=0.0
    )
    
    mlx_wavenet = MLXWaveNet(
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        n_layers=n_layers,
        gin_channels=gin_channels,
        p_dropout=0.0
    )
    
    # 创建测试输入
    batch_size = 1
    seq_len = 100
    
    # PyTorch格式: (batch, channels, seq_len)
    x_torch = torch.randn(batch_size, hidden_channels, seq_len)
    x_mask_torch = torch.ones(batch_size, 1, seq_len, dtype=torch.bool)
    g_torch = torch.randn(batch_size, gin_channels, 1)
    
    # MLX格式: (batch, seq_len, channels)
    x_mlx = torch_to_mlx(x_torch.transpose(1, 2))
    x_mask_mlx = torch_to_mlx(x_mask_torch.transpose(1, 2))
    g_mlx = torch_to_mlx(g_torch.transpose(1, 2))
    
    print(f"输入形状:")
    print(f"  PyTorch: x={x_torch.shape}, mask={x_mask_torch.shape}, g={g_torch.shape}")
    print(f"  MLX: x={x_mlx.shape}, mask={x_mask_mlx.shape}, g={g_mlx.shape}")
    
    # 创建调试器
    debugger = WaveNetDebugger()
    
    # 调试PyTorch WaveNet
    pytorch_output = debugger.debug_pytorch_wavenet(pytorch_wavenet, x_torch, x_mask_torch, g_torch)
    
    # 调试MLX WaveNet
    mlx_output = debugger.debug_mlx_wavenet(mlx_wavenet, x_mlx, x_mask_mlx, g_mlx)
    
    # 对比结果
    debugger.compare_results()
    
    # 保存调试结果
    with open('wavenet_layer_by_layer_debug.pkl', 'wb') as f:
        pickle.dump({
            'pytorch_results': debugger.pytorch_results,
            'mlx_results': debugger.mlx_results
        }, f)
    
    print("\n调试结果已保存到: wavenet_layer_by_layer_debug.pkl")

if __name__ == "__main__":
    main()