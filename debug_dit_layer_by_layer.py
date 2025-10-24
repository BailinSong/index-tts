#!/usr/bin/env python3
"""
DiT逐层对比调试工具
对比PyTorch和MLX DiT的每层中间值，定位差异源头
"""

import sys
sys.path.append('.')
import torch
import mlx.core as mx
import numpy as np
import pickle
import os
from typing import Dict, Any, Tuple
from indextts.infer_v2 import IndexTTS2
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def analyze_tensor_stats(tensor, name: str) -> Dict[str, float]:
    """分析张量统计信息"""
    if isinstance(tensor, torch.Tensor):
        tensor_np = tensor.detach().cpu().numpy()
    else:  # MLX array
        tensor_np = np.array(tensor)
    
    return {
        'name': name,
        'shape': tensor_np.shape,
        'min': float(np.min(tensor_np)),
        'max': float(np.max(tensor_np)),
        'mean': float(np.mean(tensor_np)),
        'std': float(np.std(tensor_np)),
        'range': float(np.max(tensor_np) - np.min(tensor_np))
    }

def compare_tensors(pytorch_tensor, mlx_tensor, name: str) -> Dict[str, Any]:
    """对比两个张量"""
    # 转换为numpy数组进行比较
    if isinstance(pytorch_tensor, torch.Tensor):
        pytorch_np = pytorch_tensor.detach().cpu().numpy()
    else:
        pytorch_np = pytorch_tensor
    
    if isinstance(mlx_tensor, mx.array):
        mlx_np = np.array(mlx_tensor)
    else:
        mlx_np = mlx_tensor
    
    # 确保形状一致
    if pytorch_np.shape != mlx_np.shape:
        return {
            'name': name,
            'error': f'Shape mismatch: PyTorch {pytorch_np.shape} vs MLX {mlx_np.shape}',
            'pytorch_shape': pytorch_np.shape,
            'mlx_shape': mlx_np.shape
        }
    
    # 计算差异
    diff = np.abs(pytorch_np - mlx_np)
    
    return {
        'name': name,
        'pytorch_stats': analyze_tensor_stats(pytorch_tensor, f'{name}_pytorch'),
        'mlx_stats': analyze_tensor_stats(mlx_tensor, f'{name}_mlx'),
        'diff_stats': {
            'max_diff': float(np.max(diff)),
            'mean_diff': float(np.mean(diff)),
            'std_diff': float(np.std(diff)),
            'elements_above_1e5': int(np.sum(diff > 1e-5)),
            'elements_above_1e3': int(np.sum(diff > 1e-3)),
            'elements_above_1e1': int(np.sum(diff > 1e-1)),
            'total_elements': int(diff.size),
            'pct_above_1e5': float(np.sum(diff > 1e-5) / diff.size * 100),
            'pct_above_1e3': float(np.sum(diff > 1e-3) / diff.size * 100),
            'pct_above_1e1': float(np.sum(diff > 1e-1) / diff.size * 100)
        }
    }

class DiTLayerDebugger:
    """DiT逐层调试器"""
    
    def __init__(self):
        self.pytorch_results = {}
        self.mlx_results = {}
        self.comparisons = {}
        
    def debug_pytorch_dit(self, dit_model, x, prompt_x, x_lens, t, style, cond):
        """调试PyTorch DiT的逐层输出"""
        print("=== 调试PyTorch DiT ===")
        
        # 1. Timestep embedding
        t_emb = dit_model.t_embedder(t)
        self.pytorch_results['t_embedder'] = t_emb
        print(f"t_embedder: {t_emb.shape}, range=[{t_emb.min():.6f}, {t_emb.max():.6f}]")
        
        # 2. Condition projection
        cond_proj = dit_model.cond_projection(cond)
        self.pytorch_results['cond_projection'] = cond_proj
        print(f"cond_projection: {cond_proj.shape}, range=[{cond_proj.min():.6f}, {cond_proj.max():.6f}]")
        
        # 3. Input preparation
        x_t = x.transpose(1, 2)
        prompt_x_t = prompt_x.transpose(1, 2)
        x_in = torch.cat([x_t, prompt_x_t, cond_proj], dim=-1)
        self.pytorch_results['input_concat'] = x_in
        print(f"input_concat: {x_in.shape}, range=[{x_in.min():.6f}, {x_in.max():.6f}]")
        
        # 4. Style conditioning
        if dit_model.transformer_style_condition and not dit_model.style_as_token:
            style_broadcast = style[:, None, :].repeat(1, x_in.shape[1], 1).to(x_in.device)
            x_in = torch.cat([x_in, style_broadcast], dim=-1)
            self.pytorch_results['style_conditioning'] = x_in
            print(f"style_conditioning: {x_in.shape}, range=[{x_in.min():.6f}, {x_in.max():.6f}]")
        
        # 5. Merge linear
        x_in = dit_model.cond_x_merge_linear(x_in)
        self.pytorch_results['merge_linear'] = x_in
        print(f"merge_linear: {x_in.shape}, range=[{x_in.min():.6f}, {x_in.max():.6f}]")
        
        # 6. Transformer layers
        x_mask = torch.ones(1, 1, x_in.shape[1], dtype=torch.bool, device=x.device)
        input_pos = dit_model.input_pos[:x_in.shape[1]]
        x_res = dit_model.transformer(x_in, t_emb.unsqueeze(1), input_pos, x_mask)
        self.pytorch_results['transformer_output'] = x_res
        print(f"transformer_output: {x_res.shape}, range=[{x_res.min():.6f}, {x_res.max():.6f}]")
        
        # 7. Skip connection
        if dit_model.long_skip_connection:
            x_res = dit_model.skip_linear(torch.cat([x_res, x_t], dim=-1))
            self.pytorch_results['skip_connection'] = x_res
            print(f"skip_connection: {x_res.shape}, range=[{x_res.min():.6f}, {x_res.max():.6f}]")
        
        # 8. WaveNet processing
        if dit_model.final_layer_type == 'wavenet':
            x_conv1 = dit_model.conv1(x_res)
            x_conv1_t = x_conv1.transpose(1, 2)
            t2 = dit_model.t_embedder2(t)
            x_wavenet = dit_model.wavenet(x_conv1_t, x_mask, g=t2.unsqueeze(2))
            x_wavenet_t = x_wavenet.transpose(1, 2)
            x_final = dit_model.final_layer(x_wavenet_t, t_emb).transpose(1, 2)
            x_output = dit_model.conv2(x_final)
            
            self.pytorch_results['conv1'] = x_conv1
            self.pytorch_results['wavenet_input'] = x_conv1_t
            self.pytorch_results['wavenet_output'] = x_wavenet_t
            self.pytorch_results['final_layer'] = x_final
            self.pytorch_results['conv2'] = x_output
            
            print(f"conv1: {x_conv1.shape}, range=[{x_conv1.min():.6f}, {x_conv1.max():.6f}]")
            print(f"wavenet_input: {x_conv1_t.shape}, range=[{x_conv1_t.min():.6f}, {x_conv1_t.max():.6f}]")
            print(f"wavenet_output: {x_wavenet_t.shape}, range=[{x_wavenet_t.min():.6f}, {x_wavenet_t.max():.6f}]")
            print(f"final_layer: {x_final.shape}, range=[{x_final.min():.6f}, {x_final.max():.6f}]")
            print(f"conv2: {x_output.shape}, range=[{x_output.min():.6f}, {x_output.max():.6f}]")
            
            return x_output
        else:
            x_output = dit_model.final_mlp(x_res).transpose(1, 2)
            self.pytorch_results['final_mlp'] = x_output
            print(f"final_mlp: {x_output.shape}, range=[{x_output.min():.6f}, {x_output.max():.6f}]")
            return x_output
    
    def debug_mlx_dit(self, dit_model, x, prompt_x, x_lens, t, style, cond):
        """调试MLX DiT的逐层输出"""
        print("=== 调试MLX DiT ===")
        
        # 转换输入到MLX
        if isinstance(x, torch.Tensor):
            x = torch_to_mlx(x.cpu())
            prompt_x = torch_to_mlx(prompt_x.cpu())
            x_lens = torch_to_mlx(x_lens.cpu())
            t = torch_to_mlx(t.cpu())
            style = torch_to_mlx(style.cpu())
            cond = torch_to_mlx(cond.cpu())
        
        # 1. Timestep embedding
        t_emb = dit_model.t_embedder(t)
        self.mlx_results['t_embedder'] = t_emb
        print(f"t_embedder: {t_emb.shape}, range=[{mx.min(t_emb):.6f}, {mx.max(t_emb):.6f}]")
        
        # 2. Condition projection
        cond_proj = dit_model.cond_projection(cond)
        self.mlx_results['cond_projection'] = cond_proj
        print(f"cond_projection: {cond_proj.shape}, range=[{mx.min(cond_proj):.6f}, {mx.max(cond_proj):.6f}]")
        
        # 3. Input preparation
        x_t = x.transpose(0, 2, 1)
        prompt_x_t = prompt_x.transpose(0, 2, 1)
        x_in = mx.concatenate([x_t, prompt_x_t, cond_proj], axis=-1)
        self.mlx_results['input_concat'] = x_in
        print(f"input_concat: {x_in.shape}, range=[{mx.min(x_in):.6f}, {mx.max(x_in):.6f}]")
        
        # 4. Style conditioning
        if dit_model.transformer_style_condition and not dit_model.style_as_token:
            style_broadcast = mx.broadcast_to(
                style.reshape(style.shape[0], 1, -1),
                (style.shape[0], x_in.shape[1], style.shape[-1])
            )
            x_in = mx.concatenate([x_in, style_broadcast], axis=-1)
            self.mlx_results['style_conditioning'] = x_in
            print(f"style_conditioning: {x_in.shape}, range=[{mx.min(x_in):.6f}, {mx.max(x_in):.6f}]")
        
        # 5. Merge linear
        x_in = dit_model.cond_x_merge_linear(x_in)
        self.mlx_results['merge_linear'] = x_in
        print(f"merge_linear: {x_in.shape}, range=[{mx.min(x_in):.6f}, {mx.max(x_in):.6f}]")
        
        # 6. Transformer layers
        x_mask = mx.ones((1, 1, x_in.shape[1]), dtype=mx.bool_)
        input_pos = dit_model.input_pos[:x_in.shape[1]]
        x_res = dit_model.transformer(x_in, t_emb.reshape(t_emb.shape[0], 1, -1), input_pos, x_mask)
        self.mlx_results['transformer_output'] = x_res
        print(f"transformer_output: {x_res.shape}, range=[{mx.min(x_res):.6f}, {mx.max(x_res):.6f}]")
        
        # 7. Skip connection
        if dit_model.long_skip_connection:
            x_res = dit_model.skip_linear(mx.concatenate([x_res, x_t], axis=-1))
            self.mlx_results['skip_connection'] = x_res
            print(f"skip_connection: {x_res.shape}, range=[{mx.min(x_res):.6f}, {mx.max(x_res):.6f}]")
        
        # 8. WaveNet processing (现在可以正常处理了)
        if dit_model.final_layer_type == 'wavenet':
            print("🔄 处理WaveNet...")
            # MLX WaveNet期望格式: (batch, seq_len, channels)
            # x_res已经是 (batch, seq_len, channels) 格式
            x_wavenet = x_res  # (batch, seq_len, channels)
            
            # 创建mask: (batch, 1, seq_len) -> (batch, seq_len, 1)
            x_mask = mx.ones((x_wavenet.shape[0], 1, x_wavenet.shape[1]), dtype=mx.bool_)
            
            # 调用WaveNet
            x_output = dit_model.wavenet(x_wavenet, x_mask, g=None)
            
            # 输出已经是 (batch, seq_len, channels) 格式
            self.mlx_results['wavenet_output'] = x_output
            print(f"wavenet_output: {x_output.shape}, range=[{mx.min(x_output):.6f}, {mx.max(x_output):.6f}]")
            return x_output
        else:
            x_output = dit_model.final_mlp(x_res).transpose(0, 2, 1)
            self.mlx_results['final_mlp'] = x_output
            print(f"final_mlp: {x_output.shape}, range=[{mx.min(x_output):.6f}, {mx.max(x_output):.6f}]")
            return x_output
    
    def compare_all_layers(self):
        """对比所有层的输出"""
        print("\n=== 逐层差异分析 ===")
        
        # 获取所有共同的层
        common_layers = set(self.pytorch_results.keys()) & set(self.mlx_results.keys())
        
        for layer_name in sorted(common_layers):
            print(f"\n--- {layer_name} ---")
            comparison = compare_tensors(
                self.pytorch_results[layer_name],
                self.mlx_results[layer_name],
                layer_name
            )
            self.comparisons[layer_name] = comparison
            
            if 'error' in comparison:
                print(f"❌ {comparison['error']}")
            else:
                diff_stats = comparison['diff_stats']
                print(f"最大差异: {diff_stats['max_diff']:.6f}")
                print(f"平均差异: {diff_stats['mean_diff']:.6f}")
                print(f"差异>1e-5: {diff_stats['pct_above_1e5']:.2f}%")
                print(f"差异>1e-3: {diff_stats['pct_above_1e3']:.2f}%")
                print(f"差异>1e-1: {diff_stats['pct_above_1e1']:.2f}%")
                
                if diff_stats['max_diff'] > 1e-3:
                    print("⚠️  差异较大")
                elif diff_stats['max_diff'] > 1e-5:
                    print("⚠️  差异中等")
                else:
                    print("✅ 差异较小")
    
    def save_results(self, filename: str):
        """保存调试结果"""
        results = {
            'pytorch_results': self.pytorch_results,
            'mlx_results': self.mlx_results,
            'comparisons': self.comparisons
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"\n调试结果已保存到: {filename}")

def main():
    """主函数"""
    print("=== DiT逐层对比调试工具 ===")
    
    # 设置固定种子
    torch.manual_seed(42)
    np.random.seed(42)
    mx.random.seed(42)
    
    # 初始化IndexTTS2
    print("\n1. 初始化IndexTTS2...")
    tts = IndexTTS2(
        cfg_path='checkpoints/config.yaml',
        model_dir='checkpoints',
        use_mlx=True,
        device='mps'
    )
    
    # 加载缓存输入
    print("\n2. 加载缓存输入...")
    if os.path.exists('cfm_inputs_mlx.pkl'):
        with open('cfm_inputs_mlx.pkl', 'rb') as f:
            cached_inputs = pickle.load(f)
        
        # 从缓存输入构建DiT输入
        cat_condition = cached_inputs['cat_condition']  # (1, 523, 512)
        x_lens = cached_inputs['x_lens']  # (1,)
        ref_mel = cached_inputs['ref_mel']  # (1, 80, 243)
        style = cached_inputs['style']  # (1, 192)
        
        # 构建DiT输入
        batch_size = cat_condition.shape[0]
        seq_len = cat_condition.shape[1]
        
        # 创建随机噪声x (batch, channels, seq_len)
        x = torch.randn(batch_size, 80, seq_len, device=tts.device)
        
        # prompt_x是ref_mel的扩展版本
        prompt_x = torch.zeros(batch_size, 80, seq_len, device=tts.device)
        prompt_x[:, :, :ref_mel.shape[2]] = ref_mel
        
        # cond是cat_condition，确保在正确设备上
        cond = cat_condition.to(tts.device)
        style = style.to(tts.device)
        
        t = torch.tensor([0.0], device=tts.device)
        
        print(f"输入形状: x={x.shape}, prompt_x={prompt_x.shape}, cond={cond.shape}")
    else:
        print("❌ 未找到缓存输入文件，请先运行analyze_cfm_differences_fixed.py")
        return
    
    # 创建调试器
    debugger = DiTLayerDebugger()
    
    # 调试PyTorch DiT
    print("\n3. 调试PyTorch DiT...")
    pytorch_output = debugger.debug_pytorch_dit(
        tts.s2mel.models['cfm'].estimator,
        x, prompt_x, x_lens, t, style, cond
    )
    
    # 调试MLX DiT
    print("\n4. 调试MLX DiT...")
    mlx_output = debugger.debug_mlx_dit(
        tts.mlx_s2mel_cfm.estimator,
        x, prompt_x, x_lens, t, style, cond
    )
    
    # 对比所有层
    debugger.compare_all_layers()
    
    # 保存结果
    debugger.save_results('dit_layer_by_layer_debug.pkl')
    
    print("\n=== 调试完成 ===")

if __name__ == "__main__":
    main()
