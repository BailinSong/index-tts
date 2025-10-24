#!/usr/bin/env python3
"""
模块级输入输出对比调试工具
对比PyTorch和MLX每个模块的输入输出差异
"""

import torch
import mlx.core as mx
import numpy as np
import pickle
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from indextts.infer_v2 import IndexTTS2
from indextts.utils.mlx_utils import mlx_to_torch, torch_to_mlx
from unified_random_generator import UnifiedRandomGenerator

class ModuleIOTracker:
    """模块输入输出跟踪器"""
    
    def __init__(self):
        self.pytorch_data = {}
        self.mlx_data = {}
        self.comparison_results = {}
        
    def track_pytorch_module(self, module_name, inputs, outputs):
        """跟踪PyTorch模块的输入输出"""
        self.pytorch_data[module_name] = {
            'inputs': {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()},
            'outputs': outputs.detach().cpu() if isinstance(outputs, torch.Tensor) else outputs
        }
        
    def track_mlx_module(self, module_name, inputs, outputs):
        """跟踪MLX模块的输入输出"""
        self.mlx_data[module_name] = {
            'inputs': {k: mx.array(v) if isinstance(v, np.ndarray) else v for k, v in inputs.items()},
            'outputs': outputs
        }
        
    def compare_module_io(self, module_name):
        """对比模块的输入输出"""
        if module_name not in self.pytorch_data or module_name not in self.mlx_data:
            return None
            
        pytorch_data = self.pytorch_data[module_name]
        mlx_data = self.mlx_data[module_name]
        
        # 对比输出
        pytorch_output = pytorch_data['outputs']
        mlx_output = mlx_data['outputs']
        
        # 转换MLX输出为PyTorch格式进行对比
        if isinstance(mlx_output, mx.array):
            mlx_output_torch = mlx_to_torch(mlx_output)
        else:
            mlx_output_torch = mlx_output
            
        # 计算输出差异
        if isinstance(pytorch_output, torch.Tensor) and isinstance(mlx_output_torch, torch.Tensor):
            diff = torch.abs(pytorch_output - mlx_output_torch)
            max_diff = float(diff.max())
            mean_diff = float(diff.mean())
            std_diff = float(diff.std())
            
            # 计算形状和数值范围
            pytorch_shape = list(pytorch_output.shape)
            mlx_shape = list(mlx_output_torch.shape)
            shape_match = pytorch_shape == mlx_shape
            
            pytorch_range = float(pytorch_output.max() - pytorch_output.min())
            mlx_range = float(mlx_output_torch.max() - mlx_output_torch.min())
            
            result = {
                'module_name': module_name,
                'shape_match': shape_match,
                'pytorch_shape': pytorch_shape,
                'mlx_shape': mlx_shape,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'std_diff': std_diff,
                'pytorch_range': pytorch_range,
                'mlx_range': mlx_range,
                'range_ratio': pytorch_range / mlx_range if mlx_range > 0 else float('inf'),
                'pytorch_stats': {
                    'min': float(pytorch_output.min()),
                    'max': float(pytorch_output.max()),
                    'mean': float(pytorch_output.mean()),
                    'std': float(pytorch_output.std())
                },
                'mlx_stats': {
                    'min': float(mlx_output_torch.min()),
                    'max': float(mlx_output_torch.max()),
                    'mean': float(mlx_output_torch.mean()),
                    'std': float(mlx_output_torch.std())
                }
            }
            
            self.comparison_results[module_name] = result
            return result
            
        return None

def debug_module_io_comparison():
    """主函数：对比PyTorch和MLX每个模块的输入输出"""
    
    print("=== 模块级输入输出对比调试 ===")
    
    # 初始化跟踪器
    tracker = ModuleIOTracker()
    
    # 初始化TTS系统
    print("\n1. 初始化TTS系统...")
    tts = IndexTTS2()
    
    # 生成测试输入数据
    print("\n2. 生成测试输入数据...")
    
    # 设置随机种子确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 生成测试数据
    batch_size = 1
    seq_len = 100
    in_channels = 80
    hidden_dim = 512
    style_dim = 192
    
    # 获取设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    x = torch.randn(batch_size, in_channels, seq_len, device=device)
    prompt_x = torch.randn(batch_size, in_channels, seq_len, device=device)
    x_lens = torch.tensor([seq_len], device=device)
    t = torch.zeros(batch_size, device=device)
    style = torch.randn(batch_size, style_dim, device=device)
    cond = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    print(f"✅ 生成测试数据:")
    
    print(f"输入形状:")
    print(f"  x: {x.shape}")
    print(f"  prompt_x: {prompt_x.shape}")
    print(f"  x_lens: {x_lens.shape}")
    print(f"  t: {t.shape}")
    print(f"  style: {style.shape}")
    print(f"  cond: {cond.shape}")
    
    # 设置统一随机数生成器
    unified_rng = UnifiedRandomGenerator(42)
    
    print("\n3. 测试PyTorch CFM模块级输入输出...")
    
    # 测试PyTorch CFM
    try:
        # 获取PyTorch CFM estimator
        pytorch_cfm = tts.s2mel.models.cfm
        pytorch_estimator = pytorch_cfm.estimator
        
        print(f"PyTorch CFM estimator类型: {type(pytorch_estimator)}")
        
        # 准备PyTorch输入
        pytorch_inputs = {
            'x': x,
            'prompt_x': prompt_x,
            'x_lens': x_lens,
            't': t,
            'style': style,
            'cond': cond
        }
        
        # 执行PyTorch前向传播并跟踪模块
        with torch.no_grad():
            # 手动跟踪每个模块
            print("\n--- PyTorch模块跟踪 ---")
            
            # 1. x_embedder
            x_emb = pytorch_estimator.x_embedder(x.transpose(1, 2))  # (batch, seq_len, hidden_dim)
            tracker.track_pytorch_module('x_embedder', {'x': x}, x_emb)
            print(f"x_embedder: {x_emb.shape}, min={x_emb.min():.6f}, max={x_emb.max():.6f}")
            
            # 2. cond_embedder + cond_projection
            cond_emb = pytorch_estimator.cond_embedder(cond.long())
            cond_proj = pytorch_estimator.cond_projection(cond_emb)
            tracker.track_pytorch_module('cond_embedder', {'cond': cond}, cond_emb)
            tracker.track_pytorch_module('cond_projection', {'cond_emb': cond_emb}, cond_proj)
            print(f"cond_embedder: {cond_emb.shape}, min={cond_emb.min():.6f}, max={cond_emb.max():.6f}")
            print(f"cond_projection: {cond_proj.shape}, min={cond_proj.min():.6f}, max={cond_proj.max():.6f}")
            
            # 3. t_embedder
            t_emb = pytorch_estimator.t_embedder(t)
            tracker.track_pytorch_module('t_embedder', {'t': t}, t_emb)
            print(f"t_embedder: {t_emb.shape}, min={t_emb.min():.6f}, max={t_emb.max():.6f}")
            
        # 4. 合并输入
        x_t = x.transpose(1, 2)  # (batch, seq_len, in_channels)
        prompt_x_t = prompt_x.transpose(1, 2)
        x_in = torch.cat([x_t, prompt_x_t], dim=-1)  # (batch, seq_len, in_channels*2)
        
        # 添加style
        if pytorch_estimator.transformer_style_condition:
            style_broadcast = style.unsqueeze(1).expand(-1, x_in.shape[1], -1)
            x_in = torch.cat([x_in, style_broadcast], dim=-1)
        
        print(f"x_in shape before merge: {x_in.shape}")
        print(f"cond_x_merge_linear weight shape: {pytorch_estimator.cond_x_merge_linear.weight.shape}")
        print(f"transformer_style_condition: {pytorch_estimator.transformer_style_condition}")
        print(f"in_channels: {pytorch_estimator.in_channels}")
        
        # 检查期望的输入维度
        expected_input_dim = pytorch_estimator.cond_x_merge_linear.weight.shape[1]
        actual_input_dim = x_in.shape[-1]
        print(f"期望输入维度: {expected_input_dim}")
        print(f"实际输入维度: {actual_input_dim}")
        
        # 计算期望的输入维度组成
        # 基础: x + prompt_x = 80 + 80 = 160
        # 如果transformer_style_condition=True: + style = 160 + 192 = 352
        # 但期望是864，说明还有其他输入
        print(f"输入组成分析:")
        print(f"  x + prompt_x: {pytorch_estimator.in_channels * 2} = {pytorch_estimator.in_channels * 2}")
        if pytorch_estimator.transformer_style_condition:
            print(f"  + style: {pytorch_estimator.in_channels * 2} + 192 = {pytorch_estimator.in_channels * 2 + 192}")
        print(f"  期望总维度: {expected_input_dim}")
        
        if actual_input_dim != expected_input_dim:
            print(f"❌ 输入维度不匹配！需要 {expected_input_dim}，实际 {actual_input_dim}")
            print(f"需要添加 {expected_input_dim - actual_input_dim} 个维度")
            
            # 尝试添加缺失的维度
            missing_dim = expected_input_dim - actual_input_dim
            if missing_dim > 0:
                # 添加零填充
                padding = torch.zeros(x_in.shape[0], x_in.shape[1], missing_dim, device=x_in.device)
                x_in = torch.cat([x_in, padding], dim=-1)
                print(f"添加零填充后: {x_in.shape}")
            else:
                return
        
        # 5. cond_x_merge_linear
        x_in = pytorch_estimator.cond_x_merge_linear(x_in)
        tracker.track_pytorch_module('cond_x_merge_linear', {'x_in': x_in}, x_in)
        print(f"cond_x_merge_linear: {x_in.shape}, min={x_in.min():.6f}, max={x_in.max():.6f}")
        
        # 6. Transformer
        x_res = pytorch_estimator.transformer(x_in, t_emb, None, None)
        tracker.track_pytorch_module('transformer', {'x_in': x_in, 't_emb': t_emb}, x_res)
        print(f"transformer: {x_res.shape}, min={x_res.min():.6f}, max={x_res.max():.6f}")
        
        # 7. conv1
        x_out = pytorch_estimator.conv1(x_res)
        tracker.track_pytorch_module('conv1', {'x_res': x_res}, x_out)
        print(f"conv1: {x_out.shape}, min={x_out.min():.6f}, max={x_out.max():.6f}")
        
        # 8. WaveNet
        x_mask = torch.ones(x_out.shape[0], 1, x_out.shape[1], device=x_out.device)
        g = cond_proj
        wavenet_out = pytorch_estimator.wavenet(x_out, x_mask, g)
        tracker.track_pytorch_module('wavenet', {'x_out': x_out, 'x_mask': x_mask, 'g': g}, wavenet_out)
        print(f"wavenet: {wavenet_out.shape}, min={wavenet_out.min():.6f}, max={wavenet_out.max():.6f}")
        
        # 9. 残差连接
        x_final = x_out + wavenet_out
        tracker.track_pytorch_module('residual_connection', {'x_out': x_out, 'wavenet_out': wavenet_out}, x_final)
        print(f"residual_connection: {x_final.shape}, min={x_final.min():.6f}, max={x_final.max():.6f}")
        
        # 10. final_layer
        final_out = pytorch_estimator.final_layer(x_final, t_emb)
        tracker.track_pytorch_module('final_layer', {'x_final': x_final, 't_emb': t_emb}, final_out)
        print(f"final_layer: {final_out.shape}, min={final_out.min():.6f}, max={final_out.max():.6f}")
        
        # 11. conv2
        output = pytorch_estimator.conv2(final_out)
        tracker.track_pytorch_module('conv2', {'final_out': final_out}, output)
        print(f"conv2: {output.shape}, min={output.min():.6f}, max={output.max():.6f}")
        
        # 12. 最终输出
        final_output = output.transpose(1, 2)  # (batch, in_channels, seq_len)
        tracker.track_pytorch_module('final_output', {'output': output}, final_output)
        print(f"final_output: {final_output.shape}, min={final_output.min():.6f}, max={final_output.max():.6f}")
            
    except Exception as e:
        print(f"❌ PyTorch模块跟踪失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n4. 测试MLX CFM模块级输入输出...")
    
    # 测试MLX CFM
    try:
        # 获取MLX CFM estimator
        mlx_cfm = tts.mlx_s2mel_cfm
        mlx_estimator = mlx_cfm.estimator
        
        print(f"MLX CFM estimator类型: {type(mlx_estimator)}")
        
        # 准备MLX输入
        mlx_inputs = {
            'x': torch_to_mlx(x),
            'prompt_x': torch_to_mlx(prompt_x),
            'x_lens': torch_to_mlx(x_lens),
            't': torch_to_mlx(t),
            'style': torch_to_mlx(style),
            'cond': torch_to_mlx(cond)
        }
        
        # 执行MLX前向传播并跟踪模块
        print("\n--- MLX模块跟踪 ---")
        
        # 1. x_embedder
        x_emb = mlx_estimator.x_embedder(mlx_inputs['x'].transpose(0, 2, 1))  # (batch, seq_len, hidden_dim)
        tracker.track_mlx_module('x_embedder', {'x': mlx_inputs['x']}, x_emb)
        print(f"x_embedder: {x_emb.shape}, min={x_emb.min():.6f}, max={x_emb.max():.6f}")
        
        # 2. cond_embedder + cond_projection
        cond_emb = mlx_estimator.cond_embedder(mlx_inputs['cond'].astype(mx.int32))
        cond_proj = mlx_estimator.cond_projection(cond_emb)
        tracker.track_mlx_module('cond_embedder', {'cond': mlx_inputs['cond']}, cond_emb)
        tracker.track_mlx_module('cond_projection', {'cond_emb': cond_emb}, cond_proj)
        print(f"cond_embedder: {cond_emb.shape}, min={cond_emb.min():.6f}, max={cond_emb.max():.6f}")
        print(f"cond_projection: {cond_proj.shape}, min={cond_proj.min():.6f}, max={cond_proj.max():.6f}")
        
        # 3. t_embedder
        t_emb = mlx_estimator.t_embedder(mlx_inputs['t'])
        tracker.track_mlx_module('t_embedder', {'t': mlx_inputs['t']}, t_emb)
        print(f"t_embedder: {t_emb.shape}, min={t_emb.min():.6f}, max={t_emb.max():.6f}")
        
        # 4. 合并输入
        x_t = mlx_inputs['x'].transpose(0, 2, 1)  # (batch, seq_len, in_channels)
        prompt_x_t = mlx_inputs['prompt_x'].transpose(0, 2, 1)
        x_in = mx.concatenate([x_t, prompt_x_t], axis=-1)  # (batch, seq_len, in_channels*2)
        
        # 添加style
        if mlx_estimator.transformer_style_condition:
            style_broadcast = mx.expand_dims(mlx_inputs['style'], 1)
            style_broadcast = mx.broadcast_to(style_broadcast, (x_in.shape[0], x_in.shape[1], style_broadcast.shape[-1]))
            x_in = mx.concatenate([x_in, style_broadcast], axis=-1)
        
        # 5. cond_x_merge_linear
        x_in = mlx_estimator.cond_x_merge_linear(x_in)
        tracker.track_mlx_module('cond_x_merge_linear', {'x_in': x_in}, x_in)
        print(f"cond_x_merge_linear: {x_in.shape}, min={x_in.min():.6f}, max={x_in.max():.6f}")
        
        # 6. Transformer
        x_res = mlx_estimator.transformer(x_in, t_emb, None, None)
        tracker.track_mlx_module('transformer', {'x_in': x_in, 't_emb': t_emb}, x_res)
        print(f"transformer: {x_res.shape}, min={x_res.min():.6f}, max={x_res.max():.6f}")
        
        # 7. conv1
        x_out = mlx_estimator.conv1(x_res)
        tracker.track_mlx_module('conv1', {'x_res': x_res}, x_out)
        print(f"conv1: {x_out.shape}, min={x_out.min():.6f}, max={x_out.max():.6f}")
        
        # 8. WaveNet
        x_mask = mx.ones((x_out.shape[0], 1, x_out.shape[1]))
        g = cond_proj
        wavenet_out = mlx_estimator.wavenet(x_out, x_mask, g)
        tracker.track_mlx_module('wavenet', {'x_out': x_out, 'x_mask': x_mask, 'g': g}, wavenet_out)
        print(f"wavenet: {wavenet_out.shape}, min={wavenet_out.min():.6f}, max={wavenet_out.max():.6f}")
        
        # 9. 残差连接
        x_final = x_out + wavenet_out
        tracker.track_mlx_module('residual_connection', {'x_out': x_out, 'wavenet_out': wavenet_out}, x_final)
        print(f"residual_connection: {x_final.shape}, min={x_final.min():.6f}, max={x_final.max():.6f}")
        
        # 10. final_layer
        final_out = mlx_estimator.final_layer(x_final, t_emb)
        tracker.track_mlx_module('final_layer', {'x_final': x_final, 't_emb': t_emb}, final_out)
        print(f"final_layer: {final_out.shape}, min={final_out.min():.6f}, max={final_out.max():.6f}")
        
        # 11. conv2
        output = mlx_estimator.conv2(final_out)
        tracker.track_mlx_module('conv2', {'final_out': final_out}, output)
        print(f"conv2: {output.shape}, min={output.min():.6f}, max={output.max():.6f}")
        
        # 12. 最终输出
        final_output = output.transpose(0, 2, 1)  # (batch, in_channels, seq_len)
        tracker.track_mlx_module('final_output', {'output': output}, final_output)
        print(f"final_output: {final_output.shape}, min={final_output.min():.6f}, max={final_output.max():.6f}")
        
    except Exception as e:
        print(f"❌ MLX模块跟踪失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n5. 对比分析结果...")
    
    # 对比每个模块
    modules_to_compare = [
        'x_embedder', 'cond_embedder', 'cond_projection', 't_embedder',
        'cond_x_merge_linear', 'transformer', 'conv1', 'wavenet',
        'residual_connection', 'final_layer', 'conv2', 'final_output'
    ]
    
    print("\n--- 模块对比结果 ---")
    for module_name in modules_to_compare:
        result = tracker.compare_module_io(module_name)
        if result:
            print(f"\n📊 {module_name}:")
            print(f"  形状匹配: {'✅' if result['shape_match'] else '❌'}")
            if not result['shape_match']:
                print(f"    PyTorch: {result['pytorch_shape']}")
                print(f"    MLX: {result['mlx_shape']}")
            print(f"  最大差异: {result['max_diff']:.6f}")
            print(f"  平均差异: {result['mean_diff']:.6f}")
            print(f"  标准差差异: {result['std_diff']:.6f}")
            print(f"  数值范围比例: {result['range_ratio']:.6f}")
            print(f"  PyTorch统计: min={result['pytorch_stats']['min']:.6f}, max={result['pytorch_stats']['max']:.6f}, mean={result['pytorch_stats']['mean']:.6f}")
            print(f"  MLX统计: min={result['mlx_stats']['min']:.6f}, max={result['mlx_stats']['max']:.6f}, mean={result['mlx_stats']['mean']:.6f}")
            
            # 判断差异程度
            if result['max_diff'] < 1e-5:
                print(f"  ✅ 差异极小 (< 1e-5)")
            elif result['max_diff'] < 1e-3:
                print(f"  ⚠️  差异较小 (< 1e-3)")
            elif result['max_diff'] < 1e-1:
                print(f"  ⚠️  差异中等 (< 1e-1)")
            else:
                print(f"  ❌ 差异较大 (>= 1e-1)")
        else:
            print(f"\n❌ {module_name}: 无法对比 (缺少数据)")
    
    # 保存详细结果
    results_file = "module_io_comparison_results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump({
            'pytorch_data': tracker.pytorch_data,
            'mlx_data': tracker.mlx_data,
            'comparison_results': tracker.comparison_results
        }, f)
    
    print(f"\n✅ 详细结果已保存到: {results_file}")
    
    # 总结
    print("\n=== 总结 ===")
    total_modules = len(modules_to_compare)
    compared_modules = len(tracker.comparison_results)
    print(f"总模块数: {total_modules}")
    print(f"成功对比: {compared_modules}")
    print(f"对比成功率: {compared_modules/total_modules*100:.1f}%")
    
    # 统计差异程度
    if tracker.comparison_results:
        tiny_diff = sum(1 for r in tracker.comparison_results.values() if r['max_diff'] < 1e-5)
        small_diff = sum(1 for r in tracker.comparison_results.values() if 1e-5 <= r['max_diff'] < 1e-3)
        medium_diff = sum(1 for r in tracker.comparison_results.values() if 1e-3 <= r['max_diff'] < 1e-1)
        large_diff = sum(1 for r in tracker.comparison_results.values() if r['max_diff'] >= 1e-1)
        
        print(f"\n差异程度统计:")
        print(f"  极小差异 (< 1e-5): {tiny_diff}/{compared_modules} ({tiny_diff/compared_modules*100:.1f}%)")
        print(f"  较小差异 (1e-5 ~ 1e-3): {small_diff}/{compared_modules} ({small_diff/compared_modules*100:.1f}%)")
        print(f"  中等差异 (1e-3 ~ 1e-1): {medium_diff}/{compared_modules} ({medium_diff/compared_modules*100:.1f}%)")
        print(f"  较大差异 (>= 1e-1): {large_diff}/{compared_modules} ({large_diff/compared_modules*100:.1f}%)")

if __name__ == "__main__":
    debug_module_io_comparison()
