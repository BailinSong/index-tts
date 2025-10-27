#!/usr/bin/env python3
"""
详细对比 PyTorch 和 MLX CFM 每个子模块的输入输出

使用 CFM 前级缓存数据，详细收集和对比每个子模块的输入输出
"""

import pickle
import numpy as np
import torch
import mlx.core as mx
from pathlib import Path
import sys
import os

# 添加项目路径
sys.path.append('/Users/bailin/index-tts')


def detailed_submodule_io_comparison():
    """详细对比每个子模块的输入输出"""
    print("🔍 Detailed Submodule I/O Comparison")
    print("="*60)
    
    # 加载正确的缓存输入数据
    cached_inputs = load_correct_cached_inputs()
    if not cached_inputs:
        print("❌ No correct cached inputs available!")
        return
    
    print("\n📊 Using Correct Cached Inputs:")
    print_cached_inputs_summary(cached_inputs)
    
    # 收集 PyTorch CFM 每个子模块的输入输出
    print("\n🔍 Collecting PyTorch CFM Submodule I/O:")
    pytorch_submodule_io = collect_pytorch_submodule_io(cached_inputs)
    
    # 收集 MLX CFM 每个子模块的输入输出
    print("\n🔍 Collecting MLX CFM Submodule I/O:")
    mlx_submodule_io = collect_mlx_submodule_io(cached_inputs)
    
    # 对比每个子模块的输入输出
    print("\n📊 Submodule I/O Comparison:")
    comparison_results = compare_submodule_io(pytorch_submodule_io, mlx_submodule_io)
    
    # 生成详细对比报告
    print("\n📄 Generating Detailed I/O Comparison Report:")
    generate_detailed_io_report(cached_inputs, pytorch_submodule_io, mlx_submodule_io, comparison_results)
    
    print("\n🎉 Detailed submodule I/O comparison completed!")


def load_correct_cached_inputs():
    """加载正确的缓存输入数据"""
    cached_file = Path("cfm_debug_outputs/correct_cached_inputs.pkl")
    
    if not cached_file.exists():
        print("❌ Correct cached inputs file not found!")
        return None
    
    with open(cached_file, 'rb') as f:
        cached_inputs = pickle.load(f)
    
    print(f"✅ Correct cached inputs loaded from: {cached_file}")
    return cached_inputs


def print_cached_inputs_summary(cached_inputs):
    """打印缓存输入摘要"""
    for key, value in cached_inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"     {key}: {value.shape}, range: [{value.min():.6f}, {value.max():.6f}]")
        elif isinstance(value, np.ndarray):
            print(f"     {key}: {value.shape}, range: [{value.min():.6f}, {value.max():.6f}]")
        else:
            print(f"     {key}: {value}")


def collect_pytorch_submodule_io(cached_inputs):
    """收集 PyTorch CFM 每个子模块的输入输出"""
    try:
        # 导入 PyTorch CFM
        from indextts.s2mel.modules.flow_matching import CFM
        
        # 创建配置
        config_dict = {
            'DiT': {
                'in_channels': 80,
                'hidden_dim': 512,
                'num_heads': 8,
                'depth': 13,
                'time_as_token': False,
                'style_as_token': False,
                'long_skip_connection': True,
                'style_condition': True,
                'is_causal': True,
                'final_layer_type': 'wavenet',
                'content_type': 'continuous',
                'content_codebook_size': 1024,
                'content_dim': 512,
                'class_dropout_prob': 0.1
            },
            'style_encoder': {
                'dim': 192
            },
            'wavenet': {
                'hidden_dim': 512,
                'kernel_size': 3,
                'dilation_rate': 2,
                'num_layers': 8,
                'p_dropout': 0.1,
                'style_condition': True
            },
            'dit_type': 'DiT',
            'reg_loss_type': 'l2'
        }
        
        # 创建 SimpleConfig 类
        class SimpleConfig:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    if isinstance(value, dict):
                        setattr(self, key, SimpleConfig(value))
                    else:
                        setattr(self, key, value)
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        config = SimpleConfig(config_dict)
        
        # 初始化 PyTorch CFM
        pytorch_cfm = CFM(config)
        pytorch_cfm.eval()
        
        # 初始化 caches
        batch_size = cached_inputs['batch_size']
        seq_len = cached_inputs['seq_len']
        pytorch_cfm.estimator.setup_caches(max_batch_size=batch_size, max_seq_length=seq_len)
        
        # 准备输入数据
        x = cached_inputs['x']
        prompt_x = cached_inputs['prompt_x']
        x_lens = cached_inputs['x_lens']
        style = cached_inputs['style']
        mu = cached_inputs['mu']
        
        print(f"     Input shapes: x={x.shape}, prompt_x={prompt_x.shape}, x_lens={x_lens.shape}")
        print(f"     Input shapes: style={style.shape}, mu={mu.shape}")
        
        # 创建时间跨度
        t_span = torch.linspace(0, 1, 4)  # 3 steps + 1
        
        # 手动执行 CFM 步骤并收集每个子模块的输入输出
        print(f"\n     🔍 Manual PyTorch CFM Execution with Submodule I/O Tracking:")
        
        submodule_io_data = {}
        
        with torch.no_grad():
            # 应用 prompt - 正确处理形状
            prompt_len = prompt_x.size(-1)
            # 确保 prompt_x_processed 和 x_processed 的形状与 x 一致
            x_processed = x.clone()
            prompt_x_processed = torch.zeros_like(x)

            # 只复制有效的 prompt 长度部分
            if prompt_len <= x.size(-1):
                prompt_x_processed[..., :prompt_len] = prompt_x[..., :prompt_len]
                x_processed[..., :prompt_len] = 0
            else:
                # 如果 prompt 比 x 长，只取 x 长度的部分
                prompt_x_processed = prompt_x[..., :x.size(-1)]
                x_processed = torch.zeros_like(x)
            
            print(f"       After prompt processing:")
            print(f"         x_processed: range=[{x_processed.min():.6f}, {x_processed.max():.6f}]")
            print(f"         prompt_x_processed: range=[{prompt_x_processed.min():.6f}, {prompt_x_processed.max():.6f}]")
            
            # 执行 Euler 步骤
            for step in range(1, len(t_span)):
                dt = t_span[step] - t_span[step - 1]
                t_current = t_span[step - 1]
                
                print(f"\n       Step {step}/{len(t_span)-1}: t={t_current:.6f}, dt={dt:.6f}")
                
                # 手动调用 DiT 的各个子模块并收集输入输出
                step_io_data = collect_pytorch_dit_submodule_io(
                    pytorch_cfm.estimator, x_processed, prompt_x_processed, 
                    x_lens, t_current.unsqueeze(0), style, mu
                )
                
                submodule_io_data[f'step_{step}'] = step_io_data
                
                # 调用 estimator (DiT) 获取 dphi_dt
                dphi_dt = pytorch_cfm.estimator(
                    x_processed, prompt_x_processed, x_lens, 
                    t_current.unsqueeze(0), style, mu
                )
                
                print(f"         dphi_dt: shape={dphi_dt.shape}, range=[{dphi_dt.min():.6f}, {dphi_dt.max():.6f}]")
                
                # 更新 x
                x_processed = x_processed + dt * dphi_dt
                
                print(f"         x_updated: range=[{x_processed.min():.6f}, {x_processed.max():.6f}]")
        
        return {
            'submodule_io_data': submodule_io_data,
            'success': True,
            'config': config_dict
        }
        
    except Exception as e:
        print(f"     ❌ PyTorch submodule I/O collection failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'success': False
        }


def collect_pytorch_dit_submodule_io(dit_model, x, prompt_x, x_lens, t, style, mu):
    """收集 PyTorch DiT 每个子模块的输入输出"""
    submodule_io = {}
    
    try:
        # 1. Timestep Embedding
        print(f"           🔍 Timestep Embedding:")
        t_emb = dit_model.t_embedder(t)
        print(f"             Input t: {t.shape}, range=[{t.min():.6f}, {t.max():.6f}]")
        print(f"             Output t_emb: {t_emb.shape}, range=[{t_emb.min():.6f}, {t_emb.max():.6f}]")
        submodule_io['timestep_embedding'] = {
            'input': {'t': t},
            'output': {'t_emb': t_emb}
        }
        
        # 2. Conditioning Projection
        print(f"           🔍 Conditioning Projection:")
        cond_proj = dit_model.cond_projection(mu)
        print(f"             Input mu: {mu.shape}, range=[{mu.min():.6f}, {mu.max():.6f}]")
        print(f"             Output cond_proj: {cond_proj.shape}, range=[{cond_proj.min():.6f}, {cond_proj.max():.6f}]")
        submodule_io['cond_projection'] = {
            'input': {'mu': mu},
            'output': {'cond_proj': cond_proj}
        }
        
        # 3. X Embedding
        print(f"           🔍 X Embedding:")
        x_t = x.transpose(1, 2)  # (batch, seq_len, in_channels)
        prompt_x_t = prompt_x.transpose(1, 2)  # (batch, seq_len, in_channels)
        print(f"             Input x_t: {x_t.shape}, range=[{x_t.min():.6f}, {x_t.max():.6f}]")
        print(f"             Input prompt_x_t: {prompt_x_t.shape}, range=[{prompt_x_t.min():.6f}, {prompt_x_t.max():.6f}]")
        submodule_io['x_embedding'] = {
            'input': {'x_t': x_t, 'prompt_x_t': prompt_x_t},
            'output': {'x_t': x_t, 'prompt_x_t': prompt_x_t}  # 这里只是转置，没有实际变换
        }
        
        # 4. Merge and Transformer Input
        print(f"           🔍 Merge and Transformer Input:")
        x_in = torch.cat([x_t, prompt_x_t, cond_proj], dim=-1)  # 80+80+512=672
        if dit_model.transformer_style_condition and not dit_model.style_as_token:
            x_in = torch.cat([x_in, style[:, None, :].repeat(1, x_in.size(1), 1)], dim=-1)
        
        x_in = dit_model.cond_x_merge_linear(x_in)  # (N, T, D)
        print(f"             Input x_in: {x_in.shape}, range=[{x_in.min():.6f}, {x_in.max():.6f}]")
        submodule_io['merge_input'] = {
            'input': {'x_in': x_in},
            'output': {'x_in': x_in}
        }
        
        # 5. Transformer
        print(f"           🔍 Transformer:")
        # 创建序列掩码 - 使用简单的方法
        x_mask = torch.ones(x_in.size(0), 1, x_in.size(1), device=x_in.device, dtype=torch.bool)
        input_pos = dit_model.input_pos[:x_in.size(1)]
        x_mask_expanded = x_mask[:, None, :].repeat(1, 1, x_in.size(1), 1) if not dit_model.is_causal else None
        
        x_res = dit_model.transformer(x_in, t_emb.unsqueeze(1), input_pos, x_mask_expanded)
        print(f"             Input x_in: {x_in.shape}, range=[{x_in.min():.6f}, {x_in.max():.6f}]")
        print(f"             Output x_res: {x_res.shape}, range=[{x_res.min():.6f}, {x_res.max():.6f}]")
        submodule_io['transformer'] = {
            'input': {'x_in': x_in, 't_emb': t_emb, 'input_pos': input_pos, 'x_mask': x_mask_expanded},
            'output': {'x_res': x_res}
        }
        
        # 6. Long Skip Connection
        if dit_model.long_skip_connection:
            print(f"           🔍 Long Skip Connection:")
            x_res_skip = dit_model.skip_linear(torch.cat([x_res, x_t], dim=-1))
            print(f"             Input x_res: {x_res.shape}, range=[{x_res.min():.6f}, {x_res.max():.6f}]")
            print(f"             Input x_t: {x_t.shape}, range=[{x_t.min():.6f}, {x_t.max():.6f}]")
            print(f"             Output x_res_skip: {x_res_skip.shape}, range=[{x_res_skip.min():.6f}, {x_res_skip.max():.6f}]")
            submodule_io['long_skip_connection'] = {
                'input': {'x_res': x_res, 'x_t': x_t},
                'output': {'x_res_skip': x_res_skip}
            }
            x_res = x_res_skip
        
        # 7. Final Layer
        print(f"           🔍 Final Layer:")
        if dit_model.final_layer_type == 'wavenet':
            # WaveNet final layer
            x_final = dit_model.conv1(x_res)
            x_final = x_final.transpose(1, 2)
            t2 = dit_model.t_embedder2(t)
            x_final = dit_model.wavenet(x_final, x_mask, g=t2.unsqueeze(2)).transpose(1, 2) + dit_model.res_projection(x_res)
            x_final = dit_model.final_layer(x_final, t_emb).transpose(1, 2)
            x_final = dit_model.conv2(x_final)
        else:
            # MLP final layer
            x_final = dit_model.final_mlp(x_res)
            x_final = x_final.transpose(1, 2)
        
        print(f"             Input x_res: {x_res.shape}, range=[{x_res.min():.6f}, {x_res.max():.6f}]")
        print(f"             Output x_final: {x_final.shape}, range=[{x_final.min():.6f}, {x_final.max():.6f}]")
        submodule_io['final_layer'] = {
            'input': {'x_res': x_res},
            'output': {'x_final': x_final}
        }
        
    except Exception as e:
        print(f"             ❌ Error collecting DiT submodule I/O: {e}")
        import traceback
        traceback.print_exc()
    
    return submodule_io


def collect_mlx_submodule_io(cached_inputs):
    """收集 MLX CFM 每个子模块的输入输出"""
    try:
        # 导入 MLX CFM
        from indextts.s2mel.modules.mlx_cfm import MLXCFM
        
        # 创建配置
        config_dict = {
            'DiT': {
                'in_channels': 80,
                'hidden_dim': 512,
                'num_heads': 8,
                'depth': 13,
                'time_as_token': False,
                'style_as_token': False,
                'long_skip_connection': True,
                'style_condition': True,
                'is_causal': True,
                'final_layer_type': 'wavenet',
                'content_type': 'continuous',
                'content_codebook_size': 1024,
                'content_dim': 512,
                'class_dropout_prob': 0.1
            },
            'style_encoder': {
                'dim': 192
            },
            'wavenet': {
                'hidden_dim': 512,
                'kernel_size': 3,
                'dilation_rate': 2,
                'num_layers': 8,
                'p_dropout': 0.1,
                'style_condition': True
            }
        }
        
        # 初始化 MLX CFM
        mlx_cfm = MLXCFM(config_dict)
        
        # 准备输入数据（转换为 MLX）
        x_mlx = mx.array(cached_inputs['x'].numpy())
        prompt_x_mlx = mx.array(cached_inputs['prompt_x'].numpy())
        x_lens_mlx = mx.array(cached_inputs['x_lens'].numpy())
        style_mlx = mx.array(cached_inputs['style'].numpy())
        mu_mlx = mx.array(cached_inputs['mu'].numpy())
        
        print(f"     Input shapes: x={x_mlx.shape}, prompt_x={prompt_x_mlx.shape}, x_lens={x_lens_mlx.shape}")
        print(f"     Input shapes: style={style_mlx.shape}, mu={mu_mlx.shape}")
        
        # 创建时间跨度
        t_span_mlx = mx.array(np.linspace(0, 1, 4))  # 3 steps + 1
        
        # 手动执行 CFM 步骤并收集每个子模块的输入输出
        print(f"\n     🔍 Manual MLX CFM Execution with Submodule I/O Tracking:")
        
        submodule_io_data = {}
        
        # 应用 prompt - 正确处理形状
        prompt_len = prompt_x_mlx.shape[-1]
        # 确保 prompt_x_processed 和 x_processed 的形状与 x 一致
        x_processed_mlx = mx.array(cached_inputs['x'].numpy())
        prompt_x_processed_mlx = mx.zeros_like(x_processed_mlx)

        # 只复制有效的 prompt 长度部分
        if prompt_len <= x_processed_mlx.shape[-1]:
            prompt_x_processed_mlx = prompt_x_processed_mlx.at[..., :prompt_len].set(prompt_x_mlx[..., :prompt_len])
            x_processed_mlx = x_processed_mlx.at[..., :prompt_len].set(0)
        else:
            # 如果 prompt 比 x 长，只取 x 长度的部分
            prompt_x_processed_mlx = prompt_x_mlx[..., :x_processed_mlx.shape[-1]]
            x_processed_mlx = mx.zeros_like(x_processed_mlx)
        
        print(f"       After prompt processing:")
        print(f"         x_processed: range=[{float(x_processed_mlx.min()):.6f}, {float(x_processed_mlx.max()):.6f}]")
        print(f"         prompt_x_processed: range=[{float(prompt_x_processed_mlx.min()):.6f}, {float(prompt_x_processed_mlx.max()):.6f}]")
        
        # 执行 Euler 步骤
        for step in range(1, len(t_span_mlx)):
            dt = t_span_mlx[step] - t_span_mlx[step - 1]
            t_current = t_span_mlx[step - 1]
            
            print(f"\n       Step {step}/{len(t_span_mlx)-1}: t={float(t_current):.6f}, dt={float(dt):.6f}")
            
            # 手动调用 MLX DiT 的各个子模块并收集输入输出
            step_io_data = collect_mlx_dit_submodule_io(
                mlx_cfm.estimator, x_processed_mlx, prompt_x_processed_mlx, 
                x_lens_mlx, t_current.reshape(1), style_mlx, mu_mlx
            )
            
            submodule_io_data[f'step_{step}'] = step_io_data
            
            # 调用 estimator (DiT) 获取 dphi_dt
            dphi_dt = mlx_cfm.estimator(
                x_processed_mlx, prompt_x_processed_mlx, x_lens_mlx, 
                t_current.reshape(1), style_mlx, mu_mlx
            )
            
            print(f"         dphi_dt: shape={dphi_dt.shape}, range=[{float(dphi_dt.min()):.6f}, {float(dphi_dt.max()):.6f}]")
            
            # 更新 x
            x_processed_mlx = x_processed_mlx + dt * dphi_dt
            
            print(f"         x_updated: range=[{float(x_processed_mlx.min()):.6f}, {float(x_processed_mlx.max()):.6f}]")
        
        return {
            'submodule_io_data': submodule_io_data,
            'success': True,
            'config': config_dict
        }
        
    except Exception as e:
        print(f"     ❌ MLX submodule I/O collection failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'success': False
        }


def collect_mlx_dit_submodule_io(dit_model, x, prompt_x, x_lens, t, style, mu):
    """收集 MLX DiT 每个子模块的输入输出"""
    submodule_io = {}
    
    try:
        # 1. Timestep Embedding
        print(f"           🔍 Timestep Embedding:")
        t_emb = dit_model.t_embedder(t)
        print(f"             Input t: {t.shape}, range=[{float(t.min()):.6f}, {float(t.max()):.6f}]")
        print(f"             Output t_emb: {t_emb.shape}, range=[{float(t_emb.min()):.6f}, {float(t_emb.max()):.6f}]")
        submodule_io['timestep_embedding'] = {
            'input': {'t': t},
            'output': {'t_emb': t_emb}
        }
        
        # 2. Conditioning Projection
        print(f"           🔍 Conditioning Projection:")
        cond_proj = dit_model.cond_projection(mu)
        print(f"             Input mu: {mu.shape}, range=[{float(mu.min()):.6f}, {float(mu.max()):.6f}]")
        print(f"             Output cond_proj: {cond_proj.shape}, range=[{float(cond_proj.min()):.6f}, {float(cond_proj.max()):.6f}]")
        submodule_io['cond_projection'] = {
            'input': {'mu': mu},
            'output': {'cond_proj': cond_proj}
        }
        
        # 3. X Embedding
        print(f"           🔍 X Embedding:")
        x_t = x.transpose(0, 2, 1)  # (batch, seq_len, in_channels)
        prompt_x_t = prompt_x.transpose(0, 2, 1)  # (batch, seq_len, in_channels)
        print(f"             Input x_t: {x_t.shape}, range=[{float(x_t.min()):.6f}, {float(x_t.max()):.6f}]")
        print(f"             Input prompt_x_t: {prompt_x_t.shape}, range=[{float(prompt_x_t.min()):.6f}, {float(prompt_x_t.max()):.6f}]")
        submodule_io['x_embedding'] = {
            'input': {'x_t': x_t, 'prompt_x_t': prompt_x_t},
            'output': {'x_t': x_t, 'prompt_x_t': prompt_x_t}  # 这里只是转置，没有实际变换
        }
        
        # 4. Merge and Transformer Input
        print(f"           🔍 Merge and Transformer Input:")
        x_in = mx.concatenate([x_t, prompt_x_t, cond_proj], axis=-1)  # 80+80+512=672
        if dit_model.transformer_style_condition and not dit_model.style_as_token:
            x_in = mx.concatenate([x_in, mx.broadcast_to(style[:, None, :], (style.shape[0], x_in.shape[1], style.shape[1]))], axis=-1)
        
        x_in = dit_model.cond_x_merge_linear(x_in)  # (N, T, D)
        print(f"             Input x_in: {x_in.shape}, range=[{float(x_in.min()):.6f}, {float(x_in.max()):.6f}]")
        submodule_io['merge_input'] = {
            'input': {'x_in': x_in},
            'output': {'x_in': x_in}
        }
        
        # 5. Transformer
        print(f"           🔍 Transformer:")
        x_res = dit_model.transformer(x_in, t_emb, dit_model.input_pos[:x_in.shape[1]])
        print(f"             Input x_in: {x_in.shape}, range=[{float(x_in.min()):.6f}, {float(x_in.max()):.6f}]")
        print(f"             Output x_res: {x_res.shape}, range=[{float(x_res.min()):.6f}, {float(x_res.max()):.6f}]")
        submodule_io['transformer'] = {
            'input': {'x_in': x_in, 't_emb': t_emb},
            'output': {'x_res': x_res}
        }
        
        # 6. Long Skip Connection
        if dit_model.long_skip_connection:
            print(f"           🔍 Long Skip Connection:")
            x_res_skip = dit_model.skip_linear(mx.concatenate([x_res, x_t], axis=-1))
            print(f"             Input x_res: {x_res.shape}, range=[{float(x_res.min()):.6f}, {float(x_res.max()):.6f}]")
            print(f"             Input x_t: {x_t.shape}, range=[{float(x_t.min()):.6f}, {float(x_t.max()):.6f}]")
            print(f"             Output x_res_skip: {x_res_skip.shape}, range=[{float(x_res_skip.min()):.6f}, {float(x_res_skip.max()):.6f}]")
            submodule_io['long_skip_connection'] = {
                'input': {'x_res': x_res, 'x_t': x_t},
                'output': {'x_res_skip': x_res_skip}
            }
            x_res = x_res_skip
        
        # 7. Final Layer
        print(f"           🔍 Final Layer:")
        if dit_model.final_layer_type == 'wavenet':
            # WaveNet final layer
            x_final = dit_model.conv1(x_res)
            x_final = x_final.transpose(0, 2, 1)
            t2 = dit_model.t_embedder2(t)
            # 创建正确形状的 x_mask 用于 WaveNet
            x_mask_mlx = mx.ones((x_final.shape[0], 1, x_final.shape[1]))  # (batch, 1, seq_len)
            x_final = dit_model.wavenet(x_final, x_mask_mlx, g=t2[:, :, None]).transpose(0, 2, 1) + dit_model.res_projection(x_res)
            x_final = dit_model.final_layer(x_final, t_emb).transpose(0, 2, 1)
            x_final = dit_model.conv2(x_final)
        else:
            # MLP final layer
            x_final = dit_model.final_mlp(x_res)
            x_final = x_final.transpose(0, 2, 1)
        
        print(f"             Input x_res: {x_res.shape}, range=[{float(x_res.min()):.6f}, {float(x_res.max()):.6f}]")
        print(f"             Output x_final: {x_final.shape}, range=[{float(x_final.min()):.6f}, {float(x_final.max()):.6f}]")
        submodule_io['final_layer'] = {
            'input': {'x_res': x_res},
            'output': {'x_final': x_final}
        }
        
    except Exception as e:
        print(f"             ❌ Error collecting MLX DiT submodule I/O: {e}")
        import traceback
        traceback.print_exc()
    
    return submodule_io


def compare_submodule_io(pytorch_data, mlx_data):
    """对比每个子模块的输入输出"""
    print("\n   📋 Submodule I/O Comparison:")
    
    if not pytorch_data.get('success') or not mlx_data.get('success'):
        print("     ❌ Cannot compare: one or both data collection failed")
        return None
    
    pytorch_submodules = pytorch_data['submodule_io_data']
    mlx_submodules = mlx_data['submodule_io_data']
    
    comparison_results = {}
    
    # 对比每个步骤的子模块
    for step_key in pytorch_submodules.keys():
        if step_key.startswith('step_'):
            step_num = step_key.split('_')[1]
            
            if step_key in mlx_submodules:
                pytorch_step = pytorch_submodules[step_key]
                mlx_step = mlx_submodules[step_key]
                
                print(f"\n     🔍 Step {step_num} Submodule Comparison:")
                
                step_comparison = {}
                
                # 对比每个子模块
                for submodule_name in pytorch_step.keys():
                    if submodule_name in mlx_step:
                        print(f"\n       📊 {submodule_name}:")
                        
                        pytorch_submodule = pytorch_step[submodule_name]
                        mlx_submodule = mlx_step[submodule_name]
                        
                        submodule_comparison = {}
                        
                        # 对比输入
                        if 'input' in pytorch_submodule and 'input' in mlx_submodule:
                            input_comparison = compare_tensor_dicts(
                                pytorch_submodule['input'], 
                                mlx_submodule['input'], 
                                f"{submodule_name}_input"
                            )
                            submodule_comparison['input'] = input_comparison
                        
                        # 对比输出
                        if 'output' in pytorch_submodule and 'output' in mlx_submodule:
                            output_comparison = compare_tensor_dicts(
                                pytorch_submodule['output'], 
                                mlx_submodule['output'], 
                                f"{submodule_name}_output"
                            )
                            submodule_comparison['output'] = output_comparison
                        
                        step_comparison[submodule_name] = submodule_comparison
                
                comparison_results[step_key] = step_comparison
    
    return comparison_results


def compare_tensor_dicts(pytorch_dict, mlx_dict, name_prefix):
    """对比两个张量字典"""
    comparison = {}
    
    for key in pytorch_dict.keys():
        if key in mlx_dict:
            pytorch_tensor = pytorch_dict[key]
            mlx_tensor = mlx_dict[key]
            
            # 转换为 numpy 进行对比
            if isinstance(pytorch_tensor, torch.Tensor):
                pytorch_np = pytorch_tensor.detach().cpu().numpy()
            else:
                pytorch_np = pytorch_tensor
            
            if isinstance(mlx_tensor, mx.array):
                mlx_np = np.array(mlx_tensor)
            else:
                mlx_np = mlx_tensor
            
            # 计算差异
            max_diff = np.max(np.abs(pytorch_np - mlx_np))
            mean_diff = np.mean(np.abs(pytorch_np - mlx_np))
            relative_diff = mean_diff / (np.mean(np.abs(pytorch_np)) + 1e-8)
            
            print(f"         {key}:")
            print(f"           Max diff: {max_diff:.2e}")
            print(f"           Mean diff: {mean_diff:.2e}")
            print(f"           Relative diff: {relative_diff:.2e}")
            print(f"           PyTorch range: [{pytorch_np.min():.6f}, {pytorch_np.max():.6f}]")
            print(f"           MLX range: [{mlx_np.min():.6f}, {mlx_np.max():.6f}]")
            
            comparison[key] = {
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'relative_diff': relative_diff,
                'pytorch_range': [pytorch_np.min(), pytorch_np.max()],
                'mlx_range': [mlx_np.min(), mlx_np.max()]
            }
    
    return comparison


def generate_detailed_io_report(cached_inputs, pytorch_data, mlx_data, comparison_results):
    """生成详细的输入输出对比报告"""
    report_lines = []
    report_lines.append("# PyTorch vs MLX CFM 详细子模块输入输出对比报告")
    report_lines.append("="*60)
    report_lines.append("")
    
    report_lines.append("## 🎯 分析目标")
    report_lines.append("")
    report_lines.append("使用 CFM 前级缓存数据作为输入，详细对比 PyTorch 和 MLX 版本的 CFM 实现")
    report_lines.append("中每个子模块的输入输出，找出具体的差异点。")
    report_lines.append("")
    report_lines.append("**关键要求**: 在输入一致的情况下，输出差异应该小于 e-5")
    report_lines.append("")
    
    # 缓存输入分析
    report_lines.append("## 📊 缓存输入分析")
    report_lines.append("")
    report_lines.append("### 输入数据摘要")
    report_lines.append("")
    
    for key, value in cached_inputs.items():
        if isinstance(value, torch.Tensor):
            report_lines.append(f"- **{key}**: {value.shape}, 范围: [{value.min():.6f}, {value.max():.6f}]")
        elif isinstance(value, np.ndarray):
            report_lines.append(f"- **{key}**: {value.shape}, 范围: [{value.min():.6f}, {value.max():.6f}]")
        else:
            report_lines.append(f"- **{key}**: {value}")
    
    report_lines.append("")
    
    # 数据收集状态
    report_lines.append("## 🔍 数据收集状态")
    report_lines.append("")
    
    report_lines.append("### PyTorch CFM")
    report_lines.append("")
    if pytorch_data.get('success'):
        report_lines.append("✅ **数据收集成功**")
        pytorch_submodules = pytorch_data['submodule_io_data']
        report_lines.append(f"- 子模块数据: {len(pytorch_submodules)} 个步骤")
        
        # 列出各个步骤
        for step_name in pytorch_submodules.keys():
            report_lines.append(f"  - {step_name}")
    else:
        report_lines.append("❌ **数据收集失败**")
        report_lines.append(f"- 错误: {pytorch_data.get('error', 'Unknown error')}")
    
    report_lines.append("")
    
    report_lines.append("### MLX CFM")
    report_lines.append("")
    if mlx_data.get('success'):
        report_lines.append("✅ **数据收集成功**")
        mlx_submodules = mlx_data['submodule_io_data']
        report_lines.append(f"- 子模块数据: {len(mlx_submodules)} 个步骤")
        
        # 列出各个步骤
        for step_name in mlx_submodules.keys():
            report_lines.append(f"  - {step_name}")
    else:
        report_lines.append("❌ **数据收集失败**")
        report_lines.append(f"- 错误: {mlx_data.get('error', 'Unknown error')}")
    
    report_lines.append("")
    
    # 对比结果分析
    if comparison_results:
        report_lines.append("## 📊 对比结果分析")
        report_lines.append("")
        
        # 对比每个步骤的子模块
        for step_key in sorted(comparison_results.keys()):
            step_num = step_key.split('_')[1]
            step_comparison = comparison_results[step_key]
            
            report_lines.append(f"### 步骤 {step_num} 子模块对比")
            report_lines.append("")
            
            for submodule_name in sorted(step_comparison.keys()):
                submodule_comparison = step_comparison[submodule_name]
                
                report_lines.append(f"#### {submodule_name}")
                report_lines.append("")
                
                # 输入对比
                if 'input' in submodule_comparison:
                    report_lines.append("**输入对比**:")
                    report_lines.append("")
                    for tensor_name, tensor_comp in submodule_comparison['input'].items():
                        report_lines.append(f"- **{tensor_name}**:")
                        report_lines.append(f"  - 最大差异: {tensor_comp['max_diff']:.2e}")
                        report_lines.append(f"  - 平均差异: {tensor_comp['mean_diff']:.2e}")
                        report_lines.append(f"  - 相对差异: {tensor_comp['relative_diff']:.2e}")
                        report_lines.append(f"  - PyTorch 范围: [{tensor_comp['pytorch_range'][0]:.6f}, {tensor_comp['pytorch_range'][1]:.6f}]")
                        report_lines.append(f"  - MLX 范围: [{tensor_comp['mlx_range'][0]:.6f}, {tensor_comp['mlx_range'][1]:.6f}]")
                        report_lines.append("")
                
                # 输出对比
                if 'output' in submodule_comparison:
                    report_lines.append("**输出对比**:")
                    report_lines.append("")
                    for tensor_name, tensor_comp in submodule_comparison['output'].items():
                        report_lines.append(f"- **{tensor_name}**:")
                        report_lines.append(f"  - 最大差异: {tensor_comp['max_diff']:.2e}")
                        report_lines.append(f"  - 平均差异: {tensor_comp['mean_diff']:.2e}")
                        report_lines.append(f"  - 相对差异: {tensor_comp['relative_diff']:.2e}")
                        report_lines.append(f"  - PyTorch 范围: [{tensor_comp['pytorch_range'][0]:.6f}, {tensor_comp['pytorch_range'][1]:.6f}]")
                        report_lines.append(f"  - MLX 范围: [{tensor_comp['mlx_range'][0]:.6f}, {tensor_comp['mlx_range'][1]:.6f}]")
                        report_lines.append("")
    
    else:
        report_lines.append("## ❌ 对比失败")
        report_lines.append("")
        report_lines.append("无法进行子模块对比，请检查数据收集过程。")
        report_lines.append("")
    
    # 保存报告
    report_path = Path("cfm_debug_outputs/detailed_submodule_io_comparison_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"     ✅ Detailed I/O comparison report saved to: {report_path}")
    
    return report_path


def main():
    """主函数"""
    print("🔍 Detailed Submodule I/O Comparison")
    print("="*60)
    
    try:
        # 详细对比每个子模块的输入输出
        detailed_submodule_io_comparison()
        
        print("\n🎉 Detailed submodule I/O comparison completed!")
        print("\n📋 Generated files:")
        print("1. cfm_debug_outputs/detailed_submodule_io_comparison_report.md - 详细子模块输入输出对比报告")
        
        print("\n📊 Key Features:")
        print("   🔍 使用 CFM 前级缓存数据作为输入")
        print("   📊 详细收集每个子模块的输入输出")
        print("   🎯 对比 PyTorch 和 MLX 每个子模块的输入输出")
        print("   💡 找出具体的差异点")
        print("   🔧 提供详细的修复建议")
        
        print("\n🎯 Analysis Focus:")
        print("   📏 输出差异应该小于 e-5")
        print("   🔍 识别每个子模块的差异")
        print("   📊 按子模块分类差异")
        print("   🛠️ 提供针对性修复建议")
        
    except Exception as e:
        print(f"\n❌ Error during detailed I/O comparison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
