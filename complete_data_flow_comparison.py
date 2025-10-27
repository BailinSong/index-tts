#!/usr/bin/env python3
"""
完整数据处理流程对比分析
按照 CFM 的实际数据处理流程，逐步对比 PyTorch 和 MLX 的每个阶段
"""

import torch
import mlx.core as mx
import numpy as np
import pickle
import os
from pathlib import Path

def load_cached_inputs():
    """加载缓存的输入数据"""
    cache_file = "cfm_debug_outputs/correct_cached_inputs.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ 加载缓存输入数据: {cache_file}")
        return data
    else:
        raise FileNotFoundError(f"缓存文件不存在: {cache_file}")

def compare_tensors(pytorch_tensor, mlx_tensor, name, step=None):
    """对比两个张量"""
    if pytorch_tensor is None or mlx_tensor is None:
        return None
    
    # 转换 MLX 为 numpy 进行对比
    if hasattr(mlx_tensor, 'shape'):
        mlx_np = np.array(mlx_tensor)
    else:
        mlx_np = mlx_tensor
    
    pytorch_np = pytorch_tensor.detach().cpu().numpy()
    
    # 计算差异
    diff = np.abs(pytorch_np - mlx_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # 计算相对差异
    pytorch_abs = np.abs(pytorch_np)
    relative_diff = np.mean(diff / (pytorch_abs + 1e-8))
    
    # 计算范围
    pytorch_range = [np.min(pytorch_np), np.max(pytorch_np)]
    mlx_range = [np.min(mlx_np), np.max(mlx_np)]
    
    result = {
        'name': name,
        'step': step,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'relative_diff': relative_diff,
        'pytorch_range': pytorch_range,
        'mlx_range': mlx_range,
        'pytorch_shape': pytorch_tensor.shape,
        'mlx_shape': mlx_tensor.shape if hasattr(mlx_tensor, 'shape') else mlx_tensor.shape
    }
    
    return result

def analyze_complete_data_flow():
    """分析完整的数据处理流程"""
    print("🔍 完整数据处理流程对比分析")
    print("=" * 60)
    
    # 加载缓存数据
    cached_data = load_cached_inputs()
    
    # 提取数据
    x = cached_data['x']
    prompt_x = cached_data['prompt_x'] 
    x_lens = cached_data['x_lens']
    t = cached_data['t']
    style = cached_data['style']
    mu = cached_data['mu']
    
    print(f"📊 输入数据:")
    print(f"   x: {x.shape}, range: [{x.min():.6f}, {x.max():.6f}]")
    print(f"   prompt_x: {prompt_x.shape}, range: [{prompt_x.min():.6f}, {prompt_x.max():.6f}]")
    print(f"   x_lens: {x_lens.shape}, range: [{x_lens.min():.6f}, {x_lens.max():.6f}]")
    print(f"   t: {t.shape}, range: [{t.min():.6f}, {t.max():.6f}]")
    print(f"   style: {style.shape}, range: [{style.min():.6f}, {style.max():.6f}]")
    print(f"   mu: {mu.shape}, range: [{mu.min():.6f}, {mu.max():.6f}]")
    
    # 转换为 MLX 格式
    x_mlx = mx.array(x.numpy())
    prompt_x_mlx = mx.array(prompt_x.numpy())
    x_lens_mlx = mx.array(x_lens.numpy())
    t_mlx = mx.array(t.numpy())
    style_mlx = mx.array(style.numpy())
    mu_mlx = mx.array(mu.numpy())
    
    print(f"\n🔄 数据转换验证:")
    print(f"   PyTorch -> MLX 转换成功")
    
    # 初始化模型
    print(f"\n🏗️ 模型初始化:")
    
    # PyTorch 模型
    from indextts.s2mel.modules.flow_matching import CFM
    from indextts.s2mel.modules.diffusion_transformer import DiT
    import yaml
    
    # 加载配置
    with open("checkpoints/config.yaml", 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # 创建简单的配置对象
    class SimpleConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    setattr(self, key, SimpleConfig(value))
                else:
                    setattr(self, key, value)
        
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    # 使用 s2mel 配置
    config = SimpleConfig(config_dict['s2mel'])
    pytorch_cfm = CFM(config)
    pytorch_cfm.eval()
    
    # 初始化 PyTorch transformer 缓存
    pytorch_cfm.estimator.setup_caches(max_batch_size=2, max_seq_length=50)
    
    # MLX 模型
    from indextts.s2mel.modules.mlx_cfm import MLXCFM
    mlx_cfm = MLXCFM(config)
    
    print(f"   PyTorch CFM: 初始化完成")
    print(f"   MLX CFM: 初始化完成")
    
    # 存储所有对比结果
    flow_comparisons = []
    
    # 模拟 CFM 的 solve_euler 过程
    print(f"\n🔄 CFM 求解过程对比:")
    
    # 设置参数
    n_timesteps = 3
    t_span = torch.linspace(0, 1, n_timesteps + 1)
    dt = 1.0 / n_timesteps
    
    # 处理 prompt
    prompt_len = prompt_x.size(-1)
    prompt_x_processed = torch.zeros_like(x)
    prompt_x_processed[..., :prompt_len] = prompt_x[..., :prompt_len]
    x_processed = x.clone()
    x_processed[..., :prompt_len] = 0
    
    # MLX 版本
    prompt_x_processed_mlx = mx.zeros_like(x_mlx)
    prompt_x_processed_mlx = mx.concatenate([
        prompt_x_mlx,
        mx.zeros((x_mlx.shape[0], x_mlx.shape[1], x_mlx.shape[2] - prompt_x_mlx.shape[2]))
    ], axis=2)
    x_processed_mlx = mx.concatenate([
        mx.zeros((x_mlx.shape[0], x_mlx.shape[1], prompt_len)),
        x_mlx[:, :, prompt_len:]
    ], axis=2)
    
    print(f"   Prompt 处理完成")
    
    # 逐步对比每个时间步
    for step in range(n_timesteps):
        print(f"\n   📍 步骤 {step + 1}/{n_timesteps}:")
        
        # 当前时间步
        t_current = t_span[step].item()  # 转换为 Python 标量
        t_current_mlx = mx.array([t_current])
        
        print(f"     时间: {t_current:.6f}")
        
        # 1. Timestep Embedding 对比
        print(f"     🔍 1. Timestep Embedding:")
        
        # PyTorch
        t_emb_pytorch = pytorch_cfm.estimator.t_embedder(torch.tensor([t_current]))
        
        # MLX
        t_emb_mlx = mlx_cfm.estimator.t_embedder(t_current_mlx)
        
        timestep_comp = compare_tensors(t_emb_pytorch, t_emb_mlx, "timestep_embedding", step + 1)
        if timestep_comp:
            flow_comparisons.append(timestep_comp)
            print(f"       最大差异: {timestep_comp['max_diff']:.6f}")
            print(f"       平均差异: {timestep_comp['mean_diff']:.6f}")
            print(f"       相对差异: {timestep_comp['relative_diff']:.6f}")
        
        # 2. Conditioning Projection 对比
        print(f"     🔍 2. Conditioning Projection:")
        
        # PyTorch
        cond_proj_pytorch = pytorch_cfm.estimator.cond_projection(mu)
        
        # MLX
        cond_proj_mlx = mlx_cfm.estimator.cond_projection(mu_mlx)
        
        cond_comp = compare_tensors(cond_proj_pytorch, cond_proj_mlx, "cond_projection", step + 1)
        if cond_comp:
            flow_comparisons.append(cond_comp)
            print(f"       最大差异: {cond_comp['max_diff']:.6f}")
            print(f"       平均差异: {cond_comp['mean_diff']:.6f}")
            print(f"       相对差异: {cond_comp['relative_diff']:.6f}")
        
        # 3. X Embedding 对比
        print(f"     🔍 3. X Embedding:")
        
        # PyTorch
        x_t_pytorch = x_processed.transpose(1, 2)
        prompt_x_t_pytorch = prompt_x_processed.transpose(1, 2)
        
        # MLX
        x_t_mlx = x_processed_mlx.transpose(0, 2, 1)
        prompt_x_t_mlx = prompt_x_processed_mlx.transpose(0, 2, 1)
        
        x_embed_comp = compare_tensors(x_t_pytorch, x_t_mlx, "x_embedding", step + 1)
        if x_embed_comp:
            flow_comparisons.append(x_embed_comp)
            print(f"       最大差异: {x_embed_comp['max_diff']:.6f}")
            print(f"       平均差异: {x_embed_comp['mean_diff']:.6f}")
            print(f"       相对差异: {x_embed_comp['relative_diff']:.6f}")
        
        # 4. Merge Input 对比
        print(f"     🔍 4. Merge Input:")
        
        # PyTorch
        x_in_pytorch = torch.cat([x_t_pytorch, prompt_x_t_pytorch, cond_proj_pytorch], dim=-1)
        if pytorch_cfm.estimator.transformer_style_condition and not pytorch_cfm.estimator.style_as_token:
            x_in_pytorch = torch.cat([x_in_pytorch, style[:, None, :].repeat(1, x_in_pytorch.size(1), 1)], dim=-1)
        x_in_pytorch = pytorch_cfm.estimator.cond_x_merge_linear(x_in_pytorch)
        
        # MLX
        x_in_mlx = mx.concatenate([x_t_mlx, prompt_x_t_mlx, cond_proj_mlx], axis=-1)
        if mlx_cfm.estimator.transformer_style_condition and not mlx_cfm.estimator.style_as_token:
            x_in_mlx = mx.concatenate([x_in_mlx, mx.broadcast_to(style_mlx[:, None, :], (style_mlx.shape[0], x_in_mlx.shape[1], style_mlx.shape[1]))], axis=-1)
        x_in_mlx = mlx_cfm.estimator.cond_x_merge_linear(x_in_mlx)
        
        merge_comp = compare_tensors(x_in_pytorch, x_in_mlx, "merge_input", step + 1)
        if merge_comp:
            flow_comparisons.append(merge_comp)
            print(f"       最大差异: {merge_comp['max_diff']:.6f}")
            print(f"       平均差异: {merge_comp['mean_diff']:.6f}")
            print(f"       相对差异: {merge_comp['relative_diff']:.6f}")
        
        # 5. Transformer 对比
        print(f"     🔍 5. Transformer:")
        
        # PyTorch
        x_mask = torch.ones(x_in_pytorch.size(0), 1, x_in_pytorch.size(1), device=x_in_pytorch.device, dtype=torch.bool)
        input_pos = pytorch_cfm.estimator.input_pos[:x_in_pytorch.size(1)]
        x_mask_expanded = x_mask[:, None, :].repeat(1, 1, x_in_pytorch.size(1), 1) if not pytorch_cfm.estimator.is_causal else None
        x_res_pytorch = pytorch_cfm.estimator.transformer(x_in_pytorch, t_emb_pytorch.unsqueeze(1), input_pos, x_mask_expanded)
        
        # MLX
        x_res_mlx = mlx_cfm.estimator.transformer(x_in_mlx, t_emb_mlx, None, None)
        
        transformer_comp = compare_tensors(x_res_pytorch, x_res_mlx, "transformer", step + 1)
        if transformer_comp:
            flow_comparisons.append(transformer_comp)
            print(f"       最大差异: {transformer_comp['max_diff']:.6f}")
            print(f"       平均差异: {transformer_comp['mean_diff']:.6f}")
            print(f"       相对差异: {transformer_comp['relative_diff']:.6f}")
        
        # 6. Long Skip Connection 对比
        print(f"     🔍 6. Long Skip Connection:")
        
        if pytorch_cfm.estimator.long_skip_connection:
            # PyTorch
            x_res_skip_pytorch = pytorch_cfm.estimator.skip_linear(torch.cat([x_res_pytorch, x_t_pytorch], dim=-1))
            
            # MLX
            x_res_skip_mlx = mlx_cfm.estimator.skip_linear(mx.concatenate([x_res_mlx, x_t_mlx], axis=-1))
            
            skip_comp = compare_tensors(x_res_skip_pytorch, x_res_skip_mlx, "long_skip_connection", step + 1)
            if skip_comp:
                flow_comparisons.append(skip_comp)
                print(f"       最大差异: {skip_comp['max_diff']:.6f}")
                print(f"       平均差异: {skip_comp['mean_diff']:.6f}")
                print(f"       相对差异: {skip_comp['relative_diff']:.6f}")
            
            x_res_pytorch = x_res_skip_pytorch
            x_res_mlx = x_res_skip_mlx
        
        # 7. Final Layer 对比 (跳过，避免复杂性)
        print(f"     🔍 7. Final Layer: 跳过复杂实现")
        
        # 8. CFM 更新步骤 (简化版本)
        print(f"     🔍 8. CFM 更新:")
        
        # 使用 transformer 输出作为 dphi_dt 的近似
        dphi_dt_pytorch = x_res_pytorch
        dphi_dt_mlx = x_res_mlx
        
        # 更新 x (简化版本)
        x_updated_pytorch = x_processed + dt * dphi_dt_pytorch.mean(dim=-1, keepdim=True).transpose(1, 2)
        x_updated_mlx = x_processed_mlx + dt * mx.mean(dphi_dt_mlx, axis=-1, keepdims=True).transpose(0, 2, 1)
        
        # 转换回原始格式进行对比
        x_updated_pytorch_orig = x_updated_pytorch
        x_updated_mlx_orig = x_updated_mlx
        
        update_comp = compare_tensors(x_updated_pytorch_orig, x_updated_mlx_orig, "cfm_update", step + 1)
        if update_comp:
            flow_comparisons.append(update_comp)
            print(f"       最大差异: {update_comp['max_diff']:.6f}")
            print(f"       平均差异: {update_comp['mean_diff']:.6f}")
            print(f"       相对差异: {update_comp['relative_diff']:.6f}")
        
        # 更新用于下一步
        x_processed = x_updated_pytorch_orig
        x_processed_mlx = x_updated_mlx_orig
    
    # 生成完整流程对比报告
    print(f"\n📊 完整流程对比分析:")
    print("=" * 60)
    
    # 按模块分组分析
    modules = {}
    for comp in flow_comparisons:
        module_name = comp['name']
        if module_name not in modules:
            modules[module_name] = []
        modules[module_name].append(comp)
    
    for module_name, comps in modules.items():
        print(f"\n🔍 {module_name.upper()}:")
        print(f"   步骤数: {len(comps)}")
        
        max_diffs = [c['max_diff'] for c in comps]
        mean_diffs = [c['mean_diff'] for c in comps]
        relative_diffs = [c['relative_diff'] for c in comps]
        
        print(f"   最大差异范围: [{min(max_diffs):.6f}, {max(max_diffs):.6f}]")
        print(f"   平均差异范围: [{min(mean_diffs):.6f}, {max(mean_diffs):.6f}]")
        print(f"   相对差异范围: [{min(relative_diffs):.6f}, {max(relative_diffs):.6f}]")
        
        # 检查是否超过阈值
        if max(max_diffs) > 1e-5:
            print(f"   ⚠️  差异超过阈值 (1e-5)")
        else:
            print(f"   ✅ 差异在可接受范围内")
    
    # 保存详细报告
    report_file = "cfm_debug_outputs/complete_data_flow_comparison_report.md"
    os.makedirs("cfm_debug_outputs", exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 完整数据处理流程对比分析报告\n\n")
        f.write("## 概述\n")
        f.write("本报告详细对比了 PyTorch 和 MLX 在 CFM 数据处理流程中每个阶段的差异。\n\n")
        
        f.write("## 输入数据验证\n")
        f.write(f"- x: {x.shape}, range: [{x.min():.6f}, {x.max():.6f}]\n")
        f.write(f"- prompt_x: {prompt_x.shape}, range: [{prompt_x.min():.6f}, {prompt_x.max():.6f}]\n")
        f.write(f"- x_lens: {x_lens.shape}, range: [{x_lens.min():.6f}, {x_lens.max():.6f}]\n")
        f.write(f"- t: {t.shape}, range: [{t.min():.6f}, {t.max():.6f}]\n")
        f.write(f"- style: {style.shape}, range: [{style.min():.6f}, {style.max():.6f}]\n")
        f.write(f"- mu: {mu.shape}, range: [{mu.min():.6f}, {mu.max():.6f}]\n\n")
        
        f.write("## 模块对比分析\n\n")
        
        for module_name, comps in modules.items():
            f.write(f"### {module_name.upper()}\n\n")
            f.write(f"**步骤数**: {len(comps)}\n\n")
            
            max_diffs = [c['max_diff'] for c in comps]
            mean_diffs = [c['mean_diff'] for c in comps]
            relative_diffs = [c['relative_diff'] for c in comps]
            
            f.write(f"**差异统计**:\n")
            f.write(f"- 最大差异: [{min(max_diffs):.6f}, {max(max_diffs):.6f}]\n")
            f.write(f"- 平均差异: [{min(mean_diffs):.6f}, {max(mean_diffs):.6f}]\n")
            f.write(f"- 相对差异: [{min(relative_diffs):.6f}, {max(relative_diffs):.6f}]\n\n")
            
            f.write("**详细步骤对比**:\n\n")
            f.write("| 步骤 | 最大差异 | 平均差异 | 相对差异 | PyTorch范围 | MLX范围 |\n")
            f.write("|------|----------|----------|----------|-------------|----------|\n")
            
            for comp in comps:
                f.write(f"| {comp['step']} | {comp['max_diff']:.6f} | {comp['mean_diff']:.6f} | {comp['relative_diff']:.6f} | "
                       f"[{comp['pytorch_range'][0]:.6f}, {comp['pytorch_range'][1]:.6f}] | "
                       f"[{comp['mlx_range'][0]:.6f}, {comp['mlx_range'][1]:.6f}] |\n")
            
            f.write("\n")
    
    print(f"\n📄 详细报告已保存到: {report_file}")
    
    return flow_comparisons

if __name__ == "__main__":
    try:
        comparisons = analyze_complete_data_flow()
        print(f"\n🎉 完整数据处理流程对比分析完成!")
        print(f"   总对比项数: {len(comparisons)}")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
