#!/usr/bin/env python3
"""
分析CFM推理过程中各阶段的差异
"""

import sys
import os
sys.path.append('.')

import torch
import numpy as np
import pickle
from indextts.infer_v2 import IndexTTS2
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def analyze_cfm_step_by_step():
    """逐步分析CFM推理过程中的差异"""
    
    print("=== 分析CFM推理过程中各阶段的差异 ===")
    
    # 加载缓存输入
    cache_file = 'cfm_inputs_mlx.pkl'
    if not os.path.exists(cache_file):
        print(f"❌ 缓存文件不存在: {cache_file}")
        return
    
    with open(cache_file, 'rb') as f:
        cached_data = pickle.load(f)
    
    # 初始化TTS
    tts = IndexTTS2()
    
    # 提取缓存数据
    cat_condition = cached_data['cat_condition']
    x_lens = cached_data['x_lens']
    ref_mel = cached_data['ref_mel']
    style = cached_data['style']
    diffusion_steps = cached_data['diffusion_steps']
    inference_cfg_rate = cached_data['inference_cfg_rate']
    
    # 确保所有张量在正确设备上
    device = 'mps'
    cat_condition = cat_condition.to(device)
    x_lens = x_lens.to(device)
    ref_mel = ref_mel.to(device)
    style = style.to(device)
    
    # 构造CFM输入
    mu = cat_condition  # (batch, seq_len, 512)
    prompt = ref_mel    # (batch, 80, prompt_len)
    
    # 调整prompt序列长度匹配mu的序列长度
    target_seq_len = mu.shape[1]  # 523
    if prompt.shape[-1] != target_seq_len:
        if prompt.shape[-1] < target_seq_len:
            repeat_factor = target_seq_len // prompt.shape[-1] + 1
            prompt = prompt.repeat(1, 1, repeat_factor)[:, :, :target_seq_len]
        else:
            prompt = prompt[:, :, :target_seq_len]
    
    # 生成随机噪声x
    B, T = mu.size(0), mu.size(1)
    torch.manual_seed(42)
    x = torch.randn(B, 80, T, device=device)
    
    print(f"\n📋 输入数据:")
    print(f"  mu: {mu.shape}, min={mu.min():.6f}, max={mu.max():.6f}")
    print(f"  prompt: {prompt.shape}, min={prompt.min():.6f}, max={prompt.max():.6f}")
    print(f"  x: {x.shape}, min={x.min():.6f}, max={x.max():.6f}")
    print(f"  x_lens: {x_lens}")
    print(f"  style: {style.shape}, min={style.min():.6f}, max={style.max():.6f}")
    
    # 分析PyTorch CFM的中间步骤
    print(f"\n🔥 分析PyTorch CFM中间步骤...")
    
    pytorch_cfm = tts.s2mel.models.cfm
    
    # 手动执行CFM的solve_euler步骤
    t_span = torch.linspace(0, 1, diffusion_steps + 1, device=device)
    
    # 准备prompt
    prompt_len = prompt.size(-1)
    prompt_x = torch.zeros_like(x)
    prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
    x[..., :prompt_len] = 0
    
    pytorch_steps = []
    
    # 执行前几步
    for step in range(1, min(6, len(t_span))):  # 只分析前5步
        dt = t_span[step] - t_span[step - 1]
        t = t_span[step - 1]
        
        if inference_cfg_rate > 0:
            # CFG处理
            stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
            stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)
            stacked_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
            stacked_x = torch.cat([x, x], dim=0)
            stacked_t = torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0)
            
            # 调用estimator
            stacked_dphi_dt = pytorch_cfm.estimator(
                stacked_x, stacked_prompt_x, x_lens, stacked_t, stacked_style, stacked_mu,
            )
            
            # CFG处理
            dphi_dt_cond, dphi_dt_uncond = stacked_dphi_dt.chunk(2, dim=0)
            dphi_dt = dphi_dt_uncond + inference_cfg_rate * (dphi_dt_cond - dphi_dt_uncond)
        else:
            dphi_dt = pytorch_cfm.estimator(
                x, prompt_x, x_lens, t.unsqueeze(0), style, mu,
            )
        
        # Euler步骤
        x = x + dt * dphi_dt
        
        pytorch_steps.append({
            'step': step,
            't': float(t),
            'dt': float(dt),
            'x_min': float(x.min()),
            'x_max': float(x.max()),
            'x_mean': float(x.mean()),
            'dphi_dt_min': float(dphi_dt.min()),
            'dphi_dt_max': float(dphi_dt.max()),
            'dphi_dt_mean': float(dphi_dt.mean())
        })
        
        print(f"  Step {step}: t={t:.3f}, x=[{x.min():.6f}, {x.max():.6f}], dphi_dt=[{dphi_dt.min():.6f}, {dphi_dt.max():.6f}]")
    
    # 分析MLX CFM的中间步骤
    print(f"\n🔥 分析MLX CFM中间步骤...")
    
    # 确保MLX CFM已初始化
    if tts.mlx_s2mel_cfm is None:
        print("🔧 初始化MLX CFM...")
        from indextts.s2mel.modules.mlx_cfm import MLXCFM
        tts.mlx_s2mel_cfm = MLXCFM(tts.cfg.s2mel)
        tts.mlx_s2mel_cfm.load_weights_from_pytorch(tts.s2mel.models.cfm.state_dict())
    
    mlx_cfm = tts.mlx_s2mel_cfm
    
    # 转换为MLX格式
    mu_mlx = torch_to_mlx(mu)
    x_lens_mlx = torch_to_mlx(x_lens)
    prompt_mlx = torch_to_mlx(prompt)
    style_mlx = torch_to_mlx(style)
    x_mlx = torch_to_mlx(x)
    
    # 重新生成x（因为上面被修改了）
    torch.manual_seed(42)
    x_mlx = torch_to_mlx(torch.randn(B, 80, T, device=device))
    
    # 手动执行MLX CFM的solve_euler步骤
    import mlx.core as mx
    t_span_mlx = mx.linspace(0, 1, diffusion_steps + 1)
    
    # 准备prompt
    prompt_x_mlx = mx.zeros_like(x_mlx)
    prompt_x_mlx[:, :, :prompt_len] = prompt_mlx[:, :, :prompt_len]
    x_mlx[:, :, :prompt_len] = 0
    
    mlx_steps = []
    
    # 执行前几步
    for step in range(1, min(6, len(t_span_mlx))):  # 只分析前5步
        dt = t_span_mlx[step] - t_span_mlx[step - 1]
        t = t_span_mlx[step - 1]
        
        if inference_cfg_rate > 0:
            # CFG处理
            stacked_prompt_x = mx.concatenate([prompt_x_mlx, mx.zeros_like(prompt_x_mlx)], axis=0)
            stacked_style = mx.concatenate([style_mlx, mx.zeros_like(style_mlx)], axis=0)
            stacked_mu = mx.concatenate([mu_mlx, mx.zeros_like(mu_mlx)], axis=0)
            stacked_x = mx.concatenate([x_mlx, x_mlx], axis=0)
            stacked_t = mx.concatenate([mx.array([float(t)]), mx.array([float(t)])], axis=0)
            stacked_x_lens = mx.concatenate([x_lens_mlx, x_lens_mlx], axis=0)
            
            # 调用estimator
            stacked_dphi_dt = mlx_cfm.estimator(
                stacked_x, stacked_prompt_x, stacked_x_lens, 
                stacked_t, stacked_style, stacked_mu,
                mask_content=False
            )
            
            # CFG处理
            dphi_dt_cond, dphi_dt_uncond = mx.split(stacked_dphi_dt, 2, axis=0)
            dphi_dt = dphi_dt_uncond + inference_cfg_rate * (dphi_dt_cond - dphi_dt_uncond)
        else:
            t_scalar = mx.array([float(t)])
            dphi_dt = mlx_cfm.estimator(
                x_mlx, prompt_x_mlx, x_lens_mlx, 
                t_scalar, style_mlx, mu_mlx,
                mask_content=False
            )
        
        # Euler步骤
        x_mlx = x_mlx + dt * dphi_dt
        
        mlx_steps.append({
            'step': step,
            't': float(t),
            'dt': float(dt),
            'x_min': float(x_mlx.min()),
            'x_max': float(x_mlx.max()),
            'x_mean': float(x_mlx.mean()),
            'dphi_dt_min': float(dphi_dt.min()),
            'dphi_dt_max': float(dphi_dt.max()),
            'dphi_dt_mean': float(dphi_dt.mean())
        })
        
        print(f"  Step {step}: t={t:.3f}, x=[{x_mlx.min():.6f}, {x_mlx.max():.6f}], dphi_dt=[{dphi_dt.min():.6f}, {dphi_dt.max():.6f}]")
    
    # 比较步骤差异
    print(f"\n📊 步骤差异分析:")
    
    for i, (pt_step, mlx_step) in enumerate(zip(pytorch_steps, mlx_steps)):
        print(f"\n  Step {i+1}:")
        print(f"    t差异: {abs(pt_step['t'] - mlx_step['t']):.8f}")
        print(f"    dt差异: {abs(pt_step['dt'] - mlx_step['dt']):.8f}")
        
        x_diff = abs(pt_step['x_mean'] - mlx_step['x_mean'])
        dphi_diff = abs(pt_step['dphi_dt_mean'] - mlx_step['dphi_dt_mean'])
        
        print(f"    x均值差异: {x_diff:.6f}")
        print(f"    dphi_dt均值差异: {dphi_diff:.6f}")
        
        if x_diff > 0.1 or dphi_diff > 0.1:
            print(f"    ⚠️  差异较大！")
        elif x_diff > 0.01 or dphi_diff > 0.01:
            print(f"    ⚠️  差异中等")
        else:
            print(f"    ✅ 差异较小")
    
    # 保存分析结果
    analysis_result = {
        'pytorch_steps': pytorch_steps,
        'mlx_steps': mlx_steps,
        'input_data': {
            'mu_shape': mu.shape,
            'prompt_shape': prompt.shape,
            'x_shape': x.shape,
            'diffusion_steps': diffusion_steps,
            'inference_cfg_rate': inference_cfg_rate
        }
    }
    
    with open('cfm_step_analysis.pkl', 'wb') as f:
        pickle.dump(analysis_result, f)
    
    print(f"\n💾 分析结果已保存到: cfm_step_analysis.pkl")
    
    return analysis_result

def main():
    """主函数"""
    
    print("开始分析CFM推理过程中各阶段的差异...")
    
    # 分析CFM步骤差异
    result = analyze_cfm_step_by_step()
    
    if result:
        print(f"\n🎉 分析完成！")
        print(f"已分析 {len(result['pytorch_steps'])} 个推理步骤")

if __name__ == "__main__":
    main()
