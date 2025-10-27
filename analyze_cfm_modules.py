#!/usr/bin/env python3
"""
明确产生差异的CFM流程的具体模块分析
通过逐步对比PyTorch和MLX的CFM组件来定位问题
"""

import sys
import os
sys.path.append('.')

import torch
import numpy as np
import pickle
from indextts.infer_v2 import IndexTTS2
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def analyze_cfm_module_differences():
    """分析CFM流程中具体模块的差异"""
    
    print("=== 分析CFM流程中具体模块的差异 ===")
    
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
    target_seq_len = mu.shape[1]
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
    
    # 分析PyTorch CFM的各个组件
    print(f"\n🔥 分析PyTorch CFM组件...")
    
    pytorch_cfm = tts.s2mel.models.cfm
    pytorch_estimator = pytorch_cfm.estimator
    
    # 准备CFG输入
    prompt_len = prompt.size(-1)
    prompt_x = torch.zeros_like(x)
    prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
    x[..., :prompt_len] = 0
    
    # 测试单个时间步
    t = torch.tensor([0.0], device=device)
    
    # CFG处理
    stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
    stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)
    stacked_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
    stacked_x = torch.cat([x, x], dim=0)
    stacked_t = torch.cat([t, t], dim=0)
    
    print(f"\n📊 PyTorch CFM组件分析:")
    
    # 1. 分析cond_projection
    print(f"\n1️⃣ 分析cond_projection:")
    pytorch_cond_emb = pytorch_estimator.cond_projection(stacked_mu)
    print(f"  cond_projection输出: {pytorch_cond_emb.shape}, min={pytorch_cond_emb.min():.6f}, max={pytorch_cond_emb.max():.6f}")
    
    # 2. 分析t_embedder
    print(f"\n2️⃣ 分析t_embedder:")
    pytorch_t_emb = pytorch_estimator.t_embedder(stacked_t)
    print(f"  t_embedder输出: {pytorch_t_emb.shape}, min={pytorch_t_emb.min():.6f}, max={pytorch_t_emb.max():.6f}")
    
    # 3. 分析x_embedder
    print(f"\n3️⃣ 分析x_embedder:")
    pytorch_x_emb = pytorch_estimator.x_embedder(stacked_x)
    print(f"  x_embedder输出: {pytorch_x_emb.shape}, min={pytorch_x_emb.min():.6f}, max={pytorch_x_emb.max():.6f}")
    
    # 4. 分析完整estimator
    print(f"\n4️⃣ 分析完整estimator:")
    pytorch_output = pytorch_estimator(stacked_x, stacked_prompt_x, x_lens, stacked_t, stacked_style, stacked_mu)
    print(f"  estimator输出: {pytorch_output.shape}, min={pytorch_output.min():.6f}, max={pytorch_output.max():.6f}")
    
    # 分析MLX CFM的各个组件
    print(f"\n🔥 分析MLX CFM组件...")
    
    # 确保MLX CFM已初始化
    if tts.mlx_s2mel_cfm is None:
        print("🔧 初始化MLX CFM...")
        from indextts.s2mel.modules.mlx_cfm import MLXCFM
        tts.mlx_s2mel_cfm = MLXCFM(tts.cfg.s2mel)
        tts.mlx_s2mel_cfm.load_weights_from_pytorch(tts.s2mel.models.cfm.state_dict())
    
    mlx_cfm = tts.mlx_s2mel_cfm
    mlx_estimator = mlx_cfm.estimator
    
    # 转换为MLX格式
    mu_mlx = torch_to_mlx(mu)
    x_lens_mlx = torch_to_mlx(x_lens)
    prompt_mlx = torch_to_mlx(prompt)
    style_mlx = torch_to_mlx(style)
    x_mlx = torch_to_mlx(x)
    
    # 重新生成x（因为上面被修改了）
    torch.manual_seed(42)
    x_mlx = torch_to_mlx(torch.randn(B, 80, T, device=device))
    
    # 准备MLX CFG输入
    prompt_x_mlx = mx.zeros_like(x_mlx)
    prompt_x_mlx[:, :, :prompt_len] = prompt_mlx[:, :, :prompt_len]
    x_mlx[:, :, :prompt_len] = 0
    
    # 测试单个时间步
    t_mlx = mx.array([0.0])
    
    # CFG处理
    stacked_prompt_x_mlx = mx.concatenate([prompt_x_mlx, mx.zeros_like(prompt_x_mlx)], axis=0)
    stacked_style_mlx = mx.concatenate([style_mlx, mx.zeros_like(style_mlx)], axis=0)
    stacked_mu_mlx = mx.concatenate([mu_mlx, mx.zeros_like(mu_mlx)], axis=0)
    stacked_x_mlx = mx.concatenate([x_mlx, x_mlx], axis=0)
    stacked_t_mlx = mx.concatenate([t_mlx, t_mlx], axis=0)
    stacked_x_lens_mlx = mx.concatenate([x_lens_mlx, x_lens_mlx], axis=0)
    
    print(f"\n📊 MLX CFM组件分析:")
    
    # 1. 分析cond_projection
    print(f"\n1️⃣ 分析cond_projection:")
    mlx_cond_emb = mlx_estimator.cond_projection(stacked_mu_mlx)
    print(f"  cond_projection输出: {mlx_cond_emb.shape}, min={mlx_cond_emb.min():.6f}, max={mlx_cond_emb.max():.6f}")
    
    # 2. 分析t_embedder
    print(f"\n2️⃣ 分析t_embedder:")
    mlx_t_emb = mlx_estimator.t_embedder(stacked_t_mlx)
    print(f"  t_embedder输出: {mlx_t_emb.shape}, min={mlx_t_emb.min():.6f}, max={mlx_t_emb.max():.6f}")
    
    # 3. 分析x_embedder
    print(f"\n3️⃣ 分析x_embedder:")
    mlx_x_emb = mlx_estimator.x_embedder(stacked_x_mlx)
    print(f"  x_embedder输出: {mlx_x_emb.shape}, min={mlx_x_emb.min():.6f}, max={mlx_x_emb.max():.6f}")
    
    # 4. 分析完整estimator
    print(f"\n4️⃣ 分析完整estimator:")
    mlx_output = mlx_estimator(stacked_x_mlx, stacked_prompt_x_mlx, stacked_x_lens_mlx, 
                              stacked_t_mlx, stacked_style_mlx, stacked_mu_mlx,
                              mask_content=False)
    print(f"  estimator输出: {mlx_output.shape}, min={mlx_output.min():.6f}, max={mlx_output.max():.6f}")
    
    # 比较各组件差异
    print(f"\n🔍 组件差异分析:")
    
    # 1. cond_projection差异
    cond_emb_diff = torch.abs(pytorch_cond_emb - mlx_to_torch(mlx_cond_emb))
    print(f"\n1️⃣ cond_projection差异:")
    print(f"  最大差异: {cond_emb_diff.max():.6f}")
    print(f"  平均差异: {cond_emb_diff.mean():.6f}")
    print(f"  差异标准差: {cond_emb_diff.std():.6f}")
    
    # 2. t_embedder差异
    t_emb_diff = torch.abs(pytorch_t_emb - mlx_to_torch(mlx_t_emb))
    print(f"\n2️⃣ t_embedder差异:")
    print(f"  最大差异: {t_emb_diff.max():.6f}")
    print(f"  平均差异: {t_emb_diff.mean():.6f}")
    print(f"  差异标准差: {t_emb_diff.std():.6f}")
    
    # 3. x_embedder差异
    x_emb_diff = torch.abs(pytorch_x_emb - mlx_to_torch(mlx_x_emb))
    print(f"\n3️⃣ x_embedder差异:")
    print(f"  最大差异: {x_emb_diff.max():.6f}")
    print(f"  平均差异: {x_emb_diff.mean():.6f}")
    print(f"  差异标准差: {x_emb_diff.std():.6f}")
    
    # 4. 完整estimator差异
    estimator_diff = torch.abs(pytorch_output - mlx_to_torch(mlx_output))
    print(f"\n4️⃣ 完整estimator差异:")
    print(f"  最大差异: {estimator_diff.max():.6f}")
    print(f"  平均差异: {estimator_diff.mean():.6f}")
    print(f"  差异标准差: {estimator_diff.std():.6f}")
    
    # 权重比较
    print(f"\n🔍 权重差异分析:")
    
    # 比较cond_projection权重
    pytorch_cond_weight = pytorch_estimator.cond_projection.weight
    mlx_cond_weight = mlx_to_torch(mlx_estimator.cond_projection.weight)
    cond_weight_diff = torch.abs(pytorch_cond_weight - mlx_cond_weight)
    print(f"\n1️⃣ cond_projection权重差异:")
    print(f"  最大差异: {cond_weight_diff.max():.6f}")
    print(f"  平均差异: {cond_weight_diff.mean():.6f}")
    
    # 比较t_embedder权重
    pytorch_t_weight = pytorch_estimator.t_embedder.mlp[0].weight
    mlx_t_weight = mlx_to_torch(mlx_estimator.t_embedder.mlp[0].weight)
    t_weight_diff = torch.abs(pytorch_t_weight - mlx_t_weight)
    print(f"\n2️⃣ t_embedder权重差异:")
    print(f"  最大差异: {t_weight_diff.max():.6f}")
    print(f"  平均差异: {t_weight_diff.mean():.6f}")
    
    # 比较x_embedder权重
    pytorch_x_weight = pytorch_estimator.x_embedder.weight_v
    mlx_x_weight = mlx_to_torch(mlx_estimator.x_embedder.weight_v)
    x_weight_diff = torch.abs(pytorch_x_weight - mlx_x_weight)
    print(f"\n3️⃣ x_embedder权重差异:")
    print(f"  最大差异: {x_weight_diff.max():.6f}")
    print(f"  平均差异: {x_weight_diff.mean():.6f}")
    
    # 总结
    print(f"\n📋 总结:")
    
    max_diffs = {
        'cond_projection': cond_emb_diff.max().item(),
        't_embedder': t_emb_diff.max().item(),
        'x_embedder': x_emb_diff.max().item(),
        'estimator': estimator_diff.max().item(),
        'cond_weight': cond_weight_diff.max().item(),
        't_weight': t_weight_diff.max().item(),
        'x_weight': x_weight_diff.max().item()
    }
    
    # 找出差异最大的组件
    max_diff_component = max(max_diffs, key=max_diffs.get)
    print(f"  差异最大的组件: {max_diff_component} (差异: {max_diffs[max_diff_component]:.6f})")
    
    # 保存分析结果
    analysis_result = {
        'component_diffs': {
            'cond_projection': {
                'max_diff': cond_emb_diff.max().item(),
                'mean_diff': cond_emb_diff.mean().item(),
                'std_diff': cond_emb_diff.std().item()
            },
            't_embedder': {
                'max_diff': t_emb_diff.max().item(),
                'mean_diff': t_emb_diff.mean().item(),
                'std_diff': t_emb_diff.std().item()
            },
            'x_embedder': {
                'max_diff': x_emb_diff.max().item(),
                'mean_diff': x_emb_diff.mean().item(),
                'std_diff': x_emb_diff.std().item()
            },
            'estimator': {
                'max_diff': estimator_diff.max().item(),
                'mean_diff': estimator_diff.mean().item(),
                'std_diff': estimator_diff.std().item()
            }
        },
        'weight_diffs': {
            'cond_weight': {
                'max_diff': cond_weight_diff.max().item(),
                'mean_diff': cond_weight_diff.mean().item()
            },
            't_weight': {
                'max_diff': t_weight_diff.max().item(),
                'mean_diff': t_weight_diff.mean().item()
            },
            'x_weight': {
                'max_diff': x_weight_diff.max().item(),
                'mean_diff': x_weight_diff.mean().item()
            }
        },
        'max_diff_component': max_diff_component
    }
    
    with open('cfm_module_analysis.pkl', 'wb') as f:
        pickle.dump(analysis_result, f)
    
    print(f"\n💾 分析结果已保存到: cfm_module_analysis.pkl")
    
    return analysis_result

def main():
    """主函数"""
    
    print("开始分析CFM流程中具体模块的差异...")
    
    # 分析CFM模块差异
    result = analyze_cfm_module_differences()
    
    if result:
        print(f"\n🎉 分析完成！")
        print(f"差异最大的组件: {result['max_diff_component']}")

if __name__ == "__main__":
    main()
