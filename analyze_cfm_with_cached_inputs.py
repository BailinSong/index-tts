#!/usr/bin/env python3
"""
使用前级输入缓存作为CFM输入来分析差异来源
跳过前级处理，直接从CFM开始测试
"""

import sys
import os
import torch
import numpy as np
import pickle
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from indextts.infer_v2 import IndexTTS2
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch
from unified_random_generator import UnifiedRandomGenerator

def analyze_cfm_with_cached_inputs():
    """使用前级输入缓存作为CFM输入来分析差异来源"""
    print("=== 使用前级输入缓存作为CFM输入分析差异来源 ===\n")
    
    # 初始化TTS系统
    print("1. 初始化TTS系统...")
    tts = IndexTTS2()
    
    # 设置统一随机数生成器
    unified_rng = UnifiedRandomGenerator(42)
    
    # 检查是否有缓存输入数据
    print("2. 检查缓存输入数据...")
    cache_files = [
        "cfm_inputs_mlx.pkl",
        "cfm_inputs_torch.pkl",
        "cfm_outputs_mlx.pkl", 
        "cfm_outputs_pytorch.pkl"
    ]
    
    # 优先使用PyTorch缓存输入数据作为标准
    cached_inputs = None
    preferred_cache = "cfm_inputs_torch.pkl"
    
    if os.path.exists(preferred_cache):
        print(f"   找到首选缓存文件: {preferred_cache}")
        try:
            with open(preferred_cache, 'rb') as f:
                cached_inputs = pickle.load(f)
            print(f"   使用PyTorch缓存输入: {preferred_cache}")
            print(f"   缓存数据键: {list(cached_inputs.keys())}")
        except Exception as e:
            print(f"   读取首选缓存文件失败: {e}")
            cached_inputs = None
    
    # 如果首选缓存不可用，尝试其他缓存文件
    if cached_inputs is None:
        for cache_file in cache_files:
            if cache_file == preferred_cache:
                continue  # 已经尝试过了
                
            if os.path.exists(cache_file):
                print(f"   找到备用缓存文件: {cache_file}")
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    print(f"   缓存数据键: {list(cached_data.keys())}")
                    
                    if 'x' in cached_data:
                        cached_inputs = cached_data
                        print(f"   使用备用缓存输入: {cache_file}")
                        break
                except Exception as e:
                    print(f"   读取缓存文件失败: {e}")
            else:
                print(f"   未找到缓存文件: {cache_file}")
    
    if cached_inputs is None:
        print("   ❌ 未找到可用的缓存输入数据")
        return False
    
    # 提取缓存输入
    print("3. 提取缓存输入数据...")
    if cached_inputs is None:
        print("   ❌ 缓存输入数据为空")
        return False
    
    print(f"   缓存数据键: {list(cached_inputs.keys())}")
    
    # 从PyTorch缓存数据中提取CFM输入
    cat_condition = cached_inputs['cat_condition']
    x_lens = cached_inputs['x_lens']
    ref_mel = cached_inputs['ref_mel']
    style = cached_inputs['style']
    diffusion_steps = cached_inputs['diffusion_steps']
    inference_cfg_rate = cached_inputs['inference_cfg_rate']
    
    print(f"   缓存输入形状:")
    print(f"   cat_condition: {cat_condition.shape}, 范围: [{cat_condition.min():.6f}, {cat_condition.max():.6f}]")
    print(f"   x_lens: {x_lens.shape}, 值: {x_lens}")
    print(f"   ref_mel: {ref_mel.shape}, 范围: [{ref_mel.min():.6f}, {ref_mel.max():.6f}]")
    print(f"   style: {style.shape}, 范围: [{style.min():.6f}, {style.max():.6f}]")
    print(f"   diffusion_steps: {diffusion_steps}")
    print(f"   inference_cfg_rate: {inference_cfg_rate}")
    
    # 构建CFM输入
    # 使用cat_condition作为cond
    cond = cat_condition
    
    # 对于prompt_x，我们需要调整ref_mel的维度
    # ref_mel是(1, 80, 243)，但CFM期望(1, 464, 512)
    # 我们需要将ref_mel扩展到正确的维度和序列长度
    if ref_mel.shape[-1] != 512:
        # 使用零填充将ref_mel扩展到512维
        padding_size = 512 - ref_mel.shape[-1]
        ref_mel_padded = torch.cat([ref_mel, torch.zeros(ref_mel.shape[0], ref_mel.shape[1], padding_size)], dim=-1)
    else:
        ref_mel_padded = ref_mel
    
    # 将ref_mel扩展到与x相同的序列长度(464)
    if ref_mel_padded.shape[1] != 464:
        # 使用重复或插值将ref_mel扩展到464长度
        seq_len = ref_mel_padded.shape[1]
        target_len = 464
        if seq_len < target_len:
            # 重复最后几帧来填充
            repeat_times = target_len // seq_len
            remainder = target_len % seq_len
            prompt_x = ref_mel_padded.repeat(1, repeat_times, 1)
            if remainder > 0:
                prompt_x = torch.cat([prompt_x, ref_mel_padded[:, :remainder, :]], dim=1)
        else:
            # 截断到目标长度
            prompt_x = ref_mel_padded[:, :target_len, :]
    else:
        prompt_x = ref_mel_padded
    
    # 使用cat_condition作为x (保持与cond相同的形状)
    x = cat_condition
    
    # 创建时间步t (从0开始)
    t = torch.zeros(1, dtype=torch.float32)
    
    # 转换为PyTorch张量
    print("\n4. 转换为PyTorch张量...")
    # 确保所有张量都在同一个设备上
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    x_torch = x.to(device) if hasattr(x, 'to') else torch.tensor(x, dtype=torch.float32, device=device)
    prompt_x_torch = prompt_x.to(device) if hasattr(prompt_x, 'to') else torch.tensor(prompt_x, dtype=torch.float32, device=device)
    t_torch = t.to(device) if hasattr(t, 'to') else torch.tensor(t, dtype=torch.float32, device=device)
    style_torch = style.to(device) if hasattr(style, 'to') else torch.tensor(style, dtype=torch.float32, device=device)
    cond_torch = cond.to(device) if hasattr(cond, 'to') else torch.tensor(cond, dtype=torch.float32, device=device)
    x_lens_torch = x_lens.to(device) if hasattr(x_lens, 'to') else torch.tensor(x_lens, dtype=torch.long, device=device)
    
    # 转换为MLX数组
    print("5. 转换为MLX数组...")
    x_mlx = torch_to_mlx(x_torch)
    prompt_x_mlx = torch_to_mlx(prompt_x_torch)
    t_mlx = torch_to_mlx(t_torch)
    style_mlx = torch_to_mlx(style_torch)
    cond_mlx = torch_to_mlx(cond_torch)
    
    # 验证转换一致性
    print("\n6. 验证转换一致性...")
    x_diff = np.abs(x_torch.cpu().numpy() - mlx_to_torch(x_mlx).cpu().numpy())
    prompt_x_diff = np.abs(prompt_x_torch.cpu().numpy() - mlx_to_torch(prompt_x_mlx).cpu().numpy())
    t_diff = np.abs(t_torch.cpu().numpy() - mlx_to_torch(t_mlx).cpu().numpy())
    style_diff = np.abs(style_torch.cpu().numpy() - mlx_to_torch(style_mlx).cpu().numpy())
    cond_diff = np.abs(cond_torch.cpu().numpy() - mlx_to_torch(cond_mlx).cpu().numpy())
    
    print(f"   转换差异:")
    print(f"   x_diff: 最大 {x_diff.max():.10f}, 平均 {x_diff.mean():.10f}")
    print(f"   prompt_x_diff: 最大 {prompt_x_diff.max():.10f}, 平均 {prompt_x_diff.mean():.10f}")
    print(f"   t_diff: 最大 {t_diff.max():.10f}, 平均 {t_diff.mean():.10f}")
    print(f"   style_diff: 最大 {style_diff.max():.10f}, 平均 {style_diff.mean():.10f}")
    print(f"   cond_diff: 最大 {cond_diff.max():.10f}, 平均 {cond_diff.mean():.10f}")
    
    # 直接调用PyTorch CFM
    print("\n7. 直接调用PyTorch CFM...")
    try:
        pytorch_cfm = tts.s2mel.models.cfm
        pytorch_estimator = pytorch_cfm.estimator
        
        # 单步推理
        with torch.no_grad():
            pytorch_output = pytorch_estimator(
                x=x_torch,
                prompt_x=prompt_x_torch,
                t=t_torch,
                style=style_torch,
                cond=cond_torch,
                x_lens=x_lens_torch,
                mask_content=False
            )
        
        print(f"   PyTorch CFM输出: {pytorch_output.shape}")
        print(f"   PyTorch CFM输出范围: [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]")
        print(f"   PyTorch CFM输出均值: {pytorch_output.mean():.6f}, 标准差: {pytorch_output.std():.6f}")
        
    except Exception as e:
        print(f"   ❌ PyTorch CFM调用失败: {e}")
        return False
    
    # 直接调用MLX CFM
    print("\n8. 直接调用MLX CFM...")
    try:
        mlx_cfm = tts.mlx_s2mel_cfm
        mlx_estimator = mlx_cfm.estimator
        
        # 单步推理
        mlx_output = mlx_estimator(
            x=x_mlx,
            prompt_x=prompt_x_mlx,
            t=t_mlx,
            style=style_mlx,
            cond=cond_mlx,
            x_lens=torch_to_mlx(x_lens),
            mask_content=False
        )
        
        print(f"   MLX CFM输出: {mlx_output.shape}")
        print(f"   MLX CFM输出范围: [{mlx_output.min():.6f}, {mlx_output.max():.6f}]")
        print(f"   MLX CFM输出均值: {mlx_output.mean():.6f}, 标准差: {mlx_output.std():.6f}")
        
    except Exception as e:
        print(f"   ❌ MLX CFM调用失败: {e}")
        return False
    
    # 比较输出差异
    print("\n9. 比较输出差异...")
    pytorch_output_np = pytorch_output.cpu().numpy()
    mlx_output_np = mlx_output
    
    output_diff = np.abs(pytorch_output_np - mlx_output_np)
    max_diff = output_diff.max()
    mean_diff = output_diff.mean()
    
    print(f"   输出差异:")
    print(f"   最大差异: {max_diff:.6f}")
    print(f"   平均差异: {mean_diff:.6f}")
    print(f"   差异分布:")
    print(f"   - 差异 < 1e-5: {np.sum(output_diff < 1e-5)} / {output_diff.size} ({100*np.sum(output_diff < 1e-5)/output_diff.size:.1f}%)")
    print(f"   - 差异 < 1e-3: {np.sum(output_diff < 1e-3)} / {output_diff.size} ({100*np.sum(output_diff < 1e-3)/output_diff.size:.1f}%)")
    print(f"   - 差异 < 1e-1: {np.sum(output_diff < 1e-1)} / {output_diff.size} ({100*np.sum(output_diff < 1e-1)/output_diff.size:.1f}%)")
    print(f"   - 差异 < 1.0: {np.sum(output_diff < 1.0)} / {output_diff.size} ({100*np.sum(output_diff < 1.0)/output_diff.size:.1f}%)")
    
    # 分析大差异位置
    print("\n10. 分析大差异位置...")
    large_diff_mask = output_diff > 1.0
    if np.any(large_diff_mask):
        large_diff_indices = np.where(large_diff_mask)
        print(f"   大差异位置数: {len(large_diff_indices[0])}")
        print(f"   前5个大差异位置:")
        for i in range(min(5, len(large_diff_indices[0]))):
            idx = tuple(large_diff_indices[j][i] for j in range(len(large_diff_indices)))
            pytorch_val = pytorch_output_np[idx]
            mlx_val = mlx_output_np[idx]
            diff_val = output_diff[idx]
            print(f"     位置 {idx}: PyTorch={pytorch_val:.6f}, MLX={mlx_val:.6f}, 差异={diff_val:.6f}")
    else:
        print("   ✅ 没有大差异位置")
    
    # 保存分析结果
    print("\n11. 保存分析结果...")
    analysis_result = {
        'cached_inputs': {
            'x': x,
            'prompt_x': prompt_x,
            't': t,
            'style': style,
            'cond': cond
        },
        'pytorch_output': pytorch_output_np,
        'mlx_output': mlx_output_np,
        'output_diff': output_diff,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'diff_distribution': {
            'lt_1e5': np.sum(output_diff < 1e-5),
            'lt_1e3': np.sum(output_diff < 1e-3),
            'lt_1e1': np.sum(output_diff < 1e-1),
            'lt_1_0': np.sum(output_diff < 1.0),
            'total': output_diff.size
        }
    }
    
    with open('cfm_cached_inputs_analysis.pkl', 'wb') as f:
        pickle.dump(analysis_result, f)
    
    print(f"   分析结果已保存到: cfm_cached_inputs_analysis.pkl")
    
    # 总结
    print("\n12. 总结...")
    if max_diff < 1e-5:
        print("   ✅ CFM输出差异在e-5范围内，完全一致")
    elif max_diff < 1e-3:
        print("   ✅ CFM输出差异在e-3范围内，基本一致")
    elif max_diff < 1e-1:
        print("   ⚠️ CFM输出差异在e-1范围内，有轻微差异")
    else:
        print("   ❌ CFM输出差异较大，需要进一步分析")
    
    return True

if __name__ == "__main__":
    try:
        success = analyze_cfm_with_cached_inputs()
        if success:
            print("\n✅ CFM缓存输入分析完成")
        else:
            print("\n❌ CFM缓存输入分析失败")
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
