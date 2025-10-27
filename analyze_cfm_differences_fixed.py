#!/usr/bin/env python3
"""
深度分析MLX CFM与PyTorch CFM差异的根本原因 - 修复版本
"""

import sys
sys.path.append('.')
import torch
import mlx.core as mx
import numpy as np
import os
import pickle
import librosa
from indextts.infer_v2 import IndexTTS2
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch
from unified_random_generator import UnifiedRandomGenerator

def analyze_cfm_differences():
    """使用前级缓存数据直接分析CFM差异"""
    
    print('=== 使用前级缓存数据直接分析CFM差异 ===')
    
    # 设置固定种子
    torch.manual_seed(42)
    np.random.seed(42)
    mx.random.seed(42)
    
    print(f'固定种子: 42')
    
    # 初始化 TTS 系统
    print('\\n1. 初始化 TTS 系统...')
    tts = IndexTTS2()
    
    # 加载前级缓存数据
    print('\\n2. 加载前级缓存数据...')
    cache_file = 'cfm_inputs_torch.pkl'
    
    if not os.path.exists(cache_file):
        print(f'❌ 未找到缓存文件: {cache_file}')
        return False
    
    try:
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        print(f'✅ 成功加载缓存文件: {cache_file}')
        print(f'   缓存数据键: {list(cached_data.keys())}')
    except Exception as e:
        print(f'❌ 加载缓存文件失败: {e}')
        return False
    
    # 提取缓存数据
    print('\\n3. 提取缓存数据...')
    cat_condition = cached_data['cat_condition']
    x_lens = cached_data['x_lens']
    ref_mel = cached_data['ref_mel']
    style = cached_data['style']
    diffusion_steps = cached_data['diffusion_steps']
    inference_cfg_rate = cached_data['inference_cfg_rate']
    
    print(f'   缓存数据形状:')
    print(f'   cat_condition: {cat_condition.shape}, 范围: [{cat_condition.min():.6f}, {cat_condition.max():.6f}]')
    print(f'   x_lens: {x_lens.shape}, 值: {x_lens}')
    print(f'   ref_mel: {ref_mel.shape}, 范围: [{ref_mel.min():.6f}, {ref_mel.max():.6f}]')
    print(f'   style: {style.shape}, 范围: [{style.min():.6f}, {style.max():.6f}]')
    print(f'   diffusion_steps: {diffusion_steps}')
    print(f'   inference_cfg_rate: {inference_cfg_rate}')
    
    # 构建CFM输入
    print('\\n4. 构建CFM输入...')
    
    # 确保所有张量都在MPS设备上
    device = 'mps'
    cat_condition = cat_condition.to(device)
    x_lens = x_lens.to(device)
    ref_mel = ref_mel.to(device)
    style = style.to(device)
    
    # 根据CFM inference方法的参数格式：
    # mu: (batch, seq_len, 512) - 语义条件信息
    # x_lens: (batch,) - 序列长度
    # prompt: (batch, 80, prompt_len) - 参考梅尔频谱
    # style: (batch, 192) - 风格向量
    
    # 使用cat_condition作为mu (语义条件信息)
    mu = cat_condition  # (1, 464, 512)
    
    # 使用ref_mel作为prompt (参考梅尔频谱)
    prompt = ref_mel  # (1, 80, 243)
    
    # 生成随机噪声作为初始x
    B, T = mu.size(0), mu.size(1)
    x = torch.randn(B, 80, T, device=device)  # (1, 80, 464)
    
    # 创建时间步t (从0开始)
    t = torch.zeros(1, dtype=torch.float32, device=device)
    
    print(f'   CFM输入形状:')
    print(f'   mu: {mu.shape}, 范围: [{mu.min():.6f}, {mu.max():.6f}]')
    print(f'   x_lens: {x_lens.shape}, 值: {x_lens}')
    print(f'   prompt: {prompt.shape}, 范围: [{prompt.min():.6f}, {prompt.max():.6f}]')
    print(f'   style: {style.shape}, 范围: [{style.min():.6f}, {style.max():.6f}]')
    print(f'   x: {x.shape}, 范围: [{x.min():.6f}, {x.max():.6f}]')
    print(f'   t: {t.shape}, 值: {t}')
    
    # 转换为MLX数组
    print('\\n5. 转换为MLX数组...')
    mu_mlx = torch_to_mlx(mu)
    x_lens_mlx = torch_to_mlx(x_lens)
    prompt_mlx = torch_to_mlx(prompt)
    style_mlx = torch_to_mlx(style)
    x_mlx = torch_to_mlx(x)
    t_mlx = torch_to_mlx(t)
    
    # 验证转换一致性
    print('\\n6. 验证转换一致性...')
    mu_diff = np.abs(mu.cpu().numpy() - mlx_to_torch(mu_mlx).cpu().numpy())
    x_lens_diff = np.abs(x_lens.cpu().numpy() - mlx_to_torch(x_lens_mlx).cpu().numpy())
    prompt_diff = np.abs(prompt.cpu().numpy() - mlx_to_torch(prompt_mlx).cpu().numpy())
    style_diff = np.abs(style.cpu().numpy() - mlx_to_torch(style_mlx).cpu().numpy())
    x_diff = np.abs(x.cpu().numpy() - mlx_to_torch(x_mlx).cpu().numpy())
    t_diff = np.abs(t.cpu().numpy() - mlx_to_torch(t_mlx).cpu().numpy())
    
    print(f'   转换差异:')
    print(f'   mu_diff: 最大 {mu_diff.max():.10f}, 平均 {mu_diff.mean():.10f}')
    print(f'   x_lens_diff: 最大 {x_lens_diff.max():.10f}, 平均 {x_lens_diff.mean():.10f}')
    print(f'   prompt_diff: 最大 {prompt_diff.max():.10f}, 平均 {prompt_diff.mean():.10f}')
    print(f'   style_diff: 最大 {style_diff.max():.10f}, 平均 {style_diff.mean():.10f}')
    print(f'   x_diff: 最大 {x_diff.max():.10f}, 平均 {x_diff.mean():.10f}')
    print(f'   t_diff: 最大 {t_diff.max():.10f}, 平均 {t_diff.mean():.10f}')
    
    # 直接调用PyTorch CFM
    print('\\n7. 直接调用PyTorch CFM...')
    try:
        pytorch_cfm = tts.s2mel.models.cfm
        
        # 使用CFM的inference方法
        with torch.no_grad():
            pytorch_output = pytorch_cfm.inference(
                mu=mu,
                x_lens=x_lens,
                prompt=prompt,
                style=style,
                f0=None,
                n_timesteps=25,
                temperature=1.0,
                inference_cfg_rate=0.7
            )
        
        print(f'   PyTorch CFM输出: {pytorch_output.shape}')
        print(f'   PyTorch CFM输出范围: [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]')
        print(f'   PyTorch CFM输出均值: {pytorch_output.mean():.6f}, 标准差: {pytorch_output.std():.6f}')
        
    except Exception as e:
        print(f'   ❌ PyTorch CFM调用失败: {e}')
        return False
    
    # 直接调用MLX CFM
    print('\\n8. 直接调用MLX CFM...')
    try:
        mlx_cfm = tts.mlx_s2mel_cfm
        if mlx_cfm is None:
            print("   MLX CFM未初始化，尝试手动创建...")
            from indextts.s2mel.modules.mlx_cfm import MLXCFM
            mlx_cfm = MLXCFM(tts.cfg.s2mel)
            print("   MLX CFM手动创建成功")
        
        # 使用MLX CFM的inference方法
        mlx_output = mlx_cfm.inference(
            mu=mu_mlx,
            x_lens=x_lens_mlx,
            prompt=prompt_mlx,
            style=style_mlx,
            f0=None,
            n_timesteps=25,
            temperature=1.0,
            inference_cfg_rate=0.7
        )
        
        print(f'   MLX CFM输出: {mlx_output.shape}')
        print(f'   MLX CFM输出范围: [{mlx_output.min():.6f}, {mlx_output.max():.6f}]')
        print(f'   MLX CFM输出均值: {mlx_output.mean():.6f}, 标准差: {mlx_output.std():.6f}')
        
    except Exception as e:
        print(f'   ❌ MLX CFM调用失败: {e}')
        return False
    
    # 比较输出差异
    print('\\n9. 比较输出差异...')
    pytorch_output_np = pytorch_output.cpu().numpy()
    mlx_output_np = mlx_output
    
    output_diff = np.abs(pytorch_output_np - mlx_output_np)
    max_diff = output_diff.max()
    mean_diff = output_diff.mean()
    
    print(f'   输出差异:')
    print(f'   最大差异: {max_diff:.6f}')
    print(f'   平均差异: {mean_diff:.6f}')
    print(f'   差异分布:')
    print(f'   - 差异 < 1e-5: {np.sum(output_diff < 1e-5)} / {output_diff.size} ({100*np.sum(output_diff < 1e-5)/output_diff.size:.1f}%)')
    print(f'   - 差异 < 1e-3: {np.sum(output_diff < 1e-3)} / {output_diff.size} ({100*np.sum(output_diff < 1e-3)/output_diff.size:.1f}%)')
    print(f'   - 差异 < 1e-1: {np.sum(output_diff < 1e-1)} / {output_diff.size} ({100*np.sum(output_diff < 1e-1)/output_diff.size:.1f}%)')
    print(f'   - 差异 < 1.0: {np.sum(output_diff < 1.0)} / {output_diff.size} ({100*np.sum(output_diff < 1.0)/output_diff.size:.1f}%)')
    
    # 分析大差异位置
    print('\\n10. 分析大差异位置...')
    large_diff_mask = output_diff > 1.0
    if np.any(large_diff_mask):
        large_diff_indices = np.where(large_diff_mask)
        print(f'   大差异位置数: {len(large_diff_indices[0])}')
        print(f'   前5个大差异位置:')
        for i in range(min(5, len(large_diff_indices[0]))):
            idx = tuple(large_diff_indices[j][i] for j in range(len(large_diff_indices)))
            pytorch_val = pytorch_output_np[idx]
            # 修复MLX数组索引问题
            mlx_val = float(mlx_output_np[idx])
            diff_val = output_diff[idx]
            print(f'     位置 {idx}: PyTorch={pytorch_val:.6f}, MLX={mlx_val:.6f}, 差异={diff_val:.6f}')
    else:
        print('   ✅ 没有大差异位置')
    
    # 保存分析结果
    print('\\n11. 保存分析结果...')
    analysis_result = {
        'cached_inputs': {
            'cat_condition': cat_condition.cpu().numpy(),
            'x_lens': x_lens.cpu().numpy(),
            'ref_mel': ref_mel.cpu().numpy(),
            'style': style.cpu().numpy(),
            'diffusion_steps': diffusion_steps,
            'inference_cfg_rate': inference_cfg_rate
        },
        'cfm_inputs': {
            'x': x.cpu().numpy(),
            'prompt_x': prompt_x.cpu().numpy(),
            't': t.cpu().numpy(),
            'style': style.cpu().numpy(),
            'cond': cond.cpu().numpy(),
            'x_lens': x_lens.cpu().numpy()
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
    
    print(f'   分析结果已保存到: cfm_cached_inputs_analysis.pkl')
    
    # 总结
    print('\\n12. 总结...')
    if max_diff < 1e-5:
        print('   ✅ CFM输出差异在e-5范围内，完全一致')
    elif max_diff < 1e-3:
        print('   ✅ CFM输出差异在e-3范围内，基本一致')
    elif max_diff < 1e-1:
        print('   ⚠️ CFM输出差异在e-1范围内，有轻微差异')
    else:
        print('   ❌ CFM输出差异较大，需要进一步分析')
    
    return True

if __name__ == "__main__":
    try:
        success = analyze_cfm_differences()
        if success:
            print("\\n✅ CFM缓存输入分析完成")
        else:
            print("\\n❌ CFM缓存输入分析失败")
    except Exception as e:
        print(f"\\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
