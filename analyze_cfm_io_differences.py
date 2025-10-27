#!/usr/bin/env python3
"""
分析PyTorch和MLX CFM推理的输入输出差异
使用现有的缓存数据进行分析
"""

import sys
import os
sys.path.append('.')

import torch
import numpy as np
import pickle
from indextts.infer_v2 import IndexTTS2
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def analyze_cfm_io_differences():
    """分析PyTorch和MLX CFM推理的输入输出差异"""
    
    print("=== 分析PyTorch和MLX CFM推理的输入输出差异 ===")
    
    tts = IndexTTS2()
    
    # 使用现有的缓存数据
    cache_file = 'cfm_inputs_mlx.pkl'
    if not os.path.exists(cache_file):
        print(f"❌ 缓存文件不存在: {cache_file}")
        return
    
    print(f"📁 使用缓存数据: {cache_file}")
    with open(cache_file, 'rb') as f:
        cached_data = pickle.load(f)
    
    # 提取CFM输入
    cat_condition = cached_data['cat_condition']
    x_lens = cached_data['x_lens']
    ref_mel = cached_data['ref_mel']
    style = cached_data['style']
    diffusion_steps = cached_data['diffusion_steps']
    inference_cfg_rate = cached_data['inference_cfg_rate']
    
    print(f"📊 缓存数据:")
    print(f"  cat_condition: {cat_condition.shape}, 范围: [{cat_condition.min():.6f}, {cat_condition.max():.6f}]")
    print(f"  x_lens: {x_lens.shape}, 值: {x_lens}")
    print(f"  ref_mel: {ref_mel.shape}, 范围: [{ref_mel.min():.6f}, {ref_mel.max():.6f}]")
    print(f"  style: {style.shape}, 范围: [{style.min():.6f}, {style.max():.6f}]")
    print(f"  diffusion_steps: {diffusion_steps}")
    print(f"  inference_cfg_rate: {inference_cfg_rate}")
    
    # 确保所有张量在正确的设备上
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    cat_condition = cat_condition.to(device)
    x_lens = x_lens.to(device)
    ref_mel = ref_mel.to(device)
    style = style.to(device)
    
    # 设置随机种子确保一致性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. 调用PyTorch CFM
    print(f"\n🔧 调用PyTorch CFM...")
    pytorch_cfm = tts.s2mel.models.cfm
    pytorch_output = pytorch_cfm.inference(
        mu=cat_condition,
        x_lens=x_lens,
        prompt=ref_mel,
        style=style,
        f0=None,
        n_timesteps=diffusion_steps,
        temperature=1.0,
        inference_cfg_rate=inference_cfg_rate,
        unified_random=tts.unified_random
    )
    
    print(f"✅ PyTorch CFM输出:")
    print(f"  形状: {pytorch_output.shape}")
    print(f"  范围: [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]")
    print(f"  均值: {pytorch_output.mean():.6f}")
    print(f"  标准差: {pytorch_output.std():.6f}")
    
    # 2. 初始化MLX CFM
    print(f"\n🔧 初始化MLX CFM...")
    if tts.mlx_s2mel_cfm is None:
        from indextts.s2mel.modules.mlx_cfm import MLXCFM
        tts.mlx_s2mel_cfm = MLXCFM(tts.cfg.s2mel)
        # 加载权重
        tts.mlx_s2mel_cfm.load_weights_from_pytorch(tts.s2mel.models.cfm.state_dict())
    
    # 3. 调用MLX CFM
    print(f"\n🔧 调用MLX CFM...")
    
    # 转换输入到MLX格式
    cat_condition_mlx = torch_to_mlx(cat_condition)
    x_lens_mlx = torch_to_mlx(x_lens)
    ref_mel_mlx = torch_to_mlx(ref_mel)
    style_mlx = torch_to_mlx(style)
    
    print(f"📊 MLX输入转换:")
    print(f"  cat_condition_mlx: {cat_condition_mlx.shape}, 范围: [{cat_condition_mlx.min():.6f}, {cat_condition_mlx.max():.6f}]")
    print(f"  x_lens_mlx: {x_lens_mlx.shape}, 值: {x_lens_mlx}")
    print(f"  ref_mel_mlx: {ref_mel_mlx.shape}, 范围: [{ref_mel_mlx.min():.6f}, {ref_mel_mlx.max():.6f}]")
    print(f"  style_mlx: {style_mlx.shape}, 范围: [{style_mlx.min():.6f}, {style_mlx.max():.6f}]")
    
    mlx_cfm = tts.mlx_s2mel_cfm
    mlx_output = mlx_cfm.inference(
        mu=cat_condition_mlx,
        x_lens=x_lens_mlx,
        prompt=ref_mel_mlx,
        style=style_mlx,
        f0=None,
        n_timesteps=diffusion_steps,
        temperature=1.0,
        inference_cfg_rate=inference_cfg_rate,
        unified_random=tts.unified_random
    )
    
    print(f"✅ MLX CFM输出:")
    print(f"  形状: {mlx_output.shape}")
    print(f"  范围: [{mlx_output.min():.6f}, {mlx_output.max():.6f}]")
    print(f"  均值: {mlx_output.mean():.6f}")
    print(f"  标准差: {mlx_output.std():.6f}")
    
    # 4. 比较输出差异
    print(f"\n📊 输出差异分析:")
    
    # 转换MLX输出到PyTorch格式进行比较
    mlx_output_torch = mlx_to_torch(mlx_output).to(device)
    
    # 计算差异
    diff = torch.abs(pytorch_output - mlx_output_torch)
    print(f"  最大差异: {diff.max():.6f}")
    print(f"  平均差异: {diff.mean():.6f}")
    print(f"  标准差差异: {diff.std():.6f}")
    
    # 计算信噪比
    signal_power = torch.mean(pytorch_output ** 2)
    noise_power = torch.mean(diff ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
    print(f"  信噪比: {snr:.2f} dB")
    
    # 分析差异分布
    print(f"\n📈 差异分布分析:")
    print(f"  差异 > 0.1: {(diff > 0.1).sum().item()} / {diff.numel()}")
    print(f"  差异 > 0.5: {(diff > 0.5).sum().item()} / {diff.numel()}")
    print(f"  差异 > 1.0: {(diff > 1.0).sum().item()} / {diff.numel()}")
    
    # 分析差异的空间分布
    print(f"\n🗺️ 差异空间分布:")
    # 按时间维度分析
    time_diff = diff.mean(dim=(0, 1))  # (seq_len,)
    print(f"  时间维度差异: 最大={time_diff.max():.6f}, 平均={time_diff.mean():.6f}")
    
    # 按频率维度分析
    freq_diff = diff.mean(dim=(0, 2))  # (80,)
    print(f"  频率维度差异: 最大={freq_diff.max():.6f}, 平均={freq_diff.mean():.6f}")
    
    # 保存结果用于进一步分析
    results = {
        'pytorch_output': pytorch_output.cpu().numpy(),
        'mlx_output': mlx_output_torch.cpu().numpy(),
        'diff': diff.cpu().numpy(),
        'snr': snr.item(),
        'max_diff': diff.max().item(),
        'mean_diff': diff.mean().item(),
        'std_diff': diff.std().item(),
        'time_diff': time_diff.cpu().numpy(),
        'freq_diff': freq_diff.cpu().numpy()
    }
    
    with open('cfm_io_differences.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"💾 结果已保存到 cfm_io_differences.pkl")
    
    # 分析差异是否影响音质
    if snr > 20:
        print(f"🎉 差异很小，音质应该很好")
    elif snr > 10:
        print(f"✅ 差异较小，音质应该可以接受")
    else:
        print(f"⚠️ 差异较大，可能影响音质")
    
    # 分析具体的差异模式
    print(f"\n🔍 差异模式分析:")
    
    # 检查是否有系统性的偏差
    mean_diff_per_sample = diff.mean(dim=(1, 2))  # (batch,)
    print(f"  每样本平均差异: {mean_diff_per_sample}")
    
    # 检查差异是否集中在某些频率
    high_freq_diff = freq_diff[-20:].mean()  # 高频部分
    low_freq_diff = freq_diff[:20].mean()   # 低频部分
    print(f"  低频差异: {low_freq_diff:.6f}")
    print(f"  高频差异: {high_freq_diff:.6f}")
    
    if high_freq_diff > low_freq_diff * 2:
        print(f"  ⚠️ 高频差异明显大于低频，可能导致音质问题")
    else:
        print(f"  ✅ 频率差异分布相对均匀")
    
    # 分析差异的时间模式
    print(f"\n⏰ 差异时间模式:")
    time_diff_std = time_diff.std()
    time_diff_max = time_diff.max()
    print(f"  时间差异标准差: {time_diff_std:.6f}")
    print(f"  时间差异最大值: {time_diff_max:.6f}")
    
    if time_diff_std > 0.1:
        print(f"  ⚠️ 时间维度差异变化较大，可能导致音质不稳定")
    else:
        print(f"  ✅ 时间维度差异相对稳定")

def main():
    """主函数"""
    
    print("开始分析PyTorch和MLX CFM推理的输入输出差异...")
    
    # 分析PyTorch和MLX CFM推理的输入输出差异
    analyze_cfm_io_differences()
    
    print(f"\n🎉 分析完成！")

if __name__ == "__main__":
    main()