#!/usr/bin/env python3
"""
严格按照生产环境CFM调用方式分析差异
参考infer_v2.py中的CFM调用路径
"""

import sys
import os
sys.path.append('.')

import torch
import numpy as np
import pickle
from indextts.infer_v2 import IndexTTS2
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def analyze_production_cfm_differences():
    """严格按照生产环境CFM调用方式分析差异"""
    
    print("=== 严格按照生产环境CFM调用方式分析差异 ===")
    
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
    
    print(f"\n📋 输入数据:")
    print(f"  cat_condition: {cat_condition.shape}, min={cat_condition.min():.6f}, max={cat_condition.max():.6f}")
    print(f"  x_lens: {x_lens}")
    print(f"  ref_mel: {ref_mel.shape}, min={ref_mel.min():.6f}, max={ref_mel.max():.6f}")
    print(f"  style: {style.shape}, min={style.min():.6f}, max={style.max():.6f}")
    print(f"  diffusion_steps: {diffusion_steps}")
    print(f"  inference_cfg_rate: {inference_cfg_rate}")
    
    # 严格按照生产环境调用PyTorch CFM
    print(f"\n🔥 调用PyTorch CFM (生产环境方式)...")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 调用PyTorch CFM - 严格按照infer_v2.py中的调用方式
    pytorch_output = tts.s2mel.models['cfm'].inference(
        cat_condition,
        torch.LongTensor([cat_condition.size(1)]).to(cat_condition.device),  # x_lens
        ref_mel,
        style,
        None,  # f0
        diffusion_steps,
        inference_cfg_rate=inference_cfg_rate,
        unified_random=tts.unified_random
    )
    
    print(f"✅ PyTorch CFM输出: {pytorch_output.shape}, min={pytorch_output.min():.6f}, max={pytorch_output.max():.6f}")
    
    # 确保MLX CFM已初始化
    if tts.mlx_s2mel_cfm is None:
        print("🔧 初始化MLX CFM...")
        from indextts.s2mel.modules.mlx_cfm import MLXCFM
        tts.mlx_s2mel_cfm = MLXCFM(tts.cfg.s2mel)
        tts.mlx_s2mel_cfm.load_weights_from_pytorch(tts.s2mel.models.cfm.state_dict())
    
    # 严格按照生产环境调用MLX CFM
    print(f"\n🔥 调用MLX CFM (生产环境方式)...")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 转换为MLX格式
    cat_condition_mlx = torch_to_mlx(cat_condition)
    x_lens_mlx = torch_to_mlx(torch.LongTensor([cat_condition.size(1)]).to(cat_condition.device))
    ref_mel_mlx = torch_to_mlx(ref_mel)
    style_mlx = torch_to_mlx(style)
    
    # 调用MLX CFM - 严格按照infer_v2.py中的调用方式
    mlx_output = tts.mlx_s2mel_cfm.inference(
        cat_condition_mlx,
        x_lens_mlx,
        ref_mel_mlx,
        style_mlx,
        None,  # f0
        diffusion_steps,
        inference_cfg_rate=inference_cfg_rate,
        unified_random=tts.unified_random
    )
    
    # 转换为PyTorch格式进行比较
    mlx_output_torch = mlx_to_torch(mlx_output).to(device)
    
    print(f"✅ MLX CFM输出: {mlx_output_torch.shape}, min={mlx_output_torch.min():.6f}, max={mlx_output_torch.max():.6f}")
    
    # 计算差异
    print(f"\n🔍 输出差异分析:")
    
    diff = torch.abs(pytorch_output - mlx_output_torch)
    max_diff = diff.max()
    mean_diff = diff.mean()
    std_diff = diff.std()
    
    print(f"  最大差异: {max_diff:.6f}")
    print(f"  平均差异: {mean_diff:.6f}")
    print(f"  差异标准差: {std_diff:.6f}")
    
    # 计算信噪比
    signal_power = torch.mean(pytorch_output ** 2)
    noise_power = torch.mean((pytorch_output - mlx_output_torch) ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    
    print(f"  信噪比: {snr:.2f} dB")
    
    # 分析输出分布
    print(f"\n📊 输出分布分析:")
    print(f"  PyTorch: mean={pytorch_output.mean():.6f}, std={pytorch_output.std():.6f}")
    print(f"  MLX: mean={mlx_output_torch.mean():.6f}, std={mlx_output_torch.std():.6f}")
    
    # 分析输出范围
    print(f"\n📈 输出范围分析:")
    print(f"  PyTorch: [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]")
    print(f"  MLX: [{mlx_output_torch.min():.6f}, {mlx_output_torch.max():.6f}]")
    
    # 检查是否全零
    pytorch_all_zero = torch.allclose(pytorch_output, torch.zeros_like(pytorch_output), atol=1e-6)
    mlx_all_zero = torch.allclose(mlx_output_torch, torch.zeros_like(mlx_output_torch), atol=1e-6)
    
    print(f"\n🔍 零值检查:")
    print(f"  PyTorch输出全零: {pytorch_all_zero}")
    print(f"  MLX输出全零: {mlx_all_zero}")
    
    if pytorch_all_zero or mlx_all_zero:
        print(f"  ⚠️  发现全零输出，可能存在权重加载问题")
    
    # 保存分析结果
    analysis_result = {
        'pytorch_output': {
            'shape': pytorch_output.shape,
            'min': pytorch_output.min().item(),
            'max': pytorch_output.max().item(),
            'mean': pytorch_output.mean().item(),
            'std': pytorch_output.std().item(),
            'all_zero': pytorch_all_zero
        },
        'mlx_output': {
            'shape': mlx_output_torch.shape,
            'min': mlx_output_torch.min().item(),
            'max': mlx_output_torch.max().item(),
            'mean': mlx_output_torch.mean().item(),
            'std': mlx_output_torch.std().item(),
            'all_zero': mlx_all_zero
        },
        'differences': {
            'max_diff': max_diff.item(),
            'mean_diff': mean_diff.item(),
            'std_diff': std_diff.item(),
            'snr': snr.item()
        },
        'input_data': {
            'cat_condition_shape': cat_condition.shape,
            'x_lens': x_lens.item(),
            'ref_mel_shape': ref_mel.shape,
            'style_shape': style.shape,
            'diffusion_steps': diffusion_steps,
            'inference_cfg_rate': inference_cfg_rate
        }
    }
    
    with open('production_cfm_analysis.pkl', 'wb') as f:
        pickle.dump(analysis_result, f)
    
    print(f"\n💾 分析结果已保存到: production_cfm_analysis.pkl")
    
    # 总结
    print(f"\n📋 总结:")
    if snr > 30:
        print("  ✅ CFM输出差异很小，MLX实现良好")
    elif snr > 20:
        print("  ⚠️  CFM输出差异中等，建议进一步优化")
    else:
        print("  ❌ CFM输出差异较大，需要修复MLX实现")
    
    if pytorch_all_zero or mlx_all_zero:
        print("  ❌ 发现全零输出，需要检查权重加载")
    
    return analysis_result

def main():
    """主函数"""
    
    print("开始严格按照生产环境CFM调用方式分析差异...")
    
    # 分析生产环境CFM差异
    result = analyze_production_cfm_differences()
    
    if result:
        print(f"\n🎉 分析完成！")
        print(f"信噪比: {result['differences']['snr']:.2f} dB")
        print(f"最大差异: {result['differences']['max_diff']:.6f}")
        print(f"平均差异: {result['differences']['mean_diff']:.6f}")

if __name__ == "__main__":
    main()
