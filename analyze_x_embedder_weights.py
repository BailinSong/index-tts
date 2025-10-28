#!/usr/bin/env python3
"""
详细分析 x_embedder 权重差异
检查 PyTorch 和 MLX 版本的 x_embedder 权重加载过程
"""

import torch
import mlx.core as mx
import numpy as np
from indextts.infer_v2 import IndexTTS2

def analyze_x_embedder_weights():
    """详细分析 x_embedder 权重差异"""
    print("🔍 详细分析 x_embedder 权重差异")
    print("="*60)
    
    try:
        # 加载 PyTorch DiT 模型
        print("📊 加载 PyTorch DiT 模型...")
        pytorch_tts = IndexTTS2(use_mlx=False)
        pytorch_dit = pytorch_tts.s2mel.models['cfm'].estimator
        
        # 加载 MLX DiT 模型
        print("📊 加载 MLX DiT 模型...")
        mlx_tts = IndexTTS2(use_mlx=True)
        mlx_dit = mlx_tts.mlx_s2mel_cfm.estimator
        
        print("✅ 成功加载两个模型")
        
        # 获取权重
        pytorch_weight = pytorch_dit.x_embedder.weight
        mlx_weight = mlx_dit.x_embedder.weight
        
        print(f"\n📊 权重基本信息:")
        print(f"   PyTorch shape: {pytorch_weight.shape}")
        print(f"   MLX shape:     {mlx_weight.shape}")
        
        # 转换为 numpy 进行比较
        pytorch_np = pytorch_weight.detach().cpu().numpy()
        mlx_np = np.array(mlx_weight)
        
        print(f"\n📊 权重统计信息:")
        print(f"   PyTorch: min={np.min(pytorch_np):.6f}, max={np.max(pytorch_np):.6f}, avg={np.mean(pytorch_np):.6f}")
        print(f"   MLX:     min={np.min(mlx_np):.6f}, max={np.max(mlx_np):.6f}, avg={np.mean(mlx_np):.6f}")
        
        # 计算差异
        diff = np.abs(pytorch_np - mlx_np)
        max_diff = np.max(diff)
        avg_diff = np.mean(diff)
        std_diff = np.std(diff)
        
        print(f"\n📊 差异分析:")
        print(f"   最大差异: {max_diff:.6f}")
        print(f"   平均差异: {avg_diff:.6f}")
        print(f"   标准差:   {std_diff:.6f}")
        
        # 找出差异最大的位置
        max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"   最大差异位置: {max_diff_idx}")
        print(f"   PyTorch值: {pytorch_np[max_diff_idx]:.6f}")
        print(f"   MLX值:     {mlx_np[max_diff_idx]:.6f}")
        
        # 检查是否有 weight_norm
        print(f"\n📊 Weight Norm 检查:")
        print(f"   PyTorch x_embedder 类型: {type(pytorch_dit.x_embedder)}")
        print(f"   MLX x_embedder 类型:     {type(mlx_dit.x_embedder)}")
        
        # 检查 PyTorch 是否有 weight_g 和 weight_v
        if hasattr(pytorch_dit.x_embedder, 'weight_g'):
            print(f"   PyTorch 有 weight_g: {pytorch_dit.x_embedder.weight_g.shape}")
        if hasattr(pytorch_dit.x_embedder, 'weight_v'):
            print(f"   PyTorch 有 weight_v: {pytorch_dit.x_embedder.weight_v.shape}")
        
        # 检查 MLX 是否有 weight_g 和 weight_v
        if hasattr(mlx_dit.x_embedder, 'weight_g'):
            print(f"   MLX 有 weight_g: {mlx_dit.x_embedder.weight_g.shape}")
        if hasattr(mlx_dit.x_embedder, 'weight_v'):
            print(f"   MLX 有 weight_v: {mlx_dit.x_embedder.weight_v.shape}")
        
        # 分析差异分布
        print(f"\n📊 差异分布:")
        diff_hist, diff_bins = np.histogram(diff, bins=10)
        for i in range(len(diff_hist)):
            print(f"   [{diff_bins[i]:.6f}, {diff_bins[i+1]:.6f}]: {diff_hist[i]} 个值")
        
        # 检查是否在数值精度范围内
        print(f"\n📊 数值精度分析:")
        pytorch_precision = np.finfo(pytorch_np.dtype).eps
        mlx_precision = np.finfo(mlx_np.dtype).eps
        print(f"   PyTorch 数值精度: {pytorch_precision:.2e}")
        print(f"   MLX 数值精度:     {mlx_precision:.2e}")
        
        if max_diff < max(pytorch_precision, mlx_precision) * 100:
            print(f"   ✅ 差异在数值精度范围内")
        else:
            print(f"   ❌ 差异超出数值精度范围")
        
        # 检查权重是否完全相同
        if np.array_equal(pytorch_np, mlx_np):
            print(f"   ✅ 权重完全相同")
        else:
            print(f"   ❌ 权重不完全相同")
            
            # 找出不同的位置
            different_mask = diff > 1e-10
            different_count = np.sum(different_mask)
            total_count = pytorch_np.size
            
            print(f"   不同元素数量: {different_count}/{total_count} ({different_count/total_count*100:.2f}%)")
            
            if different_count < total_count * 0.01:  # 少于1%的元素不同
                print(f"   💡 差异很小，可能是数值精度问题")
            else:
                print(f"   ⚠️  差异较大，需要检查权重加载过程")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

def check_weight_loading_process():
    """检查权重加载过程"""
    print(f"\n{'='*60}")
    print(f"🔍 检查权重加载过程")
    print(f"{'='*60}")
    
    try:
        # 检查 PyTorch 原始权重文件
        import torch
        pytorch_weights = torch.load("checkpoints/s2mel.pth", map_location='cpu')
        
        print("📊 PyTorch 原始权重文件中的 x_embedder 相关键:")
        x_embedder_keys = [k for k in pytorch_weights.keys() if 'x_embedder' in k]
        for key in x_embedder_keys:
            weight = pytorch_weights[key]
            if isinstance(weight, torch.Tensor):
                print(f"   {key}: shape={weight.shape}, dtype={weight.dtype}")
                print(f"      min={weight.min():.6f}, max={weight.max():.6f}, avg={weight.mean():.6f}")
            else:
                print(f"   {key}: {type(weight)}")
        
        # 检查 MLX 缓存文件
        import numpy as np
        mlx_weights = np.load("checkpoints/mlx/s2mel.npz")
        
        print("\n📊 MLX 缓存文件中的 x_embedder 相关键:")
        x_embedder_keys = [k for k in mlx_weights.keys() if 'x_embedder' in k]
        for key in x_embedder_keys:
            weight = mlx_weights[key]
            print(f"   {key}: shape={weight.shape}, dtype={weight.dtype}")
            print(f"      min={weight.min():.6f}, max={weight.max():.6f}, avg={weight.mean():.6f}")
        
    except Exception as e:
        print(f"❌ 检查权重加载过程失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    analyze_x_embedder_weights()
    check_weight_loading_process()

if __name__ == "__main__":
    main()
