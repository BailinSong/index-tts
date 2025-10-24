#!/usr/bin/env python3
"""
详细对比PyTorch和MLX CFM的权重差异
"""

import sys
sys.path.append('.')
import torch
import mlx.core as mx
import numpy as np
import os
import pickle
from indextts.infer_v2 import IndexTTS2
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def compare_cfm_weights():
    """详细对比PyTorch和MLX CFM的权重差异"""
    
    print('=== 详细对比PyTorch和MLX CFM的权重差异 ===')
    
    # 设置固定种子
    torch.manual_seed(42)
    np.random.seed(42)
    mx.random.seed(42)
    
    print(f'固定种子: 42')
    
    # 初始化 MLX 版本
    print('\\n1. 初始化 MLX 版本...')
    tts_mlx = IndexTTS2(
        cfg_path='checkpoints/config.yaml',
        model_dir='checkpoints',
        use_mlx=True,
        device='mps'
    )
    
    print('\\n=== 步骤1: 获取PyTorch CFM权重 ===')
    
    # 获取PyTorch CFM权重
    pytorch_cfm = tts_mlx.s2mel.models['cfm']
    pytorch_state_dict = pytorch_cfm.state_dict()
    
    print(f'PyTorch CFM 权重数量: {len(pytorch_state_dict)}')
    print(f'PyTorch CFM 类型: {type(pytorch_cfm)}')
    print(f'PyTorch CFM estimator 类型: {type(pytorch_cfm.estimator)}')
    
    print('\\n=== 步骤2: 获取MLX CFM权重 ===')
    
    # 获取MLX CFM权重
    mlx_cfm = tts_mlx.mlx_s2mel_cfm
    print(f'MLX CFM 类型: {type(mlx_cfm)}')
    print(f'MLX CFM estimator 类型: {type(mlx_cfm.estimator)}')
    
    # 尝试获取MLX权重
    try:
        # 从缓存文件加载MLX权重
        cache_path = 'checkpoints/mlx/s2mel.npz'
        if os.path.exists(cache_path):
            print(f'\\n--- 从缓存文件加载MLX权重 ---')
            mlx_state_dict = mx.load(cache_path)
            print(f'MLX 缓存权重数量: {len(mlx_state_dict)}')
            
            # 提取CFM相关权重
            cfm_weights = {}
            for key, value in mlx_state_dict.items():
                if key.startswith('models.cfm.'):
                    cfm_key = key.replace('models.cfm.', 'cfm.')
                    cfm_weights[cfm_key] = value
                    print(f'  {key} -> {cfm_key}: {value.shape}')
            
            print(f'\\n提取的CFM权重数量: {len(cfm_weights)}')
            
        else:
            print(f'❌ 缓存文件不存在: {cache_path}')
            return
            
    except Exception as e:
        print(f'❌ 获取MLX权重失败: {e}')
        return
    
    print('\\n=== 步骤3: 对比权重差异 ===')
    
    # 对比权重
    pytorch_keys = set(pytorch_state_dict.keys())
    mlx_keys = set(cfm_weights.keys())
    
    print(f'\\n--- 权重键对比 ---')
    print(f'PyTorch 权重键: {len(pytorch_keys)}')
    print(f'MLX 权重键: {len(mlx_keys)}')
    
    common_keys = pytorch_keys.intersection(mlx_keys)
    pytorch_only = pytorch_keys - mlx_keys
    mlx_only = mlx_keys - pytorch_keys
    
    print(f'共同键: {len(common_keys)}')
    print(f'仅PyTorch: {len(pytorch_only)}')
    print(f'仅MLX: {len(mlx_only)}')
    
    if pytorch_only:
        print(f'\\n仅PyTorch的键 (前20个):')
        for i, key in enumerate(list(pytorch_only)[:20]):
            print(f'  {i+1}. {key}')
        if len(pytorch_only) > 20:
            print(f'  ... 还有 {len(pytorch_only) - 20} 个')
    
    if mlx_only:
        print(f'\\n仅MLX的键 (前20个):')
        for i, key in enumerate(list(mlx_only)[:20]):
            print(f'  {i+1}. {key}')
        if len(mlx_only) > 20:
            print(f'  ... 还有 {len(mlx_only) - 20} 个')
    
    print(f'\\n=== 步骤4: 详细对比共同权重 ===')
    
    if common_keys:
        print(f'\\n--- 共同权重详细对比 (前20个) ---')
        
        weight_differences = []
        
        for i, key in enumerate(list(common_keys)[:20]):
            pytorch_weight = pytorch_state_dict[key]
            mlx_weight = cfm_weights[key]
            
            print(f'\\n{i+1}. {key}:')
            print(f'   PyTorch: {pytorch_weight.shape}, 范围 [{pytorch_weight.min():.6f}, {pytorch_weight.max():.6f}]')
            print(f'   MLX: {mlx_weight.shape}, 范围 [{mlx_weight.min():.6f}, {mlx_weight.max():.6f}]')
            
            # 转换MLX权重为PyTorch格式进行对比
            if isinstance(mlx_weight, mx.array):
                mlx_weight_torch = mlx_to_torch(mlx_weight).to(tts_mlx.device)
            else:
                mlx_weight_torch = mlx_weight
            
            # 确保形状一致
            if pytorch_weight.shape == mlx_weight_torch.shape:
                weight_diff = torch.abs(pytorch_weight - mlx_weight_torch)
                max_diff = torch.max(weight_diff).item()
                mean_diff = torch.mean(weight_diff).item()
                
                print(f'   差异: 最大 {max_diff:.6f}, 平均 {mean_diff:.6f}')
                
                if max_diff > 1e-5:
                    print(f'   ❌ 权重差异超过阈值')
                    weight_differences.append({
                        'key': key,
                        'max_diff': max_diff,
                        'mean_diff': mean_diff,
                        'pytorch_range': (pytorch_weight.min().item(), pytorch_weight.max().item()),
                        'mlx_range': (mlx_weight_torch.min().item(), mlx_weight_torch.max().item())
                    })
                else:
                    print(f'   ✅ 权重差异在阈值内')
                
                # 显示前几个大差异的位置
                if max_diff > 1e-3:
                    large_diff_indices = torch.where(weight_diff > 1e-3)
                    if len(large_diff_indices[0]) > 0:
                        print(f'   前5个大差异位置:')
                        for j in range(min(5, len(large_diff_indices[0]))):
                            idx = tuple(torch.tensor([large_diff_indices[k][j] for k in range(len(large_diff_indices))]))
                            pytorch_val = pytorch_weight[idx].item()
                            mlx_val = mlx_weight_torch[idx].item()
                            diff_val = weight_diff[idx].item()
                            print(f'     位置 {idx}: PyTorch={pytorch_val:.6f}, MLX={mlx_val:.6f}, 差异={diff_val:.6f}')
            else:
                print(f'   ❌ 权重形状不匹配 - PyTorch: {pytorch_weight.shape}, MLX: {mlx_weight_torch.shape}')
        
        # 统计差异
        print(f'\\n--- 权重差异统计 ---')
        print(f'总共同权重数: {len(common_keys)}')
        print(f'差异超过阈值的权重数: {len(weight_differences)}')
        print(f'差异比例: {len(weight_differences)/len(common_keys)*100:.2f}%')
        
        if weight_differences:
            print(f'\\n--- 差异最大的权重 (前10个) ---')
            weight_differences.sort(key=lambda x: x['max_diff'], reverse=True)
            for i, diff in enumerate(weight_differences[:10]):
                print(f'{i+1}. {diff["key"]}:')
                print(f'   最大差异: {diff["max_diff"]:.6f}')
                print(f'   平均差异: {diff["mean_diff"]:.6f}')
                print(f'   PyTorch范围: [{diff["pytorch_range"][0]:.6f}, {diff["pytorch_range"][1]:.6f}]')
                print(f'   MLX范围: [{diff["mlx_range"][0]:.6f}, {diff["mlx_range"][1]:.6f}]')
        
        # 保存详细对比结果
        comparison_results = {
            'pytorch_keys': list(pytorch_keys),
            'mlx_keys': list(mlx_keys),
            'common_keys': list(common_keys),
            'pytorch_only': list(pytorch_only),
            'mlx_only': list(mlx_only),
            'weight_differences': weight_differences
        }
        
        with open('cfm_weights_comparison.pkl', 'wb') as f:
            pickle.dump(comparison_results, f)
        print('\\n详细对比结果已保存到 cfm_weights_comparison.pkl')
    
    print(f'\\n=== 步骤5: 分析权重转换过程 ===')
    
    # 检查权重转换过程
    print(f'\\n--- 权重转换过程分析 ---')
    print(f'PyTorch权重来源: 直接从模型加载')
    print(f'MLX权重来源: 从缓存文件 {cache_path} 加载')
    print(f'缓存文件大小: {os.path.getsize(cache_path) / (1024*1024):.2f} MB')
    
    # 检查缓存文件的创建时间
    import time
    cache_time = os.path.getmtime(cache_path)
    cache_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cache_time))
    print(f'缓存文件创建时间: {cache_time_str}')
    
    print(f'\\n=== 步骤6: 检查权重数值精度 ===')
    
    # 检查权重数值精度
    print(f'\\n--- 权重数值精度分析 ---')
    
    # 检查PyTorch权重的数值范围
    pytorch_ranges = []
    for key, weight in pytorch_state_dict.items():
        if isinstance(weight, torch.Tensor) and weight.dtype.is_floating_point:
            pytorch_ranges.append((key, weight.min().item(), weight.max().item(), weight.mean().item()))
    
    # 检查MLX权重的数值范围
    mlx_ranges = []
    for key, weight in cfm_weights.items():
        if isinstance(weight, mx.array):
            mlx_ranges.append((key, weight.min(), weight.max(), weight.mean()))
    
    print(f'PyTorch权重数值范围统计:')
    pytorch_mins = [r[1] for r in pytorch_ranges]
    pytorch_maxs = [r[2] for r in pytorch_ranges]
    print(f'  最小值范围: [{min(pytorch_mins):.6f}, {max(pytorch_mins):.6f}]')
    print(f'  最大值范围: [{min(pytorch_maxs):.6f}, {max(pytorch_maxs):.6f}]')
    
    print(f'MLX权重数值范围统计:')
    mlx_mins = [r[1] for r in mlx_ranges]
    mlx_maxs = [r[2] for r in mlx_ranges]
    print(f'  最小值范围: [{min(mlx_mins):.6f}, {max(mlx_mins):.6f}]')
    print(f'  最大值范围: [{min(mlx_maxs):.6f}, {max(mlx_maxs):.6f}]')
    
    # 检查数值精度差异
    print(f'\\n--- 数值精度差异分析 ---')
    print(f'PyTorch权重数据类型: {pytorch_state_dict[list(pytorch_state_dict.keys())[0]].dtype}')
    print(f'MLX权重数据类型: {cfm_weights[list(cfm_weights.keys())[0]].dtype}')
    
    print(f'\\n=== 分析完成 ===')

if __name__ == "__main__":
    compare_cfm_weights()
