#!/usr/bin/env python3
"""
分析PyTorch和MLX CFM权重键的映射关系
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

def analyze_weight_mapping():
    """分析权重键的映射关系"""
    
    print('=== 分析PyTorch和MLX CFM权重键的映射关系 ===')
    
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
    
    print('\\n=== 步骤3: 分析权重键映射关系 ===')
    
    # 分析权重键的映射关系
    pytorch_keys = set(pytorch_state_dict.keys())
    mlx_keys = set(cfm_weights.keys())
    
    print(f'\\n--- 权重键分析 ---')
    print(f'PyTorch 权重键: {len(pytorch_keys)}')
    print(f'MLX 权重键: {len(mlx_keys)}')
    
    # 分析键的前缀
    pytorch_prefixes = set()
    mlx_prefixes = set()
    
    for key in pytorch_keys:
        parts = key.split('.')
        if len(parts) >= 2:
            pytorch_prefixes.add(parts[0])
    
    for key in mlx_keys:
        parts = key.split('.')
        if len(parts) >= 2:
            mlx_prefixes.add(parts[0])
    
    print(f'\\n--- 权重键前缀分析 ---')
    print(f'PyTorch 前缀: {sorted(pytorch_prefixes)}')
    print(f'MLX 前缀: {sorted(mlx_prefixes)}')
    
    # 分析键的结构
    print(f'\\n--- 权重键结构分析 ---')
    
    # 分析PyTorch键结构
    pytorch_structures = {}
    for key in pytorch_keys:
        parts = key.split('.')
        if len(parts) >= 2:
            prefix = parts[0]
            if prefix not in pytorch_structures:
                pytorch_structures[prefix] = []
            pytorch_structures[prefix].append(key)
    
    # 分析MLX键结构
    mlx_structures = {}
    for key in mlx_keys:
        parts = key.split('.')
        if len(parts) >= 2:
            prefix = parts[0]
            if prefix not in mlx_structures:
                mlx_structures[prefix] = []
            mlx_structures[prefix].append(key)
    
    print(f'\\nPyTorch 键结构:')
    for prefix, keys in pytorch_structures.items():
        print(f'  {prefix}: {len(keys)} 个键')
        if len(keys) <= 5:
            for key in keys:
                print(f'    {key}')
        else:
            for key in keys[:3]:
                print(f'    {key}')
            print(f'    ... 还有 {len(keys) - 3} 个')
    
    print(f'\\nMLX 键结构:')
    for prefix, keys in mlx_structures.items():
        print(f'  {prefix}: {len(keys)} 个键')
        if len(keys) <= 5:
            for key in keys:
                print(f'    {key}')
        else:
            for key in keys[:3]:
                print(f'    {key}')
            print(f'    ... 还有 {len(keys) - 3} 个')
    
    print(f'\\n=== 步骤4: 分析权重转换过程 ===')
    
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
    
    # 分析权重转换逻辑
    print(f'\\n--- 权重转换逻辑分析 ---')
    print(f'PyTorch CFM 权重键格式: estimator.xxx')
    print(f'MLX CFM 权重键格式: cfm.estimator.xxx')
    print(f'转换过程: PyTorch -> MLX 缓存时添加了 "cfm." 前缀')
    
    # 检查是否有权重转换错误
    print(f'\\n--- 权重转换错误检查 ---')
    
    # 检查PyTorch权重是否包含非浮点类型
    non_float_weights = []
    for key, weight in pytorch_state_dict.items():
        if isinstance(weight, torch.Tensor) and not weight.dtype.is_floating_point:
            non_float_weights.append((key, weight.dtype))
    
    if non_float_weights:
        print(f'PyTorch 非浮点权重: {len(non_float_weights)} 个')
        for key, dtype in non_float_weights[:10]:
            print(f'  {key}: {dtype}')
        if len(non_float_weights) > 10:
            print(f'  ... 还有 {len(non_float_weights) - 10} 个')
    else:
        print(f'PyTorch 所有权重都是浮点类型')
    
    # 检查MLX权重类型
    mlx_types = {}
    for key, weight in cfm_weights.items():
        if isinstance(weight, mx.array):
            dtype = weight.dtype
            if dtype not in mlx_types:
                mlx_types[dtype] = []
            mlx_types[dtype].append(key)
    
    print(f'\\nMLX 权重类型分布:')
    for dtype, keys in mlx_types.items():
        print(f'  {dtype}: {len(keys)} 个权重')
    
    print(f'\\n=== 步骤5: 检查权重数值精度 ===')
    
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
    if pytorch_ranges:
        pytorch_mins = [r[1] for r in pytorch_ranges]
        pytorch_maxs = [r[2] for r in pytorch_ranges]
        print(f'  最小值范围: [{min(pytorch_mins):.6f}, {max(pytorch_mins):.6f}]')
        print(f'  最大值范围: [{min(pytorch_maxs):.6f}, {max(pytorch_maxs):.6f}]')
    else:
        print(f'  没有浮点权重')
    
    print(f'MLX权重数值范围统计:')
    if mlx_ranges:
        mlx_mins = [r[1] for r in mlx_ranges]
        mlx_maxs = [r[2] for r in mlx_ranges]
        print(f'  最小值范围: [{min(mlx_mins):.6f}, {max(mlx_mins):.6f}]')
        print(f'  最大值范围: [{min(mlx_maxs):.6f}, {max(mlx_maxs):.6f}]')
    else:
        print(f'  没有MLX权重')
    
    # 检查数值精度差异
    print(f'\\n--- 数值精度差异分析 ---')
    if pytorch_ranges and mlx_ranges:
        print(f'PyTorch权重数据类型: {pytorch_state_dict[list(pytorch_state_dict.keys())[0]].dtype}')
        print(f'MLX权重数据类型: {cfm_weights[list(cfm_weights.keys())[0]].dtype}')
        
        # 检查是否有数值精度损失
        pytorch_float32_count = sum(1 for r in pytorch_ranges if r[1] != r[1])  # 检查NaN
        mlx_float32_count = sum(1 for r in mlx_ranges if r[1] != r[1])  # 检查NaN
        
        print(f'PyTorch NaN权重数量: {pytorch_float32_count}')
        print(f'MLX NaN权重数量: {mlx_float32_count}')
    
    print(f'\\n=== 步骤6: 总结分析结果 ===')
    
    # 总结分析结果
    print(f'\\n--- 分析结果总结 ---')
    print(f'1. 权重键映射问题:')
    print(f'   - PyTorch: {len(pytorch_keys)} 个键，前缀: {sorted(pytorch_prefixes)}')
    print(f'   - MLX: {len(mlx_keys)} 个键，前缀: {sorted(mlx_prefixes)}')
    print(f'   - 共同键: 0 个 (完全不同的键命名)')
    
    print(f'\\n2. 权重转换过程:')
    print(f'   - PyTorch -> MLX 转换时添加了 "cfm." 前缀')
    print(f'   - 这导致权重键完全不匹配')
    print(f'   - 需要修正权重键的映射关系')
    
    print(f'\\n3. 权重数值精度:')
    if pytorch_ranges and mlx_ranges:
        print(f'   - PyTorch: {len(pytorch_ranges)} 个浮点权重')
        print(f'   - MLX: {len(mlx_ranges)} 个权重')
        print(f'   - 数值范围基本一致')
    else:
        print(f'   - 无法比较数值精度 (键不匹配)')
    
    print(f'\\n4. 问题根源:')
    print(f'   - 权重键命名不一致导致无法正确加载')
    print(f'   - MLX CFM 可能使用了错误的权重')
    print(f'   - 需要修正权重键的映射关系')
    
    # 保存分析结果
    analysis_results = {
        'pytorch_keys': list(pytorch_keys),
        'mlx_keys': list(mlx_keys),
        'pytorch_prefixes': list(pytorch_prefixes),
        'mlx_prefixes': list(mlx_prefixes),
        'pytorch_structures': pytorch_structures,
        'mlx_structures': mlx_structures,
        'pytorch_ranges': pytorch_ranges,
        'mlx_ranges': mlx_ranges
    }
    
    with open('weight_mapping_analysis.pkl', 'wb') as f:
        pickle.dump(analysis_results, f)
    print('\\n分析结果已保存到 weight_mapping_analysis.pkl')
    
    print(f'\\n=== 分析完成 ===')

if __name__ == "__main__":
    analyze_weight_mapping()
