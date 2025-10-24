#!/usr/bin/env python3
"""
测试修复后的权重映射关系
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

def test_weight_mapping_fix():
    """测试修复后的权重映射关系"""
    
    print('=== 测试修复后的权重映射关系 ===')
    
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
                cfm_weights[key] = value
                print(f'  {key}: {value.shape}')
        
        print(f'\\n提取的CFM权重数量: {len(cfm_weights)}')
        
    else:
        print(f'❌ 缓存文件不存在: {cache_path}')
        return
    
    print('\\n=== 步骤3: 测试权重映射关系 ===')
    
    # 测试权重映射关系
    pytorch_keys = set(pytorch_state_dict.keys())
    mlx_keys = set(cfm_weights.keys())
    
    print(f'\\n--- 权重键对比 ---')
    print(f'PyTorch 权重键: {len(pytorch_keys)}')
    print(f'MLX 权重键: {len(mlx_keys)}')
    
    # 分析键的映射关系
    print(f'\\n--- 权重键映射分析 ---')
    
    # 检查PyTorch键格式
    pytorch_estimator_keys = [k for k in pytorch_keys if k.startswith('estimator.')]
    print(f'PyTorch estimator 键数量: {len(pytorch_estimator_keys)}')
    
    # 检查MLX键格式
    mlx_models_cfm_keys = [k for k in mlx_keys if k.startswith('models.cfm.estimator.')]
    print(f'MLX models.cfm.estimator 键数量: {len(mlx_models_cfm_keys)}')
    
    # 分析映射关系
    print(f'\\n--- 映射关系分析 ---')
    
    # 检查PyTorch -> MLX映射
    mapping_issues = []
    for pytorch_key in pytorch_estimator_keys:
        # 移除estimator.前缀
        rel_key = pytorch_key[len('estimator.'):]
        # 构建MLX键
        mlx_key = f'models.cfm.estimator.{rel_key}'
        
        if mlx_key in mlx_keys:
            print(f'✅ {pytorch_key} -> {mlx_key}')
        else:
            print(f'❌ {pytorch_key} -> {mlx_key} (缺失)')
            mapping_issues.append((pytorch_key, mlx_key))
    
    print(f'\\n--- 映射问题统计 ---')
    print(f'成功映射: {len(pytorch_estimator_keys) - len(mapping_issues)}')
    print(f'映射失败: {len(mapping_issues)}')
    
    if mapping_issues:
        print(f'\\n映射失败的键 (前10个):')
        for i, (pytorch_key, mlx_key) in enumerate(mapping_issues[:10]):
            print(f'  {i+1}. {pytorch_key} -> {mlx_key}')
        if len(mapping_issues) > 10:
            print(f'  ... 还有 {len(mapping_issues) - 10} 个')
    
    print(f'\\n=== 步骤4: 测试权重数值一致性 ===')
    
    # 测试权重数值一致性
    if len(mapping_issues) == 0:
        print(f'\\n--- 权重数值一致性测试 ---')
        
        # 选择几个权重进行数值对比
        test_keys = list(pytorch_estimator_keys)[:5]
        
        for pytorch_key in test_keys:
            rel_key = pytorch_key[len('estimator.'):]
            mlx_key = f'models.cfm.estimator.{rel_key}'
            
            pytorch_weight = pytorch_state_dict[pytorch_key]
            mlx_weight = cfm_weights[mlx_key]
            
            print(f'\\n测试权重: {pytorch_key}')
            print(f'  PyTorch: {pytorch_weight.shape}, 范围 [{pytorch_weight.min():.6f}, {pytorch_weight.max():.6f}]')
            print(f'  MLX: {mlx_weight.shape}, 范围 [{mlx_weight.min():.6f}, {mlx_weight.max():.6f}]')
            
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
                
                print(f'  差异: 最大 {max_diff:.6f}, 平均 {mean_diff:.6f}')
                
                if max_diff < 1e-5:
                    print(f'  ✅ 权重数值一致')
                else:
                    print(f'  ❌ 权重数值不一致')
            else:
                print(f'  ❌ 权重形状不匹配')
    
    print(f'\\n=== 步骤5: 测试CFM推理 ===')
    
    # 测试CFM推理
    print(f'\\n--- 测试CFM推理 ---')
    
    try:
        # 准备测试数据
        batch_size = 1
        seq_len = 100
        in_channels = 80
        
        # 创建测试输入
        z = torch.randn(batch_size, in_channels, seq_len, device=tts_mlx.device)
        x_lens = torch.tensor([seq_len], device=tts_mlx.device)
        prompt = torch.randn(batch_size, in_channels, seq_len, device=tts_mlx.device)
        mu = torch.randn(batch_size, in_channels, seq_len, device=tts_mlx.device)
        style = torch.randn(batch_size, 512, device=tts_mlx.device)
        f0 = torch.randn(batch_size, seq_len, device=tts_mlx.device)
        
        print(f'测试输入形状:')
        print(f'  z: {z.shape}')
        print(f'  x_lens: {x_lens.shape}')
        print(f'  prompt: {prompt.shape}')
        print(f'  mu: {mu.shape}')
        print(f'  style: {style.shape}')
        print(f'  f0: {f0.shape}')
        
        # 测试PyTorch CFM推理
        print(f'\\n--- 测试PyTorch CFM推理 ---')
        pytorch_output = pytorch_cfm.inference(
            z, x_lens, prompt, mu, style, f0, 
            unified_random=tts_mlx.unified_random
        )
        print(f'PyTorch CFM 输出形状: {pytorch_output.shape}')
        print(f'PyTorch CFM 输出范围: [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]')
        
        # 测试MLX CFM推理
        print(f'\\n--- 测试MLX CFM推理 ---')
        mlx_output = mlx_cfm.inference(
            z, x_lens, prompt, mu, style, f0,
            unified_random=tts_mlx.unified_random
        )
        print(f'MLX CFM 输出形状: {mlx_output.shape}')
        print(f'MLX CFM 输出范围: [{mlx_output.min():.6f}, {mlx_output.max():.6f}]')
        
        # 对比输出
        print(f'\\n--- 输出对比 ---')
        if pytorch_output.shape == mlx_output.shape:
            output_diff = torch.abs(pytorch_output - mlx_output)
            max_diff = torch.max(output_diff).item()
            mean_diff = torch.mean(output_diff).item()
            
            print(f'输出差异: 最大 {max_diff:.6f}, 平均 {mean_diff:.6f}')
            
            if max_diff < 1e-3:
                print(f'✅ CFM推理输出基本一致')
            else:
                print(f'❌ CFM推理输出差异较大')
        else:
            print(f'❌ CFM推理输出形状不匹配')
        
    except Exception as e:
        print(f'❌ CFM推理测试失败: {e}')
        import traceback
        traceback.print_exc()
    
    print(f'\\n=== 测试完成 ===')

if __name__ == "__main__":
    test_weight_mapping_fix()
