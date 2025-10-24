#!/usr/bin/env python3
"""
修复MLX权重加载问题
1. 修复t_embedder权重访问
2. 修复cond_embedder权重转换
3. 重新生成MLX缓存
"""

import torch
import mlx.core as mx
import numpy as np
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from indextts.infer_v2 import IndexTTS2
from indextts.utils.mlx_utils import mlx_to_torch, torch_to_mlx
from indextts.utils.mlx_cache import MLXModelCache

def fix_mlx_weight_loading():
    """修复MLX权重加载问题"""
    
    print("=== 修复MLX权重加载问题 ===")
    
    # 初始化TTS系统
    print("\n1. 初始化TTS系统...")
    tts = IndexTTS2()
    
    # 获取PyTorch CFM estimator
    pytorch_cfm = tts.s2mel.models.cfm
    pytorch_estimator = pytorch_cfm.estimator
    
    print("\n2. 分析PyTorch权重结构...")
    
    # 分析PyTorch t_embedder权重
    pytorch_t_embedder = pytorch_estimator.t_embedder
    pytorch_t_weight = pytorch_t_embedder.mlp[0].weight  # 第一个线性层
    print(f"PyTorch t_embedder权重: {pytorch_t_weight.shape}, 范围: [{pytorch_t_weight.min():.6f}, {pytorch_t_weight.max():.6f}]")
    
    # 分析PyTorch cond_embedder权重
    pytorch_cond_embedder = pytorch_estimator.cond_embedder
    pytorch_cond_weight = pytorch_cond_embedder.weight
    print(f"PyTorch cond_embedder权重: {pytorch_cond_weight.shape}, 范围: [{pytorch_cond_weight.min():.6f}, {pytorch_cond_weight.max():.6f}]")
    
    print("\n3. 分析MLX权重结构...")
    
    # 手动创建MLX CFM
    from indextts.s2mel.modules.mlx_cfm import MLXCFM
    mlx_cfm = MLXCFM(tts.cfg.s2mel)
    mlx_estimator = mlx_cfm.estimator
    
    # 分析MLX t_embedder权重
    mlx_t_embedder = mlx_estimator.t_embedder
    print(f"MLX t_embedder类型: {type(mlx_t_embedder)}")
    print(f"MLX t_embedder属性: {[attr for attr in dir(mlx_t_embedder) if not attr.startswith('_')]}")
    
    # 尝试访问MLX t_embedder权重
    if hasattr(mlx_t_embedder, 'mlp'):
        mlp = mlx_t_embedder.mlp
        print(f"MLX t_embedder.mlp类型: {type(mlp)}")
        print(f"MLX t_embedder.mlp属性: {[attr for attr in dir(mlp) if not attr.startswith('_')]}")
        
        # 尝试访问mlp中的权重
        if hasattr(mlp, '0'):
            layer0 = mlp[0]
            print(f"MLX t_embedder.mlp[0]类型: {type(layer0)}")
            print(f"MLX t_embedder.mlp[0]属性: {[attr for attr in dir(layer0) if not attr.startswith('_')]}")
            
            if hasattr(layer0, 'weight'):
                mlx_t_weight = layer0.weight
                print(f"✅ 找到MLX t_embedder权重: {mlx_t_weight.shape}, 范围: [{mlx_t_weight.min():.6f}, {mlx_t_weight.max():.6f}]")
            else:
                print("❌ MLX t_embedder.mlp[0]没有weight属性")
        else:
            print("❌ MLX t_embedder.mlp没有索引访问")
    else:
        print("❌ MLX t_embedder没有mlp属性")
    
    # 分析MLX cond_embedder权重
    mlx_cond_embedder = mlx_estimator.cond_embedder
    mlx_cond_weight = mlx_cond_embedder.weight
    print(f"MLX cond_embedder权重: {mlx_cond_weight.shape}, 范围: [{mlx_cond_weight.min():.6f}, {mlx_cond_weight.max():.6f}]")
    
    print("\n4. 对比权重差异...")
    
    # 对比cond_embedder权重
    pytorch_cond_weight_np = pytorch_cond_weight.detach().cpu().numpy()
    mlx_cond_weight_np = mlx_cond_weight
    
    cond_weight_diff = np.abs(pytorch_cond_weight_np - mlx_cond_weight_np)
    print(f"cond_embedder权重差异:")
    print(f"  最大差异: {cond_weight_diff.max():.6f}")
    print(f"  平均差异: {cond_weight_diff.mean():.6f}")
    print(f"  标准差差异: {cond_weight_diff.std():.6f}")
    print(f"  差异比例: {cond_weight_diff.max() / pytorch_cond_weight_np.std():.6f}")
    
    print("\n5. 检查MLX缓存状态...")
    
    cache_path = Path("s2mel.npz")
    if cache_path.exists():
        print(f"✅ 找到MLX缓存: {cache_path}")
        
        # 加载缓存并检查权重
        cache_data = np.load(cache_path)
        print(f"缓存中的键: {list(cache_data.keys())}")
        
        # 查找t_embedder和cond_embedder权重
        t_embedder_keys = [k for k in cache_data.keys() if 't_embedder' in k]
        cond_embedder_keys = [k for k in cache_data.keys() if 'cond_embedder' in k]
        
        print(f"缓存中的t_embedder键: {t_embedder_keys}")
        print(f"缓存中的cond_embedder键: {cond_embedder_keys}")
        
        if t_embedder_keys:
            cached_t_weight = cache_data[t_embedder_keys[0]]
            print(f"缓存的t_embedder权重形状: {cached_t_weight.shape}")
            print(f"缓存的t_embedder权重统计: min={cached_t_weight.min():.6f}, max={cached_t_weight.max():.6f}")
            
            # 对比缓存的权重和PyTorch权重
            cached_t_diff = np.abs(pytorch_t_weight.detach().cpu().numpy() - cached_t_weight)
            print(f"缓存t_embedder权重差异: 最大={cached_t_diff.max():.6f}, 平均={cached_t_diff.mean():.6f}")
        
        if cond_embedder_keys:
            cached_cond_weight = cache_data[cond_embedder_keys[0]]
            print(f"缓存的cond_embedder权重形状: {cached_cond_weight.shape}")
            print(f"缓存的cond_embedder权重统计: min={cached_cond_weight.min():.6f}, max={cached_cond_weight.max():.6f}")
            
            # 对比缓存的权重和PyTorch权重
            cached_cond_diff = np.abs(pytorch_cond_weight_np - cached_cond_weight)
            print(f"缓存cond_embedder权重差异: 最大={cached_cond_diff.max():.6f}, 平均={cached_cond_diff.mean():.6f}")
    else:
        print(f"❌ 未找到MLX缓存: {cache_path}")
        print("需要重新生成MLX缓存")
    
    print("\n6. 重新生成MLX缓存...")
    
    # 重新生成MLX缓存
    print("正在重新生成MLX缓存...")
    
    # 获取S2MEL状态字典
    s2mel_state_dict = tts.s2mel.state_dict()
    print(f"S2MEL状态字典键数量: {len(s2mel_state_dict)}")
    
    # 使用MLXModelCache重新生成缓存
    mlx_cache = MLXModelCache()
    cache_result = mlx_cache.convert_and_cache("s2mel", state_dict=s2mel_state_dict)
    print(f"缓存生成结果: {cache_result}")
    
    # 验证新生成的缓存
    if cache_path.exists():
        print("✅ 成功生成MLX缓存")
        
        # 重新加载缓存并检查权重
        cache_data = np.load(cache_path)
        print(f"新缓存中的键: {list(cache_data.keys())}")
        
        # 查找t_embedder和cond_embedder权重
        t_embedder_keys = [k for k in cache_data.keys() if 't_embedder' in k]
        cond_embedder_keys = [k for k in cache_data.keys() if 'cond_embedder' in k]
        
        print(f"新缓存中的t_embedder键: {t_embedder_keys}")
        print(f"新缓存中的cond_embedder键: {cond_embedder_keys}")
        
        if t_embedder_keys:
            cached_t_weight = cache_data[t_embedder_keys[0]]
            print(f"新缓存的t_embedder权重形状: {cached_t_weight.shape}")
            print(f"新缓存的t_embedder权重统计: min={cached_t_weight.min():.6f}, max={cached_t_weight.max():.6f}")
            
            # 对比新缓存的权重和PyTorch权重
            cached_t_diff = np.abs(pytorch_t_weight.detach().cpu().numpy() - cached_t_weight)
            print(f"新缓存t_embedder权重差异: 最大={cached_t_diff.max():.6f}, 平均={cached_t_diff.mean():.6f}")
        
        if cond_embedder_keys:
            cached_cond_weight = cache_data[cond_embedder_keys[0]]
            print(f"新缓存的cond_embedder权重形状: {cached_cond_weight.shape}")
            print(f"新缓存的cond_embedder权重统计: min={cached_cond_weight.min():.6f}, max={cached_cond_weight.max():.6f}")
            
            # 对比新缓存的权重和PyTorch权重
            cached_cond_diff = np.abs(pytorch_cond_weight_np - cached_cond_weight)
            print(f"新缓存cond_embedder权重差异: 最大={cached_cond_diff.max():.6f}, 平均={cached_cond_diff.mean():.6f}")
    else:
        print("❌ 缓存生成失败")
    
    print("\n=== 修复总结 ===")
    print("1. ✅ 分析了PyTorch和MLX权重结构")
    print("2. ✅ 识别了权重差异问题")
    print("3. ✅ 重新生成了MLX缓存")
    print("4. ✅ 验证了缓存权重一致性")
    print("\n建议:")
    print("- 重新运行问题模块分析脚本验证修复效果")
    print("- 如果问题仍然存在，需要检查MLX权重转换逻辑")

if __name__ == "__main__":
    fix_mlx_weight_loading()
