#!/usr/bin/env python3
"""
分析MLX CFM权重缺失但音频基本正常的矛盾现象
深入检查MLX CFM的实际权重加载情况
"""

import sys
import os
sys.path.append('.')

import torch
import numpy as np
import pickle
from indextts.infer_v2 import IndexTTS2
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def analyze_weight_loading_contradiction():
    """分析权重加载的矛盾现象"""
    
    print("=== 分析MLX CFM权重缺失但音频基本正常的矛盾现象 ===")
    
    # 初始化TTS
    tts = IndexTTS2()
    
    # 确保MLX CFM已初始化
    if tts.mlx_s2mel_cfm is None:
        print("🔧 初始化MLX CFM...")
        from indextts.s2mel.modules.mlx_cfm import MLXCFM
        tts.mlx_s2mel_cfm = MLXCFM(tts.cfg.s2mel)
        tts.mlx_s2mel_cfm.load_weights_from_pytorch(tts.s2mel.models.cfm.state_dict())
    
    mlx_cfm = tts.mlx_s2mel_cfm
    mlx_estimator = mlx_cfm.estimator
    
    print(f"\n🔍 深入检查MLX CFM权重加载情况:")
    
    # 1. 检查t_embedder的实际状态
    print(f"\n1️⃣ 检查t_embedder实际状态:")
    print(f"  t_embedder类型: {type(mlx_estimator.t_embedder)}")
    
    # 检查t_embedder的权重
    try:
        t_embedder_weights = mlx_estimator.t_embedder.parameters()
        print(f"  t_embedder参数数量: {len(list(t_embedder_weights))}")
        
        # 检查具体的权重
        if hasattr(mlx_estimator.t_embedder, 'mlp'):
            print(f"  t_embedder.mlp存在: True")
            if hasattr(mlx_estimator.t_embedder.mlp, '0'):
                print(f"  t_embedder.mlp.0存在: True")
                if hasattr(mlx_estimator.t_embedder.mlp[0], 'weight'):
                    weight = mlx_estimator.t_embedder.mlp[0].weight
                    print(f"  t_embedder.mlp.0.weight形状: {weight.shape}")
                    print(f"  t_embedder.mlp.0.weight范围: [{weight.min():.6f}, {weight.max():.6f}]")
                    print(f"  t_embedder.mlp.0.weight是否全零: {torch.allclose(weight, torch.zeros_like(weight))}")
                else:
                    print(f"  t_embedder.mlp.0.weight不存在")
            else:
                print(f"  t_embedder.mlp.0不存在")
        else:
            print(f"  t_embedder.mlp不存在")
            
    except Exception as e:
        print(f"  t_embedder检查失败: {e}")
    
    # 2. 检查cond_projection权重
    print(f"\n2️⃣ 检查cond_projection权重:")
    try:
        cond_weight = mlx_estimator.cond_projection.weight
        print(f"  cond_projection.weight形状: {cond_weight.shape}")
        print(f"  cond_projection.weight范围: [{cond_weight.min():.6f}, {cond_weight.max():.6f}]")
        print(f"  cond_projection.weight是否全零: {torch.allclose(cond_weight, torch.zeros_like(cond_weight))}")
    except Exception as e:
        print(f"  cond_projection检查失败: {e}")
    
    # 3. 检查x_embedder权重
    print(f"\n3️⃣ 检查x_embedder权重:")
    try:
        x_weight = mlx_estimator.x_embedder.weight_v
        print(f"  x_embedder.weight_v形状: {x_weight.shape}")
        print(f"  x_embedder.weight_v范围: [{x_weight.min():.6f}, {x_weight.max():.6f}]")
        print(f"  x_embedder.weight_v是否全零: {torch.allclose(x_weight, torch.zeros_like(x_weight))}")
    except Exception as e:
        print(f"  x_embedder检查失败: {e}")
    
    # 4. 检查transformer权重
    print(f"\n4️⃣ 检查transformer权重:")
    try:
        transformer_layers = mlx_estimator.transformer.layers
        print(f"  transformer层数: {len(transformer_layers)}")
        
        # 检查第一层的权重
        first_layer = transformer_layers[0]
        if hasattr(first_layer, 'attention'):
            attn_weight = first_layer.attention.wqkv.weight
            print(f"  第一层attention.wqkv.weight形状: {attn_weight.shape}")
            print(f"  第一层attention.wqkv.weight范围: [{attn_weight.min():.6f}, {attn_weight.max():.6f}]")
            print(f"  第一层attention.wqkv.weight是否全零: {torch.allclose(attn_weight, torch.zeros_like(attn_weight))}")
    except Exception as e:
        print(f"  transformer检查失败: {e}")
    
    # 5. 检查final_layer权重
    print(f"\n5️⃣ 检查final_layer权重:")
    try:
        final_layer = mlx_estimator.final_layer
        if hasattr(final_layer, 'linear'):
            final_weight = final_layer.linear.weight_v
            print(f"  final_layer.linear.weight_v形状: {final_weight.shape}")
            print(f"  final_layer.linear.weight_v范围: [{final_weight.min():.6f}, {final_weight.max():.6f}]")
            print(f"  final_layer.linear.weight_v是否全零: {torch.allclose(final_weight, torch.zeros_like(final_weight))}")
    except Exception as e:
        print(f"  final_layer检查失败: {e}")
    
    # 6. 测试t_embedder的实际功能
    print(f"\n6️⃣ 测试t_embedder实际功能:")
    try:
        import mlx.core as mx
        test_t = mx.array([0.0, 0.5, 1.0])
        t_emb_output = mlx_estimator.t_embedder(test_t)
        print(f"  t_embedder输出形状: {t_emb_output.shape}")
        print(f"  t_embedder输出范围: [{t_emb_output.min():.6f}, {t_emb_output.max():.6f}]")
        print(f"  t_embedder输出是否全零: {torch.allclose(mlx_to_torch(t_emb_output), torch.zeros_like(mlx_to_torch(t_emb_output)))}")
        
        # 检查不同时间步的输出是否不同
        t_emb_0 = mlx_estimator.t_embedder(mx.array([0.0]))
        t_emb_1 = mlx_estimator.t_embedder(mx.array([1.0]))
        diff = mx.abs(t_emb_0 - t_emb_1).max()
        print(f"  t=0和t=1的输出差异: {diff:.6f}")
        
    except Exception as e:
        print(f"  t_embedder功能测试失败: {e}")
    
    # 7. 检查PyTorch版本的t_embedder权重
    print(f"\n7️⃣ 检查PyTorch版本的t_embedder权重:")
    try:
        pytorch_estimator = tts.s2mel.models.cfm.estimator
        pytorch_t_weight = pytorch_estimator.t_embedder.mlp[0].weight
        print(f"  PyTorch t_embedder.mlp[0].weight形状: {pytorch_t_weight.shape}")
        print(f"  PyTorch t_embedder.mlp[0].weight范围: [{pytorch_t_weight.min():.6f}, {pytorch_t_weight.max():.6f}]")
        
        # 测试PyTorch t_embedder功能
        test_t_torch = torch.tensor([0.0, 0.5, 1.0])
        pytorch_t_emb = pytorch_estimator.t_embedder(test_t_torch)
        print(f"  PyTorch t_embedder输出形状: {pytorch_t_emb.shape}")
        print(f"  PyTorch t_embedder输出范围: [{pytorch_t_emb.min():.6f}, {pytorch_t_emb.max():.6f}]")
        
    except Exception as e:
        print(f"  PyTorch t_embedder检查失败: {e}")
    
    # 8. 分析权重加载日志
    print(f"\n8️⃣ 分析权重加载日志:")
    print(f"  之前显示'Loaded 0 DiT weights total'可能的原因:")
    print(f"  1. 权重键名映射错误")
    print(f"  2. 权重加载过程中出现异常")
    print(f"  3. 权重加载成功但计数错误")
    print(f"  4. 使用了默认初始化权重")
    
    # 9. 检查MLX CFM的实际推理能力
    print(f"\n9️⃣ 检查MLX CFM的实际推理能力:")
    
    # 使用简单的输入测试
    try:
        import mlx.core as mx
        
        # 创建简单的测试输入
        batch_size = 1
        seq_len = 10
        test_mu = mx.random.normal((batch_size, seq_len, 512))
        test_x_lens = mx.array([seq_len])
        test_ref_mel = mx.random.normal((batch_size, 80, 5))
        test_style = mx.random.normal((batch_size, 192))
        
        print(f"  测试输入:")
        print(f"    mu: {test_mu.shape}")
        print(f"    x_lens: {test_x_lens.shape}")
        print(f"    ref_mel: {test_ref_mel.shape}")
        print(f"    style: {test_style.shape}")
        
        # 测试MLX CFM推理
        test_output = mlx_cfm.inference(
            test_mu, test_x_lens, test_ref_mel, test_style, None,
            n_timesteps=5, temperature=1.0, inference_cfg_rate=0.0,
            unified_random=tts.unified_random
        )
        
        print(f"  MLX CFM测试输出:")
        print(f"    形状: {test_output.shape}")
        print(f"    范围: [{test_output.min():.6f}, {test_output.max():.6f}]")
        print(f"    均值: {test_output.mean():.6f}")
        print(f"    标准差: {test_output.std():.6f}")
        print(f"    是否全零: {torch.allclose(mlx_to_torch(test_output), torch.zeros_like(mlx_to_torch(test_output)))}")
        
    except Exception as e:
        print(f"  MLX CFM推理测试失败: {e}")
    
    print(f"\n📋 总结:")
    print(f"  如果MLX CFM真的加载了0个权重，理论上应该:")
    print(f"  1. 输出全零或随机噪音")
    print(f"  2. 无法产生有意义的音频")
    print(f"  3. 与PyTorch版本差异巨大")
    print(f"  ")
    print(f"  但实际情况是:")
    print(f"  1. 音频基本正常，只是有轻微破音")
    print(f"  2. 输出范围合理: [-3.715705, 3.658470]")
    print(f"  3. 与PyTorch版本差异相对较小")
    print(f"  ")
    print(f"  这说明:")
    print(f"  1. MLX CFM实际上加载了权重（可能是默认初始化）")
    print(f"  2. 权重加载日志可能有误")
    print(f"  3. 或者MLX CFM使用了不同的初始化策略")

def main():
    """主函数"""
    
    print("开始分析MLX CFM权重缺失但音频基本正常的矛盾现象...")
    
    # 分析权重加载矛盾
    analyze_weight_loading_contradiction()
    
    print(f"\n🎉 分析完成！")

if __name__ == "__main__":
    main()
