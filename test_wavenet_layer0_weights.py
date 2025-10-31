#!/usr/bin/env python3
"""
测试 WaveNet 第0层的实际权重转换
"""

import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pickle

def test_wavenet_layer0_weights():
    """测试 WaveNet 第0层的实际权重转换"""
    
    # 加载实际的调试数据
    debug_file = "/Users/bailin/index-tts/cfm_debug_outputs/complete_dit_debug_analysis.pkl"
    
    try:
        with open(debug_file, 'rb') as f:
            debug_data = pickle.load(f)
        
        print("✅ 成功加载调试数据")
        
        # 获取 wavenet_l0_in 的数据
        pytorch_data = debug_data.get('pytorch', {})
        mlx_data = debug_data.get('mlx', {})
        
        wavenet_l0_in_pt = pytorch_data.get('step_0_layer_0_wavenet_l0_in', {})
        wavenet_l0_in_mlx = mlx_data.get('step_0_layer_0_wavenet_l0_in', {})
        
        if not wavenet_l0_in_pt or not wavenet_l0_in_mlx:
            print("❌ 未找到 wavenet_l0_in 数据")
            return
        
        print(f"\n📊 wavenet_l0_in 数据分析:")
        
        # 分析 x_masked (输入到卷积层的数据)
        x_masked_pt = wavenet_l0_in_pt.get('x_masked', {})
        x_masked_mlx = wavenet_l0_in_mlx.get('x_masked', {})
        
        if x_masked_pt and x_masked_mlx:
            print(f"  x_masked:")
            print(f"    PyTorch 形状: {x_masked_pt.get('shape', 'N/A')}")
            print(f"    MLX 形状: {x_masked_mlx.get('shape', 'N/A')}")
            print(f"    PyTorch head: {x_masked_pt.get('head', 'N/A')}")
            print(f"    MLX head: {x_masked_mlx.get('head', 'N/A')}")
            print(f"    PyTorch tail: {x_masked_pt.get('tail', 'N/A')}")
            print(f"    MLX tail: {x_masked_mlx.get('tail', 'N/A')}")
        
        # 分析 x_in (卷积层输出的数据)
        x_in_pt = wavenet_l0_in_pt.get('x_in', {})
        x_in_mlx = wavenet_l0_in_mlx.get('x_in', {})
        
        if x_in_pt and x_in_mlx:
            print(f"\n  x_in (卷积输出):")
            print(f"    PyTorch 形状: {x_in_pt.get('shape', 'N/A')}")
            print(f"    MLX 形状: {x_in_mlx.get('shape', 'N/A')}")
            print(f"    PyTorch head: {x_in_pt.get('head', 'N/A')}")
            print(f"    MLX head: {x_in_mlx.get('head', 'N/A')}")
            print(f"    PyTorch tail: {x_in_pt.get('tail', 'N/A')}")
            print(f"    MLX tail: {x_in_mlx.get('tail', 'N/A')}")
            
            # 计算数值差异
            if 'data' in x_in_pt and 'data' in x_in_mlx:
                pt_data = x_in_pt['data']
                mlx_data = x_in_mlx['data']
                
                if hasattr(pt_data, 'numpy'):
                    pt_np = pt_data.numpy()
                else:
                    pt_np = np.array(pt_data)
                
                if hasattr(mlx_data, 'numpy'):
                    mlx_np = mlx_data.numpy()
                else:
                    mlx_np = np.array(mlx_data)
                
                # 转换为相同格式进行比较
                if len(pt_np.shape) == 3 and len(mlx_np.shape) == 3:
                    # PyTorch: (batch, channels, seq_len)
                    # MLX: (batch, seq_len, channels)
                    if pt_np.shape[0] == mlx_np.shape[0] and pt_np.shape[1] == mlx_np.shape[2] and pt_np.shape[2] == mlx_np.shape[1]:
                        mlx_np_transposed = mlx_np.transpose(0, 2, 1)
                        diff = np.abs(pt_np - mlx_np_transposed)
                        print(f"    数值差异:")
                        print(f"      最大差异: {diff.max():.6f}")
                        print(f"      平均差异: {diff.mean():.6f}")
                        print(f"      标准差: {diff.std():.6f}")
                        
                        # 分析差异分布
                        large_diff = diff > 1e-5
                        print(f"      大差异(>1e-5)数量: {large_diff.sum()}/{diff.size} ({100*large_diff.sum()/diff.size:.1f}%)")
                        
                        if large_diff.sum() > 0:
                            print(f"      最大差异位置: {np.unravel_index(diff.argmax(), diff.shape)}")
                            print(f"      最大差异值: {diff.max():.6f}")
        
        # 检查权重加载过程
        print(f"\n🔍 检查权重加载过程:")
        
        # 尝试从 s2mel.pth 加载权重
        try:
            s2mel_path = "/Users/bailin/index-tts/checkpoints/s2mel.pth"
            s2mel_state = torch.load(s2mel_path, map_location='cpu')
            
            # 查找 WaveNet 相关权重
            wavenet_keys = [k for k in s2mel_state.keys() if 'wavenet' in k.lower() or 'final_layer' in k.lower()]
            print(f"  找到 {len(wavenet_keys)} 个 WaveNet 相关权重:")
            for key in wavenet_keys[:10]:  # 只显示前10个
                print(f"    {key}: {s2mel_state[key].shape if hasattr(s2mel_state[key], 'shape') else type(s2mel_state[key])}")
            
            # 查找第0层卷积权重
            layer0_keys = [k for k in s2mel_state.keys() if 'in_layers.0' in k or 'final_layer' in k]
            print(f"\n  第0层相关权重:")
            for key in layer0_keys:
                if hasattr(s2mel_state[key], 'shape'):
                    print(f"    {key}: {s2mel_state[key].shape}")
                    if 'weight' in key:
                        weight = s2mel_state[key]
                        print(f"      权重范围: [{weight.min().item():.6f}, {weight.max().item():.6f}]")
                        print(f"      权重均值: {weight.mean().item():.6f}")
                        print(f"      权重标准差: {weight.std().item():.6f}")
                        
                        # 检查权重转换
                        if len(weight.shape) == 3:  # Conv1d 权重
                            print(f"      原始形状: {weight.shape}")
                            converted = np.transpose(weight.numpy(), (0, 2, 1))
                            print(f"      转换后形状: {converted.shape}")
                            
                            # 检查转换后的数值
                            diff_conv = np.abs(weight.numpy() - np.transpose(converted, (0, 2, 1)))
                            print(f"      转换差异: {diff_conv.max():.10f}")
                
        except Exception as e:
            print(f"  ❌ 无法加载 s2mel.pth: {e}")
        
    except Exception as e:
        print(f"❌ 无法加载调试数据: {e}")

if __name__ == "__main__":
    test_wavenet_layer0_weights()
