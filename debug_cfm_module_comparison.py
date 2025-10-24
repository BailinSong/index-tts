#!/usr/bin/env python3
"""
简化的CFM模块对比调试工具
直接使用CFM推理方法进行对比
"""

import torch
import mlx.core as mx
import numpy as np
import pickle
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from indextts.infer_v2 import IndexTTS2
from indextts.utils.mlx_utils import mlx_to_torch, torch_to_mlx
from unified_random_generator import UnifiedRandomGenerator

def debug_cfm_module_comparison():
    """主函数：对比PyTorch和MLX CFM的模块级输入输出"""
    
    print("=== CFM模块级输入输出对比调试 ===")
    
    # 初始化TTS系统
    print("\n1. 初始化TTS系统...")
    tts = IndexTTS2()
    
    # 生成测试输入数据
    print("\n2. 生成测试输入数据...")
    
    # 设置随机种子确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 生成测试数据
    batch_size = 1
    seq_len = 100
    in_channels = 80
    hidden_dim = 512
    style_dim = 192
    
    # 获取设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    x = torch.randn(batch_size, in_channels, seq_len, device=device)
    prompt_x = torch.randn(batch_size, in_channels, seq_len, device=device)
    x_lens = torch.tensor([seq_len], device=device)
    t = torch.zeros(batch_size, device=device)
    style = torch.randn(batch_size, style_dim, device=device)
    cond = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    print(f"✅ 生成测试数据:")
    print(f"输入形状:")
    print(f"  x: {x.shape}")
    print(f"  prompt_x: {prompt_x.shape}")
    print(f"  x_lens: {x_lens.shape}")
    print(f"  t: {t.shape}")
    print(f"  style: {style.shape}")
    print(f"  cond: {cond.shape}")
    
    # 设置统一随机数生成器
    unified_rng = UnifiedRandomGenerator(42)
    
    print("\n3. 测试PyTorch CFM单步推理...")
    
    # 测试PyTorch CFM单步推理
    try:
        # 获取PyTorch CFM
        pytorch_cfm = tts.s2mel.models.cfm
        pytorch_estimator = pytorch_cfm.estimator
        
        print(f"PyTorch CFM estimator类型: {type(pytorch_estimator)}")
        
        # 执行PyTorch CFM单步推理
        with torch.no_grad():
            pytorch_output = pytorch_estimator(x, prompt_x, x_lens, t, style, cond)
            print(f"PyTorch CFM输出: {pytorch_output.shape}, min={pytorch_output.min():.6f}, max={pytorch_output.max():.6f}")
            
    except Exception as e:
        print(f"❌ PyTorch CFM推理失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n4. 测试MLX CFM单步推理...")
    
    # 测试MLX CFM单步推理
    try:
        # 获取MLX CFM
        mlx_cfm = tts.mlx_s2mel_cfm
        print(f"MLX CFM类型: {type(mlx_cfm)}")
        print(f"MLX CFM是否为None: {mlx_cfm is None}")
        
        if mlx_cfm is None:
            print("❌ MLX CFM未初始化，检查初始化状态...")
            print(f"use_mlx: {tts.use_mlx}")
            print(f"mlx_s2mel_gpt_layer: {tts.mlx_s2mel_gpt_layer}")
            print(f"mlx_s2mel_length_regulator: {tts.mlx_s2mel_length_regulator}")
            print(f"mlx_s2mel_cfm: {tts.mlx_s2mel_cfm}")
            
            # 尝试手动初始化MLX CFM
            try:
                from indextts.s2mel.modules.mlx_cfm import MLXCFM
                print("尝试手动创建MLX CFM...")
                mlx_cfm = MLXCFM(tts.cfg.s2mel)
                print(f"手动创建MLX CFM成功: {type(mlx_cfm)}")
            except Exception as e:
                print(f"❌ 手动创建MLX CFM失败: {e}")
                return
        
        if mlx_cfm is None:
            print("❌ MLX CFM仍然为None，无法继续")
            return
            
        mlx_estimator = mlx_cfm.estimator
        print(f"MLX CFM estimator类型: {type(mlx_estimator)}")
        
        # 准备MLX输入
        mlx_x = torch_to_mlx(x)
        mlx_prompt_x = torch_to_mlx(prompt_x)
        mlx_x_lens = torch_to_mlx(x_lens)
        mlx_t = torch_to_mlx(t)
        mlx_style = torch_to_mlx(style)
        mlx_cond = torch_to_mlx(cond)
        
        # 执行MLX CFM单步推理
        mlx_output = mlx_estimator(mlx_x, mlx_prompt_x, mlx_x_lens, mlx_t, mlx_style, mlx_cond)
        print(f"MLX CFM输出: {mlx_output.shape}, min={mlx_output.min():.6f}, max={mlx_output.max():.6f}")
        
    except Exception as e:
        print(f"❌ MLX CFM推理失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n5. 对比分析结果...")
    
    # 转换MLX输出为PyTorch格式进行对比
    mlx_output_torch = mlx_to_torch(mlx_output)
    
    # 确保两个张量在同一设备上
    if pytorch_output.device != mlx_output_torch.device:
        mlx_output_torch = mlx_output_torch.to(pytorch_output.device)
    
    # 计算输出差异
    if isinstance(pytorch_output, torch.Tensor) and isinstance(mlx_output_torch, torch.Tensor):
        diff = torch.abs(pytorch_output - mlx_output_torch)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        std_diff = float(diff.std())
        
        # 计算形状和数值范围
        pytorch_shape = list(pytorch_output.shape)
        mlx_shape = list(mlx_output_torch.shape)
        shape_match = pytorch_shape == mlx_shape
        
        pytorch_range = float(pytorch_output.max() - pytorch_output.min())
        mlx_range = float(mlx_output_torch.max() - mlx_output_torch.min())
        
        print(f"\n--- 输出对比结果 ---")
        print(f"形状匹配: {'✅' if shape_match else '❌'}")
        if not shape_match:
            print(f"  PyTorch: {pytorch_shape}")
            print(f"  MLX: {mlx_shape}")
        print(f"最大差异: {max_diff:.6f}")
        print(f"平均差异: {mean_diff:.6f}")
        print(f"标准差差异: {std_diff:.6f}")
        print(f"数值范围比例: {pytorch_range / mlx_range if mlx_range > 0 else float('inf'):.6f}")
        print(f"PyTorch统计: min={pytorch_output.min():.6f}, max={pytorch_output.max():.6f}, mean={pytorch_output.mean():.6f}")
        print(f"MLX统计: min={mlx_output_torch.min():.6f}, max={mlx_output_torch.max():.6f}, mean={mlx_output_torch.mean():.6f}")
        
        # 判断差异程度
        if max_diff < 1e-5:
            print(f"✅ 差异极小 (< 1e-5)")
        elif max_diff < 1e-3:
            print(f"⚠️  差异较小 (< 1e-3)")
        elif max_diff < 1e-1:
            print(f"⚠️  差异中等 (< 1e-1)")
        else:
            print(f"❌ 差异较大 (>= 1e-1)")
            
        # 统计大差异元素
        large_diff_mask = diff > 1e-3
        large_diff_count = int(large_diff_mask.sum())
        total_elements = int(diff.numel())
        large_diff_ratio = large_diff_count / total_elements if total_elements > 0 else 0
        
        print(f"\n大差异元素统计:")
        print(f"  大差异元素数 (> 1e-3): {large_diff_count}/{total_elements} ({large_diff_ratio*100:.2f}%)")
        
        if large_diff_count > 0:
            # 找到前10个大差异位置
            large_diff_indices = torch.nonzero(large_diff_mask, as_tuple=False)
            if len(large_diff_indices) > 0:
                print(f"\n前10个大差异位置:")
                for i, idx in enumerate(large_diff_indices[:10]):
                    pytorch_val = pytorch_output[tuple(idx)]
                    mlx_val = mlx_output_torch[tuple(idx)]
                    diff_val = diff[tuple(idx)]
                    print(f"  位置 {tuple(idx)}: PyTorch={pytorch_val:.6f}, MLX={mlx_val:.6f}, 差异={diff_val:.6f}")
    
    print("\n=== 总结 ===")
    print("✅ CFM模块级对比完成")
    print("📊 详细差异分析已输出")

if __name__ == "__main__":
    debug_cfm_module_comparison()
