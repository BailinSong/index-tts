#!/usr/bin/env python3
"""
DiT 转置差异分析工具
重点分析 PyTorch 和 MLX DiT 中的转置操作差异
"""

import os
import pickle
import torch
import mlx.core as mx
import numpy as np

def load_dit_input_data():
    """加载 DiT 输入数据"""
    import glob
    
    # 找到最新的 DiT 输入缓存
    pytorch_files = glob.glob("cfm_production_cache/cfm_pytorch_step_*_dit_input_*.pkl")
    mlx_files = glob.glob("cfm_production_cache/cfm_mlx_step_*_dit_input_*.pkl")
    
    if not pytorch_files or not mlx_files:
        print("❌ 未找到 DiT 输入缓存文件")
        return None, None
        
    # 加载最新的文件
    pytorch_file = max(pytorch_files, key=os.path.getmtime)
    mlx_file = max(mlx_files, key=os.path.getmtime)
    
    with open(pytorch_file, 'rb') as f:
        pytorch_data = pickle.load(f)
    with open(mlx_file, 'rb') as f:
        mlx_data = pickle.load(f)
        
    return pytorch_data, mlx_data

def simulate_pytorch_dit_forward(x, prompt_x, x_lens, t, style, cond):
    """模拟 PyTorch DiT 前向传播的关键步骤"""
    print(f"\n{'='*60}")
    print(f"🔍 模拟 PyTorch DiT 前向传播")
    print(f"{'='*60}")
    
    # 转换为 PyTorch 张量
    if isinstance(x, mx.array):
        x = torch.from_numpy(np.array(x))
        prompt_x = torch.from_numpy(np.array(prompt_x))
        x_lens = torch.from_numpy(np.array(x_lens))
        t = torch.from_numpy(np.array(t))
        style = torch.from_numpy(np.array(style))
        cond = torch.from_numpy(np.array(cond))
    
    print(f"输入形状:")
    print(f"   x: {x.shape} (B, C, T)")
    print(f"   prompt_x: {prompt_x.shape} (B, C, T)")
    print(f"   cond: {cond.shape} (B, T, C)")
    print(f"   style: {style.shape} (B, C)")
    print(f"   t: {t.shape} (B,)")
    print(f"   x_lens: {x_lens.shape} (B,)")
    
    # 关键步骤1: 转置 x 和 prompt_x
    print(f"\n步骤1: 转置 x 和 prompt_x")
    x_t = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
    prompt_x_t = prompt_x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
    
    print(f"   x_t: {x_t.shape} (B, T, C)")
    print(f"   prompt_x_t: {prompt_x_t.shape} (B, T, C)")
    
    # 关键步骤2: 拼接输入
    print(f"\n步骤2: 拼接输入")
    x_in = torch.cat([x_t, prompt_x_t, cond], dim=-1)
    print(f"   x_in: {x_in.shape} (B, T, C+C+C)")
    
    # 关键步骤3: 添加风格条件
    print(f"\n步骤3: 添加风格条件")
    B, T, _ = x_in.shape
    style_broadcast = style[:, None, :].repeat(1, T, 1)
    x_in = torch.cat([x_in, style_broadcast], dim=-1)
    print(f"   style_broadcast: {style_broadcast.shape} (B, T, C)")
    print(f"   x_in: {x_in.shape} (B, T, C+C+C+C)")
    
    # 关键步骤4: 最终输出转置
    print(f"\n步骤4: 最终输出转置")
    # 假设经过 transformer 后的输出是 (B, T, C)
    x_res = torch.randn(B, T, 80)  # 模拟 transformer 输出
    x_out = x_res.transpose(1, 2)  # (B, T, C) -> (B, C, T)
    print(f"   x_res: {x_res.shape} (B, T, C)")
    print(f"   x_out: {x_out.shape} (B, C, T)")
    
    return x_out

def simulate_mlx_dit_forward(x, prompt_x, x_lens, t, style, cond):
    """模拟 MLX DiT 前向传播的关键步骤"""
    print(f"\n{'='*60}")
    print(f"🔍 模拟 MLX DiT 前向传播")
    print(f"{'='*60}")
    
    # 转换为 MLX 数组
    if isinstance(x, torch.Tensor):
        x = mx.array(x.numpy())
        prompt_x = mx.array(prompt_x.numpy())
        x_lens = mx.array(x_lens.numpy())
        t = mx.array(t.numpy())
        style = mx.array(style.numpy())
        cond = mx.array(cond.numpy())
    elif isinstance(x, np.ndarray):
        x = mx.array(x)
        prompt_x = mx.array(prompt_x)
        x_lens = mx.array(x_lens)
        t = mx.array(t)
        style = mx.array(style)
        cond = mx.array(cond)
    
    print(f"输入形状:")
    print(f"   x: {x.shape} (B, C, T)")
    print(f"   prompt_x: {prompt_x.shape} (B, C, T)")
    print(f"   cond: {cond.shape} (B, T, C)")
    print(f"   style: {style.shape} (B, C)")
    print(f"   t: {t.shape} (B,)")
    print(f"   x_lens: {x_lens.shape} (B,)")
    
    # 关键步骤1: 转置 x 和 prompt_x (MLX 使用 transpose(0, 2, 1))
    print(f"\n步骤1: 转置 x 和 prompt_x")
    x_t = x.transpose(0, 2, 1)  # (B, C, T) -> (B, T, C)
    prompt_x_t = prompt_x.transpose(0, 2, 1)  # (B, C, T) -> (B, T, C)
    
    print(f"   x_t: {x_t.shape} (B, T, C)")
    print(f"   prompt_x_t: {prompt_x_t.shape} (B, T, C)")
    
    # 关键步骤2: 拼接输入
    print(f"\n步骤2: 拼接输入")
    x_in = mx.concatenate([x_t, prompt_x_t, cond], -1)
    print(f"   x_in: {x_in.shape} (B, T, C+C+C)")
    
    # 关键步骤3: 添加风格条件
    print(f"\n步骤3: 添加风格条件")
    B, T, _ = x_in.shape
    style_broadcast = mx.broadcast_to(style.reshape(B, 1, -1), (B, T, style.shape[-1]))
    x_in = mx.concatenate([x_in, style_broadcast], -1)
    print(f"   style_broadcast: {style_broadcast.shape} (B, T, C)")
    print(f"   x_in: {x_in.shape} (B, T, C+C+C+C)")
    
    # 关键步骤4: 最终输出转置
    print(f"\n步骤4: 最终输出转置")
    # 假设经过 transformer 后的输出是 (B, T, C)
    x_res = mx.random.normal((B, T, 80))  # 模拟 transformer 输出
    x_out = x_res.transpose(0, 2, 1)  # (B, T, C) -> (B, C, T)
    print(f"   x_res: {x_res.shape} (B, T, C)")
    print(f"   x_out: {x_out.shape} (B, C, T)")
    
    return x_out

def compare_transpose_operations():
    """比较转置操作"""
    print(f"\n{'='*60}")
    print(f"🔍 转置操作对比")
    print(f"{'='*60}")
    
    # 创建测试数据
    B, C, T = 2, 80, 415
    
    # PyTorch 转置
    x_pt = torch.randn(B, C, T)
    x_pt_t = x_pt.transpose(1, 2)  # (B, C, T) -> (B, T, C)
    x_pt_back = x_pt_t.transpose(1, 2)  # (B, T, C) -> (B, C, T)
    
    # MLX 转置
    x_mx = mx.array(x_pt.numpy())
    x_mx_t = x_mx.transpose(0, 2, 1)  # (B, C, T) -> (B, T, C)
    x_mx_back = x_mx_t.transpose(0, 2, 1)  # (B, T, C) -> (B, C, T)
    
    print(f"PyTorch 转置:")
    print(f"   原始: {x_pt.shape}")
    print(f"   转置: {x_pt_t.shape}")
    print(f"   恢复: {x_pt_back.shape}")
    
    print(f"\nMLX 转置:")
    print(f"   原始: {x_mx.shape}")
    print(f"   转置: {x_mx_t.shape}")
    print(f"   恢复: {x_mx_back.shape}")
    
    # 检查是否一致
    x_pt_np = x_pt.numpy()
    x_mx_np = np.array(x_mx)
    x_pt_t_np = x_pt_t.numpy()
    x_mx_t_np = np.array(x_mx_t)
    
    print(f"\n转置结果比较:")
    if np.allclose(x_pt_t_np, x_mx_t_np):
        print(f"   ✅ 转置结果一致")
    else:
        print(f"   ❌ 转置结果不一致")
        diff = np.abs(x_pt_t_np - x_mx_t_np)
        print(f"   最大差异: {np.max(diff):.6f}")
        print(f"   平均差异: {np.mean(diff):.6f}")
    
    # 检查恢复是否一致
    x_pt_back_np = x_pt_back.numpy()
    x_mx_back_np = np.array(x_mx_back)
    
    print(f"\n恢复结果比较:")
    if np.allclose(x_pt_back_np, x_mx_back_np):
        print(f"   ✅ 恢复结果一致")
    else:
        print(f"   ❌ 恢复结果不一致")
        diff = np.abs(x_pt_back_np - x_mx_back_np)
        print(f"   最大差异: {np.max(diff):.6f}")
        print(f"   平均差异: {np.mean(diff):.6f}")

def main():
    print("🔍 DiT 转置差异分析工具")
    print("="*60)
    
    # 比较转置操作
    compare_transpose_operations()
    
    # 加载实际数据并模拟
    pt_data, mlx_data = load_dit_input_data()
    if pt_data is not None and mlx_data is not None:
        print(f"\n使用实际数据模拟:")
        
        # 模拟 PyTorch 前向传播
        pt_output = simulate_pytorch_dit_forward(
            pt_data['x'], pt_data['prompt_x'], pt_data['x_lens'],
            pt_data['t'], pt_data['style'], pt_data['mu']
        )
        
        # 模拟 MLX 前向传播
        mlx_output = simulate_mlx_dit_forward(
            mlx_data['x'], mlx_data['prompt_x'], mlx_data['x_lens'],
            mlx_data['t'], mlx_data['style'], mlx_data['mu']
        )
        
        # 比较输出
        print(f"\n输出比较:")
        print(f"   PyTorch: {pt_output.shape}")
        print(f"   MLX:     {mlx_output.shape}")
        
        if isinstance(pt_output, torch.Tensor):
            pt_np = pt_output.numpy()
        else:
            pt_np = pt_output
            
        if isinstance(mlx_output, mx.array):
            mlx_np = np.array(mlx_output)
        else:
            mlx_np = mlx_output
        
        if np.allclose(pt_np, mlx_np):
            print(f"   ✅ 输出一致")
        else:
            print(f"   ❌ 输出不一致")
            diff = np.abs(pt_np - mlx_np)
            print(f"   最大差异: {np.max(diff):.6f}")
            print(f"   平均差异: {np.mean(diff):.6f}")

if __name__ == "__main__":
    main()
