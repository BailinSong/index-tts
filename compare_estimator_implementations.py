#!/usr/bin/env python3
"""
详细对比PyTorch DiT和MLX MLXDiTRewritten的实现差异
"""

import sys
sys.path.append('.')
import torch
import mlx.core as mx
import numpy as np
from indextts.infer_v2 import IndexTTS2
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def compare_estimator_implementations():
    """对比两个estimator的实现差异"""
    
    print('=== 对比PyTorch DiT和MLX MLXDiTRewritten的实现差异 ===')
    
    # 设置固定种子
    torch.manual_seed(42)
    np.random.seed(42)
    mx.random.seed(42)
    
    # 初始化 MLX 版本
    print('\n1. 初始化 MLX 版本...')
    tts_mlx = IndexTTS2(
        cfg_path='checkpoints/config.yaml',
        model_dir='checkpoints',
        use_mlx=True,
        device='mps'
    )
    
    # 获取PyTorch和MLX的estimator
    pytorch_estimator = tts_mlx.s2mel.models['cfm'].estimator
    mlx_estimator = tts_mlx.mlx_s2mel_cfm.estimator
    
    print(f'\n2. Estimator类型对比...')
    print(f'PyTorch estimator: {type(pytorch_estimator)}')
    print(f'MLX estimator: {type(mlx_estimator)}')
    
    # 对比关键属性
    print(f'\n3. 关键属性对比...')
    
    # PyTorch DiT属性
    print(f'\n--- PyTorch DiT 属性 ---')
    pytorch_attrs = [
        'time_as_token', 'style_as_token', 'uvit_skip_connection',
        'in_channels', 'out_channels', 'num_heads',
        'transformer_style_condition', 'class_dropout_prob',
        'long_skip_connection', 'final_layer_type', 'is_causal'
    ]
    
    for attr in pytorch_attrs:
        if hasattr(pytorch_estimator, attr):
            value = getattr(pytorch_estimator, attr)
            print(f'  {attr}: {value} ({type(value)})')
        else:
            print(f'  {attr}: 不存在')
    
    # MLX MLXDiTRewritten属性
    print(f'\n--- MLX MLXDiTRewritten 属性 ---')
    for attr in pytorch_attrs:
        if hasattr(mlx_estimator, attr):
            value = getattr(mlx_estimator, attr)
            print(f'  {attr}: {value} ({type(value)})')
        else:
            print(f'  {attr}: 不存在')
    
    # 对比模块结构
    print(f'\n4. 模块结构对比...')
    
    # PyTorch模块
    print(f'\n--- PyTorch DiT 模块 ---')
    pytorch_modules = [
        'transformer', 'x_embedder', 'cond_embedder', 'cond_projection',
        't_embedder', 't_embedder2', 'conv1', 'conv2', 'wavenet',
        'final_layer', 'skip_linear', 'cond_x_merge_linear'
    ]
    
    for module_name in pytorch_modules:
        if hasattr(pytorch_estimator, module_name):
            module = getattr(pytorch_estimator, module_name)
            print(f'  {module_name}: {type(module)}')
        else:
            print(f'  {module_name}: 不存在')
    
    # MLX模块
    print(f'\n--- MLX MLXDiTRewritten 模块 ---')
    for module_name in pytorch_modules:
        if hasattr(mlx_estimator, module_name):
            module = getattr(mlx_estimator, module_name)
            print(f'  {module_name}: {type(module)}')
        else:
            print(f'  {module_name}: 不存在')
    
    # 对比forward方法的参数
    print(f'\n5. Forward方法参数对比...')
    
    # 创建测试输入
    batch_size = 1
    seq_len = 100
    in_channels = 80
    hidden_dim = 512
    style_dim = 192
    
    # PyTorch输入
    x_pytorch = torch.randn(batch_size, in_channels, seq_len, device=tts_mlx.device)
    prompt_x_pytorch = torch.randn(batch_size, in_channels, seq_len, device=tts_mlx.device)
    x_lens_pytorch = torch.tensor([seq_len], device=tts_mlx.device)
    t_pytorch = torch.tensor([0.0], device=tts_mlx.device)
    style_pytorch = torch.randn(batch_size, style_dim, device=tts_mlx.device)
    cond_pytorch = torch.randn(batch_size, seq_len, hidden_dim, device=tts_mlx.device)
    
    print(f'PyTorch输入形状:')
    print(f'  x: {x_pytorch.shape}')
    print(f'  prompt_x: {prompt_x_pytorch.shape}')
    print(f'  x_lens: {x_lens_pytorch.shape}')
    print(f'  t: {t_pytorch.shape}')
    print(f'  style: {style_pytorch.shape}')
    print(f'  cond: {cond_pytorch.shape}')
    
    # 转换到MLX
    x_mlx = torch_to_mlx(x_pytorch.cpu())
    prompt_x_mlx = torch_to_mlx(prompt_x_pytorch.cpu())
    x_lens_mlx = torch_to_mlx(x_lens_pytorch.cpu())
    t_mlx = torch_to_mlx(t_pytorch.cpu())
    style_mlx = torch_to_mlx(style_pytorch.cpu())
    cond_mlx = torch_to_mlx(cond_pytorch.cpu())
    
    print(f'\nMLX输入形状:')
    print(f'  x: {x_mlx.shape}')
    print(f'  prompt_x: {prompt_x_mlx.shape}')
    print(f'  x_lens: {x_lens_mlx.shape}')
    print(f'  t: {t_mlx.shape}')
    print(f'  style: {style_mlx.shape}')
    print(f'  cond: {cond_mlx.shape}')
    
    # 测试PyTorch forward
    print(f'\n6. 测试PyTorch forward...')
    try:
        with torch.no_grad():
            pytorch_output = pytorch_estimator.forward(
                x_pytorch, prompt_x_pytorch, x_lens_pytorch, t_pytorch,
                style_pytorch, cond_pytorch, mask_content=False
            )
        print(f'PyTorch输出: {pytorch_output.shape}, 范围 [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]')
    except Exception as e:
        print(f'PyTorch forward失败: {e}')
        pytorch_output = None
    
    # 测试MLX forward
    print(f'\n7. 测试MLX forward...')
    try:
        mlx_output = mlx_estimator(
            x_mlx, prompt_x_mlx, x_lens_mlx, t_mlx,
            style_mlx, cond_mlx, mask_content=False
        )
        print(f'MLX输出: {mlx_output.shape}, 范围 [{mlx_output.min():.6f}, {mlx_output.max():.6f}]')
    except Exception as e:
        print(f'MLX forward失败: {e}')
        mlx_output = None
    
    # 对比输出
    if pytorch_output is not None and mlx_output is not None:
        print(f'\n8. 输出对比...')
        mlx_output_torch = mlx_to_torch(mlx_output).to(tts_mlx.device)
        
        if pytorch_output.shape == mlx_output_torch.shape:
            output_diff = torch.abs(pytorch_output - mlx_output_torch)
            max_diff = torch.max(output_diff).item()
            mean_diff = torch.mean(output_diff).item()
            
            print(f'输出差异: 最大 {max_diff:.6f}, 平均 {mean_diff:.6f}')
            
            if max_diff > 1e-5:
                print(f'❌ 输出差异超过阈值')
            else:
                print(f'✅ 输出差异在阈值内')
        else:
            print(f'❌ 输出形状不匹配: PyTorch {pytorch_output.shape}, MLX {mlx_output_torch.shape}')
    
    # 分析关键差异
    print(f'\n9. 关键差异分析...')
    
    # 检查class_dropout_prob
    pytorch_dropout = getattr(pytorch_estimator, 'class_dropout_prob', None)
    mlx_dropout = getattr(mlx_estimator, 'class_dropout_prob', None)
    print(f'class_dropout_prob: PyTorch={pytorch_dropout}, MLX={mlx_dropout}')
    
    # 检查training状态
    pytorch_training = pytorch_estimator.training
    mlx_training = getattr(mlx_estimator, 'training', None)
    print(f'training状态: PyTorch={pytorch_training}, MLX={mlx_training}')
    
    # 检查transformer_style_condition
    pytorch_style_cond = getattr(pytorch_estimator, 'transformer_style_condition', None)
    mlx_style_cond = getattr(mlx_estimator, 'transformer_style_condition', None)
    print(f'transformer_style_condition: PyTorch={pytorch_style_cond}, MLX={mlx_style_cond}')
    
    # 检查long_skip_connection
    pytorch_skip = getattr(pytorch_estimator, 'long_skip_connection', None)
    mlx_skip = getattr(mlx_estimator, 'long_skip_connection', None)
    print(f'long_skip_connection: PyTorch={pytorch_skip}, MLX={mlx_skip}')
    
    print(f'\n=== 总结 ===')
    print(f'主要差异可能来自:')
    print(f'1. 模块结构差异')
    print(f'2. 属性值差异')
    print(f'3. 实现细节差异')
    print(f'4. 数值计算精度差异')

if __name__ == "__main__":
    compare_estimator_implementations()
