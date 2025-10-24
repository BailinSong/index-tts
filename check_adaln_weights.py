#!/usr/bin/env python3
"""
检查MLX和PyTorch的adaLN_modulation权重是否完全一致
"""

import sys
sys.path.append('.')
import torch
import mlx.core as mx
import numpy as np
from indextts.infer_v2 import IndexTTS2
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def check_adaln_weights():
    """检查adaLN_modulation权重的一致性"""
    
    print('=== 检查MLX和PyTorch的adaLN_modulation权重一致性 ===')
    
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
    
    # 获取PyTorch和MLX的final_layer
    pytorch_final_layer = tts_mlx.s2mel.models['cfm'].estimator.final_layer
    mlx_final_layer = tts_mlx.mlx_s2mel_cfm.estimator.final_layer
    
    print(f'\n2. 分析final_layer结构...')
    print(f'PyTorch final_layer类型: {type(pytorch_final_layer)}')
    print(f'MLX final_layer类型: {type(mlx_final_layer)}')
    
    # 检查PyTorch的adaLN_modulation结构
    print(f'\n3. 分析PyTorch adaLN_modulation结构...')
    pytorch_adaln = pytorch_final_layer.adaLN_modulation
    print(f'PyTorch adaLN_modulation类型: {type(pytorch_adaln)}')
    print(f'PyTorch adaLN_modulation结构: {pytorch_adaln}')
    
    # 检查MLX的adaLN_modulation结构
    print(f'\n4. 分析MLX adaLN_modulation结构...')
    mlx_adaln = mlx_final_layer.adaLN_modulation
    print(f'MLX adaLN_modulation类型: {type(mlx_adaln)}')
    print(f'MLX adaLN_modulation结构: {mlx_adaln}')
    
    # 检查PyTorch的权重
    print(f'\n5. 检查PyTorch权重...')
    pytorch_weights = {}
    for name, param in pytorch_adaln.named_parameters():
        pytorch_weights[name] = param
        print(f'  {name}: {param.shape}, 范围 [{param.min():.6f}, {param.max():.6f}]')
    
    # 检查MLX的权重
    print(f'\n6. 检查MLX权重...')
    mlx_weights = {}
    # MLX Sequential没有named_parameters，需要手动访问
    for i, layer in enumerate(mlx_adaln.layers):
        if hasattr(layer, 'weight'):
            name = f"{i}.weight"
            param = layer.weight
            mlx_weights[name] = param
            print(f'  {name}: {param.shape}, 范围 [{param.min():.6f}, {param.max():.6f}]')
        if hasattr(layer, 'bias'):
            name = f"{i}.bias"
            param = layer.bias
            mlx_weights[name] = param
            print(f'  {name}: {param.shape}, 范围 [{param.min():.6f}, {param.max():.6f}]')
    
    # 对比权重
    print(f'\n7. 权重对比...')
    weight_diffs = {}
    
    for name in pytorch_weights.keys():
        if name in mlx_weights:
            pytorch_weight = pytorch_weights[name]
            mlx_weight = mlx_weights[name]
            
            # 转换MLX权重为PyTorch格式
            if isinstance(mlx_weight, mx.array):
                mlx_weight_torch = mlx_to_torch(mlx_weight).to(tts_mlx.device)
            else:
                mlx_weight_torch = mlx_weight
            
            # 计算差异
            if pytorch_weight.shape == mlx_weight_torch.shape:
                diff = torch.abs(pytorch_weight - mlx_weight_torch)
                max_diff = torch.max(diff).item()
                mean_diff = torch.mean(diff).item()
                weight_diffs[name] = {'max': max_diff, 'mean': mean_diff}
                
                print(f'  {name}:')
                print(f'    PyTorch: {pytorch_weight.shape}, 范围 [{pytorch_weight.min():.6f}, {pytorch_weight.max():.6f}]')
                print(f'    MLX: {mlx_weight_torch.shape}, 范围 [{mlx_weight_torch.min():.6f}, {mlx_weight_torch.max():.6f}]')
                print(f'    差异: 最大 {max_diff:.6f}, 平均 {mean_diff:.6f}')
                
                if max_diff > 1e-5:
                    print(f'    ❌ 权重差异超过阈值')
                else:
                    print(f'    ✅ 权重差异在阈值内')
            else:
                print(f'  {name}: 形状不匹配 - PyTorch: {pytorch_weight.shape}, MLX: {mlx_weight_torch.shape}')
        else:
            print(f'  {name}: 仅在PyTorch中存在')
    
    # 检查处理流程
    print(f'\n8. 检查处理流程...')
    
    # 创建测试输入
    test_c = torch.randn(1, 512, device=tts_mlx.device)
    test_c_mlx = torch_to_mlx(test_c.cpu())
    
    print(f'测试输入 c: {test_c.shape}, 范围 [{test_c.min():.6f}, {test_c.max():.6f}]')
    
    # PyTorch处理流程
    print(f'\n9. PyTorch处理流程...')
    with torch.no_grad():
        pytorch_c_emb = pytorch_adaln(test_c)
        print(f'PyTorch c_emb: {pytorch_c_emb.shape}, 范围 [{pytorch_c_emb.min():.6f}, {pytorch_c_emb.max():.6f}]')
        print(f'PyTorch c_emb mean: {pytorch_c_emb.mean():.6f}, std: {pytorch_c_emb.std():.6f}')
    
    # MLX处理流程
    print(f'\n10. MLX处理流程...')
    mlx_c_emb = mlx_adaln(test_c_mlx)
    print(f'MLX c_emb: {mlx_c_emb.shape}, 范围 [{mlx_c_emb.min():.6f}, {mlx_c_emb.max():.6f}]')
    print(f'MLX c_emb mean: {mlx_c_emb.mean():.6f}, std: {mlx_c_emb.std():.6f}')
    
    # 对比输出
    print(f'\n11. 输出对比...')
    mlx_c_emb_torch = mlx_to_torch(mlx_c_emb).to(tts_mlx.device)
    output_diff = torch.abs(pytorch_c_emb - mlx_c_emb_torch)
    max_output_diff = torch.max(output_diff).item()
    mean_output_diff = torch.mean(output_diff).item()
    
    print(f'输出差异: 最大 {max_output_diff:.6f}, 平均 {mean_output_diff:.6f}')
    
    if max_output_diff > 1e-5:
        print(f'❌ 输出差异超过阈值')
    else:
        print(f'✅ 输出差异在阈值内')
    
    # 详细分析差异
    if max_output_diff > 1e-5:
        print(f'\n12. 详细差异分析...')
        large_diff_indices = torch.where(output_diff > 1e-5)
        if len(large_diff_indices[0]) > 0:
            print(f'大差异元素数 (>1e-5): {len(large_diff_indices[0])} / {output_diff.numel()}')
            print(f'前10个大差异位置:')
            for i in range(min(10, len(large_diff_indices[0]))):
                idx = tuple(torch.tensor([large_diff_indices[j][i] for j in range(len(large_diff_indices))]))
                pytorch_val = pytorch_c_emb[idx].item()
                mlx_val = mlx_c_emb_torch[idx].item()
                diff_val = output_diff[idx].item()
                print(f'  位置 {idx}: PyTorch={pytorch_val:.6f}, MLX={mlx_val:.6f}, 差异={diff_val:.6f}')
    
    # 总结
    print(f'\n=== 总结 ===')
    print(f'权重差异: {weight_diffs}')
    print(f'输出差异: 最大 {max_output_diff:.6f}, 平均 {mean_output_diff:.6f}')
    
    if max_output_diff > 1e-5:
        print(f'❌ 发现差异，需要进一步修复')
        return False
    else:
        print(f'✅ 权重和输出完全一致')
        return True

if __name__ == "__main__":
    check_adaln_weights()
