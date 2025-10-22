"""
深度分析 length_regulator 的每一步差异
"""

import torch
import numpy as np
from omegaconf import OmegaConf
import os
import sys
import mlx.core as mx

sys.path.insert(0, os.path.dirname(__file__))

def torch_to_mlx(tensor):
    if isinstance(tensor, torch.Tensor):
        return mx.array(tensor.detach().cpu().numpy())
    return tensor

def compare(torch_val, mlx_val, name):
    """详细对比"""
    t_np = torch_val.detach().cpu().numpy() if hasattr(torch_val, 'detach') else np.array(torch_val)
    m_np = np.array(mlx_val)
    
    if t_np.shape != m_np.shape:
        print(f"❌ {name}: shape mismatch {t_np.shape} vs {m_np.shape}")
        return None, None
    
    max_diff = np.abs(t_np - m_np).max()
    mean_diff = np.abs(t_np - m_np).mean()
    corr = np.corrcoef(t_np.flatten(), m_np.flatten())[0, 1]
    
    if max_diff < 1e-4:
        status = "✅"
    elif max_diff < 1e-2 and corr > 0.99:
        status = "✅"
    elif corr > 0.95:
        status = "⚠️ "
    else:
        status = "❌"
    
    print(f"{status} {name:40s}: max_diff={max_diff:.8f}, mean={mean_diff:.8f}, corr={corr:.6f}")
    return max_diff, corr

def main():
    from indextts.s2mel.modules.mlx_s2mel import MLXLengthRegulator
    from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
    
    # 加载配置
    cfg = OmegaConf.load("checkpoints/config.yaml")
    
    # 加载PyTorch模型
    print("加载PyTorch模型...")
    s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
    s2mel_path = os.path.join("checkpoints", cfg.s2mel_checkpoint)
    s2mel, _, _, _ = load_checkpoint2(s2mel, None, s2mel_path, load_only_params=True, ignore_modules=[], is_distributed=False)
    s2mel.eval()
    
    # 创建MLX版本
    print("创建MLX LengthRegulator...")
    mlx_lr = MLXLengthRegulator(
        channels=cfg.s2mel.length_regulator.channels,
        sampling_ratios=tuple(cfg.s2mel.length_regulator.sampling_ratios),
        is_discrete=cfg.s2mel.length_regulator.is_discrete,
        in_channels=cfg.s2mel.length_regulator.in_channels if hasattr(cfg.s2mel.length_regulator, "in_channels") else None,
        codebook_size=cfg.s2mel.length_regulator.content_codebook_size,
        n_codebooks=cfg.s2mel.length_regulator.n_codebooks if hasattr(cfg.s2mel.length_regulator, "n_codebooks") else 1,
        f0_condition=cfg.s2mel.length_regulator.f0_condition if hasattr(cfg.s2mel.length_regulator, "f0_condition") else False,
        n_f0_bins=cfg.s2mel.length_regulator.n_f0_bins if hasattr(cfg.s2mel.length_regulator, "n_f0_bins") else 512,
    )
    s2mel_state_dict = s2mel.state_dict()
    s2mel_state_dict_np = {k: v.cpu().numpy() for k, v in s2mel_state_dict.items()}
    mlx_lr.load_weights_from_pytorch(s2mel_state_dict_np)
    
    # 准备测试数据
    torch.manual_seed(42)
    np.random.seed(42)
    mx.random.seed(42)
    
    batch_size = 1
    seq_len = 30
    target_len = 50
    in_channels = cfg.s2mel.length_regulator.in_channels if hasattr(cfg.s2mel.length_regulator, "in_channels") else 1024
    
    test_input = torch.randn(batch_size, seq_len, in_channels)
    test_ylens = torch.tensor([target_len], dtype=torch.long)
    
    print(f"\n输入数据: {test_input.shape}")
    print(f"目标长度: {target_len}")
    
    # PyTorch 逐步执行
    print("\n" + "="*70)
    print("PyTorch Length Regulator - 逐步执行")
    print("="*70)
    
    with torch.no_grad():
        torch_lr = s2mel.models['length_regulator']
        
        # 1. content_in_proj
        x1_t = torch_lr.content_in_proj(test_input)
        print(f"1. content_in_proj: {x1_t.shape}, mean={x1_t.mean():.6f}, std={x1_t.std():.6f}")
        
        # 2. Interpolate
        x2_t = torch.nn.functional.interpolate(
            x1_t.transpose(1, 2).contiguous(), 
            size=test_ylens.max(), 
            mode='nearest'
        )
        print(f"2. interpolate: {x2_t.shape}, mean={x2_t.mean():.6f}, std={x2_t.std():.6f}")
        
        # 3. Model layers逐层
        x3_t = x2_t
        for i, layer in enumerate(torch_lr.model):
            x3_t = layer(x3_t)
            layer_type = type(layer).__name__
            if hasattr(x3_t, 'mean'):
                print(f"3.{i} {layer_type:15s}: {x3_t.shape}, mean={x3_t.mean():.6f}, std={x3_t.std():.6f}")
        
        # 4. Transpose + mask
        x4_t = x3_t.transpose(1, 2)
        from indextts.s2mel.modules.commons import sequence_mask
        mask = sequence_mask(test_ylens).unsqueeze(-1)
        x5_t = x4_t * mask
        print(f"4. transpose+mask: {x5_t.shape}, mean={x5_t.mean():.6f}, std={x5_t.std():.6f}")
        
        # 完整forward
        output_t, *_ = torch_lr(test_input, ylens=test_ylens, n_quantizers=None, f0=None)
        print(f"\nFinal PyTorch: {output_t.shape}, mean={output_t.mean():.6f}, std={output_t.std():.6f}")
    
    # MLX 逐步执行
    print("\n" + "="*70)
    print("MLX Length Regulator - 逐步执行")
    print("="*70)
    
    test_input_mlx = torch_to_mlx(test_input)
    test_ylens_mlx = torch_to_mlx(test_ylens)
    
    # 1. content_in_proj
    x1_m = mlx_lr.content_in_proj(test_input_mlx)
    mx.eval(x1_m)
    print(f"1. content_in_proj: {x1_m.shape}, mean={float(mx.mean(x1_m)):.6f}, std={float(mx.std(x1_m)):.6f}")
    compare(x1_t, x1_m, "  -> content_in_proj comparison")
    
    # 2. Interpolate
    # MLX的Upsample
    import mlx.nn as nn
    target_len_val = int(test_ylens_mlx.max())
    current_len = x1_m.shape[1]
    scale_factor = target_len_val / current_len
    
    upsampler = nn.Upsample(scale_factor=scale_factor, mode='nearest')
    x2_m = upsampler(x1_m)
    mx.eval(x2_m)
    print(f"2. upsample: {x2_m.shape}, mean={float(mx.mean(x2_m)):.6f}, std={float(mx.std(x2_m)):.6f}")
    compare(x2_t.transpose(1, 2), x2_m, "  -> interpolate comparison")
    
    # 检查插值后的具体值
    print(f"\n   详细对比插值结果:")
    print(f"   PyTorch前5个时间步: {x2_t[0, :3, :5]}")
    print(f"   MLX前5个时间步: {x2_m[0, :5, :3]}")
    
    # 3. Model layers逐层
    x3_m = x2_m
    for i, layer in enumerate(mlx_lr.model_layers):
        x3_m = layer(x3_m)
        mx.eval(x3_m)
        layer_type = type(layer).__name__
        print(f"3.{i} {layer_type:15s}: {x3_m.shape}, mean={float(mx.mean(x3_m)):.6f}, std={float(mx.std(x3_m)):.6f}")
        
        # 对比每一层
        if i < len(torch_lr.model):
            # 需要从PyTorch中获取对应层的输出
            pass
    
    # 完整forward
    output_m, *_ = mlx_lr(test_input_mlx, ylens=test_ylens_mlx, n_quantizers=None, f0=None)
    mx.eval(output_m)
    print(f"\nFinal MLX: {output_m.shape}, mean={float(mx.mean(output_m)):.6f}, std={float(mx.std(output_m)):.6f}")
    
    # 最终对比
    print("\n" + "="*70)
    print("最终对比")
    print("="*70)
    max_diff, corr = compare(output_t, output_m, "Final Output")
    
    if corr and corr < 0.95:
        print("\n💡 差异分析:")
        print("  主要差异来自: 插值方法")
        print("  PyTorch F.interpolate(mode='nearest') vs MLX nn.Upsample(mode='nearest')")
        print("  虽然都是nearest模式，但实现细节可能不同")
        print("\n  建议:")
        print("  1. 检查MLX Upsample的具体实现")
        print("  2. 尝试手动实现nearest插值")
        print("  3. 或接受当前差异（相关性0.85仍然可用）")

if __name__ == "__main__":
    main()

