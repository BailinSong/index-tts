"""
对比标准MLX WaveNet和改进版MLX WaveNet
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
    """快速对比"""
    t_np = torch_val.detach().cpu().numpy() if hasattr(torch_val, 'detach') else np.array(torch_val)
    m_np = np.array(mlx_val)
    
    if t_np.shape != m_np.shape:
        print(f"❌ {name}: shape mismatch {t_np.shape} vs {m_np.shape}")
        return None, None
    
    max_diff = np.abs(t_np - m_np).max()
    mean_diff = np.abs(t_np - m_np).mean()
    corr = np.corrcoef(t_np.flatten(), m_np.flatten())[0, 1]
    
    if max_diff < 0.01 and corr > 0.99:
        status = "✅"
    elif max_diff < 0.1 and corr > 0.95:
        status = "⚠️ "
    else:
        status = "❌"
    
    print(f"{status} {name:35s}: max_diff={max_diff:.6f}, mean={mean_diff:.6f}, corr={corr:.6f}")
    return max_diff, corr

def main():
    from indextts.s2mel.modules.mlx_wavenet import MLXWaveNet
    from indextts.s2mel.modules.mlx_wavenet_improved import MLXWaveNetImproved
    from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
    
    # 加载配置
    cfg = OmegaConf.load("checkpoints/config.yaml")
    
    # 加载PyTorch模型
    print("加载PyTorch WaveNet...")
    s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
    s2mel_path = os.path.join("checkpoints", cfg.s2mel_checkpoint)
    s2mel, _, _, _ = load_checkpoint2(s2mel, None, s2mel_path, load_only_params=True, ignore_modules=[], is_distributed=False)
    s2mel.eval()
    
    pytorch_wavenet = s2mel.models['cfm'].estimator.wavenet
    s2mel_state_dict = s2mel.state_dict()
    s2mel_state_dict_np = {k: v.cpu().numpy() for k, v in s2mel_state_dict.items()}
    
    # 创建两个MLX版本
    wavenet_cfg = cfg.s2mel.wavenet
    
    print("\n创建标准MLX WaveNet...")
    mlx_wn_standard = MLXWaveNet(
        hidden_channels=wavenet_cfg.hidden_dim,
        kernel_size=wavenet_cfg.kernel_size,
        dilation_rate=wavenet_cfg.dilation_rate,
        n_layers=wavenet_cfg.num_layers,
        gin_channels=wavenet_cfg.hidden_dim,
        p_dropout=wavenet_cfg.p_dropout
    )
    
    # Load weights with weight_norm handling
    print("  加载权重到标准版...")
    prefix = "models.cfm.estimator.wavenet."
    
    # 检查可用的keys
    available_keys = [k for k in s2mel_state_dict_np.keys() if k.startswith(prefix)]
    print(f"  找到 {len(available_keys)} 个WaveNet相关的key")
    if len(available_keys) > 0:
        print(f"  示例: {available_keys[:3]}")
    
    from indextts.s2mel.modules.mlx_dit_weights import load_weight_norm, convert_conv1d_weight
    
    loaded_std = 0
    
    # in_layers - 需要处理 weight_norm
    for i in range(mlx_wn_standard.n_layers):
        # PyTorch路径: in_layers.{i}.conv.conv.weight_g/weight_v
        w = load_weight_norm(s2mel_state_dict_np, f"{prefix}in_layers.{i}.conv.conv")
        if w is not None:
            mlx_wn_standard.in_layers[i].weight = mx.array(convert_conv1d_weight(w))
            loaded_std += 1
        
        b_key = f"{prefix}in_layers.{i}.conv.conv.bias"
        if b_key in s2mel_state_dict_np:
            mlx_wn_standard.in_layers[i].bias = mx.array(s2mel_state_dict_np[b_key])
            loaded_std += 1
    
    # res_skip_layers - 需要处理 weight_norm
    for i in range(mlx_wn_standard.n_layers):
        w = load_weight_norm(s2mel_state_dict_np, f"{prefix}res_skip_layers.{i}.conv.conv")
        if w is not None:
            mlx_wn_standard.res_skip_layers[i].weight = mx.array(convert_conv1d_weight(w))
            loaded_std += 1
        
        b_key = f"{prefix}res_skip_layers.{i}.conv.conv.bias"
        if b_key in s2mel_state_dict_np:
            mlx_wn_standard.res_skip_layers[i].bias = mx.array(s2mel_state_dict_np[b_key])
            loaded_std += 1
    
    # cond_layer - 需要处理 weight_norm
    if mlx_wn_standard.cond_layer is not None:
        w = load_weight_norm(s2mel_state_dict_np, f"{prefix}cond_layer.conv.conv")
        if w is not None:
            mlx_wn_standard.cond_layer.weight = mx.array(convert_conv1d_weight(w))
            loaded_std += 1
        
        b_key = f"{prefix}cond_layer.conv.conv.bias"
        if b_key in s2mel_state_dict_np:
            mlx_wn_standard.cond_layer.bias = mx.array(s2mel_state_dict_np[b_key])
            loaded_std += 1
    
    print(f"  加载了 {loaded_std} 个权重")
    
    print("\n创建改进版MLX WaveNet...")
    mlx_wn_improved = MLXWaveNetImproved(
        hidden_channels=wavenet_cfg.hidden_dim,
        kernel_size=wavenet_cfg.kernel_size,
        dilation_rate=wavenet_cfg.dilation_rate,
        n_layers=wavenet_cfg.num_layers,
        gin_channels=wavenet_cfg.hidden_dim,
        p_dropout=wavenet_cfg.p_dropout,
        causal=False
    )
    
    # Load weights
    loaded_imp = mlx_wn_improved.load_weights_from_pytorch(s2mel_state_dict_np, prefix)
    
    # 准备测试数据
    torch.manual_seed(42)
    np.random.seed(42)
    
    batch_size = 1
    seq_len = 30
    hidden_channels = wavenet_cfg.hidden_dim
    
    # WaveNet期望 PyTorch格式: (batch, channels, seq_len)
    # MLX格式: (batch, seq_len, channels)
    x_torch = torch.randn(batch_size, hidden_channels, seq_len)
    x_mask_torch = torch.ones(batch_size, 1, seq_len).bool()
    g_torch = torch.randn(batch_size, hidden_channels, 1)
    
    print(f"\n输入数据:")
    print(f"  x: {x_torch.shape}")
    print(f"  mask: {x_mask_torch.shape}")
    print(f"  g: {g_torch.shape}")
    
    # PyTorch推理
    print("\n=== PyTorch WaveNet ===")
    with torch.no_grad():
        output_torch = pytorch_wavenet(x_torch, x_mask_torch, g=g_torch)
    print(f"Output: {output_torch.shape}, mean={output_torch.mean():.4f}, std={output_torch.std():.4f}")
    
    # MLX推理 - 标准版
    print("\n=== MLX WaveNet (标准版) ===")
    x_mlx = torch_to_mlx(x_torch).transpose(0, 2, 1)  # -> (batch, seq, channels)
    x_mask_mlx = torch_to_mlx(x_mask_torch)
    g_mlx = torch_to_mlx(g_torch).transpose(0, 2, 1)  # -> (batch, seq, channels)
    
    output_std = mlx_wn_standard(x_mlx, x_mask_mlx, g=g_mlx)
    mx.eval(output_std)
    print(f"Output: {output_std.shape}, mean={float(mx.mean(output_std)):.4f}, std={float(mx.std(output_std)):.4f}")
    
    # 对比
    compare(output_torch.transpose(1, 2), output_std, "标准版 vs PyTorch")
    
    # MLX推理 - 改进版
    print("\n=== MLX WaveNet (改进版 - SConv1d-like) ===")
    output_imp = mlx_wn_improved(x_mlx, x_mask_mlx, g=g_mlx)
    mx.eval(output_imp)
    print(f"Output: {output_imp.shape}, mean={float(mx.mean(output_imp)):.4f}, std={float(mx.std(output_imp)):.4f}")
    
    # 对比
    max_diff, corr = compare(output_torch.transpose(1, 2), output_imp, "改进版 vs PyTorch")
    
    # 对比两个MLX版本
    print("\n=== MLX版本对比 ===")
    compare(output_std, output_imp, "标准版 vs 改进版")
    
    print("\n" + "="*70)
    if corr and corr > 0.99:
        print("✅ 改进版显著提升了一致性！")
    elif corr and corr > 0.95:
        print("⚠️  改进版略有提升")
    else:
        print("❌ 改进版未能显著提升一致性")
    print("="*70)

if __name__ == "__main__":
    main()

