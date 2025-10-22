"""
深度调试WaveNet的每一层差异
"""

import torch
import numpy as np
from omegaconf import OmegaConf
import os
import sys
import mlx.core as mx
import mlx.nn as nn

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
    
    print(f"{status} {name:45s}: max={max_diff:.6f}, mean={mean_diff:.6f}, corr={corr:.6f}")
    return max_diff, corr

def main():
    from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
    from indextts.s2mel.modules.mlx_dit_weights import load_weight_norm, convert_conv1d_weight
    
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
    
    # 创建MLX WaveNet
    from indextts.s2mel.modules.mlx_wavenet import MLXWaveNet
    wavenet_cfg = cfg.s2mel.wavenet
    
    print("\n创建MLX WaveNet...")
    mlx_wn = MLXWaveNet(
        hidden_channels=wavenet_cfg.hidden_dim,
        kernel_size=wavenet_cfg.kernel_size,
        dilation_rate=wavenet_cfg.dilation_rate,
        n_layers=wavenet_cfg.num_layers,
        gin_channels=wavenet_cfg.hidden_dim,
        p_dropout=wavenet_cfg.p_dropout
    )
    
    # 加载权重
    print("加载权重...")
    prefix = "models.cfm.estimator.wavenet."
    loaded = 0
    
    # in_layers
    for i in range(mlx_wn.n_layers):
        w = load_weight_norm(s2mel_state_dict_np, f"{prefix}in_layers.{i}.conv.conv")
        if w is not None:
            mlx_wn.in_layers[i].weight = mx.array(convert_conv1d_weight(w))
            loaded += 1
        
        b_key = f"{prefix}in_layers.{i}.conv.conv.bias"
        if b_key in s2mel_state_dict_np:
            mlx_wn.in_layers[i].bias = mx.array(s2mel_state_dict_np[b_key])
            loaded += 1
    
    # res_skip_layers
    for i in range(mlx_wn.n_layers):
        w = load_weight_norm(s2mel_state_dict_np, f"{prefix}res_skip_layers.{i}.conv.conv")
        if w is not None:
            mlx_wn.res_skip_layers[i].weight = mx.array(convert_conv1d_weight(w))
            loaded += 1
        
        b_key = f"{prefix}res_skip_layers.{i}.conv.conv.bias"
        if b_key in s2mel_state_dict_np:
            mlx_wn.res_skip_layers[i].bias = mx.array(s2mel_state_dict_np[b_key])
            loaded += 1
    
    # cond_layer
    if mlx_wn.cond_layer is not None:
        w = load_weight_norm(s2mel_state_dict_np, f"{prefix}cond_layer.conv.conv")
        if w is not None:
            mlx_wn.cond_layer.weight = mx.array(convert_conv1d_weight(w))
            loaded += 1
        
        b_key = f"{prefix}cond_layer.conv.conv.bias"
        if b_key in s2mel_state_dict_np:
            mlx_wn.cond_layer.bias = mx.array(s2mel_state_dict_np[b_key])
            loaded += 1
    
    print(f"  加载了 {loaded} 个权重")
    
    # 准备测试数据
    torch.manual_seed(42)
    np.random.seed(42)
    
    batch_size = 1
    seq_len = 30
    hidden_channels = wavenet_cfg.hidden_dim
    
    x_torch = torch.randn(batch_size, hidden_channels, seq_len)
    x_mask_torch = torch.ones(batch_size, 1, seq_len).bool()
    g_torch = torch.randn(batch_size, hidden_channels, 1)
    
    print(f"\n输入数据: x={x_torch.shape}, g={g_torch.shape}")
    
    # PyTorch 逐层执行
    print("\n" + "="*70)
    print("PyTorch WaveNet - 逐层执行")
    print("="*70)
    
    with torch.no_grad():
        # Process global conditioning
        g_cond_t = pytorch_wavenet.cond_layer(g_torch)
        print(f"0. cond_layer: {g_cond_t.shape}, mean={g_cond_t.mean():.6f}")
        
        x_t = x_torch
        output_t = torch.zeros_like(x_torch)
        n_channels_tensor = torch.IntTensor([hidden_channels])
        
        for i in range(pytorch_wavenet.n_layers):
            # In layer
            x_in_t = pytorch_wavenet.in_layers[i](x_t)
            print(f"\n{i}.1 in_layer[{i}]: {x_in_t.shape}, mean={x_in_t.mean():.6f}, std={x_in_t.std():.6f}")
            
            # Get conditioning
            cond_offset = i * 2 * hidden_channels
            g_l_t = g_cond_t[:, cond_offset:cond_offset + 2 * hidden_channels, :]
            print(f"{i}.2 g_l[{i}]: mean={g_l_t.mean():.6f}")
            
            # Fused activation
            from indextts.s2mel.modules.commons import fused_add_tanh_sigmoid_multiply
            acts_t = fused_add_tanh_sigmoid_multiply(x_in_t, g_l_t, n_channels_tensor)
            print(f"{i}.3 acts[{i}]: {acts_t.shape}, mean={acts_t.mean():.6f}, std={acts_t.std():.6f}")
            
            # Res/skip
            res_skip_t = pytorch_wavenet.res_skip_layers[i](acts_t)
            print(f"{i}.4 res_skip[{i}]: {res_skip_t.shape}, mean={res_skip_t.mean():.6f}")
            
            if i < pytorch_wavenet.n_layers - 1:
                res_acts_t = res_skip_t[:, :hidden_channels, :]
                skip_acts_t = res_skip_t[:, hidden_channels:, :]
                x_t = (x_t + res_acts_t) * x_mask_torch
                output_t = output_t + skip_acts_t
                print(f"{i}.5 x after res: mean={x_t.mean():.6f}")
                print(f"{i}.6 output累积: mean={output_t.mean():.6f}")
            else:
                output_t = output_t + res_skip_t
                print(f"{i}.5 final output: mean={output_t.mean():.6f}")
        
        final_t = output_t * x_mask_torch
        print(f"\nFinal PyTorch WaveNet: {final_t.shape}, mean={final_t.mean():.6f}, std={final_t.std():.6f}")
    
    # MLX 逐层执行
    print("\n" + "="*70)
    print("MLX WaveNet - 逐层执行")
    print("="*70)
    
    x_mlx = torch_to_mlx(x_torch).transpose(0, 2, 1)  # (B,T,C)
    x_mask_mlx = torch_to_mlx(x_mask_torch)
    x_mask_mlx_t = x_mask_mlx.transpose(0, 2, 1)  # (B,T,1)
    g_mlx = torch_to_mlx(g_torch).transpose(0, 2, 1)  # (B,T,C)
    
    # Process global conditioning
    if mlx_wn.cond_layer is not None:
        g_cond_m = mlx_wn.cond_layer(g_mlx)
        mx.eval(g_cond_m)
        print(f"0. cond_layer: {g_cond_m.shape}, mean={float(mx.mean(g_cond_m)):.6f}")
        compare(g_cond_t.transpose(1, 2), g_cond_m, "  -> cond_layer comparison")
    else:
        g_cond_m = None
    
    x_m = x_mlx
    output_m = mx.zeros_like(x_mlx)
    
    for i in range(mlx_wn.n_layers):
        # In layer
        x_in_m = mlx_wn.in_layers[i](x_m * x_mask_mlx_t)
        mx.eval(x_in_m)
        print(f"\n{i}.1 in_layer[{i}]: {x_in_m.shape}, mean={float(mx.mean(x_in_m)):.6f}, std={float(mx.std(x_in_m)):.6f}")
        
        # 对比in_layer输出
        x_in_t_transposed = pytorch_wavenet.in_layers[i](x_torch if i == 0 else x_t).transpose(1, 2)
        compare(x_in_t_transposed, x_in_m, f"  -> in_layer[{i}] comparison")
        
        # Get conditioning
        if g_cond_m is not None:
            cond_offset = i * 2 * hidden_channels
            g_l_m = g_cond_m[:, :, cond_offset:cond_offset + 2 * hidden_channels]
        else:
            g_l_m = mx.zeros_like(x_in_m)
        
        # Fused activation
        in_act = x_in_m + g_l_m
        t_act = mx.tanh(in_act[:, :, :hidden_channels])
        s_act = mx.sigmoid(in_act[:, :, hidden_channels:])
        acts_m = t_act * s_act
        mx.eval(acts_m)
        print(f"{i}.3 acts[{i}]: {acts_m.shape}, mean={float(mx.mean(acts_m)):.6f}, std={float(mx.std(acts_m)):.6f}")
        
        # Res/skip
        res_skip_m = mlx_wn.res_skip_layers[i](acts_m)
        mx.eval(res_skip_m)
        print(f"{i}.4 res_skip[{i}]: {res_skip_m.shape}, mean={float(mx.mean(res_skip_m)):.6f}")
        
        if i < mlx_wn.n_layers - 1:
            res_acts_m = res_skip_m[:, :, :hidden_channels]
            skip_acts_m = res_skip_m[:, :, hidden_channels:]
            x_m = (x_m + res_acts_m) * x_mask_mlx_t
            output_m = output_m + skip_acts_m
            print(f"{i}.5 x after res: mean={float(mx.mean(x_m)):.6f}")
            print(f"{i}.6 output累积: mean={float(mx.mean(output_m)):.6f}")
        else:
            output_m = output_m + res_skip_m
            print(f"{i}.5 final output: mean={float(mx.mean(output_m)):.6f}")
    
    final_m = output_m * x_mask_mlx_t
    mx.eval(final_m)
    print(f"\nFinal MLX WaveNet: {final_m.shape}, mean={float(mx.mean(final_m)):.6f}, std={float(mx.std(final_m)):.6f}")
    
    # 最终对比
    print("\n" + "="*70)
    print("最终对比")
    print("="*70)
    max_diff, corr = compare(final_t.transpose(1, 2), final_m, "Final WaveNet Output")
    
    if corr and corr < 0.99:
        print("\n💡 需要检查:")
        print("  1. in_layers的卷积实现（dilation处理）")
        print("  2. res_skip_layers的1x1卷积")
        print("  3. 门控激活函数的数值精度")
        print("  4. 累积误差的来源")

if __name__ == "__main__":
    main()

