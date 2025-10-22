"""
调试final_layer和conv2的差异
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
    t_np = torch_val.detach().cpu().numpy() if hasattr(torch_val, 'detach') else np.array(torch_val)
    m_np = np.array(mlx_val)
    
    if t_np.shape != m_np.shape:
        print(f"❌ {name}: shape mismatch {t_np.shape} vs {m_np.shape}")
        return None, None
    
    max_diff = np.abs(t_np - m_np).max()
    mean_diff = np.abs(t_np - m_np).mean()
    corr = np.corrcoef(t_np.flatten(), m_np.flatten())[0, 1]
    
    if max_diff < 1e-3 and corr > 0.999:
        status = "✅"
    elif corr > 0.99:
        status = "✅"
    elif corr > 0.95:
        status = "⚠️ "
    else:
        status = "❌"
    
    print(f"{status} {name:40s}: max={max_diff:.6f}, mean={mean_diff:.6f}, corr={corr:.6f}")
    return max_diff, corr

def main():
    from indextts.s2mel.modules.mlx_cfm import MLXCFM
    from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
    
    # 加载配置
    cfg = OmegaConf.load("checkpoints/config.yaml")
    
    # 加载PyTorch模型
    print("加载PyTorch模型...")
    s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
    s2mel_path = os.path.join("checkpoints", cfg.s2mel_checkpoint)
    s2mel, _, _, _ = load_checkpoint2(s2mel, None, s2mel_path, load_only_params=True, ignore_modules=[], is_distributed=False)
    s2mel.eval()
    s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
    
    # 创建MLX CFM
    print("创建MLX CFM...")
    mlx_cfm = MLXCFM(cfg.s2mel)
    s2mel_state_dict = s2mel.state_dict()
    s2mel_state_dict_np = {k: v.cpu().numpy() for k, v in s2mel_state_dict.items()}
    mlx_cfm.load_weights_from_pytorch(s2mel_state_dict_np, prefix="models.cfm.")
    
    # 准备测试数据
    torch.manual_seed(42)
    np.random.seed(42)
    
    batch_size = 1
    seq_len = 30
    
    # 创建一个中间状态的输入（WaveNet输出后）
    # 这样我们可以单独测试final_layer
    x_before_final = torch.randn(batch_size, seq_len, 512)  # (B, T, wavenet_dim)
    t = torch.tensor([0.5])
    
    print(f"\n输入数据: x={x_before_final.shape}, t={t}")
    
    # PyTorch final_layer
    print("\n=== PyTorch Final Layer ===")
    with torch.no_grad():
        dit_t = s2mel.models['cfm'].estimator
        t1_t = dit_t.t_embedder(t)
        
        # Final layer
        final_out_t = dit_t.final_layer(x_before_final, t1_t)
        print(f"final_layer output: {final_out_t.shape}, mean={final_out_t.mean():.6f}, std={final_out_t.std():.6f}")
        
        # 查看内部步骤
        # norm_final
        x_norm_t = dit_t.final_layer.norm_final(x_before_final)
        print(f"  1. norm_final: mean={x_norm_t.mean():.6f}, std={x_norm_t.std():.6f}")
        
        # adaLN_modulation
        ada_out_t = dit_t.final_layer.adaLN_modulation(t1_t)
        shift_t, scale_t = ada_out_t.chunk(2, dim=1)
        print(f"  2. shift: mean={shift_t.mean():.6f}, scale: mean={scale_t.mean():.6f}")
        
        # modulate
        from indextts.s2mel.modules.diffusion_transformer import modulate
        x_mod_t = modulate(x_norm_t, shift_t, scale_t)
        print(f"  3. modulate: mean={x_mod_t.mean():.6f}, std={x_mod_t.std():.6f}")
        
        # linear
        x_linear_t = dit_t.final_layer.linear(x_mod_t)
        print(f"  4. linear: mean={x_linear_t.mean():.6f}, std={x_linear_t.std():.6f}")
    
    # MLX final_layer
    print("\n=== MLX Final Layer ===")
    x_before_final_mlx = torch_to_mlx(x_before_final)
    t1_m = mlx_cfm.estimator.t_embedder(torch_to_mlx(t))
    mx.eval(t1_m)
    
    # Final layer
    final_out_m = mlx_cfm.estimator.final_layer(x_before_final_mlx, t1_m)
    mx.eval(final_out_m)
    print(f"final_layer output: {final_out_m.shape}, mean={float(mx.mean(final_out_m)):.6f}, std={float(mx.std(final_out_m)):.6f}")
    
    # 查看内部步骤
    # norm_final
    x_norm_m = mlx_cfm.estimator.final_layer.norm_final(x_before_final_mlx)
    mx.eval(x_norm_m)
    print(f"  1. norm_final: mean={float(mx.mean(x_norm_m)):.6f}, std={float(mx.std(x_norm_m)):.6f}")
    compare(x_norm_t, x_norm_m, "    -> norm comparison")
    
    # adaLN_0
    ada_out_m = mlx_cfm.estimator.final_layer.adaLN_0(t1_m)
    mx.eval(ada_out_m)
    ada_out_m = nn.silu(ada_out_m)
    mx.eval(ada_out_m)
    shift_m = ada_out_m[:, :512]
    scale_m = ada_out_m[:, 512:]
    mx.eval(shift_m)
    mx.eval(scale_m)
    print(f"  2. shift: mean={float(mx.mean(shift_m)):.6f}, scale: mean={float(mx.mean(scale_m)):.6f}")
    
    # Compare shift and scale
    compare(shift_t, shift_m, "    -> shift comparison")
    compare(scale_t, scale_m, "    -> scale comparison")
    
    # modulate
    from indextts.s2mel.modules.mlx_cfm import mlx_modulate
    x_mod_m = mlx_modulate(x_norm_m, shift_m, scale_m)
    mx.eval(x_mod_m)
    print(f"  3. modulate: mean={float(mx.mean(x_mod_m)):.6f}, std={float(mx.std(x_mod_m)):.6f}")
    compare(x_mod_t, x_mod_m, "    -> modulate comparison")
    
    # linear
    x_linear_m = mlx_cfm.estimator.final_layer.linear(x_mod_m)
    mx.eval(x_linear_m)
    print(f"  4. linear: mean={float(mx.mean(x_linear_m)):.6f}, std={float(mx.std(x_linear_m)):.6f}")
    compare(x_linear_t, x_linear_m, "    -> linear comparison")
    
    # 最终对比
    print("\n最终对比:")
    compare(final_out_t, final_out_m, "Final Layer Output")

if __name__ == "__main__":
    main()

