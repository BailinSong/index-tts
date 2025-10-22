"""
逐层调试CFM DiT的差异
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
    t_np = torch_val.detach().cpu().numpy()
    m_np = np.array(mlx_val)
    max_diff = np.abs(t_np - m_np).max()
    corr = np.corrcoef(t_np.flatten(), m_np.flatten())[0, 1]
    print(f"{name:30s}: max_diff={max_diff:.6f}, corr={corr:.6f}")
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
    in_channels = 80
    content_dim = 512
    
    x = torch.randn(batch_size, in_channels, seq_len)
    prompt_x = torch.zeros_like(x)
    prompt_x[:, :, :10] = torch.randn(batch_size, in_channels, 10)
    cond = torch.randn(batch_size, seq_len, content_dim)
    x_lens = torch.tensor([seq_len])
    t = torch.tensor([0.5])
    style = torch.randn(batch_size, 192)
    
    print(f"\n输入数据:")
    print(f"  x: {x.shape}")
    print(f"  cond: {cond.shape}")
    print(f"  style: {style.shape}")
    
    # PyTorch 逐层
    print("\n=== PyTorch DiT Forward ===")
    with torch.no_grad():
        dit_torch = s2mel.models['cfm'].estimator
        
        # 1. x_embedder
        x_emb_t = dit_torch.x_embedder(x.transpose(1, 2))  # Input (B,T,C)
        print(f"1. x_embedder: {x_emb_t.shape}, mean={x_emb_t.mean():.4f}")
        
        # 2. cond_projection
        cond_proj_t = dit_torch.cond_projection(cond)
        print(f"2. cond_projection: {cond_proj_t.shape}, mean={cond_proj_t.mean():.4f}")
        
        # 3. Timestep embedder
        t_emb_t = dit_torch.t_embedder(t)
        print(f"3. t_embedder: {t_emb_t.shape}, mean={t_emb_t.mean():.4f}")
        
        # 完整forward
        output_t = dit_torch(x, prompt_x, x_lens, t, style, cond, mask_content=False)
        print(f"\nFinal PyTorch: {output_t.shape}, mean={output_t.mean():.4f}, std={output_t.std():.4f}")
    
    # MLX 逐层
    print("\n=== MLX DiT Forward ===")
    x_mlx = torch_to_mlx(x)
    prompt_x_mlx = torch_to_mlx(prompt_x)
    cond_mlx = torch_to_mlx(cond)
    x_lens_mlx = torch_to_mlx(x_lens)
    t_mlx = torch_to_mlx(t)
    style_mlx = torch_to_mlx(style)
    
    dit_mlx = mlx_cfm.estimator
    
    # 1. x_embedder
    x_emb_m = dit_mlx.x_embedder(x_mlx.transpose(0, 2, 1))
    mx.eval(x_emb_m)
    print(f"1. x_embedder: {x_emb_m.shape}, mean={float(mx.mean(x_emb_m)):.4f}")
    compare(x_emb_t, x_emb_m, "  -> x_embedder comparison")
    
    # 2. cond_projection
    print(f"   MLX has cond_projection: {hasattr(dit_mlx, 'cond_projection')}")
    print(f"   MLX has cond_embedder: {hasattr(dit_mlx, 'cond_embedder')}")
    cond_proj_m = dit_mlx.cond_projection(cond_mlx)
    mx.eval(cond_proj_m)
    print(f"2. cond_projection: {cond_proj_m.shape}, mean={float(mx.mean(cond_proj_m)):.4f}")
    compare(cond_proj_t, cond_proj_m, "  -> cond_projection comparison")
    
    # 3. t_embedder
    t_emb_m = dit_mlx.t_embedder(t_mlx)
    mx.eval(t_emb_m)
    print(f"3. t_embedder: {t_emb_m.shape}, mean={float(mx.mean(t_emb_m)):.4f}")
    compare(t_emb_t, t_emb_m, "  -> t_embedder comparison")
    
    # 完整forward
    output_m = dit_mlx(x_mlx, prompt_x_mlx, x_lens_mlx, t_mlx, style_mlx, cond_mlx, mask_content=False)
    mx.eval(output_m)
    print(f"\nFinal MLX: {output_m.shape}, mean={float(mx.mean(output_m)):.4f}, std={float(mx.std(output_m)):.4f}")
    
    # 最终对比
    print("\n=== 最终对比 ===")
    max_diff, corr = compare(output_t, output_m, "Final Output")
    
    if corr < 0.95:
        print("\n⚠️  显著差异！可能的原因：")
        print("  1. Transformer层的实现差异")
        print("  2. WaveNet的实现差异")
        print("  3. LayerNorm/GroupNorm的数值精度")
        print("  4. 激活函数的实现差异")

if __name__ == "__main__":
    main()

