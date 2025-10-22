"""
深度调试CFM的Transformer和WaveNet层
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
    mean_diff = np.abs(t_np - m_np).mean()
    corr = np.corrcoef(t_np.flatten(), m_np.flatten())[0, 1]
    
    # 判断状态
    if max_diff < 1e-4 and corr > 0.9999:
        status = "✅"
    elif max_diff < 1e-2 and corr > 0.99:
        status = "✅"
    elif max_diff < 0.1 and corr > 0.95:
        status = "⚠️ "
    else:
        status = "❌"
    
    print(f"{status} {name:35s}: max_diff={max_diff:.6f}, mean={mean_diff:.6f}, corr={corr:.6f}")
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
    
    print(f"\n输入数据: x={x.shape}, cond={cond.shape}")
    
    # PyTorch forward - 细分步骤
    print("\n=== PyTorch DiT - 细分步骤 ===")
    with torch.no_grad():
        dit_t = s2mel.models['cfm'].estimator
        
        # 准备输入
        t1 = dit_t.t_embedder(t)
        cond_t = dit_t.cond_projection(cond)
        x_t = x.transpose(1, 2)
        prompt_x_t = prompt_x.transpose(1, 2)
        
        # Concatenate
        x_in_t = torch.cat([x_t, prompt_x_t, cond_t], dim=-1)
        x_in_t = torch.cat([x_in_t, style[:, None, :].repeat(1, seq_len, 1)], dim=-1)
        
        print(f"1. After concat: {x_in_t.shape}, mean={x_in_t.mean():.4f}")
        
        # Merge
        x_merged_t = dit_t.cond_x_merge_linear(x_in_t)
        print(f"2. After merge: {x_merged_t.shape}, mean={x_merged_t.mean():.4f}")
        
        # Transformer
        from indextts.s2mel.modules.commons import sequence_mask
        x_mask = sequence_mask(x_lens).to(x.device).unsqueeze(1)
        input_pos = dit_t.input_pos[:x_merged_t.size(1)]
        x_mask_expanded = x_mask[:, None, :].repeat(1, 1, x_merged_t.size(1), 1)
        x_res_t = dit_t.transformer(x_merged_t, t1.unsqueeze(1), input_pos, x_mask_expanded)
        print(f"3. After transformer: {x_res_t.shape}, mean={x_res_t.mean():.4f}, std={x_res_t.std():.4f}")
        
        # Skip connection
        x_skip_t = dit_t.skip_linear(torch.cat([x_res_t, x_t], dim=-1))
        print(f"4. After skip: {x_skip_t.shape}, mean={x_skip_t.mean():.4f}")
        
        # WaveNet
        x_wn_in = dit_t.conv1(x_skip_t)
        print(f"5. WaveNet input: {x_wn_in.shape}, mean={x_wn_in.mean():.4f}")
        
        x_wn_in_t2 = x_wn_in.transpose(1, 2)
        t2 = dit_t.t_embedder2(t)
        x_wn_out = dit_t.wavenet(x_wn_in_t2, x_mask, g=t2.unsqueeze(2))
        print(f"6. WaveNet output: {x_wn_out.shape}, mean={x_wn_out.mean():.4f}")
        
        # After wavenet + residual
        x_wn_out_t = x_wn_out.transpose(1, 2)
        x_with_res_t = x_wn_out_t + dit_t.res_projection(x_res_t)
        print(f"7. After WN + res: {x_with_res_t.shape}, mean={x_with_res_t.mean():.4f}")
        
        # Final layer
        x_final_t = dit_t.final_layer(x_with_res_t, t1)
        print(f"8. After final_layer: {x_final_t.shape}, mean={x_final_t.mean():.4f}")
        
        x_final_t2 = x_final_t.transpose(1, 2)
        x_conv2_t = dit_t.conv2(x_final_t2)
        print(f"9. After conv2: {x_conv2_t.shape}, mean={x_conv2_t.mean():.4f}")
        
        # Complete
        output_t = dit_t(x, prompt_x, x_lens, t, style, cond, mask_content=False)
        print(f"\nFinal PyTorch: mean={output_t.mean():.4f}, std={output_t.std():.4f}")
    
    # MLX forward - 细分步骤
    print("\n=== MLX DiT - 细分步骤 ===")
    x_mlx = torch_to_mlx(x)
    prompt_x_mlx = torch_to_mlx(prompt_x)
    cond_mlx = torch_to_mlx(cond)
    x_lens_mlx = torch_to_mlx(x_lens)
    t_mlx = torch_to_mlx(t)
    style_mlx = torch_to_mlx(style)
    
    dit_m = mlx_cfm.estimator
    
    # 准备输入
    t1_m = dit_m.t_embedder(t_mlx)
    cond_m = dit_m.cond_projection(cond_mlx)
    x_m = x_mlx.transpose(0, 2, 1)
    prompt_x_m = prompt_x_mlx.transpose(0, 2, 1)
    
    # Concatenate
    x_in_m = mx.concatenate([x_m, prompt_x_m, cond_m], axis=-1)
    style_broadcast = mx.broadcast_to(style_mlx.reshape(batch_size, 1, -1), (batch_size, seq_len, style_mlx.shape[-1]))
    x_in_m = mx.concatenate([x_in_m, style_broadcast], axis=-1)
    mx.eval(x_in_m)
    
    print(f"1. After concat: {x_in_m.shape}, mean={float(mx.mean(x_in_m)):.4f}")
    compare(x_in_t, x_in_m, "  -> Concat comparison")
    
    # Merge
    x_merged_m = dit_m.cond_x_merge_linear(x_in_m)
    mx.eval(x_merged_m)
    print(f"2. After merge: {x_merged_m.shape}, mean={float(mx.mean(x_merged_m)):.4f}")
    compare(x_merged_t, x_merged_m, "  -> Merge comparison")
    
    # Transformer
    input_pos_m = dit_m.input_pos[:x_merged_m.shape[1]]
    positions = mx.arange(seq_len).reshape(1, 1, 1, -1)
    lens_reshaped = x_lens_mlx.reshape(-1, 1, 1, 1)
    mask_expanded_m = positions < lens_reshaped
    mask_expanded_m = mx.broadcast_to(mask_expanded_m, (batch_size, 1, seq_len, seq_len))
    
    x_res_m = dit_m.transformer(x_merged_m, t1_m, input_pos=input_pos_m, mask=mask_expanded_m)
    mx.eval(x_res_m)
    print(f"3. After transformer: {x_res_m.shape}, mean={float(mx.mean(x_res_m)):.4f}, std={float(mx.std(x_res_m)):.4f}")
    compare(x_res_t, x_res_m, "  -> Transformer comparison")
    
    # Skip connection
    x_skip_m = dit_m.skip_linear(mx.concatenate([x_res_m, x_m], axis=-1))
    mx.eval(x_skip_m)
    print(f"4. After skip: {x_skip_m.shape}, mean={float(mx.mean(x_skip_m)):.4f}")
    compare(x_skip_t, x_skip_m, "  -> Skip comparison")
    
    # WaveNet
    x_wn_in_m = dit_m.conv1(x_skip_m)
    mx.eval(x_wn_in_m)
    print(f"5. WaveNet input: {x_wn_in_m.shape}, mean={float(mx.mean(x_wn_in_m)):.4f}")
    compare(x_wn_in, x_wn_in_m, "  -> WaveNet input comparison")
    
    # WaveNet (MLX)
    # MLX WaveNet input needs to be (batch, seq_len, channels)
    # But we have (batch, seq_len, wavenet_dim), so it's already correct
    
    # Create mask for WaveNet
    positions_wn = mx.arange(x_wn_in_m.shape[1]).reshape(1, 1, -1)
    lens_wn = x_lens_mlx.reshape(-1, 1, 1)
    x_mask_wn = positions_wn < lens_wn
    
    t2_m = dit_m.t_embedder2(t_mlx)
    t2_m_expanded = mx.broadcast_to(t2_m.reshape(batch_size, 1, -1), (batch_size, x_wn_in_m.shape[1], t2_m.shape[-1]))
    
    x_wn_out_m = dit_m.wavenet(x_wn_in_m, x_mask_wn, g=t2_m_expanded)
    mx.eval(x_wn_out_m)
    print(f"6. WaveNet output: {x_wn_out_m.shape}, mean={float(mx.mean(x_wn_out_m)):.4f}")
    compare(x_wn_out.transpose(1, 2), x_wn_out_m, "  -> WaveNet output comparison")
    
    # After residual
    x_with_res_m = x_wn_out_m + dit_m.res_projection(x_res_m)
    mx.eval(x_with_res_m)
    print(f"7. After WN + res: {x_with_res_m.shape}, mean={float(mx.mean(x_with_res_m)):.4f}")
    compare(x_with_res_t, x_with_res_m, "  -> After residual comparison")
    
    # Final layer
    x_final_m = dit_m.final_layer(x_with_res_m, t1_m)
    mx.eval(x_final_m)
    print(f"8. After final_layer: {x_final_m.shape}, mean={float(mx.mean(x_final_m)):.4f}")
    compare(x_final_t, x_final_m, "  -> Final layer comparison")
    
    # Conv2
    x_conv2_m = dit_m.conv2(x_final_m)
    mx.eval(x_conv2_m)
    print(f"9. After conv2: {x_conv2_m.shape}, mean={float(mx.mean(x_conv2_m)):.4f}")
    # PyTorch conv2 expects (B,C,T) and outputs (B,C,T)
    # MLX conv2 expects (B,T,C) and outputs (B,T,C)
    # So we need to transpose for comparison
    compare(x_conv2_t.transpose(1, 2), x_conv2_m, "  -> Conv2 comparison")
    
    # Complete
    output_m = dit_m(x_mlx, prompt_x_mlx, x_lens_mlx, t_mlx, style_mlx, cond_mlx, mask_content=False)
    mx.eval(output_m)
    print(f"\nFinal MLX: mean={float(mx.mean(output_m)):.4f}, std={float(mx.std(output_m)):.4f}")
    
    # 最终对比
    print("\n=== 最终对比 ===")
    compare(output_t, output_m, "Final Output")

if __name__ == "__main__":
    main()

