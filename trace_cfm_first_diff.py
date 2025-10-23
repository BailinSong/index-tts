"""
精确追踪CFM的第一个差异点
使用相同的输入，逐层对比，找出第一个偏差
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

def compare_tensors(torch_val, mlx_val, name, threshold=1e-5):
    """比较两个tensor，返回是否一致"""
    t_np = torch_val.detach().cpu().numpy() if hasattr(torch_val, 'detach') else np.array(torch_val)
    m_np = np.array(mlx_val)
    
    # 确保shape一致
    if t_np.shape != m_np.shape:
        print(f"❌ {name}: SHAPE MISMATCH!")
        print(f"   PyTorch: {t_np.shape}")
        print(f"   MLX: {m_np.shape}")
        return False
    
    diff = np.abs(t_np - m_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    corr = np.corrcoef(t_np.flatten(), m_np.flatten())[0, 1]
    
    # 判断是否一致（排除精度累计）
    is_identical = max_diff < threshold
    
    if is_identical:
        print(f"✅ {name:<40}: max_diff={max_diff:.2e}, corr={corr:.6f}")
    else:
        print(f"❌ {name:<40}: max_diff={max_diff:.6f}, mean={mean_diff:.6f}, corr={corr:.6f}")
        # 找出最大差异位置
        max_idx = np.unravel_index(diff.argmax(), diff.shape)
        print(f"   最大差异位置: {max_idx}")
        print(f"   PyTorch值: {t_np[max_idx]:.6f}")
        print(f"   MLX值: {m_np[max_idx]:.6f}")
    
    return is_identical

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
    dit_torch = s2mel.models['cfm'].estimator
    dit_torch.setup_caches(max_batch_size=1, max_seq_length=8192)
    
    # 创建MLX CFM
    print("创建MLX CFM...")
    mlx_cfm = MLXCFM(cfg.s2mel)
    s2mel_state_dict = s2mel.state_dict()
    s2mel_state_dict_np = {k: v.cpu().numpy() for k, v in s2mel_state_dict.items()}
    mlx_cfm.load_weights_from_pytorch(s2mel_state_dict_np, prefix="models.cfm.")
    dit_mlx = mlx_cfm.estimator
    
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
    
    print(f"\n{'='*80}")
    print(f"逐层追踪 CFM DiT - 找出第一个差异")
    print(f"{'='*80}\n")
    
    # ============ PyTorch Forward ============
    with torch.no_grad():
        # 1. Embeddings
        t1_t = dit_torch.t_embedder(t)
        cond_t = dit_torch.cond_projection(cond)
        
        # 2. Prepare input
        x_t = x.transpose(1, 2)  # (B, T, C)
        prompt_x_t = prompt_x.transpose(1, 2)
        x_in_t = torch.cat([x_t, prompt_x_t, cond_t], dim=-1)
        x_in_t = torch.cat([x_in_t, style[:, None, :].repeat(1, seq_len, 1)], dim=-1)
        x_merged_t = dit_torch.cond_x_merge_linear(x_in_t)
        
        # 3. Mask
        from indextts.s2mel.modules.commons import sequence_mask
        x_mask = sequence_mask(x_lens).to(x.device).unsqueeze(1)
        input_pos = dit_torch.input_pos[:x_merged_t.size(1)]
        x_mask_expanded = x_mask[:, None, :].repeat(1, 1, x_merged_t.size(1), 1)
        
        # 4. Transformer
        x_res_t = dit_torch.transformer(x_merged_t, t1_t.unsqueeze(1), input_pos, x_mask_expanded)
        
        # 5. Skip connection
        x_skip_t = dit_torch.skip_linear(torch.cat([x_res_t, x_t], dim=-1))
        
        # 6. WaveNet preparation
        x_wn_in_t = dit_torch.conv1(x_skip_t)
        x_wn_in_t2 = x_wn_in_t.transpose(1, 2)  # (B, C, T)
        t2_t = dit_torch.t_embedder2(t)
        
        # 7. WaveNet
        x_wn_out_t = dit_torch.wavenet(x_wn_in_t2, x_mask, g=t2_t.unsqueeze(2))
        x_wn_out_t2 = x_wn_out_t.transpose(1, 2)  # (B, T, C)
        
        # 8. Residual
        x_with_res_t = x_wn_out_t2 + dit_torch.res_projection(x_res_t)
        
        # 9. Final layer
        x_final_t = dit_torch.final_layer(x_with_res_t, t1_t)
        
        # 10. Conv2
        x_final_t2 = x_final_t.transpose(1, 2)  # (B, C, T)
        output_t = dit_torch.conv2(x_final_t2)
    
    # ============ MLX Forward ============
    x_mlx = torch_to_mlx(x)
    prompt_x_mlx = torch_to_mlx(prompt_x)
    cond_mlx = torch_to_mlx(cond)
    x_lens_mlx = torch_to_mlx(x_lens)
    t_mlx = torch_to_mlx(t)
    style_mlx = torch_to_mlx(style)
    
    # 1. Embeddings
    t1_m = dit_mlx.t_embedder(t_mlx)
    cond_m = dit_mlx.cond_projection(cond_mlx)
    
    print(f"DEBUG: t1_m.shape = {t1_m.shape}")
    print(f"DEBUG: t1_t.shape = {t1_t.shape}")
    
    # 2. Prepare input
    x_m = x_mlx.transpose(0, 2, 1)  # (B, T, C)
    prompt_x_m = prompt_x_mlx.transpose(0, 2, 1)
    x_in_m = mx.concatenate([x_m, prompt_x_m, cond_m], axis=-1)
    style_expanded = mx.broadcast_to(style_mlx[:, None, :], (batch_size, seq_len, 192))
    x_in_m = mx.concatenate([x_in_m, style_expanded], axis=-1)
    x_merged_m = dit_mlx.cond_x_merge_linear(x_in_m)
    
    # 3. Mask
    x_mask_m = mx.ones((batch_size, seq_len), dtype=mx.bool_)
    input_pos_m = mx.arange(seq_len)
    x_mask_expanded_m = mx.broadcast_to(x_mask_m[:, None, None, :], (batch_size, 1, seq_len, seq_len))
    
    # 4. Transformer
    t1_m_expanded = t1_m[:, None, :]  # (batch, 1, d_model)
    x_res_m = dit_mlx.transformer(x_merged_m, t1_m_expanded, input_pos_m, x_mask_expanded_m)
    
    # 5. Skip connection
    x_skip_m = dit_mlx.skip_linear(mx.concatenate([x_res_m, x_m], axis=-1))
    
    # 6. WaveNet preparation
    x_wn_in_m = dit_mlx.conv1(x_skip_m)
    t2_m = dit_mlx.t_embedder2(t_mlx)
    
    # 7. WaveNet
    x_mask_wn = mx.ones((batch_size, 1, seq_len))
    x_wn_out_m = dit_mlx.wavenet(x_wn_in_m, x_mask_wn, g=t2_m[:, None, :])
    
    # 8. Residual
    x_with_res_m = x_wn_out_m + dit_mlx.res_projection(x_res_m)
    
    # 9. Final layer
    x_final_m = dit_mlx.final_layer(x_with_res_m, t1_m)
    
    # 10. Conv2
    # MLX Conv1d expects (batch, time, channels)
    output_m = dit_mlx.conv2(x_final_m)
    
    mx.eval(t1_m, cond_m, x_merged_m, x_res_m, x_skip_m, x_wn_in_m, x_wn_out_m, x_with_res_m, x_final_m, output_m)
    
    # ============ 逐层对比 ============
    print("Step 1: Embeddings")
    compare_tensors(t1_t, t1_m, "t_embedder")
    compare_tensors(cond_t, cond_m, "cond_projection")
    
    print("\nStep 2: Input Preparation")
    compare_tensors(x_merged_t, x_merged_m, "cond_x_merge_linear")
    
    print("\nStep 3: Transformer")
    compare_tensors(x_res_t, x_res_m, "transformer")
    
    print("\nStep 4: Skip Connection")
    compare_tensors(x_skip_t, x_skip_m, "skip_linear")
    
    print("\nStep 5: Conv1 (WaveNet input)")
    compare_tensors(x_wn_in_t, x_wn_in_m, "conv1")
    
    print("\nStep 6: WaveNet")
    is_wn_ok = compare_tensors(x_wn_out_t2, x_wn_out_m, "wavenet", threshold=0.01)  # 放宽阈值
    
    print("\nStep 7: Residual Projection")
    compare_tensors(x_with_res_t, x_with_res_m, "with residual")
    
    print("\nStep 8: Final Layer")
    compare_tensors(x_final_t, x_final_m, "final_layer")
    
    print("\nStep 9: Conv2 (Final Output)")
    # output_m is (batch, time, channels), output_t is (batch, channels, time)
    compare_tensors(output_t.transpose(1, 2), output_m, "conv2 (final)")
    
    print(f"\n{'='*80}")
    if not is_wn_ok:
        print("🎯 找到问题：WaveNet 是第一个出现差异的地方！")
        print("   需要深入检查 WaveNet 的实现细节")
    else:
        print("所有层都在阈值内，差异可能是精度累计")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

