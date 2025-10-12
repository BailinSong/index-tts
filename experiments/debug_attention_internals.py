#!/usr/bin/env python3
"""
详细检查 Attention 内部的每一步
"""
import sys
sys.path.insert(0, '/Users/bailin/index-tts')

import torch
import mlx.core as mx
import numpy as np
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch

def compute_correlation(a, b):
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    if isinstance(a, mx.array):
        a = np.array(a)
    if isinstance(b, mx.array):
        b = np.array(b)
    a_flat = a.flatten()
    b_flat = b.flatten()
    corr = np.corrcoef(a_flat, b_flat)[0, 1]
    max_diff = np.abs(a_flat - b_flat).max()
    return corr, max_diff

def main():
    print("=" * 80)
    print("🔬 详细检查 Attention 内部的每一步")
    print("=" * 80)
    
    # Load models
    from indextts.infer_v2 import IndexTTS2
    
    print("\n加载模型...")
    pt_model = IndexTTS2(device="mps", use_mlx=False)
    mlx_model = IndexTTS2(device="mps", use_mlx=True)
    
    # Create test input (smaller for detailed analysis)
    torch.manual_seed(42)
    np.random.seed(42)
    
    batch, seq_len, dim = 1, 60, 512
    pt_x = torch.randn(batch, seq_len, dim, device=pt_model.device)
    mlx_x = torch_to_mlx(pt_x.cpu())
    
    pt_pos_emb = torch.randn(batch, seq_len, dim, device=pt_model.device)
    mlx_pos_emb = torch_to_mlx(pt_pos_emb.cpu())
    
    print(f"\n测试输入: {pt_x.shape}")
    
    # ========================================================================
    # Get Attention modules
    # ========================================================================
    pt_attn = pt_model.gpt.conditioning_encoder.encoders[0].self_attn
    mlx_attn = mlx_model.mlx_transformer.conditioning_module.conformer.blocks[0].attn
    
    # ========================================================================
    # Step 1: Q, K, V projections
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 1: Q, K, V Projections")
    print("=" * 80)
    
    with torch.no_grad():
        pt_q = pt_attn.linear_q(pt_x)
        pt_k = pt_attn.linear_k(pt_x)
        pt_v = pt_attn.linear_v(pt_x)
    
    mlx_q = mlx_attn.q_proj(mlx_x)
    mlx_k = mlx_attn.k_proj(mlx_x)
    mlx_v = mlx_attn.v_proj(mlx_x)
    
    corr_q, max_diff_q = compute_correlation(pt_q.cpu(), mlx_to_torch(mlx_q, "cpu"))
    corr_k, max_diff_k = compute_correlation(pt_k.cpu(), mlx_to_torch(mlx_k, "cpu"))
    corr_v, max_diff_v = compute_correlation(pt_v.cpu(), mlx_to_torch(mlx_v, "cpu"))
    
    print(f"\nQ: corr={corr_q:.6f}, max_diff={max_diff_q:.6f}")
    print(f"K: corr={corr_k:.6f}, max_diff={max_diff_k:.6f}")
    print(f"V: corr={corr_v:.6f}, max_diff={max_diff_v:.6f}")
    
    # ========================================================================
    # Step 2: Positional encoding projection
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 2: Positional Encoding Projection")
    print("=" * 80)
    
    with torch.no_grad():
        pt_p = pt_attn.linear_pos(pt_pos_emb)
    
    mlx_p = mlx_attn.pos_proj(mlx_pos_emb)
    
    corr_p, max_diff_p = compute_correlation(pt_p.cpu(), mlx_to_torch(mlx_p, "cpu"))
    print(f"\nP (pos encoding): corr={corr_p:.6f}, max_diff={max_diff_p:.6f}")
    
    # ========================================================================
    # Step 3: Reshape for multi-head attention
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 3: Reshape for Multi-Head Attention")
    print("=" * 80)
    
    num_heads = 8
    head_dim = dim // num_heads
    
    # PyTorch
    pt_q = pt_q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    pt_k = pt_k.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    pt_v = pt_v.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    pt_p = pt_p.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # MLX
    mlx_q_reshaped = mlx_q.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    mlx_k_reshaped = mlx_k.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    mlx_v_reshaped = mlx_v.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    mlx_p_reshaped = mlx_p.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    
    corr_q_reshape, _ = compute_correlation(pt_q.cpu(), mlx_to_torch(mlx_q_reshaped, "cpu"))
    print(f"\nQ (reshaped): corr={corr_q_reshape:.6f}")
    
    # ========================================================================
    # Step 4: Compute attention scores (Relative Positional Attention)
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 4: Attention Scores (Transformer-XL style)")
    print("=" * 80)
    
    # PyTorch
    # Get pos_bias_u and pos_bias_v
    pt_pos_bias_u = pt_attn.pos_bias_u.view(1, num_heads, 1, head_dim)
    pt_pos_bias_v = pt_attn.pos_bias_v.view(1, num_heads, 1, head_dim)
    
    pt_q_with_u = pt_q + pt_pos_bias_u
    pt_q_with_v = pt_q + pt_pos_bias_v
    
    pt_matrix_ac = torch.matmul(pt_q_with_u, pt_k.transpose(-2, -1))
    pt_matrix_bd = torch.matmul(pt_q_with_v, pt_p.transpose(-2, -1))
    
    pt_scores = (pt_matrix_ac + pt_matrix_bd) / np.sqrt(head_dim)
    
    # MLX
    mlx_pos_bias_u = mlx_attn.pos_bias_u.reshape(1, num_heads, 1, head_dim)
    mlx_pos_bias_v = mlx_attn.pos_bias_v.reshape(1, num_heads, 1, head_dim)
    
    mlx_q_with_u = mlx_q_reshaped + mlx_pos_bias_u
    mlx_q_with_v = mlx_q_reshaped + mlx_pos_bias_v
    
    mlx_matrix_ac = mlx_q_with_u @ mlx_k_reshaped.transpose(0, 1, 3, 2)
    mlx_matrix_bd = mlx_q_with_v @ mlx_p_reshaped.transpose(0, 1, 3, 2)
    
    mlx_scores = (mlx_matrix_ac + mlx_matrix_bd) / np.sqrt(head_dim)
    
    corr_scores, max_diff_scores = compute_correlation(
        pt_scores.cpu(), 
        mlx_to_torch(mlx_scores, "cpu")
    )
    print(f"\nAttention scores: corr={corr_scores:.6f}, max_diff={max_diff_scores:.6f}")
    
    # ========================================================================
    # Step 5: Softmax
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 5: Softmax")
    print("=" * 80)
    
    pt_attn_weights = torch.softmax(pt_scores, dim=-1)
    mlx_attn_weights = mx.softmax(mlx_scores, axis=-1)
    
    corr_attn_weights, max_diff_attn_weights = compute_correlation(
        pt_attn_weights.cpu(),
        mlx_to_torch(mlx_attn_weights, "cpu")
    )
    print(f"\nAttention weights: corr={corr_attn_weights:.6f}, max_diff={max_diff_attn_weights:.6f}")
    
    # ========================================================================
    # Step 6: Apply attention to values
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 6: Apply Attention to Values")
    print("=" * 80)
    
    pt_attn_output = torch.matmul(pt_attn_weights, pt_v)
    mlx_attn_output = mlx_attn_weights @ mlx_v_reshaped
    
    corr_attn_output, max_diff_attn_output = compute_correlation(
        pt_attn_output.cpu(),
        mlx_to_torch(mlx_attn_output, "cpu")
    )
    print(f"\nAttention output (before out_proj): corr={corr_attn_output:.6f}, max_diff={max_diff_attn_output:.6f}")
    
    # ========================================================================
    # Step 7: Reshape and output projection
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 7: Reshape and Output Projection")
    print("=" * 80)
    
    # Reshape back
    pt_attn_output = pt_attn_output.transpose(1, 2).contiguous().view(batch, seq_len, dim)
    mlx_attn_output = mlx_attn_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, dim)
    
    corr_reshaped, _ = compute_correlation(
        pt_attn_output.cpu(),
        mlx_to_torch(mlx_attn_output, "cpu")
    )
    print(f"\nAfter reshape: corr={corr_reshaped:.6f}")
    
    # Output projection
    with torch.no_grad():
        pt_final = pt_attn.linear_out(pt_attn_output)
    
    mlx_final = mlx_attn.out_proj(mlx_attn_output)
    
    corr_final, max_diff_final = compute_correlation(
        pt_final.cpu(),
        mlx_to_torch(mlx_final, "cpu")
    )
    print(f"After out_proj: corr={corr_final:.6f}, max_diff={max_diff_final:.6f}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 总结")
    print("=" * 80)
    
    print("\nCorrelation at each step:")
    print(f"  1. Q/K/V projections: {min(corr_q, corr_k, corr_v):.6f}")
    print(f"  2. Pos encoding proj: {corr_p:.6f}")
    print(f"  3. Reshape: {corr_q_reshape:.6f}")
    print(f"  4. Attention scores: {corr_scores:.6f}")
    print(f"  5. Softmax (weights): {corr_attn_weights:.6f}")
    print(f"  6. Apply to values: {corr_attn_output:.6f}")
    print(f"  7. Output projection: {corr_final:.6f}")
    
    print("\n问题定位:")
    if corr_scores < 0.999:
        print(f"  ❌ Attention scores 计算有问题 (corr={corr_scores:.6f}, max_diff={max_diff_scores:.6f})")
        print("     可能是 matrix multiplication 的数值精度")
    elif corr_attn_weights < 0.999:
        print(f"  ❌ Softmax 有问题 (corr={corr_attn_weights:.6f})")
    elif corr_attn_output < 0.999:
        print(f"  ❌ Apply attention to values 有问题 (corr={corr_attn_output:.6f})")
    elif corr_final < 0.999:
        print(f"  ❌ Output projection 有问题 (corr={corr_final:.6f})")
    else:
        print("  ✅ 所有步骤的 correlation 都很高！")
        print("     误差可能是浮点精度的微小累积")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

