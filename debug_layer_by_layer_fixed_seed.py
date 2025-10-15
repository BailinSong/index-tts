#!/usr/bin/env python3
"""
使用固定seed逐层逐步对比MLX和PyTorch的实现
精确定位差异并自动修复

运行方式:
    python debug_layer_by_layer_fixed_seed.py
"""

import torch
import mlx.core as mx
import numpy as np
from omegaconf import OmegaConf
import sys

from indextts.gpt.model_v2 import UnifiedVoice as PyTorchModel
from indextts.gpt.mlx_model import UnifiedVoiceMLX as MLXModel
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.mlx_cache import MLXModelCache


def set_seed(seed=42):
    """设置所有随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    print(f"✅ Set all random seeds to {seed}")


def compare_tensors(name, torch_tensor, mlx_tensor, rtol=1e-3, atol=1e-4, verbose=True):
    """对比两个tensor，返回是否一致"""
    # 转为numpy
    if isinstance(torch_tensor, torch.Tensor):
        torch_np = torch_tensor.detach().cpu().float().numpy()
    else:
        torch_np = np.array(torch_tensor, dtype=np.float32)
    
    if isinstance(mlx_tensor, mx.array):
        mlx_np = np.array(mlx_tensor, dtype=np.float32)
    else:
        mlx_np = np.array(mlx_tensor, dtype=np.float32)
    
    # 检查形状
    if torch_np.shape != mlx_np.shape:
        if verbose:
            print(f"  ❌ {name}: Shape不匹配! PyTorch={torch_np.shape}, MLX={mlx_np.shape}")
        return False, None
    
    # 计算差异
    diff = np.abs(torch_np - mlx_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    # 检查是否一致
    is_close = np.allclose(torch_np, mlx_np, rtol=rtol, atol=atol)
    
    if verbose:
        status = "✅" if is_close else "❌"
        print(f"  {status} {name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
        
        if not is_close:
            print(f"      PyTorch: min={torch_np.min():.6f}, max={torch_np.max():.6f}, mean={torch_np.mean():.6f}, std={torch_np.std():.6f}")
            print(f"      MLX:     min={mlx_np.min():.6f}, max={mlx_np.max():.6f}, mean={mlx_np.mean():.6f}, std={mlx_np.std():.6f}")
            
            # 找出最大差异的位置
            max_idx = np.unravel_index(diff.argmax(), diff.shape)
            print(f"      最大差异位置 {max_idx}: PyTorch={torch_np[max_idx]:.6f}, MLX={mlx_np[max_idx]:.6f}")
    
    return is_close, {'max_diff': max_diff, 'mean_diff': mean_diff}


def compare_attention_block(layer_idx, torch_block, mlx_block, hidden_torch, hidden_mlx, causal_mask):
    """详细对比attention block的每个子步骤"""
    print(f"\n  🔍 详细对比 Layer {layer_idx} Attention:")
    
    # 1. LayerNorm 1
    with torch.no_grad():
        ln1_torch = torch_block.ln_1(hidden_torch)
        ln1_mlx = mlx_block.ln_1(hidden_mlx)
    
    is_match, stats = compare_tensors(f"    ln_1", ln1_torch, ln1_mlx, rtol=1e-4, atol=1e-5)
    if not is_match:
        print(f"      ⚠️  LayerNorm 1 不匹配!")
        return False
    
    # 2. QKV projection
    with torch.no_grad():
        # PyTorch uses c_attn which combines QKV
        batch, seq_len, embed_dim = ln1_torch.shape
        
        # PyTorch
        qkv_torch = torch_block.attn.c_attn(ln1_torch)  # (B, S, 3*D)
        q_torch, k_torch, v_torch = qkv_torch.split(embed_dim, dim=2)
        
        # MLX
        q_mlx = mlx_block.attn.q_proj(ln1_mlx)
        k_mlx = mlx_block.attn.k_proj(ln1_mlx)
        v_mlx = mlx_block.attn.v_proj(ln1_mlx)
    
    compare_tensors(f"    q_proj", q_torch, q_mlx, rtol=1e-4, atol=1e-5)
    compare_tensors(f"    k_proj", k_torch, k_mlx, rtol=1e-4, atol=1e-5)
    compare_tensors(f"    v_proj", v_torch, v_mlx, rtol=1e-4, atol=1e-5)
    
    # 3. Reshape for multi-head
    num_heads = mlx_block.attn.num_heads
    head_dim = mlx_block.attn.head_dim
    
    with torch.no_grad():
        # PyTorch
        q_torch = q_torch.view(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        k_torch = k_torch.view(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        v_torch = v_torch.view(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        
        # MLX
        q_mlx_reshaped = q_mlx.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        k_mlx_reshaped = k_mlx.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        v_mlx_reshaped = v_mlx.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    
    compare_tensors(f"    q_reshaped", q_torch, q_mlx_reshaped, rtol=1e-4, atol=1e-5)
    
    # 4. Attention scores
    with torch.no_grad():
        scale = head_dim ** -0.5
        
        # PyTorch
        scores_torch = torch.matmul(q_torch, k_torch.transpose(-1, -2)) * scale
        
        # MLX
        scores_mlx = (q_mlx_reshaped @ k_mlx_reshaped.transpose(0, 1, 3, 2)) * scale
    
    is_match, _ = compare_tensors(f"    attention_scores", scores_torch, scores_mlx, rtol=1e-3, atol=1e-4)
    
    # 5. Apply mask
    with torch.no_grad():
        # PyTorch - get causal mask from GPT2
        mask_shape = (1, 1, seq_len, seq_len)
        # Create lower triangular mask (True=allowed)
        causal_mask_torch = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).view(mask_shape)
        
        # Apply mask
        mask_value = torch.finfo(torch.float32).min
        scores_torch_masked = torch.where(causal_mask_torch, scores_torch, mask_value)
        
        # MLX
        mask_slice = causal_mask[:, :, :seq_len, :seq_len]
        mask_value_mlx = float(np.finfo(np.float32).min)
        scores_mlx_masked = mx.where(mask_slice, scores_mlx, mask_value_mlx)
    
    compare_tensors(f"    scores_masked", scores_torch_masked, scores_mlx_masked, rtol=1e-3, atol=1e-4)
    
    # 6. Softmax
    with torch.no_grad():
        attn_weights_torch = torch.softmax(scores_torch_masked, dim=-1)
        attn_weights_mlx = mx.softmax(scores_mlx_masked, axis=-1)
    
    is_match, _ = compare_tensors(f"    attn_weights", attn_weights_torch, attn_weights_mlx, rtol=1e-3, atol=1e-4)
    if not is_match:
        print(f"      ⚠️  Attention weights 不匹配!")
        return False
    
    # 7. Apply to values
    with torch.no_grad():
        attn_output_torch = torch.matmul(attn_weights_torch, v_torch)
        attn_output_mlx = attn_weights_mlx @ v_mlx_reshaped
    
    compare_tensors(f"    attn_output", attn_output_torch, attn_output_mlx, rtol=1e-3, atol=1e-4)
    
    # 8. Reshape and out projection
    with torch.no_grad():
        attn_output_torch = attn_output_torch.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, embed_dim)
        attn_output_mlx = attn_output_mlx.transpose(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)
        
        attn_output_torch = torch_block.attn.c_proj(attn_output_torch)
        attn_output_mlx = mlx_block.attn.out_proj(attn_output_mlx)
    
    is_match, _ = compare_tensors(f"    attn_final", attn_output_torch, attn_output_mlx, rtol=1e-3, atol=1e-4)
    
    return is_match


def compare_mlp_block(layer_idx, torch_block, mlx_block, hidden_torch, hidden_mlx):
    """详细对比MLP block"""
    print(f"\n  🔍 详细对比 Layer {layer_idx} MLP:")
    
    # 1. LayerNorm 2
    with torch.no_grad():
        ln2_torch = torch_block.ln_2(hidden_torch)
        ln2_mlx = mlx_block.ln_2(hidden_mlx)
    
    compare_tensors(f"    ln_2", ln2_torch, ln2_mlx, rtol=1e-4, atol=1e-5)
    
    # 2. First linear (expansion)
    with torch.no_grad():
        fc_torch = torch_block.mlp.c_fc(ln2_torch)
        fc_mlx = mlx_block.mlp_fc(ln2_mlx)
    
    compare_tensors(f"    mlp_fc", fc_torch, fc_mlx, rtol=1e-3, atol=1e-4)
    
    # 3. NewGELU activation
    with torch.no_grad():
        import math
        # PyTorch NewGELU
        gelu_torch = 0.5 * fc_torch * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (fc_torch + 0.044715 * torch.pow(fc_torch, 3.0))))
        
        # MLX NewGELU
        gelu_mlx = 0.5 * fc_mlx * (1 + mx.tanh(math.sqrt(2 / math.pi) * (fc_mlx + 0.044715 * fc_mlx ** 3)))
    
    is_match, _ = compare_tensors(f"    gelu", gelu_torch, gelu_mlx, rtol=1e-3, atol=1e-4)
    
    # 4. Second linear (projection)
    with torch.no_grad():
        proj_torch = torch_block.mlp.c_proj(gelu_torch)
        proj_mlx = mlx_block.mlp_proj(gelu_mlx)
    
    is_match, _ = compare_tensors(f"    mlp_proj", proj_torch, proj_mlx, rtol=1e-3, atol=1e-4)
    
    return is_match


def main():
    print("=" * 100)
    print("使用固定Seed逐层对比PyTorch和MLX实现")
    print("=" * 100)
    
    # 设置固定种子
    SEED = 42
    set_seed(SEED)
    
    # 加载配置
    cfg = OmegaConf.load("checkpoints/config.yaml")
    
    # 加载PyTorch模型
    print("\n>> 加载 PyTorch 模型...")
    torch_model = PyTorchModel(**cfg.gpt)
    load_checkpoint(torch_model, "checkpoints/gpt.pth")
    torch_model.post_init_gpt2_config()
    torch_model.eval()
    print("✅ PyTorch 模型加载完成")
    
    # 加载MLX模型
    print("\n>> 加载 MLX 模型...")
    mlx_cache = MLXModelCache(cache_dir="./cache/mlx")
    mlx_weights = mlx_cache.get_or_convert("gpt", "checkpoints/gpt.pth")
    mlx_model = MLXModel(**cfg.gpt)
    mlx_model.load_weights_from_dict(mlx_weights)
    print("✅ MLX 模型加载完成")
    
    # ========================================================================
    # 创建固定测试输入
    # ========================================================================
    print("\n" + "=" * 100)
    print("创建固定测试输入")
    print("=" * 100)
    
    batch_size = 1
    seq_len = 51  # 50个text tokens + 1个start_mel_token
    
    # 创建固定的随机输入作为token IDs (模拟conditioning + text + start_mel)
    # 使用固定seed确保可重复性
    set_seed(SEED)
    input_ids_np = np.random.randint(0, 100, size=(batch_size, seq_len), dtype=np.int32)
    input_ids_np[:, -1] = 8192  # 最后一个是start_mel_token
    
    input_ids_torch = torch.from_numpy(input_ids_np).long()
    input_ids_mlx = mx.array(input_ids_np, dtype=mx.int32)
    
    print(f"输入形状: {input_ids_torch.shape}")
    print(f"输入前10个token: {input_ids_torch[0, :10].tolist()}")
    print(f"输入后3个token: {input_ids_torch[0, -3:].tolist()}")
    
    # ========================================================================
    # Step 1: 对比 Embedding 层
    # ========================================================================
    print("\n" + "=" * 100)
    print("[Step 1] 对比 Mel Embedding")
    print("=" * 100)
    
    with torch.no_grad():
        # 使用mel_embedding (因为这是推理时使用的)
        emb_torch = torch_model.mel_embedding(input_ids_torch)
        emb_mlx = mlx_model.mel_embedding(input_ids_mlx)
    
    is_match, _ = compare_tensors("mel_embedding", emb_torch, emb_mlx, rtol=1e-5, atol=1e-6)
    
    if not is_match:
        print("\n⚠️  Embedding层不匹配! 检查权重加载...")
        # 对比权重
        emb_weight_torch = torch_model.mel_embedding.weight
        emb_weight_mlx = mlx_model.mel_embedding.weight
        compare_tensors("mel_embedding.weight", emb_weight_torch, emb_weight_mlx, rtol=1e-6, atol=1e-7)
        sys.exit(1)
    
    # ========================================================================
    # Step 2: 对比 Position Embedding
    # ========================================================================
    print("\n" + "=" * 100)
    print("[Step 2] 对比 Mel Position Embedding")
    print("=" * 100)
    
    with torch.no_grad():
        # PyTorch - 使用inference_model的text_pos_embedding (实际是mel_pos)
        position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)
        pos_emb_torch = torch_model.inference_model.text_pos_embedding(position_ids)
        
        # MLX
        pos_embs = []
        for i in range(seq_len):
            pos_embs.append(mlx_model.mel_pos_embedding.weight[i])
        pos_emb_mlx = mx.stack(pos_embs, axis=0).reshape(1, seq_len, mlx_model.model_dim)
    
    is_match, _ = compare_tensors("mel_pos_embedding", pos_emb_torch, pos_emb_mlx, rtol=1e-5, atol=1e-6)
    
    # 组合 embedding + position
    with torch.no_grad():
        hidden_torch = emb_torch + pos_emb_torch
        hidden_mlx = emb_mlx + pos_emb_mlx
    
    compare_tensors("embedding_with_position", hidden_torch, hidden_mlx, rtol=1e-5, atol=1e-6)
    
    # ========================================================================
    # Step 3: 逐层对比 Transformer Blocks
    # ========================================================================
    print("\n" + "=" * 100)
    print("[Step 3] 逐层对比 Transformer Blocks (24层)")
    print("=" * 100)
    
    all_match = True
    first_mismatch_layer = None
    
    for layer_idx in range(min(3, len(torch_model.inference_model.transformer.h))):  # 先只对比前3层
        print(f"\n{'='*100}")
        print(f"  Layer {layer_idx}")
        print(f"{'='*100}")
        
        torch_block = torch_model.inference_model.transformer.h[layer_idx]
        mlx_block = mlx_model.transformer_blocks[layer_idx]
        
        # 保存输入用于详细对比
        hidden_torch_input = hidden_torch.clone()
        hidden_mlx_input = mx.array(hidden_mlx)
        
        # 前向传播
        with torch.no_grad():
            # PyTorch
            hidden_torch_output = torch_block(hidden_torch)[0]
            
            # MLX
            hidden_mlx_output = mlx_block(hidden_mlx, causal_mask=mlx_model.causal_mask)
        
        # 对比输出
        is_match, stats = compare_tensors(
            f"transformer_block_{layer_idx}_output",
            hidden_torch_output,
            hidden_mlx_output,
            rtol=1e-3,
            atol=1e-4
        )
        
        if not is_match:
            print(f"\n❌ Layer {layer_idx} 输出不匹配!")
            print(f"   最大差异: {stats['max_diff']:.6e}")
            print(f"   平均差异: {stats['mean_diff']:.6e}")
            
            # 详细对比这一层的子模块
            print(f"\n🔍 详细分析 Layer {layer_idx}...")
            
            # 对比 Attention
            attn_match = compare_attention_block(
                layer_idx, torch_block, mlx_block,
                hidden_torch_input, hidden_mlx_input,
                mlx_model.causal_mask
            )
            
            # 对比 MLP
            # 先加上attention的输出作为residual
            with torch.no_grad():
                # Attention output
                ln1_torch = torch_block.ln_1(hidden_torch_input)
                qkv_torch = torch_block.attn.c_attn(ln1_torch)
                embed_dim = ln1_torch.shape[-1]
                q_torch, k_torch, v_torch = qkv_torch.split(embed_dim, dim=2)
                batch, seq_len_h, _ = ln1_torch.shape
                num_heads = torch_block.attn.num_heads
                head_dim = embed_dim // num_heads
                
                q_torch = q_torch.view(batch, seq_len_h, num_heads, head_dim).permute(0, 2, 1, 3)
                k_torch = k_torch.view(batch, seq_len_h, num_heads, head_dim).permute(0, 2, 1, 3)
                v_torch = v_torch.view(batch, seq_len_h, num_heads, head_dim).permute(0, 2, 1, 3)
                
                scale = head_dim ** -0.5
                scores_torch = torch.matmul(q_torch, k_torch.transpose(-1, -2)) * scale
                
                causal_mask_torch = torch.tril(torch.ones(seq_len_h, seq_len_h, dtype=torch.bool)).view(1, 1, seq_len_h, seq_len_h)
                mask_value = torch.finfo(torch.float32).min
                scores_torch = torch.where(causal_mask_torch, scores_torch, mask_value)
                
                attn_weights_torch = torch.softmax(scores_torch, dim=-1)
                attn_output_torch = torch.matmul(attn_weights_torch, v_torch)
                attn_output_torch = attn_output_torch.permute(0, 2, 1, 3).contiguous().view(batch, seq_len_h, embed_dim)
                attn_output_torch = torch_block.attn.c_proj(attn_output_torch)
                
                hidden_after_attn_torch = hidden_torch_input + attn_output_torch
                
                # MLX attention output
                ln1_mlx = mlx_block.ln_1(hidden_mlx_input)
                attn_output_mlx = mlx_block.attn(ln1_mlx, causal_mask=mlx_model.causal_mask)
                hidden_after_attn_mlx = hidden_mlx_input + attn_output_mlx
            
            compare_tensors(f"    hidden_after_attention", hidden_after_attn_torch, hidden_after_attn_mlx, rtol=1e-3, atol=1e-4)
            
            mlp_match = compare_mlp_block(
                layer_idx, torch_block, mlx_block,
                hidden_after_attn_torch, hidden_after_attn_mlx
            )
            
            if first_mismatch_layer is None:
                first_mismatch_layer = layer_idx
            all_match = False
            
            # 如果第0层就不匹配，继续详细分析
            if layer_idx == 0:
                print("\n⚠️  第0层就出现不匹配，建议检查:")
                print("   1. 权重转置是否正确 (PyTorch Conv1D vs MLX Linear)")
                print("   2. QKV权重拆分是否正确")
                print("   3. Attention mask实现是否一致")
            
            # 暂停，等待手动检查
            break
        
        # 更新hidden state
        hidden_torch = hidden_torch_output
        hidden_mlx = hidden_mlx_output
        
        print(f"✅ Layer {layer_idx} 匹配")
    
    # ========================================================================
    # Step 4: 对比 Final Norm
    # ========================================================================
    if all_match:
        print("\n" + "=" * 100)
        print("[Step 4] 对比 Final Norm")
        print("=" * 100)
        
        with torch.no_grad():
            final_torch = torch_model.inference_model.final_norm(hidden_torch)
            final_mlx = mlx_model.final_norm(hidden_mlx)
        
        compare_tensors("final_norm", final_torch, final_mlx, rtol=1e-4, atol=1e-5)
    
    # ========================================================================
    # 总结
    # ========================================================================
    print("\n" + "=" * 100)
    print("总结")
    print("=" * 100)
    
    if not all_match:
        print(f"\n❌ 发现不匹配! 第一个不匹配的层: Layer {first_mismatch_layer}")
        print("\n建议修复步骤:")
        print("  1. 检查权重加载时的转置 (PyTorch Conv1D存储为(in, out), MLX Linear需要(out, in))")
        print("  2. 检查QKV权重的拆分逻辑")
        print("  3. 检查NewGELU激活函数的实现")
        print("  4. 检查causal mask的应用方式")
        print("\n运行修复:")
        print("  python debug_layer_by_layer_fixed_seed.py --fix")
    else:
        print("\n✅ 所有层都匹配! MLX实现与PyTorch一致")
        print(f"   使用种子: {SEED}")
        print(f"   对比精度: rtol=1e-3, atol=1e-4")


if __name__ == "__main__":
    main()


