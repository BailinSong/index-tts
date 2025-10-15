#!/usr/bin/env python3
"""
逐token对比MLX和PyTorch的生成过程
使用固定seed确保可重复性

运行方式:
    python debug_generation_step_by_step.py
"""

import torch
import mlx.core as mx
import numpy as np
from omegaconf import OmegaConf
import os

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
    # 设置环境变量用于MLX内部seed
    os.environ['MLX_FIXED_SEED'] = str(seed)
    print(f"✅ Set all random seeds to {seed}")


def compare_tensors(name, torch_tensor, mlx_tensor, rtol=1e-3, atol=1e-4, verbose=True):
    """对比两个tensor"""
    if isinstance(torch_tensor, torch.Tensor):
        torch_np = torch_tensor.detach().cpu().float().numpy()
    else:
        torch_np = np.array(torch_tensor, dtype=np.float32)
    
    if isinstance(mlx_tensor, mx.array):
        mlx_np = np.array(mlx_tensor, dtype=np.float32)
    else:
        mlx_np = np.array(mlx_tensor, dtype=np.float32)
    
    if torch_np.shape != mlx_np.shape:
        if verbose:
            print(f"  ❌ {name}: Shape不匹配! PyTorch={torch_np.shape}, MLX={mlx_np.shape}")
        return False, None
    
    diff = np.abs(torch_np - mlx_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    is_close = np.allclose(torch_np, mlx_np, rtol=rtol, atol=atol)
    
    if verbose:
        status = "✅" if is_close else "❌"
        print(f"  {status} {name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
        
        if not is_close:
            print(f"      PyTorch: min={torch_np.min():.6f}, max={torch_np.max():.6f}, mean={torch_np.mean():.6f}")
            print(f"      MLX:     min={mlx_np.min():.6f}, max={mlx_np.max():.6f}, mean={mlx_np.mean():.6f}")
    
    return is_close, {'max_diff': max_diff, 'mean_diff': mean_diff}


def main():
    print("=" * 100)
    print("逐Token对比PyTorch和MLX的生成过程 (固定Seed)")
    print("=" * 100)
    
    SEED = 42
    MAX_TOKENS_TO_COMPARE = 10  # 对比前10个生成的token
    
    set_seed(SEED)
    
    # 加载配置
    cfg = OmegaConf.load("checkpoints/config.yaml")
    
    # 加载PyTorch模型
    print("\n>> 加载 PyTorch 模型...")
    torch_model = PyTorchModel(**cfg.gpt)
    load_checkpoint(torch_model, "checkpoints/gpt.pth")
    torch_model.post_init_gpt2_config()
    torch_model.eval()
    
    # 加载MLX模型
    print("\n>> 加载 MLX 模型...")
    mlx_cache = MLXModelCache(cache_dir="./cache/mlx")
    mlx_weights = mlx_cache.get_or_convert("gpt", "checkpoints/gpt.pth")
    mlx_model = MLXModel(**cfg.gpt, use_mlx_conditioning=False)  # 暂时不用MLX conditioning
    mlx_model.load_weights_from_dict(mlx_weights)
    print("✅ 模型加载完成\n")
    
    # ========================================================================
    # 创建固定的输入
    # ========================================================================
    print("=" * 100)
    print("创建测试输入")
    print("=" * 100)
    
    batch_size = 1
    context_len = 50  # conditioning + text
    
    # 创建fake conditioning (全1)
    set_seed(SEED)
    fake_inputs_torch = torch.full((batch_size, context_len), 1, dtype=torch.long)
    fake_inputs_mlx = mx.full((batch_size, context_len), 1, dtype=mx.int32)
    
    print(f"输入: {context_len}个fake tokens (value=1)")
    print(f"Start token: {torch_model.start_mel_token} (8192)")
    
    # ========================================================================
    # PyTorch 生成过程
    # ========================================================================
    print("\n" + "=" * 100)
    print("PyTorch 生成过程 (Greedy Decoding)")
    print("=" * 100)
    
    set_seed(SEED)
    
    with torch.no_grad():
        # 准备输入
        text_emb_torch = torch_model.text_embedding(fake_inputs_torch)
        text_pos_torch = torch_model.text_pos_embedding(torch.arange(context_len).unsqueeze(0))
        text_with_pos_torch = text_emb_torch + text_pos_torch
        
        # Start token
        start_token_torch = torch.full((batch_size, 1), torch_model.start_mel_token, dtype=torch.long)
        start_emb_torch = torch_model.mel_embedding(start_token_torch)
        start_pos_torch = torch_model.inference_model.text_pos_embedding(torch.tensor([[context_len]]))
        start_with_pos_torch = start_emb_torch + start_pos_torch
        
        # 初始序列
        sequence_torch = torch.cat([text_with_pos_torch, start_with_pos_torch], dim=1)
        
        # 第一次forward (full context)
        hidden_torch = sequence_torch
        past_kvs_torch = []
        for block in torch_model.inference_model.transformer.h:
            outputs = block(hidden_torch, use_cache=True)
            if isinstance(outputs, tuple):
                hidden_torch = outputs[0]
                if len(outputs) > 1:
                    past_kvs_torch.append(outputs[1])
            else:
                hidden_torch = outputs
        
        hidden_torch = torch_model.inference_model.final_norm(hidden_torch)
        logits_torch = torch_model.mel_head(hidden_torch[:, -1, :])
        
        # 第一个token
        first_token_torch = torch.argmax(logits_torch, dim=-1)
        pytorch_tokens = [int(first_token_torch[0])]
        
        print(f"Token 0: {pytorch_tokens[0]}")
        print(f"  Logits top-5: {torch.topk(logits_torch[0], k=5).indices.tolist()}")
        
        # 后续tokens (使用KV cache)
        for step in range(1, MAX_TOKENS_TO_COMPARE):
            # Embed new token
            next_token_torch = torch.tensor([[pytorch_tokens[-1]]], dtype=torch.long)
            next_emb_torch = torch_model.mel_embedding(next_token_torch)
            next_pos_torch = torch_model.inference_model.text_pos_embedding(
                torch.tensor([[context_len + step]])
            )
            next_with_pos_torch = next_emb_torch + next_pos_torch
            
            # Forward with KV cache
            hidden_torch = next_with_pos_torch
            new_past_kvs_torch = []
            for i, block in enumerate(torch_model.inference_model.transformer.h):
                if i < len(past_kvs_torch):
                    outputs = block(hidden_torch, past_key_values=past_kvs_torch[i], use_cache=True)
                else:
                    outputs = block(hidden_torch, use_cache=True)
                if isinstance(outputs, tuple):
                    hidden_torch = outputs[0]
                    if len(outputs) > 1:
                        new_past_kvs_torch.append(outputs[1])
                else:
                    hidden_torch = outputs
            
            past_kvs_torch = new_past_kvs_torch
            
            hidden_torch = torch_model.inference_model.final_norm(hidden_torch)
            logits_torch = torch_model.mel_head(hidden_torch[:, -1, :])
            
            next_token_id = int(torch.argmax(logits_torch, dim=-1)[0])
            pytorch_tokens.append(next_token_id)
            
            print(f"Token {step}: {next_token_id}")
            
            if next_token_id == torch_model.stop_mel_token:
                print(f"  (Stop token reached)")
                break
    
    # ========================================================================
    # MLX 生成过程
    # ========================================================================
    print("\n" + "=" * 100)
    print("MLX 生成过程 (Greedy Decoding)")
    print("=" * 100)
    
    set_seed(SEED)
    
    # 准备输入
    text_emb_mlx = mlx_model.text_embedding(fake_inputs_mlx)
    text_pos_list = []
    for i in range(context_len):
        text_pos_list.append(mlx_model.text_pos_embedding.weight[i])
    text_pos_mlx = mx.stack(text_pos_list, axis=0).reshape(1, context_len, mlx_model.model_dim)
    text_with_pos_mlx = text_emb_mlx + text_pos_mlx
    
    # Start token
    start_token_mlx = mx.full((batch_size, 1), mlx_model.start_mel_token, dtype=mx.int32)
    start_emb_mlx = mlx_model.mel_embedding(start_token_mlx)
    start_pos_mlx = mlx_model.mel_pos_embedding.weight[context_len:context_len+1]
    start_with_pos_mlx = start_emb_mlx + start_pos_mlx
    
    # 初始序列
    sequence_mlx = mx.concatenate([text_with_pos_mlx, start_with_pos_mlx], axis=1)
    
    # 第一次forward (full context)
    hidden_mlx = sequence_mlx
    past_kvs_mlx = []
    for block in mlx_model.transformer_blocks:
        hidden_mlx, kv = block(hidden_mlx, causal_mask=mlx_model.causal_mask, use_cache=True)
        past_kvs_mlx.append(kv)
    
    hidden_mlx = mlx_model.final_norm(hidden_mlx)
    logits_mlx = mlx_model.mel_head(hidden_mlx[:, -1:, :])[:, 0, :]
    
    # 第一个token
    first_token_mlx = mx.argmax(logits_mlx, axis=-1)
    mlx_tokens = [int(first_token_mlx[0])]
    
    print(f"Token 0: {mlx_tokens[0]}")
    logits_mlx_np = np.array(logits_mlx[0])
    top5_indices = np.argsort(logits_mlx_np)[-5:][::-1]
    print(f"  Logits top-5: {top5_indices.tolist()}")
    
    # 后续tokens (使用KV cache)
    for step in range(1, MAX_TOKENS_TO_COMPARE):
        # Embed new token
        next_token_mlx = mx.array([[mlx_tokens[-1]]], dtype=mx.int32)
        next_emb_mlx = mlx_model.mel_embedding(next_token_mlx)
        next_pos_mlx = mlx_model.mel_pos_embedding.weight[context_len + step:context_len + step + 1]
        next_with_pos_mlx = next_emb_mlx + next_pos_mlx
        
        # Forward with KV cache
        hidden_mlx = next_with_pos_mlx
        new_past_kvs_mlx = []
        for i, block in enumerate(mlx_model.transformer_blocks):
            hidden_mlx, kv = block(hidden_mlx, causal_mask=mlx_model.causal_mask, 
                                  past_kv=past_kvs_mlx[i], use_cache=True)
            new_past_kvs_mlx.append(kv)
        
        past_kvs_mlx = new_past_kvs_mlx
        
        hidden_mlx = mlx_model.final_norm(hidden_mlx)
        logits_mlx = mlx_model.mel_head(hidden_mlx[:, -1:, :])[:, 0, :]
        
        next_token_id = int(mx.argmax(logits_mlx, axis=-1)[0])
        mlx_tokens.append(next_token_id)
        
        print(f"Token {step}: {next_token_id}")
        
        if next_token_id == mlx_model.stop_mel_token:
            print(f"  (Stop token reached)")
            break
    
    # ========================================================================
    # 对比结果
    # ========================================================================
    print("\n" + "=" * 100)
    print("对比结果")
    print("=" * 100)
    
    min_len = min(len(pytorch_tokens), len(mlx_tokens))
    
    print(f"\nPyTorch tokens ({len(pytorch_tokens)}): {pytorch_tokens}")
    print(f"MLX tokens     ({len(mlx_tokens)}): {mlx_tokens}")
    
    # 逐token对比
    print(f"\n逐Token对比 (前{min_len}个):")
    all_match = True
    first_mismatch = None
    
    for i in range(min_len):
        match = pytorch_tokens[i] == mlx_tokens[i]
        status = "✅" if match else "❌"
        print(f"  Token {i}: PyTorch={pytorch_tokens[i]}, MLX={mlx_tokens[i]} {status}")
        
        if not match and first_mismatch is None:
            first_mismatch = i
            all_match = False
    
    print("\n" + "=" * 100)
    print("总结")
    print("=" * 100)
    
    if all_match:
        print("✅ 所有tokens完全匹配!")
        print(f"   使用种子: {SEED}")
        print(f"   对比了 {min_len} 个tokens")
    else:
        print(f"❌ 发现不匹配! 第一个不匹配的token位置: {first_mismatch}")
        print(f"   PyTorch token {first_mismatch}: {pytorch_tokens[first_mismatch]}")
        print(f"   MLX token {first_mismatch}: {mlx_tokens[first_mismatch]}")
        print("\n可能的原因:")
        print("  1. 随机数生成器差异 (如果使用sampling)")
        print("  2. Logits processor实现差异")
        print("  3. 数值累积误差")
        print("\n建议:")
        print("  - 检查logits是否一致 (在第一个不匹配的位置)")
        print("  - 使用更高的数值精度")
        print("  - 确保两边都使用greedy decoding (argmax)")


if __name__ == "__main__":
    main()

