#!/usr/bin/env python3
"""
对比第一步的logits是否一致
"""

import torch
import mlx.core as mx
import numpy as np
from omegaconf import OmegaConf

from indextts.gpt.model_v2 import UnifiedVoice as PyTorchModel
from indextts.gpt.mlx_model import UnifiedVoiceMLX as MLXModel
from indextts.utils.mlx_utils import torch_to_mlx
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.mlx_cache import MLXModelCache


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def main():
    SEED = 42
    set_seed(SEED)
    
    # 加载配置和模型
    cfg = OmegaConf.load("checkpoints/config.yaml")
    
    print(">> 加载 PyTorch 模型...")
    torch_model = PyTorchModel(**cfg.gpt)
    load_checkpoint(torch_model, "checkpoints/gpt.pth")
    torch_model.post_init_gpt2_config()
    torch_model.eval()
    
    print(">> 加载 MLX 模型...")
    mlx_cache = MLXModelCache(cache_dir="./cache/mlx")
    mlx_weights = mlx_cache.get_or_convert("gpt", "checkpoints/gpt.pth")
    mlx_model = MLXModel(**cfg.gpt, use_mlx_conditioning=False)
    mlx_model.load_weights_from_dict(mlx_weights)
    
    # 创建测试输入
    batch_size = 1
    context_len = 50
    
    print(f"\n>> 创建测试输入: {context_len} tokens + start_mel_token")
    
    # 创建输入
    set_seed(SEED)
    input_ids_torch = torch.full((batch_size, context_len), 1, dtype=torch.long)
    input_ids_mlx = mx.full((batch_size, context_len), 1, dtype=mx.int32)
    
    # ========================================================================
    # PyTorch forward
    # ========================================================================
    print("\n>> PyTorch forward pass...")
    with torch.no_grad():
        # Text embeddings
        text_emb = torch_model.text_embedding(input_ids_torch)
        text_pos = torch_model.text_pos_embedding(torch.arange(context_len).unsqueeze(0))
        text_with_pos = text_emb + text_pos
        
        # Start token
        start_token = torch.full((1, 1), 8192, dtype=torch.long)
        start_emb = torch_model.mel_embedding(start_token)
        start_pos = torch_model.inference_model.text_pos_embedding(torch.tensor([[context_len]]))
        start_with_pos = start_emb + start_pos
        
        # 合并
        sequence = torch.cat([text_with_pos, start_with_pos], dim=1)
        
        # Forward through transformer (包含 gpt.ln_f)
        # 方法1: 直接调用 GPT2Model (推荐)
        gpt_outputs = torch_model.inference_model.transformer(inputs_embeds=sequence)
        hidden = gpt_outputs[0]  # 已经包含 gpt.ln_f
        
        # 方法2 (等效): 手动遍历 + 应用 gpt.ln_f
        # hidden = sequence
        # for block in torch_model.inference_model.transformer.h:
        #     outputs = block(hidden, use_cache=False)
        #     hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        # hidden = torch_model.gpt.ln_f(hidden)  # 应用 gpt.ln_f
        
        # Final norm (lm_head中的LayerNorm)
        hidden = torch_model.inference_model.final_norm(hidden)
        
        # Mel head
        logits_torch = torch_model.mel_head(hidden[:, -1, :])
        
        # Top-10 tokens
        top10_torch = torch.topk(logits_torch[0], k=10)
        print(f"\nPyTorch Top-10:")
        for i, (idx, val) in enumerate(zip(top10_torch.indices, top10_torch.values)):
            print(f"  {i+1}. token {idx.item()}: {val.item():.6f}")
    
    # ========================================================================
    # MLX forward
    # ========================================================================
    print("\n>> MLX forward pass...")
    
    # Text embeddings
    text_emb_mlx = mlx_model.text_embedding(input_ids_mlx)
    text_pos_list = []
    for i in range(context_len):
        text_pos_list.append(mlx_model.text_pos_embedding.weight[i])
    text_pos_mlx = mx.stack(text_pos_list, axis=0).reshape(1, context_len, mlx_model.model_dim)
    text_with_pos_mlx = text_emb_mlx + text_pos_mlx
    
    # Start token
    start_token_mlx = mx.full((1, 1), 8192, dtype=mx.int32)
    start_emb_mlx = mlx_model.mel_embedding(start_token_mlx)
    start_pos_mlx = mlx_model.mel_pos_embedding.weight[context_len:context_len+1]
    start_with_pos_mlx = start_emb_mlx + start_pos_mlx
    
    # 合并
    sequence_mlx = mx.concatenate([text_with_pos_mlx, start_with_pos_mlx], axis=1)
    
    # Forward through transformer
    hidden_mlx = sequence_mlx
    for block in mlx_model.transformer_blocks:
        hidden_mlx = block(hidden_mlx, causal_mask=mlx_model.causal_mask, use_cache=False)
    
    # Two LayerNorms
    hidden_mlx = mlx_model.gpt_ln_f(hidden_mlx)
    hidden_mlx = mlx_model.final_norm(hidden_mlx)
    
    # Mel head
    logits_mlx = mlx_model.mel_head(hidden_mlx[:, -1:, :])[:, 0, :]
    
    # Top-10 tokens
    logits_mlx_np = np.array(logits_mlx[0])
    top10_indices = np.argsort(logits_mlx_np)[-10:][::-1]
    print(f"\nMLX Top-10:")
    for i, idx in enumerate(top10_indices):
        print(f"  {i+1}. token {idx}: {logits_mlx_np[idx]:.6f}")
    
    # ========================================================================
    # 对比
    # ========================================================================
    print("\n" + "="*80)
    print("对比结果")
    print("="*80)
    
    logits_torch_np = logits_torch[0].numpy()
    diff = np.abs(logits_torch_np - logits_mlx_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    is_close = np.allclose(logits_torch_np, logits_mlx_np, rtol=1e-3, atol=1e-4)
    
    print(f"\nLogits 对比:")
    print(f"  Shape: PyTorch={logits_torch_np.shape}, MLX={logits_mlx_np.shape}")
    print(f"  Max diff: {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")
    print(f"  Close? {is_close}")
    
    # 对比top token
    pt_top = int(top10_torch.indices[0])
    mlx_top = int(top10_indices[0])
    
    print(f"\nTop-1 token:")
    print(f"  PyTorch: {pt_top} (logit={logits_torch_np[pt_top]:.6f})")
    print(f"  MLX:     {mlx_top} (logit={logits_mlx_np[mlx_top]:.6f})")
    print(f"  Match? {'✅' if pt_top == mlx_top else '❌'}")


if __name__ == "__main__":
    main()

