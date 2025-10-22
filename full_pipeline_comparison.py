#!/usr/bin/env python3
"""
完整Pipeline逐层对比 - 从输入到最终输出
控制变量，精确定位每一步的差异
"""

import os
import torch
import mlx.core as mx
import numpy as np

os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'

def compare(pt, mlx, name, show_detail=False):
    """对比两个tensor"""
    from indextts.utils.mlx_utils import mlx_to_torch
    
    if isinstance(mlx, mx.array):
        mlx = mlx_to_torch(mlx, device='cpu')
    if pt.device.type != 'cpu':
        pt = pt.cpu()
    
    diff = torch.abs(pt - mlx)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    pt_flat = pt.detach().numpy().flatten()
    mlx_flat = mlx.detach().numpy().flatten()
    corr = np.corrcoef(pt_flat, mlx_flat)[0, 1] if len(pt_flat) > 1 else 1.0
    
    status = "✅" if max_diff < 1e-4 else ("⚠️" if max_diff < 0.01 else "❌")
    print(f"{status} {name}: max={max_diff:.8f}, mean={mean_diff:.8f}, corr={corr:.6f}")
    
    if show_detail and max_diff > 1e-4:
        print(f"     PyTorch mean={pt.mean().item():.6f}, std={pt.std().item():.6f}")
        print(f"     MLX mean={mlx.mean().item():.6f}, std={mlx.std().item():.6f}")
    
    return max_diff, corr

def main():
    from omegaconf import OmegaConf
    from indextts.gpt.model_v2 import UnifiedVoice
    from indextts.gpt.mlx_model import UnifiedVoiceMLX
    from indextts.utils.checkpoint import load_checkpoint
    from indextts.utils.mlx_cache import MLXModelCache
    from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch
    
    print("🔬 完整Pipeline逐层对比")
    print("=" * 80)
    
    # 加载模型
    cfg = OmegaConf.load('checkpoints/config.yaml')
    gpt_pt = UnifiedVoice(**cfg.gpt)
    load_checkpoint(gpt_pt, 'checkpoints/gpt.pth')
    gpt_pt = gpt_pt.to('mps')
    gpt_pt.eval()
    
    mlx_cache = MLXModelCache(cache_dir='checkpoints/mlx')
    gpt_mlx = UnifiedVoiceMLX(use_mlx_conditioning=True, **cfg.gpt)
    gpt_mlx.load_weights_from_dict(mlx_cache.get_or_convert('gpt', 'checkpoints/gpt.pth'))
    
    # 生成控制变量
    torch.manual_seed(42)
    np.random.seed(42)
    
    speech_conditioning = torch.randn(1, 50, 1024, device='mps')
    emo_speech_conditioning = torch.randn(1, 50, 1024, device='mps')
    text_tokens = torch.randint(0, 1000, (1, 20), device='mps')
    cond_lengths = torch.tensor([50], device='mps')
    
    print(f"控制变量:")
    print(f"  speech_conditioning: {speech_conditioning.shape}")
    print(f"  emo_speech_conditioning: {emo_speech_conditioning.shape}")
    print(f"  text_tokens: {text_tokens.shape}")
    print(f"  cond_lengths: {cond_lengths}")
    
    # =====================================================================
    # PART 1: Speech Conditioning (Speaker)
    # =====================================================================
    print("\n" + "=" * 80)
    print("PART 1: Speech Conditioning (Speaker)")
    print("=" * 80)
    
    with torch.no_grad():
        pt_speech_cond = gpt_pt.get_conditioning(speech_conditioning.transpose(1, 2), cond_lengths)
    
    mlx_speech_cond = gpt_mlx.get_conditioning(speech_conditioning.cpu(), cond_lengths.cpu())
    
    compare(pt_speech_cond, mlx_speech_cond, "Speech Conditioning输出")
    
    # =====================================================================
    # PART 2: Emotion Conditioning
    # =====================================================================
    print("\n" + "=" * 80)
    print("PART 2: Emotion Conditioning")
    print("=" * 80)
    
    # 2.1 Emotion Conformer
    print("\n2.1 Emotion Conformer:")
    
    with torch.no_grad():
        pt_emo_conf_out, pt_emo_mask = gpt_pt.emo_conditioning_encoder(
            emo_speech_conditioning, cond_lengths
        )
    
    mlx_emo_input = torch_to_mlx(emo_speech_conditioning.cpu())
    mlx_emo_lengths = torch_to_mlx(cond_lengths.cpu())
    mlx_emo_conf_out, mlx_emo_mask = gpt_mlx.emo_conditioning_module.conformer(
        mlx_emo_input, mlx_emo_lengths
    )
    
    compare(pt_emo_conf_out, mlx_emo_conf_out, "  Emotion Conformer输出")
    
    # 2.2 Emotion Perceiver
    print("\n2.2 Emotion Perceiver:")
    
    with torch.no_grad():
        pt_emo_mask_sq = pt_emo_mask.squeeze(1)
        pt_emo_conds_mask = gpt_pt.emo_cond_mask_pad(pt_emo_mask_sq)
        pt_emo_perceiver_out = gpt_pt.emo_perceiver_encoder(pt_emo_conf_out, pt_emo_conds_mask)
    
    mlx_emo_mask_sq = mlx_emo_mask.squeeze(1) if len(mlx_emo_mask.shape) > 2 else mlx_emo_mask
    mlx_emo_conds_mask = mx.concatenate([
        mx.ones((mlx_emo_mask_sq.shape[0], 1), dtype=mx.bool_),
        mlx_emo_mask_sq
    ], axis=1)
    mlx_emo_perceiver_out = gpt_mlx.emo_conditioning_module.perceiver(
        mlx_emo_conf_out, mlx_emo_conds_mask
    )
    
    compare(pt_emo_perceiver_out, mlx_emo_perceiver_out, "  Emotion Perceiver输出")
    
    # 2.3 Final squeeze
    print("\n2.3 Emotion Conditioning最终输出:")
    
    with torch.no_grad():
        pt_emo_cond = pt_emo_perceiver_out.squeeze(1)
    
    mlx_emo_cond = mx.squeeze(mlx_emo_perceiver_out, axis=1)
    
    compare(pt_emo_cond, mlx_emo_cond, "  Emotion Conditioning")
    
    # =====================================================================
    # PART 3: Text Embedding
    # =====================================================================
    print("\n" + "=" * 80)
    print("PART 3: Text Embedding")
    print("=" * 80)
    
    with torch.no_grad():
        pt_text_emb = gpt_pt.text_embedding(text_tokens)
    
    mlx_text_tokens = torch_to_mlx(text_tokens.cpu())
    mlx_text_emb = gpt_mlx.text_embedding(mlx_text_tokens)
    
    compare(pt_text_emb, mlx_text_emb, "Text Embedding")
    
    # =====================================================================
    # PART 4: Position Embedding
    # =====================================================================
    print("\n" + "=" * 80)
    print("PART 4: Position Embedding")
    print("=" * 80)
    
    with torch.no_grad():
        pt_text_pos = gpt_pt.text_pos_embedding(text_tokens)
    
    mlx_text_pos = gpt_mlx.text_pos_embedding(mlx_text_tokens)
    
    compare(pt_text_pos, mlx_text_pos, "Text Position Embedding")
    
    # =====================================================================
    # PART 5: Emotion Vector (emovec_layer)
    # =====================================================================
    print("\n" + "=" * 80)
    print("PART 5: Emotion Vector Projection")
    print("=" * 80)
    
    with torch.no_grad():
        pt_emovec = gpt_pt.emovec_layer(pt_emo_cond)
    
    mlx_emovec = gpt_mlx.emovec_layer(mlx_emo_cond)
    
    compare(pt_emovec, mlx_emovec, "Emotion Vector (emovec_layer)")
    
    # =====================================================================
    # PART 6: Emotion Layer
    # =====================================================================
    print("\n" + "=" * 80)
    print("PART 6: Emotion Layer Projection")
    print("=" * 80)
    
    with torch.no_grad():
        pt_emo = gpt_pt.emo_layer(pt_emovec)
    
    mlx_emo = gpt_mlx.emo_layer(mlx_emovec)
    
    compare(pt_emo, mlx_emo, "Emotion (emo_layer)")
    
    # =====================================================================
    # PART 7: 完整的get_emo_conditioning对比
    # =====================================================================
    print("\n" + "=" * 80)
    print("PART 7: 完整get_emo_conditioning (端到端)")
    print("=" * 80)
    
    with torch.no_grad():
        pt_emo_full = gpt_pt.get_emo_conditioning(emo_speech_conditioning, cond_lengths)
    
    mlx_emo_full = gpt_mlx.get_emo_conditioning(emo_speech_conditioning.cpu(), cond_lengths.cpu())
    
    max_diff, corr = compare(pt_emo_full, mlx_emo_full, "get_emo_conditioning (完整)", show_detail=True)
    
    # =====================================================================
    # 总结
    # =====================================================================
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    
    if max_diff < 1e-4:
        print("✅ 所有组件完全一致！纯MLX实现成功！")
        print(f"   最大差异: {max_diff:.8f} (可忽略的数值精度)")
        print(f"   相关系数: {corr:.6f} (完美)")
    elif max_diff < 0.01:
        print("✅ 差异极小，实用性优秀")
        print(f"   最大差异: {max_diff:.6f}")
        print(f"   相关系数: {corr:.6f}")
    elif max_diff < 0.1:
        print("⚠️ 有中等差异，需要进一步优化")
        print(f"   最大差异: {max_diff:.6f}")
        print(f"   相关系数: {corr:.6f}")
    else:
        print("❌ 差异显著，需要调试")
        print(f"   最大差异: {max_diff:.6f}")
        print(f"   相关系数: {corr:.6f}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

