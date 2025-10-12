#!/usr/bin/env python3
"""
详细分析 S2MEL 各组件的性能，特别是 Length Regulator 周边的数据流
"""

import torch
import time
import sys
sys.path.insert(0, '/Users/bailin/index-tts')

from indextts.infer_v2 import IndexTTS2


def profile_s2mel_pipeline():
    print("=" * 60)
    print("S2MEL 详细性能分析")
    print("=" * 60)
    
    # 初始化模型
    print("\n>> 加载模型...")
    model = IndexTTS2(
        device='mps',
        gpt_pth='checkpoints/gpt.pth',
        use_mlx_conditioning=True  # Pure MLX mode
    )
    
    # 准备测试数据
    print("\n>> 准备测试数据...")
    ref_audio_path = 'examples/voice_01.wav'
    text = "今天天气真不错"
    
    # 1. 生成 codes (使用实际推理流程)
    print("\n>> Step 1: 生成 codes...")
    t0 = time.perf_counter()
    
    # 使用 model 的实际方法准备数据
    from indextts.utils.front import Front
    from indextts.utils.audio import load_audio
    import torchaudio
    
    # 准备参考音频
    ref_mel, ref_mel_lens, ref_audio = model._prepare_ref_audio(ref_audio_path)
    
    # 准备 speaker conditioning
    speech_conditioning_latent, emo_cond_emb, emo_vec = model._prepare_speaker_conditioning(
        ref_audio_path, ref_mel, None
    )
    
    # 准备文本
    text_tokens = model._prepare_text(text)
    
    print(f"   准备数据耗时: {time.perf_counter() - t0:.4f}s")
    
    # 2. 生成 codes (GPT autoregressive generation)
    print("\n>> Step 2: GPT 生成 codes...")
    t0 = time.perf_counter()
    
    # 简化版本：使用固定长度的 dummy codes
    # 实际推理中会通过 GPT 生成
    codes = torch.randint(0, 1024, (1, 100), device=model.device)
    code_lens = torch.tensor([100], device=model.device)
    
    print(f"   codes shape: {codes.shape}")
    print(f"   code_lens: {code_lens}")
    print(f"   生成 codes 耗时: {time.perf_counter() - t0:.4f}s")
    
    # 3. S2MEL Pipeline 开始
    print("\n>> Step 3: S2MEL Pipeline (重点分析)")
    print("-" * 60)
    
    # 3.1 GPT Layer
    t0 = time.perf_counter()
    latent = torch.randn(1, codes.shape[1], 1280, device=model.device)  # Dummy latent
    torch.mps.synchronize()
    t_prepare_latent = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    latent = model.s2mel.models['gpt_layer'](latent)
    torch.mps.synchronize()
    t_gpt_layer = time.perf_counter() - t0
    
    print(f"   3.1 GPT Layer: {t_gpt_layer:.4f}s")
    
    # 3.2 VQ Embedding (semantic_codec.quantizer.vq2emb)
    t0 = time.perf_counter()
    S_infer = model.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
    torch.mps.synchronize()
    t_vq2emb = time.perf_counter() - t0
    
    print(f"   3.2 VQ2EMB: {t_vq2emb:.4f}s")
    print(f"       S_infer shape: {S_infer.shape}")
    
    # 3.3 Transpose
    t0 = time.perf_counter()
    S_infer = S_infer.transpose(1, 2)
    torch.mps.synchronize()
    t_transpose = time.perf_counter() - t0
    
    print(f"   3.3 Transpose: {t_transpose:.4f}s")
    
    # 3.4 Add latent
    t0 = time.perf_counter()
    S_infer = S_infer + latent
    torch.mps.synchronize()
    t_add = time.perf_counter() - t0
    
    print(f"   3.4 Add latent: {t_add:.4f}s")
    
    # 3.5 Compute target_lengths
    t0 = time.perf_counter()
    target_lengths = (code_lens * 1.72).long()
    torch.mps.synchronize()
    t_target_lengths = time.perf_counter() - t0
    
    print(f"   3.5 Compute target_lengths: {t_target_lengths:.4f}s")
    print(f"       target_lengths: {target_lengths}")
    
    # 3.6 Length Regulator (关键!)
    print("\n   3.6 Length Regulator (详细分析):")
    print("   " + "-" * 56)
    
    # Warm-up
    for _ in range(3):
        _ = model.s2mel.models['length_regulator'](
            S_infer, 
            ylens=target_lengths, 
            n_quantizers=3, 
            f0=None
        )
        torch.mps.synchronize()
    
    # 实际测试（多次取平均）
    times = []
    for i in range(10):
        t0 = time.perf_counter()
        cond, olens, _, _, _ = model.s2mel.models['length_regulator'](
            S_infer, 
            ylens=target_lengths, 
            n_quantizers=3, 
            f0=None
        )
        torch.mps.synchronize()
        t = time.perf_counter() - t0
        times.append(t)
        if i < 3:
            print(f"       Run {i+1}: {t:.4f}s")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"       Average: {avg_time:.4f}s (over {len(times)} runs)")
    print(f"       Min: {min_time:.4f}s, Max: {max_time:.4f}s")
    print(f"       cond shape: {cond.shape}")
    
    # 3.7 CFM (Continuous Flow Matching)
    print("\n   3.7 CFM (Diffusion):")
    
    # 准备 prompt_condition (从 cache 或重新计算)
    if model.cache_s2mel_prompt is None:
        # 重新计算 (简化版本)
        prompt_condition = torch.randn(1, target_lengths.max(), 512, device=model.device)
    else:
        prompt_condition = model.cache_s2mel_prompt
    
    # CFM forward
    t0 = time.perf_counter()
    diffusion_steps = 15
    for step in range(diffusion_steps):
        # 简化版本：只计时，不实际运行 CFM
        pass
    torch.mps.synchronize()
    t_cfm = time.perf_counter() - t0
    
    print(f"       CFM placeholder: {t_cfm:.4f}s")
    
    # 总结
    print("\n" + "=" * 60)
    print("性能总结")
    print("=" * 60)
    print(f"GPT Layer:        {t_gpt_layer:.4f}s")
    print(f"VQ2EMB:           {t_vq2emb:.4f}s")
    print(f"Transpose:        {t_transpose:.6f}s")
    print(f"Add latent:       {t_add:.6f}s")
    print(f"Target lengths:   {t_target_lengths:.6f}s")
    print(f"Length Regulator: {avg_time:.4f}s (avg)")
    print(f"CFM placeholder:  {t_cfm:.4f}s")
    print("-" * 60)
    total = t_gpt_layer + t_vq2emb + t_transpose + t_add + t_target_lengths + avg_time + t_cfm
    print(f"Total:            {total:.4f}s")
    print("=" * 60)
    
    # 分析
    print("\n🔍 瓶颈分析:")
    components = [
        ("GPT Layer", t_gpt_layer),
        ("VQ2EMB", t_vq2emb),
        ("Length Regulator", avg_time),
        ("CFM", t_cfm),
    ]
    components.sort(key=lambda x: x[1], reverse=True)
    
    for name, t in components:
        pct = (t / total) * 100
        print(f"   {name:20s}: {t:6.4f}s ({pct:5.1f}%)")


if __name__ == '__main__':
    profile_s2mel_pipeline()

