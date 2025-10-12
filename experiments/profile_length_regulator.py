#!/usr/bin/env python3
"""
详细 profiling Length Regulator 的性能瓶颈
"""
import sys
sys.path.insert(0, '/Users/bailin/index-tts')

import torch
import time
import numpy as np

def profile_length_regulator():
    """Profile Length Regulator 各部分的时间"""
    print("="*80)
    print("🔬 Length Regulator 性能分析")
    print("="*80)
    
    from indextts.infer_v2 import IndexTTS2
    
    # Load model
    print("\n加载模型...")
    model = IndexTTS2(device="mps", use_mlx=True)
    
    # Get length regulator
    length_reg = model.s2mel.models['length_regulator']
    
    # Print model structure
    print(f"\n模型结构:")
    print(f"  Interpolate: {length_reg.interpolate}")
    print(f"  Sampling ratios: {length_reg.sampling_ratios}")
    print(f"  F0 condition: {length_reg.f0_condition}")
    print(f"  Model layers: {len(length_reg.model)}")
    
    # Create realistic test input (matching actual inference)
    batch = 1
    seq_len = 164  # Actual from real inference
    target_len = 282  # Actual target length (164 * 1.72)
    channels = 1024  # Input channels (in_channels)
    
    print(f"\n测试参数 (实际推理数据):")
    print(f"  Input seq_len: {seq_len}")
    print(f"  Target length: {target_len}")
    print(f"  Channels: {channels} (input)")
    print(f"  Internal channels: 512 (processing)")
    
    # Create test data (continuous embeddings, not discrete codes)
    # Simulate output from semantic_codec.quantizer.vq2emb
    x = torch.randn(batch, seq_len, channels, device=model.device)  # (B, T, D)
    ylens = torch.tensor([target_len], device=model.device)
    
    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        with torch.no_grad():
            _ = length_reg(x, ylens=ylens, n_quantizers=3, f0=None)
    
    if model.device == "mps":
        torch.mps.synchronize()
    
    # ========================================================================
    # Full forward timing
    # ========================================================================
    print("\n" + "="*80)
    print("完整 Forward Pass")
    print("="*80)
    
    times = []
    for i in range(10):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = length_reg(x, ylens=ylens, n_quantizers=3, f0=None)
        if model.device == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if i == 0:
            print(f"  Output shape: {out[0].shape}")
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"\n完整 forward:")
    print(f"  Mean: {mean_time:.4f}s")
    print(f"  Std:  {std_time:.4f}s")
    print(f"  Min:  {min(times):.4f}s")
    print(f"  Max:  {max(times):.4f}s")
    
    # ========================================================================
    # Component-wise profiling
    # ========================================================================
    print("\n" + "="*80)
    print("逐步 Profiling")
    print("="*80)
    
    with torch.no_grad():
        # Step 1: Input projection (for continuous input)
        print("\n1️⃣ Input Projection")
        times_proj = []
        x_proj = x  # Already in correct shape for continuous input
        for _ in range(10):
            t0 = time.perf_counter()
            # Length regulator doesn't do embedding for continuous input
            # x stays as is
            if model.device == "mps":
                torch.mps.synchronize()
            times_proj.append(time.perf_counter() - t0)
        
        print(f"  Time: {np.mean(times_proj):.4f}s ± {np.std(times_proj):.4f}s")
        print(f"  Shape: {x_proj.shape}")
        
        # Step 2: F.interpolate
        print("\n2️⃣ F.interpolate (序列长度调整)")
        times_interp = []
        for _ in range(10):
            x_t = x_proj.transpose(1, 2).contiguous()
            t0 = time.perf_counter()
            x_interp = torch.nn.functional.interpolate(
                x_t, size=ylens.max(), mode='nearest'
            )
            if model.device == "mps":
                torch.mps.synchronize()
            times_interp.append(time.perf_counter() - t0)
        
        print(f"  Time: {np.mean(times_interp):.4f}s ± {np.std(times_interp):.4f}s")
        print(f"  Input shape: {x_t.shape}")
        print(f"  Output shape: {x_interp.shape}")
        print(f"  Upsampling ratio: {target_len / seq_len:.2f}x")
        
        # Step 3: Conv1d layers
        print("\n3️⃣ Conv1d Layers")
        times_conv = []
        for _ in range(10):
            t0 = time.perf_counter()
            x_conv = length_reg.model(x_interp)
            if model.device == "mps":
                torch.mps.synchronize()
            times_conv.append(time.perf_counter() - t0)
        
        print(f"  Time: {np.mean(times_conv):.4f}s ± {np.std(times_conv):.4f}s")
        print(f"  Input shape: {x_interp.shape}")
        print(f"  Output shape: {x_conv.shape}")
        print(f"  Layers: {len(length_reg.model)}")
        
        # Individual Conv1d layers
        print(f"\n  逐层分析:")
        x_layer = x_interp
        for i, layer in enumerate(length_reg.model):
            times_layer = []
            for _ in range(10):
                t0 = time.perf_counter()
                x_layer_out = layer(x_layer)
                if model.device == "mps":
                    torch.mps.synchronize()
                times_layer.append(time.perf_counter() - t0)
            
            layer_time = np.mean(times_layer)
            layer_type = type(layer).__name__
            print(f"    Layer {i} ({layer_type:15s}): {layer_time:.4f}s")
            x_layer = x_layer_out
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("📊 性能总结")
    print("="*80)
    
    total_components = np.mean(times_proj) + np.mean(times_interp) + np.mean(times_conv)
    
    print(f"\n组件时间占比:")
    print(f"  Input proj:     {np.mean(times_proj):.4f}s  ({np.mean(times_proj)/mean_time*100:.1f}%)")
    print(f"  F.interpolate:  {np.mean(times_interp):.4f}s  ({np.mean(times_interp)/mean_time*100:.1f}%)")
    print(f"  Conv1d layers:  {np.mean(times_conv):.4f}s  ({np.mean(times_conv)/mean_time*100:.1f}%)")
    print(f"  Other overhead: {mean_time - total_components:.4f}s  ({(mean_time - total_components)/mean_time*100:.1f}%)")
    print(f"  ──────────────────────────────")
    print(f"  Total:          {mean_time:.4f}s  (100.0%)")
    
    # Identify bottleneck
    component_times = {
        "Input proj": np.mean(times_proj),
        "F.interpolate": np.mean(times_interp),
        "Conv1d layers": np.mean(times_conv)
    }
    
    bottleneck = max(component_times, key=component_times.get)
    bottleneck_time = component_times[bottleneck]
    
    print(f"\n主要瓶颈: {bottleneck}")
    print(f"  时间: {bottleneck_time:.4f}s")
    print(f"  占比: {bottleneck_time/mean_time*100:.1f}%")
    
    # Optimization suggestions
    print("\n" + "="*80)
    print("🚀 优化建议")
    print("="*80)
    
    if bottleneck == "F.interpolate":
        print("\n1️⃣ F.interpolate 瓶颈 (最耗时)")
        print("  优化方案:")
        print("  A. torch.compile - 可能 20-30% 加速")
        print("  B. 使用更快的插值方法 (linear vs nearest)")
        print("  C. MLX 化整个 Length Regulator")
        potential_speedup = bottleneck_time * 0.25  # Assume 25% speedup
        print(f"  预期加速: {potential_speedup:.2f}s ({potential_speedup/mean_time*100:.1f}% of total)")
    
    elif bottleneck == "Conv1d layers":
        print("\n1️⃣ Conv1d layers 瓶颈")
        print("  优化方案:")
        print("  A. torch.compile - 可能 30-40% 加速")
        print("  B. 减少层数 (但可能影响质量)")
        print("  C. 使用更高效的卷积实现")
        potential_speedup = bottleneck_time * 0.35
        print(f"  预期加速: {potential_speedup:.2f}s ({potential_speedup/mean_time*100:.1f}% of total)")
    
    print("\n2️⃣ 整体 MLX 化")
    print("  工作量: 大 (预计 5-8 小时)")
    print("  预期加速: 不确定 (可能 20-50%)")
    print("  风险: MLX 的 interpolate 性能未知")
    
    print("\n3️⃣ torch.compile (推荐)")
    print("  工作量: 小 (预计 30 分钟)")
    print("  预期加速: 20-35%")
    print("  风险: 低")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    profile_length_regulator()

