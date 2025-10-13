#!/usr/bin/env python3
"""
诊断内存泄漏和性能衰减问题
"""

import torch
import mlx.core as mx
import gc
import time
import psutil
import os
import sys
sys.path.insert(0, '/Users/bailin/index-tts')

from indextts.infer_v2 import IndexTTS2


def get_memory_usage():
    """获取内存使用情况"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    # MPS 内存
    if torch.backends.mps.is_available():
        mps_allocated = torch.mps.current_allocated_memory() / 1024**3
        mps_reserved = torch.mps.driver_allocated_memory() / 1024**3
    else:
        mps_allocated = 0
        mps_reserved = 0
    
    return {
        'cpu_rss_gb': mem_info.rss / 1024**3,
        'mps_allocated_gb': mps_allocated,
        'mps_reserved_gb': mps_reserved,
    }


def test_inference_loop():
    print("=" * 70)
    print("内存泄漏诊断测试")
    print("=" * 70)
    
    # 初始化模型
    print("\n>> 初始化模型...")
    model = IndexTTS2(device='mps', use_mlx=True, diffusion_steps=20)
    
    ref_audio = 'examples/voice_01.wav'
    test_texts = [
        "今天天气真不错",
        "我们一起去看电影吧",
        "人工智能技术发展迅速",
    ]
    
    print("\n>> 开始连续推理测试...")
    print("-" * 70)
    
    results = []
    
    for i in range(10):
        text = test_texts[i % len(test_texts)]
        
        # 记录推理前内存
        mem_before = get_memory_usage()
        
        # 推理
        t0 = time.perf_counter()
        try:
            wav, sr = model.infer(
                spk_audio_prompt=ref_audio,
                text=text,
                output_path=None,  # 不保存文件
                verbose=False
            )
            inference_time = time.perf_counter() - t0
            success = True
        except Exception as e:
            inference_time = time.perf_counter() - t0
            success = False
            print(f"\n   ❌ Error: {e}")
        
        # 记录推理后内存
        mem_after = get_memory_usage()
        
        # 计算内存变化
        mem_delta = {
            'cpu': mem_after['cpu_rss_gb'] - mem_before['cpu_rss_gb'],
            'mps_allocated': mem_after['mps_allocated_gb'] - mem_before['mps_allocated_gb'],
            'mps_reserved': mem_after['mps_reserved_gb'] - mem_before['mps_reserved_gb'],
        }
        
        result = {
            'iteration': i + 1,
            'text': text[:20],
            'time': inference_time,
            'mem_after': mem_after,
            'mem_delta': mem_delta,
            'success': success,
        }
        results.append(result)
        
        # 打印结果
        print(f"\n[Test {i+1}/10] \"{text[:20]}...\"")
        print(f"   Time: {inference_time:.2f}s")
        print(f"   CPU Memory: {mem_after['cpu_rss_gb']:.2f} GB (Δ {mem_delta['cpu']:+.3f} GB)")
        print(f"   MPS Allocated: {mem_after['mps_allocated_gb']:.2f} GB (Δ {mem_delta['mps_allocated']:+.3f} GB)")
        print(f"   MPS Reserved: {mem_after['mps_reserved_gb']:.2f} GB (Δ {mem_delta['mps_reserved']:+.3f} GB)")
        
        # 尝试清理
        if i % 3 == 2:  # 每3次推理后清理一次
            print("   >> 执行垃圾回收...")
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            mx.metal.clear_cache()
            
            mem_after_gc = get_memory_usage()
            print(f"   >> GC后 MPS: {mem_after_gc['mps_allocated_gb']:.2f} GB")
    
    # 总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    
    times = [r['time'] for r in results]
    print(f"\n推理时间趋势:")
    print(f"   Test 1-3:  {sum(times[0:3])/3:.2f}s (平均)")
    print(f"   Test 4-7:  {sum(times[3:7])/4:.2f}s (平均)")
    print(f"   Test 8-10: {sum(times[7:10])/3:.2f}s (平均)")
    
    if times[-1] > times[0] * 1.5:
        print(f"\n   ⚠️  性能衰减: {times[-1]/times[0]:.1f}x 变慢")
    else:
        print(f"\n   ✅ 性能稳定")
    
    cpu_growth = results[-1]['mem_after']['cpu_rss_gb'] - results[0]['mem_after']['cpu_rss_gb']
    mps_growth = results[-1]['mem_after']['mps_allocated_gb'] - results[0]['mem_after']['mps_allocated_gb']
    
    print(f"\n内存增长:")
    print(f"   CPU:  {cpu_growth:+.2f} GB")
    print(f"   MPS:  {mps_growth:+.2f} GB")
    
    if cpu_growth > 1.0 or mps_growth > 0.5:
        print(f"\n   ⚠️  检测到内存泄漏！")
    else:
        print(f"\n   ✅ 内存使用正常")


if __name__ == '__main__':
    test_inference_loop()

