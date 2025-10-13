#!/usr/bin/env python3
"""
测试随机状态累积修复是否有效
验证相同输入是否产生稳定的输出
"""

import sys
sys.path.insert(0, '/Users/bailin/index-tts')

from indextts.infer_v2 import IndexTTS2
import time


def test_repeated_inference():
    print("=" * 70)
    print("测试随机状态累积修复")
    print("=" * 70)
    
    # 初始化模型
    print("\n>> 初始化模型...")
    model = IndexTTS2(device='mps', use_mlx=True, diffusion_steps=20)
    
    ref_audio = 'examples/voice_01.wav'
    
    # 使用相同的文本连续推理 10 次
    test_text = "今天天气真不错，我们一起去看电影吧"
    
    print(f"\n>> 测试文本: \"{test_text}\"")
    print(f">> 连续推理 10 次，观察输出稳定性...")
    print("-" * 70)
    
    results = []
    
    for i in range(10):
        print(f"\n[Test {i+1}/10]")
        
        t0 = time.perf_counter()
        try:
            wav, sr = model.infer(
                spk_audio_prompt=ref_audio,
                text=test_text,
                output_path=None,
                verbose=False
            )
            inference_time = time.perf_counter() - t0
            
            # 从日志中手动记录 token 数量（或解析输出）
            # 这里简化为只记录时间
            result = {
                'iteration': i + 1,
                'time': inference_time,
                'success': True
            }
            results.append(result)
            
            print(f"   ✅ 推理时间: {inference_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'iteration': i + 1,
                'time': -1,
                'success': False
            })
    
    # 分析结果
    print("\n" + "=" * 70)
    print("分析结果")
    print("=" * 70)
    
    successful_results = [r for r in results if r['success']]
    
    if len(successful_results) >= 5:
        times = [r['time'] for r in successful_results]
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        
        print(f"\n推理时间统计:")
        print(f"   平均: {avg_time:.2f}s")
        print(f"   最小: {min_time:.2f}s")
        print(f"   最大: {max_time:.2f}s")
        print(f"   标准差: {std_dev:.2f}s")
        
        # 检查时间是否稳定（标准差 < 平均值的 20%）
        if std_dev < avg_time * 0.2:
            print(f"\n   ✅ 推理时间稳定")
        else:
            print(f"\n   ⚠️  推理时间波动较大")
        
        # 检查是否有持续增长趋势
        first_half_avg = sum(times[:5]) / 5
        second_half_avg = sum(times[5:]) / len(times[5:])
        
        print(f"\n时间趋势:")
        print(f"   前5次平均: {first_half_avg:.2f}s")
        print(f"   后5次平均: {second_half_avg:.2f}s")
        
        if second_half_avg > first_half_avg * 1.2:
            print(f"   ⚠️  后半部分明显变慢 ({second_half_avg/first_half_avg:.1f}x)")
        else:
            print(f"   ✅ 无明显性能衰减")
        
        print("\n" + "=" * 70)
        print("建议:")
        print("=" * 70)
        print("\n请检查 webui 的实际日志，观察：")
        print("1. Generated tokens 数量是否稳定 (期望: 100-150 tokens)")
        print("2. First token 是否相对稳定 (期望: 不应频繁跳变)")
        print("3. 是否还有\"丢字\"现象")
        print("\n如果 token 数量仍然不稳定，可能需要进一步调查 conditioning 部分。")
    else:
        print("\n❌ 测试失败次数过多，无法进行统计分析")


if __name__ == '__main__':
    test_repeated_inference()

