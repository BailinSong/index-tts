#!/usr/bin/env python3
"""
第一版优化基准测试 (Conditioning缓存后)
4次运行，忽略第1次预热，后3次计算平均值作为V1基准
"""

from indextts.infer_v2 import create_tts
import time
import statistics

print("=" * 80)
print("第一版优化基准测试（V1 Baseline with Conditioning Cache）")
print("=" * 80)

# 配置
CONFIG = {
    "texts": [
        "今天天气真不错",
        "到底应该吃什么", 
        "你为什么不愿意",
        "今天天气真不错",  # 第4次重复第1次，验证稳定性
    ],
    "voice": "examples/zh_vo_Main_Linaxita_2_4_24_6.wav",
    "seed": 42,
    "num_runs": 4,  # 总共4次
    "warmup": 1,    # 忽略第1次
}

print(f"\n测试配置:")
print(f"  文本: {CONFIG['texts']}")
print(f"  Voice: {CONFIG['voice']}")
print(f"  Seed: {CONFIG['seed']} (固定)")
print(f"  总运行次数: {CONFIG['num_runs']}")
print(f"  预热次数: {CONFIG['warmup']} (忽略第1次)")
print(f"  基准数据: 后3次的平均值")

# 初始化（只初始化一次，保持缓存）
print(f"\n初始化 IndexTTS2...")
tts = create_tts(use_mlx=True)

# 运行测试
results = []

for i in range(CONFIG['num_runs']):
    run_num = i + 1
    is_warmup = (i < CONFIG['warmup'])
    label = "预热" if is_warmup else f"Run {run_num - CONFIG['warmup']}"
    
    # 选择对应的文本
    text = CONFIG['texts'][i]
    
    print(f"\n{'='*80}")
    print(f"[{label}] 第{run_num}次运行 - 文本: '{text}'")
    print(f"{'='*80}")
    
    t0 = time.time()
    tts.infer(
        spk_audio_prompt=CONFIG['voice'],
        text=text,
        output_path=f"gen_v1_{run_num}.wav",
        num_beams=1,
        seed=CONFIG['seed']
    )
    elapsed = time.time() - t0
    
    if not is_warmup:
        results.append({
            'time': elapsed,
            'text': text,
            'run': run_num - CONFIG['warmup']
        })
        print(f"\n✅ {label}完成: {elapsed:.2f}s (计入基准)")
    else:
        print(f"\n🔥 预热完成: {elapsed:.2f}s (忽略，不计入基准)")

# 计算统计
print(f"\n{'='*80}")
print(f"V1基准统计（后3次运行）")
print(f"{'='*80}")

times = [r['time'] for r in results]
mean_time = statistics.mean(times)
median_time = statistics.median(times)
stdev_time = statistics.stdev(times) if len(times) > 1 else 0
min_time = min(times)
max_time = max(times)

print(f"\n详细数据:")
for r in results:
    print(f"  Run {r['run']}: {r['time']:.2f}s - '{r['text']}'")

print(f"\n统计:")
print(f"  平均值: {mean_time:.2f}s  ⭐ V1基准")
print(f"  中位数: {median_time:.2f}s")
print(f"  标准差: ±{stdev_time:.2f}s")
print(f"  范围: {min_time:.2f}s - {max_time:.2f}s")
print(f"  变异系数: {stdev_time/mean_time*100:.1f}%")

# 保存V1基准
print(f"\n保存V1基准数据...")
with open('BASELINE_V1_OPTIMIZED.md', 'w') as f:
    f.write("# 第一版优化基准（V1 Baseline）\n\n")
    f.write("## 优化内容\n\n")
    f.write("✅ Conditioning缓存（Pure MLX GPT + 缓存优化）\n\n")
    f.write("## 测试配置\n\n")
    f.write(f"- 文本: {CONFIG['texts'][1:]}  # 后3次\n")
    f.write(f"- Voice: {CONFIG['voice']}\n")
    f.write(f"- Seed: {CONFIG['seed']} (固定)\n")
    f.write(f"- 测试方法: 4次运行，忽略第1次预热，后3次平均\n\n")
    f.write("## V1基准数据\n\n")
    f.write(f"| 指标 | 数值 |\n")
    f.write(f"|------|------|\n")
    f.write(f"| **平均值（V1基准）** | **{mean_time:.2f}s** |\n")
    f.write(f"| 中位数 | {median_time:.2f}s |\n")
    f.write(f"| 标准差 | ±{stdev_time:.2f}s |\n")
    f.write(f"| 范围 | {min_time:.2f}s - {max_time:.2f}s |\n")
    f.write(f"| RTF | {mean_time/2.53:.2f} |\n\n")
    f.write("## 详细数据\n\n")
    f.write(f"| Run | 文本 | 时间 |\n")
    f.write(f"|-----|------|------|\n")
    for r in results:
        f.write(f"| {r['run']} | {r['text']} | {r['time']:.2f}s |\n")
    f.write(f"\n## 对比V0基准\n\n")
    f.write(f"V0基准（Pure MLX GPT，无缓存）: 16.62s (RTF=6.57)\n")
    f.write(f"V1基准（+ Conditioning缓存）: {mean_time:.2f}s (RTF={mean_time/2.53:.2f})\n\n")
    if mean_time < 16.62:
        improvement = (16.62 - mean_time) / 16.62 * 100
        f.write(f"V1 vs V0提升: {16.62-mean_time:.2f}s ({improvement:.1f}%)\n\n")
    f.write("## 后续优化参考\n\n")
    f.write(f"**所有后续优化都应该与V1基准({mean_time:.2f}s)对比，而非V0。**\n\n")
    f.write("测试命令:\n")
    f.write("```bash\n")
    f.write(f"python -m indextts.cli \"今天天气真不错\" \\\n")
    f.write(f"  -v {CONFIG['voice']} \\\n")
    f.write(f"  --force --mlx --num-beams 1 --seed {CONFIG['seed']}\n")
    f.write("```\n")

print(f"✅ V1基准已保存到: BASELINE_V1_OPTIMIZED.md")

print(f"\n{'='*80}")
print(f"V1基准建立完成")
print(f"{'='*80}")
print(f"\n📊 **V1基准值: {mean_time:.2f}s (RTF={mean_time/2.53:.2f})**")
print(f"\n后续所有优化都应该与此基准对比！")
print(f"{'='*80}")

