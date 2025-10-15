#!/usr/bin/env python3
"""
IndexTTS2 MLX性能基准测试
使用标准测试命令建立baseline
"""

import time
import subprocess
import re
import statistics
from pathlib import Path

# 标准测试配置
BENCHMARK_CONFIG = {
    "text": "今天天气真不错",
    "voice": "examples/zh_vo_Main_Linaxita_2_4_24_6.wav",
    "num_runs": 3,  # 运行次数
    "warmup": 1,    # 预热次数
    "seed": 42,     # 固定seed保证可复现
}

def parse_inference_output(output: str) -> dict:
    """解析推理输出，提取时间数据"""
    data = {}
    
    # S2MEL breakdown
    s2mel_match = re.search(
        r'S2MEL breakdown: gpt_layer=([\d.]+)s, vq2emb=([\d.]+)s, prepare=([\d.]+)s, length_reg=([\d.]+)s, cfm=([\d.]+)s',
        output
    )
    if s2mel_match:
        data['s2mel_gpt_layer'] = float(s2mel_match.group(1))
        data['s2mel_vq2emb'] = float(s2mel_match.group(2))
        data['s2mel_prepare'] = float(s2mel_match.group(3))
        data['s2mel_length_reg'] = float(s2mel_match.group(4))
        data['s2mel_cfm'] = float(s2mel_match.group(5))
    
    # 主要时间
    patterns = {
        'gpt_gen_time': r'gpt_gen_time: ([\d.]+) seconds',
        'gpt_forward_time': r'gpt_forward_time: ([\d.]+) seconds',
        's2mel_time': r's2mel_time: ([\d.]+) seconds',
        'bigvgan_time': r'bigvgan_time: ([\d.]+) seconds',
        'total_time': r'Total inference time: ([\d.]+) seconds',
        'audio_length': r'Generated audio length: ([\d.]+) seconds',
        'rtf': r'RTF: ([\d.]+)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            data[key] = float(match.group(1))
    
    return data

def run_benchmark(use_mlx: bool, num_runs: int = 3, warmup: int = 1) -> list:
    """运行基准测试"""
    mode = "MLX" if use_mlx else "PyTorch"
    print(f"\n{'='*80}")
    print(f"运行 {mode} 基准测试")
    print(f"配置: {num_runs}次运行 + {warmup}次预热")
    print(f"{'='*80}")
    
    cmd_base = [
        "conda", "run", "-n", "indextts2",
        "python", "-m", "indextts.cli",
        BENCHMARK_CONFIG["text"],
        "-v", BENCHMARK_CONFIG["voice"],
        "--force",
        "--num-beams", "1",
        "--seed", str(BENCHMARK_CONFIG["seed"]),  # 固定seed
    ]
    
    if use_mlx:
        cmd_base.append("--mlx")
    
    results = []
    
    # 预热
    if warmup > 0:
        print(f"\n🔥 预热 {warmup} 次...")
        for i in range(warmup):
            print(f"  预热 {i+1}/{warmup}...", end=' ')
            subprocess.run(
                cmd_base,
                capture_output=True,
                text=True,
                cwd="/Users/bailin/index-tts"
            )
            print("完成")
    
    # 正式测试
    print(f"\n📊 正式测试 {num_runs} 次...")
    for i in range(num_runs):
        print(f"\n  Run {i+1}/{num_runs}:")
        result = subprocess.run(
            cmd_base,
            capture_output=True,
            text=True,
            cwd="/Users/bailin/index-tts"
        )
        
        if result.returncode != 0:
            print(f"    ❌ 失败: {result.stderr[:200]}")
            continue
        
        data = parse_inference_output(result.stdout + result.stderr)
        if data:
            results.append(data)
            print(f"    ✓ Total: {data.get('total_time', 0):.2f}s")
            print(f"      GPT: {data.get('gpt_gen_time', 0):.2f}s")
            print(f"      S2MEL: {data.get('s2mel_time', 0):.2f}s")
        else:
            print(f"    ⚠️  无法解析输出")
    
    return results

def calculate_statistics(results: list) -> dict:
    """计算统计数据"""
    if not results:
        return {}
    
    stats = {}
    keys = results[0].keys()
    
    for key in keys:
        values = [r[key] for r in results if key in r]
        if values:
            stats[key] = {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'stdev': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
            }
    
    return stats

def print_statistics(stats: dict, title: str):
    """打印统计结果"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    # 主要指标
    main_metrics = [
        ('total_time', '总推理时间'),
        ('gpt_gen_time', 'GPT生成时间'),
        ('gpt_forward_time', 'GPT前向时间'),
        ('s2mel_time', 'S2MEL时间'),
        ('bigvgan_time', 'BigVGAN时间'),
        ('audio_length', '音频长度'),
        ('rtf', 'RTF'),
    ]
    
    print("\n主要性能指标:")
    print("-" * 80)
    for key, label in main_metrics:
        if key in stats:
            s = stats[key]
            print(f"{label:20s}: {s['mean']:7.2f}s ± {s['stdev']:5.2f}s "
                  f"(范围: {s['min']:.2f}-{s['max']:.2f}s)")
    
    # S2MEL详细分解
    s2mel_metrics = [
        ('s2mel_gpt_layer', 'GPT Layer'),
        ('s2mel_vq2emb', 'VQ2EMB'),
        ('s2mel_prepare', 'Prepare'),
        ('s2mel_length_reg', 'Length Reg'),
        ('s2mel_cfm', 'CFM'),
    ]
    
    print("\nS2MEL详细分解:")
    print("-" * 80)
    for key, label in s2mel_metrics:
        if key in stats:
            s = stats[key]
            print(f"{label:20s}: {s['mean']:7.2f}s ± {s['stdev']:5.2f}s")

def save_baseline(stats: dict, mode: str):
    """保存baseline到文件"""
    output_file = Path(f"BASELINE_{mode}.md")
    
    with open(output_file, 'w') as f:
        f.write(f"# IndexTTS2 {mode} Baseline\n\n")
        f.write(f"测试配置:\n")
        f.write(f"- 文本: {BENCHMARK_CONFIG['text']}\n")
        f.write(f"- 音频: {BENCHMARK_CONFIG['voice']}\n")
        f.write(f"- Seed: {BENCHMARK_CONFIG['seed']} (固定)\n")
        f.write(f"- 运行次数: {BENCHMARK_CONFIG['num_runs']}\n")
        f.write(f"- 预热次数: {BENCHMARK_CONFIG['warmup']}\n\n")
        
        f.write(f"## 性能数据\n\n")
        f.write(f"| 指标 | 平均值 | 标准差 | 最小值 | 最大值 |\n")
        f.write(f"|------|--------|--------|--------|--------|\n")
        
        for key in ['total_time', 'gpt_gen_time', 'gpt_forward_time', 
                    's2mel_time', 'bigvgan_time', 'rtf']:
            if key in stats:
                s = stats[key]
                f.write(f"| {key} | {s['mean']:.2f}s | {s['stdev']:.2f}s | "
                       f"{s['min']:.2f}s | {s['max']:.2f}s |\n")
        
        f.write(f"\n## S2MEL分解\n\n")
        f.write(f"| 模块 | 平均值 | 标准差 |\n")
        f.write(f"|------|--------|--------|\n")
        
        for key in ['s2mel_gpt_layer', 's2mel_vq2emb', 's2mel_prepare',
                    's2mel_length_reg', 's2mel_cfm']:
            if key in stats:
                s = stats[key]
                f.write(f"| {key} | {s['mean']:.2f}s | {s['stdev']:.2f}s |\n")
    
    print(f"\n✅ Baseline已保存到: {output_file}")

def main():
    print("=" * 80)
    print("IndexTTS2 性能基准测试")
    print("=" * 80)
    print(f"\n测试配置:")
    print(f"  文本: {BENCHMARK_CONFIG['text']}")
    print(f"  音频: {BENCHMARK_CONFIG['voice']}")
    print(f"  Seed: {BENCHMARK_CONFIG['seed']} (固定)")
    print(f"  运行次数: {BENCHMARK_CONFIG['num_runs']}")
    print(f"  预热次数: {BENCHMARK_CONFIG['warmup']}")
    
    # 测试MLX版本
    mlx_results = run_benchmark(use_mlx=True, 
                                num_runs=BENCHMARK_CONFIG['num_runs'],
                                warmup=BENCHMARK_CONFIG['warmup'])
    
    if mlx_results:
        mlx_stats = calculate_statistics(mlx_results)
        print_statistics(mlx_stats, "MLX性能统计")
        save_baseline(mlx_stats, "MLX")
    
    # 可选：测试PyTorch版本进行对比
    # pytorch_results = run_benchmark(use_mlx=False, 
    #                                 num_runs=BENCHMARK_CONFIG['num_runs'],
    #                                 warmup=BENCHMARK_CONFIG['warmup'])
    # if pytorch_results:
    #     pytorch_stats = calculate_statistics(pytorch_results)
    #     print_statistics(pytorch_stats, "PyTorch性能统计")
    #     save_baseline(pytorch_stats, "PyTorch")
    
    print("\n" + "=" * 80)
    print("基准测试完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()

