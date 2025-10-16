#!/usr/bin/env python3
"""
IndexTTS2 性能基线测试脚本

批量测试不同配置下的推理性能，收集详细的性能指标。
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 添加项目路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from indextts.infer_v2 import IndexTTS2
from indextts.utils.performance_monitor import (
    PerformanceMonitor, 
    BenchmarkCollector, 
    InferenceMetrics
)


def check_conda_environment():
    """检查是否在正确的conda环境中"""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env != "indextts2":
        print(f"⚠️  Warning: Not running in 'indextts2' conda environment!")
        print(f"   Current environment: {conda_env or '(none)'}")
        print(f"   Please activate: conda activate indextts2")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        print(f"✅ Running in conda environment: {conda_env}")


def get_system_info() -> Dict[str, Any]:
    """获取系统信息"""
    info = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        info["device_type"] = "Apple MPS"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        info["device_type"] = "Intel XPU"
    else:
        info["device_type"] = "CPU"
    
    return info


def load_test_cases(cases_file: str = "examples/cases.jsonl") -> List[Dict[str, Any]]:
    """加载测试用例"""
    cases = []
    if not os.path.exists(cases_file):
        print(f"❌ Test cases file not found: {cases_file}")
        return cases
    
    with open(cases_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
                case['line_number'] = line_num
                cases.append(case)
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Failed to parse line {line_num}: {e}")
    
    print(f"✅ Loaded {len(cases)} test cases from {cases_file}")
    return cases


def run_single_test(
    tts: IndexTTS2,
    monitor: PerformanceMonitor,
    test_case: Dict[str, Any],
    test_id: int,
    output_dir: str,
    max_text_tokens_per_segment: int = 120,
    verbose: bool = False
) -> InferenceMetrics:
    """运行单个测试用例"""
    
    # 准备测试参数
    prompt_audio = os.path.join("examples", test_case.get("prompt_audio", "voice_01.wav"))
    text = test_case.get("text", "")
    emo_mode = test_case.get("emo_mode", 0)
    emo_audio = test_case.get("emo_audio")
    emo_weight = test_case.get("emo_weight", 1.0)
    
    if emo_audio:
        emo_audio = os.path.join("examples", emo_audio)
    
    # 情感向量
    emo_vector = None
    if emo_mode == 2:
        emo_vector = [
            test_case.get(f"emo_vec_{i}", 0) 
            for i in range(1, 9)
        ]
    
    # 情感文本
    use_emo_text = (emo_mode == 3)
    emo_text = test_case.get("emo_text", None) if use_emo_text else None
    
    output_path = os.path.join(output_dir, f"test_{test_id:03d}.wav")
    
    test_name = f"Case {test_id}: {text[:50]}..."
    
    # 开始性能监控
    metrics = monitor.start_inference(
        test_name=test_name,
        text_length=len(text),
        use_fp16=tts.use_fp16,
        use_cuda_kernel=tts.use_cuda_kernel,
        dtype=str(tts.dtype) if tts.dtype else "float32"
    )
    
    # 文本处理计时
    with monitor.timer("text_processing_time"):
        text_tokens_list = tts.tokenizer.tokenize(text)
        segments = tts.tokenizer.split_segments(
            text_tokens_list, 
            max_text_tokens_per_segment
        )
    
    metrics.text_tokens_count = len(text_tokens_list)
    metrics.segments_count = len(segments)
    
    print(f"\n{'='*70}")
    print(f"🧪 Test {test_id}: {test_case.get('prompt_audio', 'unknown')}")
    print(f"   Text: {text[:60]}{'...' if len(text) > 60 else ''}")
    print(f"   Tokens: {metrics.text_tokens_count}, Segments: {metrics.segments_count}")
    print(f"{'='*70}")
    
    # 推理
    start_time = time.perf_counter()
    
    try:
        # 使用自定义的推理逻辑以捕获更多性能数据
        result = tts.infer(
            spk_audio_prompt=prompt_audio,
            text=text,
            output_path=output_path,
            emo_audio_prompt=emo_audio,
            emo_alpha=emo_weight,
            emo_vector=emo_vector,
            use_emo_text=use_emo_text,
            emo_text=emo_text,
            verbose=verbose,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
            # 使用默认参数
            do_sample=True,
            top_p=0.8,
            top_k=30,
            temperature=0.8,
            max_mel_tokens=1500,
        )
        
        end_time = time.perf_counter()
        metrics.total_time = end_time - start_time
        
        # 读取生成的音频文件以获取时长
        if os.path.exists(output_path):
            import torchaudio
            waveform, sample_rate = torchaudio.load(output_path)
            metrics.audio_duration = waveform.shape[1] / sample_rate
        
        # 从IndexTTS2的输出中提取模块耗时
        # 注意：这些时间已经在infer_v2.py中打印出来了
        # 我们需要从控制台输出中解析，或者修改infer_v2.py返回这些数据
        # 暂时使用估算
        
    except Exception as e:
        print(f"❌ Test {test_id} failed: {e}")
        import traceback
        traceback.print_exc()
        metrics.total_time = time.perf_counter() - start_time
        return None
    
    # 结束性能监控
    metrics = monitor.end_inference()
    
    # 打印摘要
    monitor.print_summary(metrics)
    
    return metrics


def run_benchmark(
    model_dir: str = "checkpoints",
    config_path: str = None,
    test_cases: List[Dict[str, Any]] = None,
    output_dir: str = "outputs/benchmark",
    fp16: bool = False,
    cuda_kernel: bool = False,
    deepspeed: bool = False,
    max_text_tokens_per_segment: int = 120,
    warmup: bool = True,
    verbose: bool = False,
    device: str = None,
) -> BenchmarkCollector:
    """运行基准测试"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化收集器
    collector = BenchmarkCollector()
    
    # 设置元数据
    system_info = get_system_info()
    collector.set_metadata(
        timestamp=datetime.now().isoformat(),
        system_info=system_info,
        model_dir=model_dir,
        config={
            "fp16": fp16,
            "cuda_kernel": cuda_kernel,
            "deepspeed": deepspeed,
            "max_text_tokens_per_segment": max_text_tokens_per_segment,
            "device": device,
        }
    )
    
    print("\n" + "="*70)
    print("🚀 IndexTTS2 Performance Baseline Benchmark")
    print("="*70)
    print(f"\n📋 Configuration:")
    print(f"   Model Dir: {model_dir}")
    print(f"   Config: {config_path or 'default'}")
    print(f"   Device: {device or 'auto-detect'}")
    print(f"   FP16: {fp16}")
    print(f"   CUDA Kernel: {cuda_kernel}")
    print(f"   DeepSpeed: {deepspeed}")
    print(f"   Max Tokens/Segment: {max_text_tokens_per_segment}")
    print(f"   Test Cases: {len(test_cases)}")
    print(f"   Warmup: {warmup}")
    
    # 加载模型
    print(f"\n⏳ Loading model...")
    model_load_start = time.perf_counter()
    
    if config_path is None:
        config_path = os.path.join(model_dir, "config.yaml")
    
    tts = IndexTTS2(
        cfg_path=config_path,
        model_dir=model_dir,
        use_fp16=fp16,
        device=device,
        use_cuda_kernel=cuda_kernel,
        use_deepspeed=deepspeed,
    )
    
    model_load_time = time.perf_counter() - model_load_start
    print(f"✅ Model loaded in {model_load_time:.2f}s")
    print(f"   Device: {tts.device}")
    print(f"   FP16: {tts.use_fp16}")
    
    # 创建性能监控器
    monitor = PerformanceMonitor(device=tts.device, enable_profiling=True)
    
    # Warmup run
    if warmup and test_cases:
        print(f"\n🔥 Running warmup...")
        warmup_case = test_cases[0]
        try:
            run_single_test(
                tts, monitor, warmup_case, 0, output_dir,
                max_text_tokens_per_segment, verbose=False
            )
            print(f"✅ Warmup completed")
            monitor.reset()
        except Exception as e:
            print(f"⚠️  Warmup failed: {e}")
    
    # 运行所有测试
    print(f"\n{'='*70}")
    print(f"🧪 Running {len(test_cases)} test cases...")
    print(f"{'='*70}\n")
    
    for i, test_case in enumerate(test_cases, 1):
        metrics = run_single_test(
            tts, monitor, test_case, i, output_dir,
            max_text_tokens_per_segment, verbose
        )
        
        if metrics:
            collector.add_result(metrics)
        
        # 清理GPU缓存
        if tts.device.startswith("cuda"):
            torch.cuda.empty_cache()
        
        monitor.reset()
    
    return collector


def main():
    parser = argparse.ArgumentParser(
        description="IndexTTS2 Performance Baseline Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model_dir", type=str, default="checkpoints",
        help="Model checkpoints directory"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (default: {model_dir}/config.yaml)"
    )
    parser.add_argument(
        "--cases", type=str, default="examples/cases.jsonl",
        help="Path to test cases file"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for results (default: outputs/benchmark_{timestamp})"
    )
    parser.add_argument(
        "--fp16", action="store_true", default=False,
        help="Use FP16 for inference"
    )
    parser.add_argument(
        "--cuda_kernel", action="store_true", default=False,
        help="Use CUDA kernel for BigVGAN"
    )
    parser.add_argument(
        "--deepspeed", action="store_true", default=False,
        help="Use DeepSpeed acceleration"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=120,
        help="Max tokens per segment"
    )
    parser.add_argument(
        "--no_warmup", action="store_true", default=False,
        help="Skip warmup run"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Enable verbose output"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (cuda:0, mps, cpu, etc.)"
    )
    parser.add_argument(
        "--skip_env_check", action="store_true", default=False,
        help="Skip conda environment check"
    )
    
    args = parser.parse_args()
    
    # 环境检查
    if not args.skip_env_check:
        check_conda_environment()
    
    # 加载测试用例
    test_cases = load_test_cases(args.cases)
    if not test_cases:
        print("❌ No test cases loaded. Exiting.")
        sys.exit(1)
    
    # 输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/benchmark_{timestamp}"
    
    # 运行基准测试
    collector = run_benchmark(
        model_dir=args.model_dir,
        config_path=args.config,
        test_cases=test_cases,
        output_dir=args.output_dir,
        fp16=args.fp16,
        cuda_kernel=args.cuda_kernel,
        deepspeed=args.deepspeed,
        max_text_tokens_per_segment=args.max_tokens,
        warmup=not args.no_warmup,
        verbose=args.verbose,
        device=args.device,
    )
    
    # 打印汇总统计
    print("\n" + "="*70)
    print("📊 Benchmark Summary")
    print("="*70)
    
    summary = collector.get_summary_stats()
    if summary:
        print(f"\n✅ Completed {summary['count']} tests")
        print(f"\n⏱️  Total Time:")
        print(f"   Mean:   {summary['total_time']['mean']:.3f}s")
        print(f"   Median: {summary['total_time']['median']:.3f}s")
        print(f"   Std:    {summary['total_time']['std']:.3f}s")
        print(f"   Range:  {summary['total_time']['min']:.3f}s - {summary['total_time']['max']:.3f}s")
        
        if summary['rtf']['mean'] > 0:
            print(f"\n🎯 RTF (Real-Time Factor):")
            print(f"   Mean:   {summary['rtf']['mean']:.4f}")
            print(f"   Median: {summary['rtf']['median']:.4f}")
            print(f"   Range:  {summary['rtf']['min']:.4f} - {summary['rtf']['max']:.4f}")
        
        if summary['gpu_memory_mb']['mean'] > 0:
            print(f"\n💾 GPU Memory:")
            print(f"   Mean Peak: {summary['gpu_memory_mb']['mean']:.1f} MB")
            print(f"   Max Peak:  {summary['gpu_memory_mb']['max']:.1f} MB")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.output_dir, f"benchmark_baseline_{timestamp}.json")
    collector.save_json(json_path)
    
    print(f"\n{'='*70}")
    print(f"✅ Benchmark completed!")
    print(f"📁 Results saved to: {args.output_dir}")
    print(f"📄 JSON report: {json_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

