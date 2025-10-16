#!/usr/bin/env python3
"""
IndexTTS2 性能分析和优化建议脚本

分析基准测试结果，识别性能瓶颈，提供优化建议。
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np


class BenchmarkAnalyzer:
    """基准测试结果分析器"""
    
    def __init__(self, results_file: str):
        """
        Args:
            results_file: 基准测试结果JSON文件路径
        """
        self.results_file = results_file
        self.data = None
        self.results = []
        self.metadata = {}
        self.summary = {}
        
        self.load_results()
    
    def load_results(self):
        """加载测试结果"""
        if not os.path.exists(self.results_file):
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        
        with open(self.results_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.metadata = self.data.get("metadata", {})
        self.summary = self.data.get("summary", {})
        self.results = self.data.get("results", [])
        
        print(f"✅ Loaded {len(self.results)} test results from {self.results_file}")
    
    def analyze_module_timing(self) -> Dict[str, Any]:
        """分析各模块的耗时"""
        module_times = {
            "gpt_gen_time": [],
            "gpt_forward_time": [],
            "s2mel_time": [],
            "bigvgan_time": [],
            "text_processing_time": [],
        }
        
        for result in self.results:
            for module in module_times.keys():
                if module in result:
                    module_times[module].append(result[module])
        
        # 计算统计
        analysis = {}
        total_module_time = 0
        
        for module, times in module_times.items():
            if times:
                times_arr = np.array(times)
                module_mean = np.mean(times_arr)
                total_module_time += module_mean
                
                analysis[module] = {
                    "mean": float(module_mean),
                    "median": float(np.median(times_arr)),
                    "std": float(np.std(times_arr)),
                    "min": float(np.min(times_arr)),
                    "max": float(np.max(times_arr)),
                    "total": float(np.sum(times_arr)),
                }
        
        # 计算百分比
        if total_module_time > 0:
            for module, stats in analysis.items():
                stats["percentage"] = (stats["mean"] / total_module_time) * 100
        
        # 排序找出最慢的模块
        sorted_modules = sorted(
            analysis.items(),
            key=lambda x: x[1]["mean"],
            reverse=True
        )
        
        return {
            "module_stats": analysis,
            "sorted_modules": [(m, s["mean"], s.get("percentage", 0)) for m, s in sorted_modules],
            "total_module_time": total_module_time,
        }
    
    def analyze_memory_usage(self) -> Dict[str, Any]:
        """分析内存使用情况"""
        gpu_mem_peaks = []
        gpu_mem_allocated = []
        cpu_mem_peaks = []
        
        for result in self.results:
            if "memory_peak" in result:
                mem_peak = result["memory_peak"]
                if mem_peak.get("gpu_max_allocated_mb", 0) > 0:
                    gpu_mem_peaks.append(mem_peak["gpu_max_allocated_mb"])
                if mem_peak.get("gpu_allocated_mb", 0) > 0:
                    gpu_mem_allocated.append(mem_peak["gpu_allocated_mb"])
                if mem_peak.get("cpu_memory_mb", 0) > 0:
                    cpu_mem_peaks.append(mem_peak["cpu_memory_mb"])
        
        analysis = {}
        
        if gpu_mem_peaks:
            analysis["gpu_max_allocated"] = {
                "mean": float(np.mean(gpu_mem_peaks)),
                "median": float(np.median(gpu_mem_peaks)),
                "max": float(np.max(gpu_mem_peaks)),
                "min": float(np.min(gpu_mem_peaks)),
                "unit": "MB"
            }
        
        if gpu_mem_allocated:
            analysis["gpu_allocated"] = {
                "mean": float(np.mean(gpu_mem_allocated)),
                "median": float(np.median(gpu_mem_allocated)),
                "max": float(np.max(gpu_mem_allocated)),
                "min": float(np.min(gpu_mem_allocated)),
                "unit": "MB"
            }
        
        if cpu_mem_peaks:
            analysis["cpu_memory"] = {
                "mean": float(np.mean(cpu_mem_peaks)),
                "median": float(np.median(cpu_mem_peaks)),
                "max": float(np.max(cpu_mem_peaks)),
                "min": float(np.min(cpu_mem_peaks)),
                "unit": "MB"
            }
        
        return analysis
    
    def analyze_text_length_correlation(self) -> Dict[str, Any]:
        """分析文本长度与性能的关系"""
        data_points = []
        
        for result in self.results:
            if "text_tokens_count" in result and "total_time" in result:
                data_points.append({
                    "tokens": result["text_tokens_count"],
                    "segments": result.get("segments_count", 0),
                    "total_time": result["total_time"],
                    "rtf": result.get("rtf", 0),
                    "audio_duration": result.get("audio_duration", 0),
                })
        
        if not data_points:
            return {}
        
        # 按token数排序
        data_points.sort(key=lambda x: x["tokens"])
        
        tokens = [p["tokens"] for p in data_points]
        times = [p["total_time"] for p in data_points]
        
        # 简单线性相关分析
        if len(tokens) > 1:
            correlation = np.corrcoef(tokens, times)[0, 1]
        else:
            correlation = 0
        
        return {
            "data_points": data_points,
            "correlation": float(correlation),
            "tokens_range": [min(tokens), max(tokens)],
            "time_range": [min(times), max(times)],
        }
    
    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """识别性能瓶颈"""
        bottlenecks = []
        
        # 分析模块耗时
        module_analysis = self.analyze_module_timing()
        if module_analysis["sorted_modules"]:
            top_module = module_analysis["sorted_modules"][0]
            module_name, module_time, module_pct = top_module
            
            if module_pct > 40:  # 如果某个模块占比超过40%
                bottlenecks.append({
                    "type": "module_bottleneck",
                    "severity": "high" if module_pct > 60 else "medium",
                    "module": module_name,
                    "percentage": module_pct,
                    "mean_time": module_time,
                    "description": f"{module_name} 占用了 {module_pct:.1f}% 的推理时间",
                })
        
        # 分析RTF
        if self.summary and "rtf" in self.summary:
            mean_rtf = self.summary["rtf"].get("mean", 0)
            if mean_rtf > 0.5:  # RTF > 0.5 说明不够实时
                bottlenecks.append({
                    "type": "rtf_bottleneck",
                    "severity": "high" if mean_rtf > 1.0 else "medium",
                    "value": mean_rtf,
                    "description": f"平均RTF为 {mean_rtf:.3f}，{'不能' if mean_rtf > 1.0 else '接近不能'}实时处理",
                })
        
        # 分析内存使用
        memory_analysis = self.analyze_memory_usage()
        if "gpu_max_allocated" in memory_analysis:
            max_mem = memory_analysis["gpu_max_allocated"]["max"]
            if max_mem > 8000:  # 超过8GB
                bottlenecks.append({
                    "type": "memory_bottleneck",
                    "severity": "medium",
                    "value": max_mem,
                    "description": f"峰值GPU内存使用 {max_mem:.0f} MB，可能限制批处理能力",
                })
        
        return bottlenecks
    
    def generate_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """生成优化建议"""
        suggestions = []
        
        bottlenecks = self.identify_bottlenecks()
        module_analysis = self.analyze_module_timing()
        config = self.metadata.get("config", {})
        
        # 基于配置的建议
        if not config.get("fp16", False):
            suggestions.append({
                "priority": "high",
                "category": "precision",
                "title": "启用FP16推理",
                "description": "当前使用FP32，启用FP16可以：",
                "benefits": [
                    "减少约50%的GPU内存占用",
                    "提升推理速度（通常20-40%）",
                    "音质损失极小"
                ],
                "implementation": "在初始化时设置 use_fp16=True",
                "estimated_improvement": "20-40%",
            })
        
        if not config.get("cuda_kernel", False):
            system_info = self.metadata.get("system_info", {})
            if system_info.get("cuda_available", False):
                suggestions.append({
                    "priority": "high",
                    "category": "optimization",
                    "title": "启用BigVGAN CUDA内核",
                    "description": "使用优化的CUDA内核可以显著提升BigVGAN性能",
                    "benefits": [
                        "BigVGAN模块加速30-50%",
                        "减少内存占用"
                    ],
                    "implementation": "在初始化时设置 use_cuda_kernel=True",
                    "estimated_improvement": "10-20% (整体)",
                })
        
        # 基于瓶颈的建议
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "module_bottleneck":
                module = bottleneck["module"]
                
                if "gpt" in module.lower():
                    suggestions.append({
                        "priority": "high",
                        "category": "model",
                        "title": "优化GPT模块",
                        "description": f"GPT模块是主要瓶颈（{bottleneck['percentage']:.1f}%）",
                        "benefits": [
                            "使用KV-cache（已部分实现）",
                            "考虑DeepSpeed加速",
                            "尝试torch.compile（PyTorch 2.0+）"
                        ],
                        "implementation": "使用 use_deepspeed=True 或探索 torch.compile",
                        "estimated_improvement": "15-30%",
                    })
                
                elif "s2mel" in module.lower():
                    suggestions.append({
                        "priority": "medium",
                        "category": "model",
                        "title": "优化S2Mel扩散模块",
                        "description": f"S2Mel模块耗时较长（{bottleneck['percentage']:.1f}%）",
                        "benefits": [
                            "减少扩散步数（如从25降到15-20）",
                            "使用更高效的采样器（DDIM）",
                            "算子融合优化"
                        ],
                        "implementation": "调整 diffusion_steps 参数",
                        "estimated_improvement": "10-20%",
                    })
                
                elif "bigvgan" in module.lower():
                    suggestions.append({
                        "priority": "medium",
                        "category": "vocoder",
                        "title": "优化BigVGAN声码器",
                        "description": f"BigVGAN耗时占比 {bottleneck['percentage']:.1f}%",
                        "benefits": [
                            "启用CUDA内核（如果未启用）",
                            "考虑使用更轻量的声码器"
                        ],
                        "implementation": "use_cuda_kernel=True",
                        "estimated_improvement": "15-30%",
                    })
        
        # 通用优化建议
        suggestions.append({
            "priority": "medium",
            "category": "inference",
            "title": "分句策略优化",
            "description": "调整max_text_tokens_per_segment以平衡质量和速度",
            "benefits": [
                "更大的值减少分句，提升速度但可能影响质量",
                "更小的值提升质量但增加推理时间",
                "建议值：80-200"
            ],
            "implementation": f"当前值：{config.get('max_text_tokens_per_segment', 120)}，可尝试调整",
            "estimated_improvement": "5-15%",
        })
        
        suggestions.append({
            "priority": "low",
            "category": "system",
            "title": "流式推理",
            "description": "对于长文本，考虑使用流式推理",
            "benefits": [
                "降低首字延迟",
                "提升用户体验",
                "更好的内存管理"
            ],
            "implementation": "使用 stream_return=True 参数",
            "estimated_improvement": "用户体验提升",
        })
        
        # 排序：高优先级在前
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        return suggestions
    
    def print_analysis(self):
        """打印完整分析报告"""
        print("\n" + "="*70)
        print("📊 IndexTTS2 Performance Analysis Report")
        print("="*70)
        
        # 基本信息
        print(f"\n📋 Test Information:")
        print(f"   Test Date: {self.metadata.get('timestamp', 'unknown')}")
        system_info = self.metadata.get("system_info", {})
        print(f"   Conda Env: {system_info.get('conda_env', 'unknown')}")
        print(f"   Device: {system_info.get('gpu_name', system_info.get('device_type', 'unknown'))}")
        print(f"   PyTorch: {system_info.get('pytorch_version', 'unknown')}")
        
        config = self.metadata.get("config", {})
        print(f"\n⚙️  Configuration:")
        print(f"   FP16: {config.get('fp16', False)}")
        print(f"   CUDA Kernel: {config.get('cuda_kernel', False)}")
        print(f"   DeepSpeed: {config.get('deepspeed', False)}")
        print(f"   Max Tokens/Segment: {config.get('max_text_tokens_per_segment', 120)}")
        
        # 汇总统计
        if self.summary:
            print(f"\n📈 Overall Performance:")
            print(f"   Tests Completed: {self.summary.get('count', 0)}")
            
            if "total_time" in self.summary:
                tt = self.summary["total_time"]
                print(f"   Mean Total Time: {tt.get('mean', 0):.3f}s (±{tt.get('std', 0):.3f}s)")
                print(f"   Range: {tt.get('min', 0):.3f}s - {tt.get('max', 0):.3f}s")
            
            if "rtf" in self.summary and self.summary["rtf"].get("mean", 0) > 0:
                rtf = self.summary["rtf"]
                print(f"   Mean RTF: {rtf.get('mean', 0):.4f}")
                print(f"   Range: {rtf.get('min', 0):.4f} - {rtf.get('max', 0):.4f}")
        
        # 模块耗时分析
        print(f"\n⏱️  Module Timing Analysis:")
        module_analysis = self.analyze_module_timing()
        if module_analysis["sorted_modules"]:
            print(f"   Total Module Time: {module_analysis['total_module_time']:.3f}s (mean)")
            print(f"\n   Breakdown:")
            for module, time, pct in module_analysis["sorted_modules"]:
                bar_length = int(pct / 2)  # 50% = 25 chars
                bar = "█" * bar_length
                print(f"   {module:25s} {time:6.3f}s {bar:25s} {pct:5.1f}%")
        
        # 内存分析
        print(f"\n💾 Memory Usage:")
        memory_analysis = self.analyze_memory_usage()
        if "gpu_max_allocated" in memory_analysis:
            gpu = memory_analysis["gpu_max_allocated"]
            print(f"   GPU Peak: {gpu['mean']:.1f} MB (mean), {gpu['max']:.1f} MB (max)")
        if "cpu_memory" in memory_analysis:
            cpu = memory_analysis["cpu_memory"]
            print(f"   CPU: {cpu['mean']:.1f} MB (mean), {cpu['max']:.1f} MB (max)")
        
        # 瓶颈分析
        print(f"\n🔍 Bottleneck Analysis:")
        bottlenecks = self.identify_bottlenecks()
        if bottlenecks:
            for i, bottleneck in enumerate(bottlenecks, 1):
                severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(bottleneck["severity"], "⚪")
                print(f"\n   {severity_emoji} Bottleneck {i} [{bottleneck['severity'].upper()}]:")
                print(f"      {bottleneck['description']}")
        else:
            print("   ✅ No significant bottlenecks detected")
        
        # 优化建议
        print(f"\n💡 Optimization Suggestions:")
        suggestions = self.generate_optimization_suggestions()
        
        for i, sug in enumerate(suggestions, 1):
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(sug["priority"], "⚪")
            print(f"\n   {priority_emoji} Suggestion {i} [{sug['priority'].upper()}] - {sug['category'].upper()}")
            print(f"      Title: {sug['title']}")
            print(f"      {sug['description']}")
            
            if "benefits" in sug:
                print(f"      Benefits:")
                for benefit in sug["benefits"]:
                    print(f"        • {benefit}")
            
            if "implementation" in sug:
                print(f"      Implementation: {sug['implementation']}")
            
            if "estimated_improvement" in sug:
                print(f"      Est. Improvement: {sug['estimated_improvement']}")
        
        print("\n" + "="*70)
        print("📄 For detailed optimization directions, see docs/optimization_directions.md")
        print("="*70 + "\n")
    
    def save_analysis_report(self, output_file: str):
        """保存分析报告为JSON"""
        report = {
            "metadata": self.metadata,
            "summary": self.summary,
            "module_timing": self.analyze_module_timing(),
            "memory_usage": self.analyze_memory_usage(),
            "text_length_correlation": self.analyze_text_length_correlation(),
            "bottlenecks": self.identify_bottlenecks(),
            "optimization_suggestions": self.generate_optimization_suggestions(),
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Analysis report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze IndexTTS2 benchmark results and provide optimization suggestions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to benchmark results JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for analysis report (JSON)"
    )
    
    args = parser.parse_args()
    
    # 分析结果
    analyzer = BenchmarkAnalyzer(args.results_file)
    analyzer.print_analysis()
    
    # 保存分析报告
    if args.output:
        analyzer.save_analysis_report(args.output)
    else:
        # 默认保存到与结果文件相同的目录
        base_name = os.path.splitext(args.results_file)[0]
        output_file = f"{base_name}_analysis.json"
        analyzer.save_analysis_report(output_file)


if __name__ == "__main__":
    main()


