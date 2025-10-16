"""
性能监控工具类

用于追踪和记录IndexTTS推理过程中的性能指标，包括时间、GPU/CPU内存使用等。
"""

import time
import json
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
import torch


@dataclass
class TimingStats:
    """时间统计数据结构"""
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    
    def __post_init__(self):
        if self.duration == 0.0 and self.end_time > 0.0:
            self.duration = self.end_time - self.start_time


@dataclass
class MemoryStats:
    """内存统计数据结构"""
    # GPU 内存 (bytes)
    gpu_allocated: int = 0
    gpu_reserved: int = 0
    gpu_max_allocated: int = 0
    gpu_max_reserved: int = 0
    
    # CPU 内存 (bytes)
    cpu_memory: int = 0
    cpu_memory_percent: float = 0.0
    
    def to_mb(self, value: int) -> float:
        """转换字节到MB"""
        return value / (1024 * 1024)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，自动转换为MB单位"""
        return {
            "gpu_allocated_mb": self.to_mb(self.gpu_allocated),
            "gpu_reserved_mb": self.to_mb(self.gpu_reserved),
            "gpu_max_allocated_mb": self.to_mb(self.gpu_max_allocated),
            "gpu_max_reserved_mb": self.to_mb(self.gpu_max_reserved),
            "cpu_memory_mb": self.to_mb(self.cpu_memory),
            "cpu_memory_percent": self.cpu_memory_percent,
        }


@dataclass
class InferenceMetrics:
    """单次推理的完整性能指标"""
    # 基本信息
    test_name: str = ""
    text_length: int = 0
    text_tokens_count: int = 0
    segments_count: int = 0
    audio_duration: float = 0.0  # 生成的音频时长（秒）
    
    # 总体时间
    total_time: float = 0.0
    rtf: float = 0.0  # Real-Time Factor
    
    # 模块耗时
    model_load_time: float = 0.0
    text_processing_time: float = 0.0
    gpt_gen_time: float = 0.0
    gpt_forward_time: float = 0.0
    s2mel_time: float = 0.0
    bigvgan_time: float = 0.0
    audio_save_time: float = 0.0
    
    # 内存统计
    memory_start: MemoryStats = field(default_factory=MemoryStats)
    memory_peak: MemoryStats = field(default_factory=MemoryStats)
    memory_end: MemoryStats = field(default_factory=MemoryStats)
    
    # 其他信息
    device: str = ""
    dtype: str = ""
    use_fp16: bool = False
    use_cuda_kernel: bool = False
    use_deepspeed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        # 转换内存统计为更友好的格式
        result["memory_start"] = self.memory_start.to_dict()
        result["memory_peak"] = self.memory_peak.to_dict()
        result["memory_end"] = self.memory_end.to_dict()
        return result


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, device: str = "cuda", enable_profiling: bool = True):
        """
        Args:
            device: 设备类型 (cuda, mps, xpu, cpu)
            enable_profiling: 是否启用性能分析
        """
        self.device = device
        self.enable_profiling = enable_profiling
        self.timing_records: Dict[str, TimingStats] = {}
        self.current_metrics: Optional[InferenceMetrics] = None
        
        # 检测设备类型
        self.is_cuda = device.startswith("cuda")
        self.is_mps = device == "mps"
        self.is_xpu = device == "xpu"
        
        # 检查是否有psutil用于CPU内存监控
        self.has_psutil = False
        try:
            import psutil
            self.psutil = psutil
            self.has_psutil = True
        except ImportError:
            self.psutil = None
    
    def reset(self):
        """重置所有计时记录"""
        self.timing_records.clear()
        self.current_metrics = None
    
    def start_inference(self, test_name: str = "", **kwargs) -> InferenceMetrics:
        """开始一次推理的性能监控"""
        if not self.enable_profiling:
            return None
        
        self.current_metrics = InferenceMetrics(
            test_name=test_name,
            device=self.device,
            **kwargs
        )
        
        # 记录初始内存状态
        self.current_metrics.memory_start = self.get_memory_stats()
        
        # 重置GPU内存统计（如果是CUDA）
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()
        
        return self.current_metrics
    
    def end_inference(self) -> Optional[InferenceMetrics]:
        """结束推理监控并返回指标"""
        if not self.enable_profiling or self.current_metrics is None:
            return None
        
        # 记录峰值和结束时内存状态
        self.current_metrics.memory_peak = self.get_peak_memory_stats()
        self.current_metrics.memory_end = self.get_memory_stats()
        
        # 计算RTF
        if self.current_metrics.audio_duration > 0:
            self.current_metrics.rtf = (
                self.current_metrics.total_time / self.current_metrics.audio_duration
            )
        
        metrics = self.current_metrics
        self.current_metrics = None
        return metrics
    
    @contextmanager
    def timer(self, name: str):
        """计时上下文管理器"""
        if not self.enable_profiling:
            yield
            return
        
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            duration = end - start
            self.timing_records[name] = TimingStats(
                name=name,
                start_time=start,
                end_time=end,
                duration=duration
            )
            
            # 如果有当前指标，更新对应字段
            if self.current_metrics is not None:
                field_name = name.replace(" ", "_").lower()
                if hasattr(self.current_metrics, field_name):
                    setattr(self.current_metrics, field_name, duration)
    
    def get_memory_stats(self) -> MemoryStats:
        """获取当前内存统计"""
        stats = MemoryStats()
        
        # GPU内存
        if self.is_cuda:
            stats.gpu_allocated = torch.cuda.memory_allocated()
            stats.gpu_reserved = torch.cuda.memory_reserved()
        elif self.is_mps:
            stats.gpu_allocated = torch.mps.current_allocated_memory()
        elif self.is_xpu:
            stats.gpu_allocated = torch.xpu.memory_allocated()
        
        # CPU内存
        if self.has_psutil:
            process = self.psutil.Process()
            mem_info = process.memory_info()
            stats.cpu_memory = mem_info.rss
            stats.cpu_memory_percent = process.memory_percent()
        
        return stats
    
    def get_peak_memory_stats(self) -> MemoryStats:
        """获取峰值内存统计"""
        stats = self.get_memory_stats()
        
        # GPU峰值内存
        if self.is_cuda:
            stats.gpu_max_allocated = torch.cuda.max_memory_allocated()
            stats.gpu_max_reserved = torch.cuda.max_memory_reserved()
        elif self.is_mps:
            stats.gpu_max_allocated = torch.mps.driver_allocated_memory()
        elif self.is_xpu:
            stats.gpu_max_allocated = torch.xpu.max_memory_allocated()
        
        return stats
    
    def get_timing_summary(self) -> Dict[str, float]:
        """获取所有计时记录的汇总"""
        return {
            name: record.duration
            for name, record in self.timing_records.items()
        }
    
    def print_summary(self, metrics: InferenceMetrics):
        """打印性能摘要"""
        if not self.enable_profiling or metrics is None:
            return
        
        print("\n" + "="*70)
        print(f"Performance Summary: {metrics.test_name}")
        print("="*70)
        
        # 基本信息
        print(f"\n📊 Basic Info:")
        print(f"  Device: {metrics.device}")
        print(f"  Text Length: {metrics.text_length} chars, {metrics.text_tokens_count} tokens")
        print(f"  Segments: {metrics.segments_count}")
        print(f"  Audio Duration: {metrics.audio_duration:.2f}s")
        print(f"  Use FP16: {metrics.use_fp16}")
        
        # 时间统计
        print(f"\n⏱️  Timing:")
        print(f"  Total Time: {metrics.total_time:.3f}s")
        print(f"  RTF: {metrics.rtf:.4f}")
        
        if metrics.gpt_gen_time > 0:
            print(f"\n  Module Breakdown:")
            print(f"    GPT Generation: {metrics.gpt_gen_time:.3f}s ({metrics.gpt_gen_time/metrics.total_time*100:.1f}%)")
            print(f"    GPT Forward: {metrics.gpt_forward_time:.3f}s ({metrics.gpt_forward_time/metrics.total_time*100:.1f}%)")
            print(f"    S2Mel: {metrics.s2mel_time:.3f}s ({metrics.s2mel_time/metrics.total_time*100:.1f}%)")
            print(f"    BigVGAN: {metrics.bigvgan_time:.3f}s ({metrics.bigvgan_time/metrics.total_time*100:.1f}%)")
        
        # 内存统计
        if self.is_cuda or self.is_mps or self.is_xpu:
            print(f"\n💾 Memory:")
            mem_peak = metrics.memory_peak
            print(f"  GPU Peak Allocated: {mem_peak.to_mb(mem_peak.gpu_allocated):.1f} MB")
            if self.is_cuda:
                print(f"  GPU Max Allocated: {mem_peak.to_mb(mem_peak.gpu_max_allocated):.1f} MB")
                print(f"  GPU Max Reserved: {mem_peak.to_mb(mem_peak.gpu_max_reserved):.1f} MB")
        
        if self.has_psutil:
            print(f"  CPU Memory: {metrics.memory_peak.to_mb(metrics.memory_peak.cpu_memory):.1f} MB")
        
        print("="*70 + "\n")


class BenchmarkCollector:
    """基准测试结果收集器"""
    
    def __init__(self):
        self.results: List[InferenceMetrics] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_result(self, metrics: InferenceMetrics):
        """添加一个测试结果"""
        if metrics is not None:
            self.results.append(metrics)
    
    def set_metadata(self, **kwargs):
        """设置元数据"""
        self.metadata.update(kwargs)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """计算汇总统计"""
        if not self.results:
            return {}
        
        import numpy as np
        
        total_times = [r.total_time for r in self.results]
        rtfs = [r.rtf for r in self.results if r.rtf > 0]
        gpu_mems = [r.memory_peak.gpu_max_allocated for r in self.results]
        
        return {
            "count": len(self.results),
            "total_time": {
                "mean": float(np.mean(total_times)),
                "median": float(np.median(total_times)),
                "std": float(np.std(total_times)),
                "min": float(np.min(total_times)),
                "max": float(np.max(total_times)),
            },
            "rtf": {
                "mean": float(np.mean(rtfs)) if rtfs else 0.0,
                "median": float(np.median(rtfs)) if rtfs else 0.0,
                "std": float(np.std(rtfs)) if rtfs else 0.0,
                "min": float(np.min(rtfs)) if rtfs else 0.0,
                "max": float(np.max(rtfs)) if rtfs else 0.0,
            },
            "gpu_memory_mb": {
                "mean": float(np.mean([m / (1024**2) for m in gpu_mems])) if gpu_mems else 0.0,
                "max": float(np.max([m / (1024**2) for m in gpu_mems])) if gpu_mems else 0.0,
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "metadata": self.metadata,
            "summary": self.get_summary_stats(),
            "results": [r.to_dict() for r in self.results]
        }
    
    def save_json(self, filepath: str):
        """保存为JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"✅ Benchmark results saved to: {filepath}")
    
    def load_json(self, filepath: str):
        """从JSON文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.metadata = data.get("metadata", {})
        # Note: results are loaded as dicts, not InferenceMetrics objects
        self.results = data.get("results", [])
        return data


