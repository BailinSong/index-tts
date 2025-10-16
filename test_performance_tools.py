#!/usr/bin/env python3
"""
快速验证性能工具是否正常工作
"""

import sys
import os

def test_imports():
    """测试导入"""
    print("🧪 Testing imports...")
    
    try:
        from indextts.utils.performance_monitor import (
            PerformanceMonitor,
            InferenceMetrics,
            BenchmarkCollector
        )
        print("   ✅ performance_monitor imported successfully")
    except Exception as e:
        print(f"   ❌ Failed to import performance_monitor: {e}")
        return False
    
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__} available")
    except Exception as e:
        print(f"   ❌ PyTorch import failed: {e}")
        return False
    
    return True


def test_performance_monitor():
    """测试性能监控器"""
    print("\n🧪 Testing PerformanceMonitor...")
    
    try:
        from indextts.utils.performance_monitor import PerformanceMonitor
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        monitor = PerformanceMonitor(device=device, enable_profiling=True)
        
        # 测试基本功能
        metrics = monitor.start_inference(test_name="test")
        
        with monitor.timer("test_operation"):
            import time
            time.sleep(0.1)
        
        monitor.end_inference()
        
        print("   ✅ PerformanceMonitor works correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ PerformanceMonitor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_benchmark_collector():
    """测试结果收集器"""
    print("\n🧪 Testing BenchmarkCollector...")
    
    try:
        from indextts.utils.performance_monitor import BenchmarkCollector, InferenceMetrics
        
        collector = BenchmarkCollector()
        collector.set_metadata(test="validation")
        
        # 添加测试结果
        metrics = InferenceMetrics(
            test_name="test1",
            total_time=1.5,
            rtf=0.5
        )
        collector.add_result(metrics)
        
        # 获取统计
        stats = collector.get_summary_stats()
        
        print("   ✅ BenchmarkCollector works correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ BenchmarkCollector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scripts_exist():
    """检查脚本文件是否存在"""
    print("\n🧪 Checking script files...")
    
    files = {
        "benchmark_baseline.py": "基线测试脚本",
        "analyze_optimization.py": "分析脚本",
        "docs/optimization_directions.md": "优化方向文档",
        "docs/performance_benchmark_guide.md": "使用指南",
        "PERFORMANCE_BASELINE_README.md": "总体说明",
    }
    
    all_exist = True
    for file, desc in files.items():
        if os.path.exists(file):
            print(f"   ✅ {desc}: {file}")
        else:
            print(f"   ❌ {desc} 不存在: {file}")
            all_exist = False
    
    return all_exist


def main():
    print("="*70)
    print("IndexTTS2 Performance Tools - Validation Test")
    print("="*70)
    
    results = []
    
    # 运行测试
    results.append(("Imports", test_imports()))
    results.append(("PerformanceMonitor", test_performance_monitor()))
    results.append(("BenchmarkCollector", test_benchmark_collector()))
    results.append(("Script Files", test_scripts_exist()))
    
    # 打印总结
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:25s} {status}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 All tests passed! The performance tools are ready to use.")
        print("\nNext steps:")
        print("  1. Read PERFORMANCE_BASELINE_README.md for overview")
        print("  2. Read docs/performance_benchmark_guide.md for detailed usage")
        print("  3. Run: python benchmark_baseline.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


