#!/usr/bin/env python3
"""
简化的测试脚本：直接对比PyTorch和MLX版本的get_conditioning方法输出

这个脚本将：
1. 加载PyTorch和MLX版本的GPT模型
2. 使用相同的输入调用get_conditioning方法
3. 对比输出结果的差异
4. 生成详细的对比报告
"""

import os
import sys
import time
import torch
import numpy as np
import json
import argparse
from pathlib import Path

# 设置环境变量
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'

def compare_tensors(tensor1, tensor2, name, tolerance=1e-3, verbose=True):
    """对比两个tensor的一致性"""
    if tensor1 is None or tensor2 is None:
        result = {
            'is_close': tensor1 is None and tensor2 is None,
            'error': "One or both tensors are None"
        }
        if verbose:
            print(f"  ❌ {name}: One or both tensors are None")
        return result
    
    # 确保在CPU上对比
    if isinstance(tensor1, torch.Tensor):
        tensor1 = tensor1.cpu().detach()
    if isinstance(tensor2, torch.Tensor):
        tensor2 = tensor2.cpu().detach()
    
    # 形状检查
    if tensor1.shape != tensor2.shape:
        result = {
            'is_close': False,
            'error': f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}",
            'shape1': list(tensor1.shape),
            'shape2': list(tensor2.shape)
        }
        if verbose:
            print(f"  ❌ {name}: Shape mismatch {tensor1.shape} vs {tensor2.shape}")
        return result
    
    # 转换为numpy进行数值比较
    try:
        arr1 = tensor1.numpy() if isinstance(tensor1, torch.Tensor) else tensor1
        arr2 = tensor2.numpy() if isinstance(tensor2, torch.Tensor) else tensor2
        
        # 计算差异
        diff = np.abs(arr1 - arr2)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # 计算相关系数
        arr1_flat = arr1.flatten()
        arr2_flat = arr2.flatten()
        correlation = np.corrcoef(arr1_flat, arr2_flat)[0, 1] if len(arr1_flat) > 1 else 1.0
        
        is_close = max_diff < tolerance and not np.isnan(correlation)
        
        result = {
            'is_close': is_close,
            'max_diff': float(max_diff),
            'mean_diff': float(mean_diff),
            'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'tolerance': tolerance,
            'shape': list(arr1.shape)
        }
        
        if verbose:
            status = "✅" if is_close else "⚠️"
            print(f"  {status} {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, corr={correlation:.4f}")
            if not is_close:
                print(f"    Tolerance: {tolerance}, Maximum difference: {max_diff:.8f}")
        
        return result
        
    except Exception as e:
        result = {
            'is_close': False,
            'error': f"Comparison failed: {str(e)}"
        }
        if verbose:
            print(f"  ❌ {name}: Comparison failed - {e}")
        return result

class ConditioningConsistencyTester:
    def __init__(self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", device="mps"):
        self.cfg_path = cfg_path
        self.model_dir = model_dir
        self.device = device
        self.results = {}
        
    def load_models(self):
        """加载PyTorch和MLX版本的GPT模型"""
        print("=" * 80)
        print("Loading GPT Models for Conditioning Consistency Test")
        print("=" * 80)
        
        # 导入必要的模块
        from omegaconf import OmegaConf
        from indextts.gpt.model_v2 import UnifiedVoice
        from indextts.gpt.mlx_model import UnifiedVoiceMLX
        from indextts.utils.checkpoint import load_checkpoint
        
        # 加载配置
        self.cfg = OmegaConf.load(self.cfg_path)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        
        print(f"Config: {self.cfg_path}")
        print(f"Model dir: {self.model_dir}")
        print(f"GPT checkpoint: {self.gpt_path}")
        
        # 加载PyTorch版本
        print("\n🔄 Loading PyTorch GPT model...")
        start_time = time.time()
        
        self.gpt_pytorch = UnifiedVoice(**self.cfg.gpt)
        load_checkpoint(self.gpt_pytorch, self.gpt_path)
        self.gpt_pytorch = self.gpt_pytorch.to(self.device)
        self.gpt_pytorch.eval()
        
        pytorch_load_time = time.time() - start_time
        print(f"✅ PyTorch GPT loaded in {pytorch_load_time:.2f}s")
        
        # 加载MLX版本
        print("\n🔄 Loading MLX GPT model...")
        start_time = time.time()
        
        # 加载MLX权重
        from indextts.utils.mlx_utils import check_mlx_available
        from indextts.utils.mlx_cache import MLXModelCache
        
        if not check_mlx_available():
            print("❌ MLX not available, cannot test MLX consistency")
            return False
        
        mlx_cache = MLXModelCache(cache_dir=os.path.join(self.model_dir, "mlx"))
        mlx_gpt_weights = mlx_cache.get_or_convert("gpt", self.gpt_path)
        
        self.gpt_mlx = UnifiedVoiceMLX(
            use_mlx_conditioning=True,
            **self.cfg.gpt
        )
        self.gpt_mlx.load_weights_from_dict(mlx_gpt_weights)
        
        mlx_load_time = time.time() - start_time
        print(f"✅ MLX GPT loaded in {mlx_load_time:.2f}s")
        
        self.results['model_load_times'] = {
            'pytorch': pytorch_load_time,
            'mlx': mlx_load_time
        }
        
        return True
    
    def generate_test_input(self):
        """生成测试用的输入数据"""
        print("\n🔄 Generating test input data...")
        
        # 创建模拟的semantic features输入
        # 实际的shape应该是 (batch_size, time, 1024) 或 (batch_size, 1024, time)
        batch_size = 1
        time_steps = 100  # 模拟100个时间步
        feature_dim = 1024
        
        # 生成随机但确定性的输入数据
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 创建输入 tensor (batch_size, time, feature_dim)
        speech_conditioning_input = torch.randn(batch_size, time_steps, feature_dim, device=self.device)
        
        # 创建长度信息
        cond_mel_lengths = torch.tensor([time_steps], device=self.device)
        
        print(f"  Input shape: {speech_conditioning_input.shape}")
        print(f"  Lengths: {cond_mel_lengths}")
        
        return speech_conditioning_input, cond_mel_lengths
    
    def test_conditioning_consistency(self, verbose=True):
        """测试get_conditioning方法的一致性"""
        print("\n" + "=" * 80)
        print("Testing get_conditioning Method Consistency")
        print("=" * 80)
        
        # 生成测试输入
        speech_conditioning_input, cond_mel_lengths = self.generate_test_input()
        
        # 设置相同的随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        try:
            # 测试PyTorch版本
            print("\n🔄 Computing PyTorch get_conditioning...")
            start_time = time.time()
            
            # PyTorch版本期望 (batch_size, feature_dim, time) 格式
            # 我们的输入是 (batch_size, time, feature_dim)，需要转换为PyTorch期望的格式
            pytorch_input = speech_conditioning_input.transpose(1, 2)  # (b, t, 1024) -> (b, 1024, t)
            
            with torch.no_grad():
                pytorch_conditioning = self.gpt_pytorch.get_conditioning(pytorch_input, cond_mel_lengths)
            
            pytorch_time = time.time() - start_time
            print(f"✅ PyTorch get_conditioning completed in {pytorch_time:.2f}s")
            print(f"  Input shape: {pytorch_input.shape}")
            print(f"  Output shape: {pytorch_conditioning.shape}")
            print(f"  Output dtype: {pytorch_conditioning.dtype}")
            
            # 重置随机种子以确保一致性
            torch.manual_seed(42)
            np.random.seed(42)
            
            # 测试MLX版本
            print("\n🔄 Computing MLX get_conditioning...")
            start_time = time.time()
            
            # MLX版本可以直接使用 (batch_size, time, feature_dim) 格式
            # 如果输入格式不对，MLX版本会自动转换
            mlx_input = speech_conditioning_input  # 直接使用 (b, t, 1024)
            
            mlx_conditioning = self.gpt_mlx.get_conditioning(mlx_input, cond_mel_lengths)
            
            mlx_time = time.time() - start_time
            print(f"✅ MLX get_conditioning completed in {mlx_time:.2f}s")
            print(f"  Input shape: {mlx_input.shape}")
            print(f"  Output shape: {mlx_conditioning.shape}")
            print(f"  Output dtype: {mlx_conditioning.dtype}")
            
            # 对比结果
            print("\n📊 Comparing conditioning outputs...")
            conditioning_comparison = compare_tensors(
                pytorch_conditioning, mlx_conditioning, 
                "get_conditioning Output", tolerance=1e-3, verbose=verbose
            )
            
            # 存储结果
            self.results['conditioning_test'] = {
                'comparison': conditioning_comparison,
                'pytorch_time': pytorch_time,
                'mlx_time': mlx_time,
                'input_shape': list(speech_conditioning_input.shape),
                'pytorch_output_shape': list(pytorch_conditioning.shape),
                'mlx_output_shape': list(mlx_conditioning.shape)
            }
            
            return conditioning_comparison['is_close']
            
        except Exception as e:
            print(f"❌ Conditioning consistency test failed: {e}")
            import traceback
            traceback.print_exc()
            self.results['conditioning_test'] = {
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            return False
    
    def run_multiple_tests(self, num_tests=5):
        """运行多次测试以验证一致性"""
        print("\n" + "=" * 80)
        print(f"Running Multiple Tests ({num_tests} iterations)")
        print("=" * 80)
        
        test_results = []
        
        for i in range(num_tests):
            print(f"\n🔄 Test {i+1}/{num_tests}")
            
            # 为每次测试生成不同的输入
            torch.manual_seed(42 + i)
            np.random.seed(42 + i)
            
            speech_conditioning_input, cond_mel_lengths = self.generate_test_input()
            
            try:
                # PyTorch
                torch.manual_seed(42 + i)
                pytorch_input = speech_conditioning_input.transpose(1, 2)  # (b, t, 1024) -> (b, 1024, t)
                
                with torch.no_grad():
                    pytorch_conditioning = self.gpt_pytorch.get_conditioning(pytorch_input, cond_mel_lengths)
                
                # MLX
                torch.manual_seed(42 + i)
                np.random.seed(42 + i)
                mlx_input = speech_conditioning_input  # 直接使用 (b, t, 1024)
                
                mlx_conditioning = self.gpt_mlx.get_conditioning(mlx_input, cond_mel_lengths)
                
                # 比较
                comparison = compare_tensors(
                    pytorch_conditioning, mlx_conditioning, 
                    f"Test {i+1}", tolerance=1e-3, verbose=False
                )
                
                test_results.append(comparison)
                
                status = "✅" if comparison['is_close'] else "❌"
                print(f"  {status} Test {i+1}: max_diff={comparison.get('max_diff', 'N/A'):.6f}")
                
            except Exception as e:
                print(f"  ❌ Test {i+1} failed: {e}")
                test_results.append({'is_close': False, 'error': str(e)})
        
        # 统计结果
        successful_tests = sum(1 for r in test_results if r.get('is_close', False))
        print(f"\n📊 Results: {successful_tests}/{num_tests} tests passed")
        
        self.results['multiple_tests'] = {
            'total_tests': num_tests,
            'successful_tests': successful_tests,
            'results': test_results
        }
        
        return successful_tests == num_tests
    
    def generate_report(self):
        """生成测试报告"""
        print("\n" + "=" * 80)
        print("CONDITIONING CONSISTENCY TEST REPORT")
        print("=" * 80)
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config_path': self.cfg_path,
            'model_dir': self.model_dir,
            'device': self.device,
            'results': self.results,
            'summary': {}
        }
        
        # 分析结果
        print("\n📋 Summary:")
        
        if 'model_load_times' in self.results:
            times = self.results['model_load_times']
            print(f"  Model Loading Times:")
            print(f"    PyTorch: {times['pytorch']:.2f}s")
            print(f"    MLX: {times['mlx']:.2f}s")
            if times['mlx'] > 0:
                print(f"    Speedup: {times['pytorch']/times['mlx']:.2f}x")
        
        if 'conditioning_test' in self.results:
            test_result = self.results['conditioning_test']
            if 'comparison' in test_result:
                comp = test_result['comparison']
                status = "✅ PASS" if comp['is_close'] else "❌ FAIL"
                print(f"  Conditioning Consistency: {status}")
                if not comp['is_close']:
                    print(f"    Max difference: {comp.get('max_diff', 'N/A'):.8f}")
                    print(f"    Correlation: {comp.get('correlation', 'N/A'):.4f}")
            
            if 'pytorch_time' in test_result:
                print(f"  Computation Times:")
                print(f"    PyTorch: {test_result['pytorch_time']:.3f}s")
                print(f"    MLX: {test_result['mlx_time']:.3f}s")
        
        if 'multiple_tests' in self.results:
            multi = self.results['multiple_tests']
            print(f"  Multiple Tests: {multi['successful_tests']}/{multi['total_tests']} passed")
        
        # 保存报告
        report_path = "conditioning_consistency_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Full report saved to: {report_path}")
        print("=" * 80)
        
        return report

def main():
    parser = argparse.ArgumentParser(description="Test PyTorch-MLX conditioning consistency for IndexTTS2")
    parser.add_argument("--config", type=str, default="checkpoints/config.yaml",
                       help="Path to config file")
    parser.add_argument("--model_dir", type=str, default="checkpoints",
                       help="Path to model directory")
    parser.add_argument("--device", type=str, default="mps",
                       help="Device to use (mps, cuda, cpu)")
    parser.add_argument("--num_tests", type=int, default=3,
                       help="Number of test iterations to run")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # 检查必要的文件
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.model_dir):
        print(f"❌ Model directory not found: {args.model_dir}")
        sys.exit(1)
    
    print("🚀 Starting PyTorch-MLX Conditioning Consistency Test")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建测试器
    tester = ConditioningConsistencyTester(
        cfg_path=args.config,
        model_dir=args.model_dir,
        device=args.device
    )
    
    try:
        # 加载模型
        if not tester.load_models():
            print("❌ Failed to load models")
            sys.exit(1)
        
        # 运行基本一致性测试
        print("\n" + "="*50)
        print("BASIC CONSISTENCY TEST")
        print("="*50)
        success1 = tester.test_conditioning_consistency(verbose=args.verbose)
        
        # 运行多次测试
        print("\n" + "="*50)
        print("MULTIPLE TEST ITERATIONS")
        print("="*50)
        success2 = tester.run_multiple_tests(num_tests=args.num_tests)
        
        # 生成报告
        report = tester.generate_report()
        
        # 总结
        if success1 and success2:
            print("\n✅ All consistency tests PASSED!")
            print("PyTorch and MLX versions produce consistent conditioning outputs.")
        else:
            print("\n⚠️  Some consistency tests FAILED!")
            print("There may be differences between PyTorch and MLX implementations.")
        
        return 0 if (success1 and success2) else 1
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
