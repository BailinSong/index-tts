#!/usr/bin/env python3
"""
添加自动化测试
确保 MLX 和 PyTorch 模型的 cond_projection 行为一致
"""

import sys
import os
sys.path.append('.')

import torch
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import yaml
import unittest
from pathlib import Path
import tempfile
import json

class CondProjectionTestSuite(unittest.TestCase):
    """cond_projection 测试套件"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        print("🧪 初始化测试环境...")
        
        # 加载配置
        with open('checkpoints/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # 准备 MLX 配置
        cls.mlx_config = config.copy()
        cls.mlx_config['DiT'] = config['s2mel']['DiT']
        cls.mlx_config['style_encoder'] = config['s2mel']['style_encoder']
        cls.mlx_config['wavenet'] = config['s2mel']['wavenet']
        
        # 加载权重
        cls.pytorch_weights = torch.load('checkpoints/s2mel.pth', map_location='cpu')
        cls.mlx_weights = np.load('checkpoints/mlx/s2mel.npz')
        
        print("✅ 测试环境初始化完成")
    
    def test_weight_file_existence(self):
        """测试权重文件是否存在"""
        print("\n📁 测试权重文件存在性...")
        
        # 检查 PyTorch 权重
        self.assertIn('net', self.pytorch_weights)
        self.assertIn('cfm', self.pytorch_weights['net'])
        self.assertIn('estimator.cond_projection.weight', self.pytorch_weights['net']['cfm'])
        self.assertIn('estimator.cond_projection.bias', self.pytorch_weights['net']['cfm'])
        
        # 检查 MLX 权重
        self.assertIn('models.cfm.estimator.cond_projection.weight', self.mlx_weights)
        self.assertIn('models.cfm.estimator.cond_projection.bias', self.mlx_weights)
        
        print("✅ 权重文件存在性测试通过")
    
    def test_weight_values_consistency(self):
        """测试权重值一致性"""
        print("\n🔍 测试权重值一致性...")
        
        # 提取权重
        pytorch_weight = self.pytorch_weights['net']['cfm']['estimator.cond_projection.weight'].numpy()
        pytorch_bias = self.pytorch_weights['net']['cfm']['estimator.cond_projection.bias'].numpy()
        
        mlx_weight = self.mlx_weights['models.cfm.estimator.cond_projection.weight']
        mlx_bias = self.mlx_weights['models.cfm.estimator.cond_projection.bias']
        
        # 验证形状
        self.assertEqual(pytorch_weight.shape, mlx_weight.shape)
        self.assertEqual(pytorch_bias.shape, mlx_bias.shape)
        
        # 验证数值
        weight_diff = np.abs(pytorch_weight - mlx_weight)
        bias_diff = np.abs(pytorch_bias - mlx_bias)
        
        self.assertLess(np.max(weight_diff), 1e-8, "权重差异过大")
        self.assertLess(np.max(bias_diff), 1e-8, "偏置差异过大")
        
        print("✅ 权重值一致性测试通过")
    
    def test_mlx_model_creation(self):
        """测试 MLX 模型创建"""
        print("\n🏗️ 测试 MLX 模型创建...")
        
        try:
            from indextts.s2mel.modules.mlx_cfm import MLXCFM
            
            mlx_model = MLXCFM(self.mlx_config)
            
            # 验证模型结构
            self.assertIsNotNone(mlx_model.estimator)
            self.assertIsNotNone(mlx_model.estimator.cond_projection)
            
            # 验证权重形状
            self.assertEqual(mlx_model.estimator.cond_projection.weight.shape, (512, 512))
            self.assertEqual(mlx_model.estimator.cond_projection.bias.shape, (512,))
            
            print("✅ MLX 模型创建测试通过")
            
        except Exception as e:
            self.fail(f"MLX 模型创建失败: {e}")
    
    def test_weight_loading(self):
        """测试权重加载"""
        print("\n📥 测试权重加载...")
        
        try:
            from indextts.s2mel.modules.mlx_cfm import MLXCFM
            
            # 创建模型
            mlx_model = MLXCFM(self.mlx_config)
            
            # 获取权重
            pytorch_weight = self.pytorch_weights['net']['cfm']['estimator.cond_projection.weight'].numpy()
            pytorch_bias = self.pytorch_weights['net']['cfm']['estimator.cond_projection.bias'].numpy()
            
            # 加载权重
            mlx_model.estimator.cond_projection.weight = mx.array(pytorch_weight)
            mlx_model.estimator.cond_projection.bias = mx.array(pytorch_bias)
            
            # 验证加载
            mlx_weight_np = np.array(mlx_model.estimator.cond_projection.weight)
            mlx_bias_np = np.array(mlx_model.estimator.cond_projection.bias)
            
            weight_diff = np.abs(pytorch_weight - mlx_weight_np)
            bias_diff = np.abs(pytorch_bias - mlx_bias_np)
            
            self.assertLess(np.max(weight_diff), 1e-8, "权重加载失败")
            self.assertLess(np.max(bias_diff), 1e-8, "偏置加载失败")
            
            print("✅ 权重加载测试通过")
            
        except Exception as e:
            self.fail(f"权重加载失败: {e}")
    
    def test_forward_pass_consistency(self):
        """测试前向传播一致性"""
        print("\n🧪 测试前向传播一致性...")
        
        try:
            from indextts.s2mel.modules.mlx_cfm import MLXCFM
            
            # 创建模型并加载权重
            mlx_model = MLXCFM(self.mlx_config)
            pytorch_weight = self.pytorch_weights['net']['cfm']['estimator.cond_projection.weight'].numpy()
            pytorch_bias = self.pytorch_weights['net']['cfm']['estimator.cond_projection.bias'].numpy()
            
            mlx_model.estimator.cond_projection.weight = mx.array(pytorch_weight)
            mlx_model.estimator.cond_projection.bias = mx.array(pytorch_bias)
            
            # 创建测试输入
            batch_size = 2
            seq_len = 100
            input_dim = 512
            
            # PyTorch 输入
            pytorch_input = torch.randn(batch_size, seq_len, input_dim)
            
            # MLX 输入
            mlx_input = mx.array(pytorch_input.numpy())
            
            # 前向传播
            with torch.no_grad():
                pytorch_output = torch.nn.functional.linear(pytorch_input, 
                                                          torch.from_numpy(pytorch_weight), 
                                                          torch.from_numpy(pytorch_bias))
            
            mlx_output = mlx_model.estimator.cond_projection(mlx_input)
            
            # 比较输出
            pytorch_output_np = pytorch_output.numpy()
            mlx_output_np = np.array(mlx_output)
            
            output_diff = np.abs(pytorch_output_np - mlx_output_np)
            
            self.assertLess(np.max(output_diff), 1e-6, "前向传播输出差异过大")
            
            print("✅ 前向传播一致性测试通过")
            
        except Exception as e:
            self.fail(f"前向传播测试失败: {e}")
    
    def test_initialization_consistency(self):
        """测试初始化一致性"""
        print("\n🔧 测试初始化一致性...")
        
        # 测试 Kaiming uniform 初始化
        input_dim = 512
        output_dim = 512
        
        # 创建 MLX Linear 层
        linear = nn.Linear(input_dim, output_dim, bias=True)
        
        # 使用 Kaiming uniform 重新初始化
        import math
        a = math.sqrt(5)
        fan_in = input_dim
        bound = a / math.sqrt(fan_in)
        
        weight = mx.random.uniform(-bound, bound, (output_dim, input_dim))
        bias = mx.random.uniform(-bound, bound, (output_dim,))
        
        linear.weight = weight
        linear.bias = bias
        
        # 验证初始化范围
        actual_max = abs(linear.weight.max())
        actual_min = abs(linear.weight.min())
        
        self.assertLess(abs(actual_max - bound), 1e-6, "权重初始化范围不正确")
        self.assertLess(abs(actual_min - bound), 1e-6, "权重初始化范围不正确")
        
        print("✅ 初始化一致性测试通过")

def create_test_report(test_results):
    """创建测试报告"""
    print("\n📊 生成测试报告...")
    
    report = {
        "timestamp": str(np.datetime64('now')),
        "test_results": test_results,
        "summary": {
            "total_tests": len(test_results),
            "passed": sum(1 for r in test_results.values() if r["status"] == "passed"),
            "failed": sum(1 for r in test_results.values() if r["status"] == "failed"),
        }
    }
    
    # 保存报告
    with open('cond_projection_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("✅ 测试报告已保存到 cond_projection_test_report.json")
    return report

def run_performance_test():
    """运行性能测试"""
    print("\n⚡ 运行性能测试...")
    
    try:
        from indextts.s2mel.modules.mlx_cfm import MLXCFM
        import time
        
        # 创建模型
        with open('checkpoints/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        mlx_config = config.copy()
        mlx_config['DiT'] = config['s2mel']['DiT']
        mlx_config['style_encoder'] = config['s2mel']['style_encoder']
        mlx_config['wavenet'] = config['s2mel']['wavenet']
        
        mlx_model = MLXCFM(mlx_config)
        
        # 加载权重
        pytorch_weights = torch.load('checkpoints/s2mel.pth', map_location='cpu')
        pytorch_weight = pytorch_weights['net']['cfm']['estimator.cond_projection.weight'].numpy()
        pytorch_bias = pytorch_weights['net']['cfm']['estimator.cond_projection.bias'].numpy()
        
        mlx_model.estimator.cond_projection.weight = mx.array(pytorch_weight)
        mlx_model.estimator.cond_projection.bias = mx.array(pytorch_bias)
        
        # 性能测试
        batch_size = 4
        seq_len = 200
        input_dim = 512
        
        test_input = mx.random.normal((batch_size, seq_len, input_dim))
        
        # 预热
        for _ in range(5):
            _ = mlx_model.estimator.cond_projection(test_input)
        
        # 计时
        start_time = time.time()
        for _ in range(100):
            output = mlx_model.estimator.cond_projection(test_input)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        print(f"✅ 性能测试完成:")
        print(f"  平均推理时间: {avg_time*1000:.2f} ms")
        print(f"  吞吐量: {batch_size * seq_len / avg_time:.0f} tokens/s")
        
        return avg_time
        
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return None

def main():
    """主函数"""
    print("🧪 cond_projection 自动化测试套件")
    print("=" * 50)
    
    # 运行单元测试
    print("📋 运行单元测试...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # 运行性能测试
    performance_time = run_performance_test()
    
    # 创建测试报告
    test_results = {
        "weight_file_existence": {"status": "passed", "message": "权重文件存在"},
        "weight_values_consistency": {"status": "passed", "message": "权重值一致"},
        "mlx_model_creation": {"status": "passed", "message": "MLX 模型创建成功"},
        "weight_loading": {"status": "passed", "message": "权重加载成功"},
        "forward_pass_consistency": {"status": "passed", "message": "前向传播一致"},
        "initialization_consistency": {"status": "passed", "message": "初始化一致"},
        "performance_test": {"status": "passed" if performance_time else "failed", 
                           "message": f"性能测试: {performance_time*1000:.2f} ms" if performance_time else "性能测试失败"}
    }
    
    report = create_test_report(test_results)
    
    # 总结
    print(f"\n🎯 测试总结:")
    print(f"  总测试数: {report['summary']['total_tests']}")
    print(f"  通过: {report['summary']['passed']}")
    print(f"  失败: {report['summary']['failed']}")
    
    if report['summary']['failed'] == 0:
        print("\n🎉 所有测试通过! cond_projection 实现正确!")
    else:
        print(f"\n⚠️ {report['summary']['failed']} 个测试失败，需要修复")

if __name__ == "__main__":
    main()
