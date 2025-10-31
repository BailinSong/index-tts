#!/usr/bin/env python3
"""
weight_norm 实现差异分析工具
深入分析 PyTorch weight_norm 与 MLX 普通 Linear 的差异
"""

import os
import sys
import pickle
import glob
import torch
import mlx.core as mx
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class WeightNormAnalyzer:
    def __init__(self, cache_dir: str = "cfm_production_cache"):
        self.cache_dir = cache_dir
        
    def analyze_weight_norm_differences(self):
        """分析 weight_norm 实现差异"""
        print("🔍 weight_norm 实现差异分析工具")
        print("="*60)
        
        # 1. 分析 weight_norm 原理
        self._analyze_weight_norm_principle()
        
        # 2. 分析 PyTorch weight_norm 实现
        self._analyze_pytorch_weight_norm()
        
        # 3. 分析 MLX Linear 实现
        self._analyze_mlx_linear()
        
        # 4. 分析权重转换方法
        self._analyze_weight_conversion()
        
        # 5. 测试修复方案
        self._test_fix_solution()
        
    def _analyze_weight_norm_principle(self):
        """分析 weight_norm 原理"""
        print(f"\n{'='*60}")
        print(f"🔍 weight_norm 原理分析")
        print(f"{'='*60}")
        
        print(f"\n📊 weight_norm 工作原理:")
        print(f"  1. weight_norm 将权重 W 分解为两个参数：")
        print(f"     - weight_g: 标量参数 (out_features, 1)")
        print(f"     - weight_v: 方向参数 (out_features, in_features)")
        print(f"  2. 实际权重计算：W = weight_g * weight_v / ||weight_v||")
        print(f"  3. 其中 ||weight_v|| 是 weight_v 的 L2 范数")
        print(f"  4. 这样可以控制权重的方向和大小")
        
        print(f"\n📊 数学公式:")
        print(f"  W[i, j] = g[i] * v[i, j] / ||v[i, :]||")
        print(f"  其中：")
        print(f"    g[i] = weight_g[i, 0]")
        print(f"    v[i, j] = weight_v[i, j]")
        print(f"    ||v[i, :]|| = sqrt(sum(v[i, j]^2))")
        
    def _analyze_pytorch_weight_norm(self):
        """分析 PyTorch weight_norm 实现"""
        print(f"\n{'='*60}")
        print(f"🔍 PyTorch weight_norm 实现分析")
        print(f"{'='*60}")
        
        try:
            # 加载生产环境模型
            from indextts.infer_v2 import IndexTTS2
            pytorch_tts = IndexTTS2(use_mlx=False)
            pytorch_dit = pytorch_tts.s2mel.models['cfm'].estimator
            pytorch_final_layer = pytorch_dit.final_layer
            
            print(f"\n📊 PyTorch FinalLayer linear 权重结构:")
            linear_layer = pytorch_final_layer.linear
            
            # 分析 weight_norm 结构
            print(f"  类型: {type(linear_layer)}")
            print(f"  权重参数:")
            for name, param in linear_layer.named_parameters():
                print(f"    {name}: {param.shape}, range=[{param.min().item():.6f}, {param.max().item():.6f}]")
            
            # 分析 weight_norm 的实际权重计算
            print(f"\n📊 weight_norm 实际权重计算:")
            weight_g = linear_layer.weight_g  # (out_features, 1)
            weight_v = linear_layer.weight_v  # (out_features, in_features)
            bias = linear_layer.bias          # (out_features,)
            
            print(f"  weight_g 形状: {weight_g.shape}")
            print(f"  weight_v 形状: {weight_v.shape}")
            print(f"  bias 形状: {bias.shape}")
            
            # 计算实际权重
            with torch.no_grad():
                # 计算 L2 范数
                weight_v_norm = torch.norm(weight_v, dim=1, keepdim=True)  # (out_features, 1)
                print(f"  weight_v_norm 形状: {weight_v_norm.shape}")
                print(f"  weight_v_norm 范围: [{weight_v_norm.min().item():.6f}, {weight_v_norm.max().item():.6f}]")
                
                # 计算实际权重
                actual_weight = weight_g * weight_v / weight_v_norm  # (out_features, in_features)
                print(f"  实际权重形状: {actual_weight.shape}")
                print(f"  实际权重范围: [{actual_weight.min().item():.6f}, {actual_weight.max().item():.6f}]")
                
                # 保存实际权重用于比较
                self.pytorch_actual_weight = actual_weight.cpu().numpy()
                self.pytorch_bias = bias.cpu().numpy()
                
        except Exception as e:
            print(f"❌ PyTorch weight_norm 分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_mlx_linear(self):
        """分析 MLX Linear 实现"""
        print(f"\n{'='*60}")
        print(f"🔍 MLX Linear 实现分析")
        print(f"{'='*60}")
        
        try:
            # 加载生产环境模型
            from indextts.infer_v2 import IndexTTS2
            mlx_tts = IndexTTS2(use_mlx=True)
            mlx_dit = mlx_tts.mlx_s2mel_cfm.estimator
            mlx_final_layer = mlx_dit.final_layer
            
            print(f"\n📊 MLX FinalLayer linear 权重结构:")
            linear_layer = mlx_final_layer.linear
            
            # 分析 Linear 结构
            print(f"  类型: {type(linear_layer)}")
            print(f"  权重参数:")
            for name, param in linear_layer.parameters().items():
                param_np = np.array(param)
                print(f"    {name}: {param.shape}, range=[{np.min(param_np):.6f}, {np.max(param_np):.6f}]")
            
            # 获取 MLX 权重
            weight = linear_layer.weight  # (in_features, out_features)
            bias = linear_layer.bias      # (out_features,)
            
            print(f"\n📊 MLX Linear 权重:")
            print(f"  weight 形状: {weight.shape}")
            print(f"  bias 形状: {bias.shape}")
            
            # 转换为 numpy 用于比较
            weight_np = np.array(weight)
            bias_np = np.array(bias)
            
            print(f"  weight 范围: [{np.min(weight_np):.6f}, {np.max(weight_np):.6f}]")
            print(f"  bias 范围: [{np.min(bias_np):.6f}, {np.max(bias_np):.6f}]")
            
            # 保存 MLX 权重用于比较
            self.mlx_weight = weight_np
            self.mlx_bias = bias_np
            
        except Exception as e:
            print(f"❌ MLX Linear 分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_weight_conversion(self):
        """分析权重转换方法"""
        print(f"\n{'='*60}")
        print(f"🔍 权重转换方法分析")
        print(f"{'='*60}")
        
        try:
            if not hasattr(self, 'pytorch_actual_weight') or not hasattr(self, 'mlx_weight'):
                print("❌ 缺少必要的权重数据")
                return
            
            print(f"\n📊 权重形状比较:")
            print(f"  PyTorch 实际权重: {self.pytorch_actual_weight.shape}")
            print(f"  MLX 权重: {self.mlx_weight.shape}")
            print(f"  PyTorch bias: {self.pytorch_bias.shape}")
            print(f"  MLX bias: {self.mlx_bias.shape}")
            
            # 比较权重
            if self.pytorch_actual_weight.shape == self.mlx_weight.shape:
                weight_diff = np.abs(self.pytorch_actual_weight - self.mlx_weight)
                max_weight_diff = np.max(weight_diff)
                mean_weight_diff = np.mean(weight_diff)
                
                print(f"\n📊 权重差异:")
                print(f"  最大差异: {max_weight_diff:.6f}")
                print(f"  平均差异: {mean_weight_diff:.6f}")
                
                if max_weight_diff < 1e-6:
                    print(f"  ✅ 权重完全一致")
                elif max_weight_diff < 0.1:
                    print(f"  ⚠️  权重存在小幅差异")
                else:
                    print(f"  ❌ 权重存在显著差异")
            else:
                print(f"  ❌ 权重形状不匹配")
            
            # 比较 bias
            if self.pytorch_bias.shape == self.mlx_bias.shape:
                bias_diff = np.abs(self.pytorch_bias - self.mlx_bias)
                max_bias_diff = np.max(bias_diff)
                mean_bias_diff = np.mean(bias_diff)
                
                print(f"\n📊 bias 差异:")
                print(f"  最大差异: {max_bias_diff:.6f}")
                print(f"  平均差异: {mean_bias_diff:.6f}")
                
                if max_bias_diff < 1e-6:
                    print(f"  ✅ bias 完全一致")
                elif max_bias_diff < 0.1:
                    print(f"  ⚠️  bias 存在小幅差异")
                else:
                    print(f"  ❌ bias 存在显著差异")
            else:
                print(f"  ❌ bias 形状不匹配")
            
        except Exception as e:
            print(f"❌ 权重转换分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _test_fix_solution(self):
        """测试修复方案"""
        print(f"\n{'='*60}")
        print(f"🔍 修复方案测试")
        print(f"{'='*60}")
        
        try:
            if not hasattr(self, 'pytorch_actual_weight') or not hasattr(self, 'mlx_weight'):
                print("❌ 缺少必要的权重数据")
                return
            
            print(f"\n📊 修复方案 1: 直接使用 PyTorch 实际权重")
            
            # 创建测试数据
            batch_size = 2
            seq_len = 415
            hidden_size = 512
            
            # PyTorch 计算（使用 weight_norm）
            from indextts.infer_v2 import IndexTTS2
            pytorch_tts = IndexTTS2(use_mlx=False)
            pytorch_dit = pytorch_tts.s2mel.models['cfm'].estimator
            pytorch_final_layer = pytorch_dit.final_layer
            
            # 创建测试输入
            device = next(pytorch_final_layer.parameters()).device
            x_pt = torch.randn(batch_size, seq_len, hidden_size).to(device)
            x_mx = mx.array(x_pt.cpu().numpy())
            
            with torch.no_grad():
                pytorch_output = pytorch_final_layer.linear(x_pt)
            
            # MLX 计算（使用普通 Linear）
            mlx_tts = IndexTTS2(use_mlx=True)
            mlx_dit = mlx_tts.mlx_s2mel_cfm.estimator
            mlx_final_layer = mlx_dit.final_layer
            
            mlx_output = mlx_final_layer.linear(x_mx)
            
            # 比较输出
            pytorch_np = pytorch_output.detach().cpu().numpy()
            mlx_np = np.array(mlx_output)
            
            print(f"  PyTorch 输出: {pytorch_output.shape}, range=[{pytorch_output.min().item():.6f}, {pytorch_output.max().item():.6f}]")
            print(f"  MLX 输出: {mlx_output.shape}, range=[{np.min(mlx_np):.6f}, {np.max(mlx_np):.6f}]")
            
            if pytorch_np.shape == mlx_np.shape:
                diff = np.abs(pytorch_np - mlx_np)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                
                print(f"  输出差异: max={max_diff:.6f}, mean={mean_diff:.6f}")
                
                if max_diff < 1e-6:
                    print(f"  ✅ 输出完全一致")
                elif max_diff < 0.1:
                    print(f"  ⚠️  输出存在小幅差异")
                else:
                    print(f"  ❌ 输出存在显著差异")
            else:
                print(f"  ❌ 输出形状不匹配")
            
            print(f"\n📊 修复方案 2: 在 MLX 中实现 weight_norm")
            self._test_mlx_weight_norm_implementation(x_mx, pytorch_output)
            
        except Exception as e:
            print(f"❌ 修复方案测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _test_mlx_weight_norm_implementation(self, x_mx, pytorch_output):
        """测试 MLX weight_norm 实现"""
        try:
            print(f"  测试 MLX weight_norm 实现...")
            
            # 获取 PyTorch weight_norm 参数
            from indextts.infer_v2 import IndexTTS2
            pytorch_tts = IndexTTS2(use_mlx=False)
            pytorch_dit = pytorch_tts.s2mel.models['cfm'].estimator
            pytorch_final_layer = pytorch_dit.final_layer
            
            with torch.no_grad():
                weight_g = pytorch_final_layer.linear.weight_g.detach().cpu().numpy()
                weight_v = pytorch_final_layer.linear.weight_v.detach().cpu().numpy()
                bias = pytorch_final_layer.linear.bias.detach().cpu().numpy()
            
            # 在 MLX 中实现 weight_norm
            weight_g_mx = mx.array(weight_g)
            weight_v_mx = mx.array(weight_v)
            bias_mx = mx.array(bias)
            
            # 计算 L2 范数
            weight_v_norm_mx = mx.sqrt(mx.sum(weight_v_mx ** 2, axis=1, keepdims=True))
            
            # 计算实际权重
            actual_weight_mx = weight_g_mx * weight_v_mx / weight_v_norm_mx
            
            # 执行线性变换
            mlx_weight_norm_output = mx.matmul(x_mx, actual_weight_mx.T) + bias_mx
            
            # 比较输出
            mlx_weight_norm_np = np.array(mlx_weight_norm_output)
            pytorch_np = pytorch_output.detach().cpu().numpy()
            
            print(f"    PyTorch 输出: {pytorch_output.shape}, range=[{pytorch_output.min().item():.6f}, {pytorch_output.max().item():.6f}]")
            print(f"    MLX weight_norm 输出: {mlx_weight_norm_output.shape}, range=[{np.min(mlx_weight_norm_np):.6f}, {np.max(mlx_weight_norm_np):.6f}]")
            
            if pytorch_np.shape == mlx_weight_norm_np.shape:
                diff = np.abs(pytorch_np - mlx_weight_norm_np)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                
                print(f"    输出差异: max={max_diff:.6f}, mean={mean_diff:.6f}")
                
                if max_diff < 1e-6:
                    print(f"    ✅ MLX weight_norm 实现完全一致")
                elif max_diff < 0.1:
                    print(f"    ⚠️  MLX weight_norm 实现存在小幅差异")
                else:
                    print(f"    ❌ MLX weight_norm 实现存在显著差异")
            else:
                print(f"    ❌ 输出形状不匹配")
                
        except Exception as e:
            print(f"    ❌ MLX weight_norm 实现测试失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    analyzer = WeightNormAnalyzer()
    analyzer.analyze_weight_norm_differences()

if __name__ == "__main__":
    main()
