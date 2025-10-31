#!/usr/bin/env python3
"""
cond_embedder 差异分析工具
专门分析 PyTorch 和 MLX cond_embedder 的实现差异
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

class CondEmbedderDiffAnalyzer:
    def __init__(self, cache_dir: str = "cfm_production_cache"):
        self.cache_dir = cache_dir
        
    def analyze_cond_embedder_diff(self):
        """分析 cond_embedder 差异"""
        print("🔍 cond_embedder 差异分析工具")
        print("="*60)
        
        # 1. 加载生产环境模型
        pytorch_dit, mlx_dit = self._load_production_models()
        if pytorch_dit is None or mlx_dit is None:
            return
        
        # 2. 准备测试数据
        test_data = self._prepare_test_data()
        if test_data is None:
            return
        
        # 3. 分析 cond_embedder 差异
        self._analyze_cond_embedder_implementation(pytorch_dit, mlx_dit, test_data)
        
    def _load_production_models(self):
        """加载生产环境模型"""
        print("🔧 加载生产环境 DiT 模型...")
        
        try:
            from indextts.infer_v2 import IndexTTS2
            
            # 加载 PyTorch DiT
            pytorch_tts = IndexTTS2(use_mlx=False)
            pytorch_dit = pytorch_tts.s2mel.models['cfm'].estimator
            
            # 加载 MLX DiT
            mlx_tts = IndexTTS2(use_mlx=True)
            mlx_dit = mlx_tts.mlx_s2mel_cfm.estimator
            
            print("✅ 成功加载生产环境 DiT 模型")
            return pytorch_dit, mlx_dit
            
        except Exception as e:
            print(f"❌ 加载生产环境模型失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _prepare_test_data(self):
        """准备测试数据"""
        print("🔧 准备测试数据...")
        
        # 查找最新的 DiT 输入缓存文件
        dit_input_files = glob.glob(os.path.join(self.cache_dir, "*_dit_input_*.pkl"))
        
        if not dit_input_files:
            print("❌ 未找到 DiT 输入缓存文件")
            return None
        
        # 加载最新的输入文件
        latest_file = max(dit_input_files, key=os.path.getctime)
        
        try:
            with open(latest_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"✅ 加载测试数据: {os.path.basename(latest_file)}")
            return data
            
        except Exception as e:
            print(f"❌ 加载测试数据失败: {e}")
            return None
    
    def _analyze_cond_embedder_implementation(self, pytorch_dit, mlx_dit, test_data):
        """分析 cond_embedder 实现差异"""
        print(f"\n{'='*60}")
        print(f"🔍 cond_embedder 实现差异分析")
        print(f"{'='*60}")
        
        # 准备输入数据
        mu = test_data['mu']
        
        # 转换为 PyTorch 格式
        device = next(pytorch_dit.parameters()).device
        mu_pt = torch.as_tensor(mu).to(device)
        
        # 转换为 MLX 格式
        mu_mx = mx.array(mu)
        
        print(f"输入数据:")
        print(f"  mu 形状: {mu_pt.shape}")
        print(f"  mu 数据类型: {mu_pt.dtype}")
        print(f"  mu 数值范围: min={mu_pt.min().item():.6f}, max={mu_pt.max().item():.6f}")
        
        # 分析 PyTorch cond_embedder
        self._analyze_pytorch_cond_embedder(pytorch_dit, mu_pt)
        
        # 分析 MLX cond_embedder
        self._analyze_mlx_cond_embedder(mlx_dit, mu_mx)
        
        # 比较两种实现
        self._compare_cond_embedder_implementations(pytorch_dit, mlx_dit, mu_pt, mu_mx)
    
    def _analyze_pytorch_cond_embedder(self, pytorch_dit, mu_pt):
        """分析 PyTorch cond_embedder"""
        print(f"\n📊 PyTorch cond_embedder 分析:")
        
        cond_embedder = pytorch_dit.cond_embedder
        print(f"  类型: {type(cond_embedder)}")
        print(f"  权重形状: {cond_embedder.weight.shape}")
        print(f"  权重数值范围: min={cond_embedder.weight.min().item():.6f}, max={cond_embedder.weight.max().item():.6f}")
        
        # 分析输入要求
        print(f"  输入要求:")
        print(f"    期望输入类型: 整数索引")
        print(f"    实际输入类型: {mu_pt.dtype}")
        print(f"    输入形状: {mu_pt.shape}")
        
        # 尝试不同的输入处理方式
        print(f"\n  尝试不同的输入处理方式:")
        
        # 方式1: 直接使用浮点输入
        try:
            with torch.no_grad():
                output1 = cond_embedder(mu_pt)
            print(f"    方式1 (直接浮点): 成功, 输出形状 {output1.shape}")
        except Exception as e:
            print(f"    方式1 (直接浮点): 失败 - {e}")
        
        # 方式2: 转换为整数索引
        try:
            # 将浮点数转换为整数索引 (0-1023)
            mu_int = torch.clamp((mu_pt * 511 + 511).long(), 0, 1023)
            with torch.no_grad():
                output2 = cond_embedder(mu_int)
            print(f"    方式2 (整数索引): 成功, 输出形状 {output2.shape}")
            print(f"    索引范围: min={mu_int.min().item()}, max={mu_int.max().item()}")
        except Exception as e:
            print(f"    方式2 (整数索引): 失败 - {e}")
        
        # 方式3: 使用线性层而不是嵌入层
        try:
            # 检查是否有线性层版本
            if hasattr(pytorch_dit, 'cond_embedder_linear'):
                with torch.no_grad():
                    output3 = pytorch_dit.cond_embedder_linear(mu_pt)
                print(f"    方式3 (线性层): 成功, 输出形状 {output3.shape}")
            else:
                print(f"    方式3 (线性层): 不存在 cond_embedder_linear")
        except Exception as e:
            print(f"    方式3 (线性层): 失败 - {e}")
    
    def _analyze_mlx_cond_embedder(self, mlx_dit, mu_mx):
        """分析 MLX cond_embedder"""
        print(f"\n📊 MLX cond_embedder 分析:")
        
        cond_embedder = mlx_dit.cond_embedder
        print(f"  类型: {type(cond_embedder)}")
        
        # 检查 MLX cond_embedder 的属性
        print(f"  属性:")
        for attr in dir(cond_embedder):
            if not attr.startswith('_'):
                try:
                    value = getattr(cond_embedder, attr)
                    if hasattr(value, 'shape'):
                        print(f"    {attr}: {value.shape}")
                    else:
                        print(f"    {attr}: {type(value)}")
                except:
                    pass
        
        # 尝试直接使用浮点输入
        try:
            output = cond_embedder(mu_mx)
            print(f"  直接浮点输入: 成功, 输出形状 {output.shape}")
        except Exception as e:
            print(f"  直接浮点输入: 失败 - {e}")
    
    def _compare_cond_embedder_implementations(self, pytorch_dit, mlx_dit, mu_pt, mu_mx):
        """比较两种 cond_embedder 实现"""
        print(f"\n📊 cond_embedder 实现比较:")
        
        # 尝试找到可以比较的输出
        pytorch_output = None
        mlx_output = None
        
        # PyTorch 输出
        try:
            # 尝试整数索引方式
            mu_int = torch.clamp((mu_pt * 511 + 511).long(), 0, 1023)
            with torch.no_grad():
                pytorch_output = pytorch_dit.cond_embedder(mu_int)
            print(f"  PyTorch 输出: {pytorch_output.shape}")
        except Exception as e:
            print(f"  PyTorch 输出: 失败 - {e}")
        
        # MLX 输出
        try:
            mlx_output = mlx_dit.cond_embedder(mu_mx)
            print(f"  MLX 输出: {mlx_output.shape}")
        except Exception as e:
            print(f"  MLX 输出: 失败 - {e}")
        
        # 比较输出
        if pytorch_output is not None and mlx_output is not None:
            if pytorch_output.shape == mlx_output.shape:
                pytorch_np = pytorch_output.detach().cpu().numpy()
                mlx_np = np.array(mlx_output)
                
                diff = np.abs(pytorch_np - mlx_np)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                
                print(f"  输出比较:")
                print(f"    PyTorch: shape={pytorch_np.shape}, min={np.min(pytorch_np):.6f}, max={np.max(pytorch_np):.6f}, avg={np.mean(pytorch_np):.6f}")
                print(f"    MLX:     shape={mlx_np.shape}, min={np.min(mlx_np):.6f}, max={np.max(mlx_np):.6f}, avg={np.mean(mlx_np):.6f}")
                print(f"    差异: max={max_diff:.6f}, mean={mean_diff:.6f}")
                
                if max_diff < 1e-6:
                    print(f"    ✅ cond_embedder 输出完全一致")
                elif max_diff < 0.1:
                    print(f"    ⚠️  cond_embedder 输出存在小幅差异")
                else:
                    print(f"    ❌ cond_embedder 输出存在显著差异")
            else:
                print(f"  ❌ 输出形状不匹配: PyTorch={pytorch_output.shape}, MLX={mlx_output.shape}")
        else:
            print(f"  ❌ 无法比较输出")
    
    def analyze_cond_embedder_usage_in_dit(self, pytorch_dit, mlx_dit, test_data):
        """分析 cond_embedder 在 DiT 中的使用方式"""
        print(f"\n{'='*60}")
        print(f"🔍 cond_embedder 在 DiT 中的使用方式分析")
        print(f"{'='*60}")
        
        # 检查 DiT 的 forward 方法
        print(f"PyTorch DiT forward 方法:")
        pytorch_forward = pytorch_dit.forward
        print(f"  方法: {pytorch_forward}")
        print(f"  源码行数: {pytorch_forward.__code__.co_argcount}")
        
        # 检查 MLX DiT 的 __call__ 方法
        print(f"\nMLX DiT __call__ 方法:")
        mlx_call = mlx_dit.__call__
        print(f"  方法: {mlx_call}")
        print(f"  源码行数: {mlx_call.__code__.co_argcount}")
        
        # 分析源码
        try:
            import inspect
            pytorch_source = inspect.getsource(pytorch_forward)
            print(f"\nPyTorch DiT forward 源码 (前20行):")
            for i, line in enumerate(pytorch_source.split('\n')[:20]):
                print(f"  {i+1:2d}: {line}")
        except Exception as e:
            print(f"  无法获取 PyTorch 源码: {e}")
        
        try:
            mlx_source = inspect.getsource(mlx_call)
            print(f"\nMLX DiT __call__ 源码 (前20行):")
            for i, line in enumerate(mlx_source.split('\n')[:20]):
                print(f"  {i+1:2d}: {line}")
        except Exception as e:
            print(f"  无法获取 MLX 源码: {e}")

def main():
    """主函数"""
    analyzer = CondEmbedderDiffAnalyzer()
    analyzer.analyze_cond_embedder_diff()

if __name__ == "__main__":
    main()







