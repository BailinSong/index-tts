#!/usr/bin/env python3
"""
DiT 层实现差异分析工具
基于 analyze_cfm_stages.py 的入口，深入分析 PyTorch 和 MLX DiT 层的实现差异
"""

import os
import sys
import pickle
import glob
import torch
import mlx.core as mx
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import inspect
import ast

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class DiTLayerImplementationAnalyzer:
    def __init__(self, cache_dir: str = "cfm_production_cache"):
        self.cache_dir = cache_dir
        
    def analyze_dit_implementation_differences(self):
        """分析 DiT 层实现差异"""
        print("🔍 DiT 层实现差异分析工具")
        print("="*60)
        
        # 1. 直接调用生产环境中的 DiT 模型进行对比
        self.analyze_dit_models_direct_comparison()
        
        # 2. 分析 DiT 前向传播实现差异
        self.analyze_dit_forward_implementation()
        
        # 3. 分析 DiT 关键组件实现差异
        self.analyze_dit_components()
        
        # 4. 分析 DiT 权重加载差异
        self.analyze_dit_weight_loading()
        
        # 5. 分析 DiT 数值计算差异
        self.analyze_dit_numerical_differences()
    
    def analyze_dit_models_direct_comparison(self):
        """直接调用生产环境中的 DiT 模型进行对比验证"""
        print(f"\n{'='*60}")
        print(f"🔍 生产环境 DiT 模型直接对比验证")
        print(f"{'='*60}")
        
        try:
            # 使用生产环境的 IndexTTS2 加载模型
            from indextts.infer_v2 import IndexTTS2
            
            print("🔧 加载生产环境 PyTorch DiT 模型...")
            pytorch_tts = IndexTTS2(use_mlx=False)
            pytorch_dit = pytorch_tts.s2mel.models['cfm'].estimator
            
            print("🔧 加载生产环境 MLX DiT 模型...")
            mlx_tts = IndexTTS2(use_mlx=True)
            mlx_dit = mlx_tts.mlx_s2mel_cfm.estimator
            
            print("✅ 成功加载生产环境 DiT 模型")
            
            # 准备测试数据（使用实际缓存数据）
            test_data = self._prepare_test_data()
            if test_data is None:
                print("❌ 无法准备测试数据")
                return
            
            # 执行 PyTorch DiT 前向传播
            print("\n🔧 执行 PyTorch DiT 前向传播...")
            pytorch_output = self._run_pytorch_dit_forward(pytorch_dit, test_data)
            
            # 执行 MLX DiT 前向传播
            print("🔧 执行 MLX DiT 前向传播...")
            mlx_output = self._run_mlx_dit_forward(mlx_dit, test_data)
            
            # 对比结果
            if pytorch_output is not None and mlx_output is not None:
                self._compare_dit_outputs(pytorch_output, mlx_output)
            
        except Exception as e:
            print(f"❌ 生产环境 DiT 模型对比失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _prepare_test_data(self):
        """准备测试数据"""
        # 查找最新的 DiT 输入缓存文件
        dit_input_files = glob.glob(os.path.join(self.cache_dir, "*_dit_input_*.pkl"))
        
        if not dit_input_files:
            print("   ⚠️  未找到 DiT 输入缓存文件，使用模拟数据")
            return self._create_mock_test_data()
        
        # 加载最新的输入文件
        latest_file = max(dit_input_files, key=os.path.getctime)
        
        try:
            with open(latest_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"   ✅ 加载测试数据: {os.path.basename(latest_file)}")
            return data
            
        except Exception as e:
            print(f"   ❌ 加载测试数据失败: {e}")
            return self._create_mock_test_data()
    
    def _create_mock_test_data(self):
        """创建模拟测试数据"""
        print("   🔧 创建模拟测试数据...")
        
        # 使用与生产环境相同的形状
        batch_size = 2
        seq_len = 415
        mel_dim = 80
        cond_dim = 512
        style_dim = 192
        
        mock_data = {
            'x': torch.randn(batch_size, mel_dim, seq_len),
            'prompt_x': torch.randn(batch_size, mel_dim, seq_len),
            'mu': torch.randn(batch_size, seq_len, cond_dim),
            'style': torch.randn(batch_size, style_dim),
            't': torch.randn(batch_size),
            'x_lens': torch.tensor([seq_len, seq_len])
        }
        
        return mock_data
    
    def _run_pytorch_dit_forward(self, pytorch_dit, test_data):
        """运行 PyTorch DiT 前向传播"""
        try:
            # 确保数据在正确的设备上
            device = next(pytorch_dit.parameters()).device
            
            # 准备输入数据 - 处理 numpy 数组
            x = test_data['x']
            prompt_x = test_data['prompt_x']
            mu = test_data['mu']
            style = test_data['style']
            t = test_data['t']
            x_lens = test_data['x_lens']
            
            # 转换为 torch.Tensor 如果还不是
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).to(device)
            else:
                x = x.to(device)
                
            if isinstance(prompt_x, np.ndarray):
                prompt_x = torch.from_numpy(prompt_x).to(device)
            else:
                prompt_x = prompt_x.to(device)
                
            if isinstance(mu, np.ndarray):
                mu = torch.from_numpy(mu).to(device)
            else:
                mu = mu.to(device)
                
            if isinstance(style, np.ndarray):
                style = torch.from_numpy(style).to(device)
            else:
                style = style.to(device)
                
            if isinstance(t, np.ndarray):
                t = torch.from_numpy(t).to(device)
            else:
                t = t.to(device)
                
            if isinstance(x_lens, np.ndarray):
                x_lens = torch.from_numpy(x_lens).to(device)
            else:
                x_lens = x_lens.to(device)
            
            # 执行前向传播
            with torch.no_grad():
                output = pytorch_dit(x, prompt_x, x_lens, t, style, mu)
            
            print(f"   ✅ PyTorch DiT 输出形状: {output.shape}")
            return output
            
        except Exception as e:
            print(f"   ❌ PyTorch DiT 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _run_mlx_dit_forward(self, mlx_dit, test_data):
        """运行 MLX DiT 前向传播"""
        try:
            # 准备输入数据（转换为 MLX 格式）
            x = test_data['x']
            prompt_x = test_data['prompt_x']
            mu = test_data['mu']
            style = test_data['style']
            t = test_data['t']
            x_lens = test_data['x_lens']
            
            # 转换为 MLX 数组
            if isinstance(x, torch.Tensor):
                x = mx.array(x.numpy())
            elif isinstance(x, np.ndarray):
                x = mx.array(x)
                
            if isinstance(prompt_x, torch.Tensor):
                prompt_x = mx.array(prompt_x.numpy())
            elif isinstance(prompt_x, np.ndarray):
                prompt_x = mx.array(prompt_x)
                
            if isinstance(mu, torch.Tensor):
                mu = mx.array(mu.numpy())
            elif isinstance(mu, np.ndarray):
                mu = mx.array(mu)
                
            if isinstance(style, torch.Tensor):
                style = mx.array(style.numpy())
            elif isinstance(style, np.ndarray):
                style = mx.array(style)
                
            if isinstance(t, torch.Tensor):
                t = mx.array(t.numpy())
            elif isinstance(t, np.ndarray):
                t = mx.array(t)
                
            if isinstance(x_lens, torch.Tensor):
                x_lens = mx.array(x_lens.numpy())
            elif isinstance(x_lens, np.ndarray):
                x_lens = mx.array(x_lens)
            
            # 执行前向传播
            output = mlx_dit(x, prompt_x, x_lens, t, style, mu)
            
            print(f"   ✅ MLX DiT 输出形状: {output.shape}")
            return output
            
        except Exception as e:
            print(f"   ❌ MLX DiT 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _compare_dit_outputs(self, pytorch_output, mlx_output):
        """对比 DiT 输出结果"""
        print(f"\n📊 DiT 输出结果对比:")
        
        # 转换为 numpy 进行比较
        pytorch_np = pytorch_output.detach().cpu().numpy()
        mlx_np = np.array(mlx_output)
        
        print(f"   PyTorch 输出: shape={pytorch_np.shape}, min={np.min(pytorch_np):.6f}, max={np.max(pytorch_np):.6f}, avg={np.mean(pytorch_np):.6f}")
        print(f"   MLX 输出:     shape={mlx_np.shape}, min={np.min(mlx_np):.6f}, max={np.max(mlx_np):.6f}, avg={np.mean(mlx_np):.6f}")
        
        # 检查形状
        if pytorch_np.shape != mlx_np.shape:
            print(f"   ❌ 输出形状不匹配")
            return
        
        # 计算差异
        diff = np.abs(pytorch_np - mlx_np)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"   输出差异: max={max_diff:.6f}, mean={mean_diff:.6f}")
        
        if max_diff < 1e-6:
            print(f"   ✅ DiT 输出完全一致")
        elif max_diff < 0.1:
            print(f"   ⚠️  DiT 输出存在小幅差异")
        else:
            print(f"   ❌ DiT 输出存在显著差异")
            
        # 分析差异分布
        self._analyze_difference_distribution(diff)
    
    def _analyze_difference_distribution(self, diff):
        """分析差异分布"""
        print(f"\n📊 差异分布分析:")
        
        # 统计差异
        diff_stats = {
            'min': np.min(diff),
            'max': np.max(diff),
            'mean': np.mean(diff),
            'std': np.std(diff),
            'median': np.median(diff),
            'p95': np.percentile(diff, 95),
            'p99': np.percentile(diff, 99)
        }
        
        for stat_name, stat_value in diff_stats.items():
            print(f"   {stat_name}: {stat_value:.6f}")
        
        # 分析差异模式
        large_diff_mask = diff > 0.01
        large_diff_count = np.sum(large_diff_mask)
        total_count = diff.size
        
        print(f"   大差异(>0.01)数量: {large_diff_count}/{total_count} ({large_diff_count/total_count*100:.2f}%)")
        
        if large_diff_count > 0:
            print(f"   大差异位置分析:")
            large_diff_indices = np.where(large_diff_mask)
            if len(large_diff_indices[0]) > 0:
                print(f"     最大差异位置: {large_diff_indices[0][0]}, {large_diff_indices[1][0]}, {large_diff_indices[2][0]}")
                print(f"     最大差异值: {diff[large_diff_indices[0][0], large_diff_indices[1][0], large_diff_indices[2][0]]:.6f}")
    
    def analyze_dit_model_structure(self):
        """分析 DiT 模型结构差异"""
        print(f"\n{'='*60}")
        print(f"🔍 DiT 模型结构分析")
        print(f"{'='*60}")
        
        try:
            # 加载 PyTorch DiT 模型
            from indextts.infer_v2 import IndexTTS2
            pytorch_tts = IndexTTS2(use_mlx=False)
            pytorch_dit = pytorch_tts.s2mel.models['cfm'].estimator
            
            # 加载 MLX DiT 模型
            mlx_tts = IndexTTS2(use_mlx=True)
            mlx_dit = mlx_tts.mlx_s2mel_cfm.estimator
            
            print("✅ 成功加载 PyTorch 和 MLX DiT 模型")
            
            # 分析模型结构
            self._compare_model_attributes(pytorch_dit, mlx_dit, "DiT")
            
            # 分析 Transformer 结构
            if hasattr(pytorch_dit, 'transformer') and hasattr(mlx_dit, 'transformer'):
                self._compare_model_attributes(pytorch_dit.transformer, mlx_dit.transformer, "Transformer")
                
                # 分析 Transformer 层
                if hasattr(pytorch_dit.transformer, 'layers') and hasattr(mlx_dit.transformer, 'layers'):
                    self._compare_transformer_layers(pytorch_dit.transformer.layers, mlx_dit.transformer.layers)
            
            # 分析其他关键组件
            key_components = ['x_embedder', 't_embedder', 'cond_projection', 'cond_embedder', 'cond_x_merge_linear', 'final_layer']
            for component in key_components:
                if hasattr(pytorch_dit, component) and hasattr(mlx_dit, component):
                    self._compare_model_attributes(
                        getattr(pytorch_dit, component), 
                        getattr(mlx_dit, component), 
                        component
                    )
                    
        except Exception as e:
            print(f"❌ DiT 模型结构分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _compare_model_attributes(self, pytorch_model, mlx_model, model_name: str):
        """比较模型属性"""
        print(f"\n📊 {model_name} 属性比较:")
        
        # 获取属性列表
        pytorch_attrs = set(dir(pytorch_model))
        mlx_attrs = set(dir(mlx_model))
        
        common_attrs = pytorch_attrs & mlx_attrs
        pytorch_only = pytorch_attrs - mlx_attrs
        mlx_only = mlx_attrs - pytorch_attrs
        
        print(f"   共同属性: {len(common_attrs)}")
        print(f"   PyTorch 独有: {len(pytorch_only)}")
        print(f"   MLX 独有: {len(mlx_only)}")
        
        if pytorch_only:
            print(f"   PyTorch 独有属性: {sorted(list(pytorch_only))[:10]}")
        if mlx_only:
            print(f"   MLX 独有属性: {sorted(list(mlx_only))[:10]}")
        
        # 分析关键属性
        key_attrs = ['weight', 'bias', 'layers', 'num_layers', 'hidden_size', 'num_heads']
        for attr in key_attrs:
            if hasattr(pytorch_model, attr) and hasattr(mlx_model, attr):
                pytorch_val = getattr(pytorch_model, attr)
                mlx_val = getattr(mlx_model, attr)
                
                if hasattr(pytorch_val, 'shape') and hasattr(mlx_val, 'shape'):
                    print(f"   {attr}: PyTorch={pytorch_val.shape}, MLX={mlx_val.shape}")
                else:
                    print(f"   {attr}: PyTorch={type(pytorch_val)}, MLX={type(mlx_val)}")
    
    def _compare_transformer_layers(self, pytorch_layers, mlx_layers):
        """比较 Transformer 层"""
        print(f"\n📊 Transformer 层比较:")
        print(f"   PyTorch 层数: {len(pytorch_layers)}")
        print(f"   MLX 层数: {len(mlx_layers)}")
        
        if len(pytorch_layers) != len(mlx_layers):
            print(f"   ❌ 层数不匹配")
            return
        
        # 分析前几层
        for i in range(min(3, len(pytorch_layers))):
            print(f"\n   📊 Layer {i}:")
            self._compare_model_attributes(pytorch_layers[i], mlx_layers[i], f"Layer_{i}")
            
            # 分析注意力机制
            if hasattr(pytorch_layers[i], 'attention') and hasattr(mlx_layers[i], 'attention'):
                self._compare_model_attributes(
                    pytorch_layers[i].attention, 
                    mlx_layers[i].attention, 
                    f"Layer_{i}_Attention"
                )
            
            # 分析前馈网络
            if hasattr(pytorch_layers[i], 'feed_forward') and hasattr(mlx_layers[i], 'feed_forward'):
                self._compare_model_attributes(
                    pytorch_layers[i].feed_forward, 
                    mlx_layers[i].feed_forward, 
                    f"Layer_{i}_FeedForward"
                )
    
    def analyze_dit_forward_implementation(self):
        """分析 DiT 前向传播实现差异"""
        print(f"\n{'='*60}")
        print(f"🔍 DiT 前向传播实现分析")
        print(f"{'='*60}")
        
        try:
            # 获取 PyTorch DiT 实现
            from indextts.s2mel.modules.diffusion_transformer import DiT
            pytorch_forward = DiT.forward
            
            # 获取 MLX DiT 实现
            from indextts.s2mel.modules.mlx_diffusion_transformer import MLXDiTRewritten
            mlx_forward = MLXDiTRewritten.__call__
            
            # 分析函数签名
            self._compare_function_signatures(pytorch_forward, mlx_forward, "DiT Forward")
            
            # 分析函数源码
            self._compare_function_source(pytorch_forward, mlx_forward, "DiT Forward")
            
        except Exception as e:
            print(f"❌ DiT 前向传播实现分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _compare_function_signatures(self, pytorch_func, mlx_func, func_name: str):
        """比较函数签名"""
        print(f"\n📊 {func_name} 函数签名比较:")
        
        try:
            pytorch_sig = inspect.signature(pytorch_func)
            mlx_sig = inspect.signature(mlx_func)
            
            print(f"   PyTorch 参数: {list(pytorch_sig.parameters.keys())}")
            print(f"   MLX 参数: {list(mlx_sig.parameters.keys())}")
            
            # 比较参数
            pytorch_params = set(pytorch_sig.parameters.keys())
            mlx_params = set(mlx_sig.parameters.keys())
            
            common_params = pytorch_params & mlx_params
            pytorch_only = pytorch_params - mlx_params
            mlx_only = mlx_params - pytorch_params
            
            print(f"   共同参数: {len(common_params)}")
            print(f"   PyTorch 独有: {pytorch_only}")
            print(f"   MLX 独有: {mlx_only}")
            
        except Exception as e:
            print(f"   ❌ 函数签名分析失败: {e}")
    
    def _compare_function_source(self, pytorch_func, mlx_func, func_name: str):
        """比较函数源码"""
        print(f"\n📊 {func_name} 源码比较:")
        
        try:
            pytorch_source = inspect.getsource(pytorch_func)
            mlx_source = inspect.getsource(mlx_func)
            
            pytorch_lines = pytorch_source.split('\n')
            mlx_lines = mlx_source.split('\n')
            
            print(f"   PyTorch 行数: {len(pytorch_lines)}")
            print(f"   MLX 行数: {len(mlx_lines)}")
            
            # 显示前几行
            print(f"\n   PyTorch 前5行:")
            for i, line in enumerate(pytorch_lines[:5]):
                print(f"     {i+1:2d}: {line}")
            
            print(f"\n   MLX 前5行:")
            for i, line in enumerate(mlx_lines[:5]):
                print(f"     {i+1:2d}: {line}")
            
            # 分析关键差异
            self._analyze_source_differences(pytorch_source, mlx_source)
            
        except Exception as e:
            print(f"   ❌ 源码分析失败: {e}")
    
    def _analyze_source_differences(self, pytorch_source: str, mlx_source: str):
        """分析源码差异"""
        print(f"\n📊 源码差异分析:")
        
        # 简单的关键词分析
        pytorch_keywords = {
            'transpose': pytorch_source.count('transpose'),
            'concatenate': pytorch_source.count('concatenate'),
            'cat': pytorch_source.count('cat'),
            'matmul': pytorch_source.count('matmul'),
            'bmm': pytorch_source.count('bmm'),
            'softmax': pytorch_source.count('softmax'),
            'layer_norm': pytorch_source.count('layer_norm'),
            'linear': pytorch_source.count('linear'),
        }
        
        mlx_keywords = {
            'transpose': mlx_source.count('transpose'),
            'concatenate': mlx_source.count('concatenate'),
            'cat': mlx_source.count('cat'),
            'matmul': mlx_source.count('matmul'),
            'bmm': mlx_source.count('bmm'),
            'softmax': mlx_source.count('softmax'),
            'layer_norm': mlx_source.count('layer_norm'),
            'linear': mlx_source.count('linear'),
        }
        
        print(f"   关键词统计:")
        for keyword in pytorch_keywords:
            pt_count = pytorch_keywords[keyword]
            mx_count = mlx_keywords[keyword]
            diff = abs(pt_count - mx_count)
            status = "✅" if diff == 0 else "⚠️" if diff <= 2 else "❌"
            print(f"     {keyword}: PyTorch={pt_count}, MLX={mx_count}, 差异={diff} {status}")
    
    def analyze_dit_components(self):
        """分析 DiT 关键组件实现差异"""
        print(f"\n{'='*60}")
        print(f"🔍 DiT 关键组件实现分析")
        print(f"{'='*60}")
        
        try:
            # 分析注意力机制
            self._analyze_attention_implementation()
            
            # 分析层归一化
            self._analyze_layer_norm_implementation()
            
            # 分析前馈网络
            self._analyze_feed_forward_implementation()
            
        except Exception as e:
            print(f"❌ DiT 组件分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_attention_implementation(self):
        """分析注意力机制实现"""
        print(f"\n📊 注意力机制实现分析:")
        
        try:
            # 获取 PyTorch 注意力实现
            from indextts.s2mel.modules.gpt_fast.model import Attention
            pytorch_attention = Attention
            
            # 获取 MLX 注意力实现
            from indextts.s2mel.modules.mlx_gpt_fast_model import MLXAttentionGPTFast as MLXAttention
            mlx_attention = MLXAttention
            
            # 比较前向传播
            pytorch_forward = pytorch_attention.forward
            mlx_forward = mlx_attention.__call__
            
            self._compare_function_signatures(pytorch_forward, mlx_forward, "Attention Forward")
            self._compare_function_source(pytorch_forward, mlx_forward, "Attention Forward")
            
        except Exception as e:
            print(f"   ❌ 注意力机制分析失败: {e}")
    
    def _analyze_layer_norm_implementation(self):
        """分析层归一化实现"""
        print(f"\n📊 层归一化实现分析:")
        
        try:
            # 分析 AdaLN 实现
            from indextts.s2mel.modules.gpt_fast.model import AdaptiveLayerNorm as AdaLN
            pytorch_adaln = AdaLN
            
            from indextts.s2mel.modules.mlx_gpt_fast_model import MLXAdaptiveLayerNorm as MLXAdaLN
            mlx_adaln = MLXAdaLN
            
            pytorch_forward = pytorch_adaln.forward
            mlx_forward = mlx_adaln.__call__
            
            self._compare_function_signatures(pytorch_forward, mlx_forward, "AdaLN Forward")
            self._compare_function_source(pytorch_forward, mlx_forward, "AdaLN Forward")
            
        except Exception as e:
            print(f"   ❌ 层归一化分析失败: {e}")
    
    def _analyze_feed_forward_implementation(self):
        """分析前馈网络实现"""
        print(f"\n📊 前馈网络实现分析:")
        
        try:
            from indextts.s2mel.modules.gpt_fast.model import FeedForward
            pytorch_ff = FeedForward
            
            from indextts.s2mel.modules.mlx_gpt_fast_model import MLXFeedForward
            mlx_ff = MLXFeedForward
            
            pytorch_forward = pytorch_ff.forward
            mlx_forward = mlx_ff.__call__
            
            self._compare_function_signatures(pytorch_forward, mlx_forward, "FeedForward Forward")
            self._compare_function_source(pytorch_forward, mlx_forward, "FeedForward Forward")
            
        except Exception as e:
            print(f"   ❌ 前馈网络分析失败: {e}")
    
    def analyze_dit_weight_loading(self):
        """分析 DiT 权重加载差异"""
        print(f"\n{'='*60}")
        print(f"🔍 DiT 权重加载分析")
        print(f"{'='*60}")
        
        try:
            # 加载 PyTorch 权重
            pytorch_checkpoint = torch.load("checkpoints/s2mel.pth", map_location='cpu')
            pytorch_dit_weights = pytorch_checkpoint['net']['cfm']
            
            # 加载 MLX 权重
            mlx_checkpoint = np.load("checkpoints/mlx/s2mel.npz", allow_pickle=True)
            mlx_dit_weights = {}
            for key, value in mlx_checkpoint.items():
                if key.startswith('cfm.estimator.'):
                    clean_key = key[len('cfm.estimator.'):]
                    mlx_dit_weights[clean_key] = value
            
            print(f"   PyTorch 权重数量: {len(pytorch_dit_weights)}")
            print(f"   MLX 权重数量: {len(mlx_dit_weights)}")
            
            # 比较权重键
            pytorch_keys = set(pytorch_dit_weights.keys())
            mlx_keys = set(mlx_dit_weights.keys())
            
            common_keys = pytorch_keys & mlx_keys
            pytorch_only = pytorch_keys - mlx_keys
            mlx_only = mlx_keys - pytorch_keys
            
            print(f"   共同权重: {len(common_keys)}")
            print(f"   PyTorch 独有: {len(pytorch_only)}")
            print(f"   MLX 独有: {len(mlx_only)}")
            
            if pytorch_only:
                print(f"   PyTorch 独有权重: {sorted(list(pytorch_only))[:10]}")
            if mlx_only:
                print(f"   MLX 独有权重: {sorted(list(mlx_only))[:10]}")
            
            # 分析关键权重
            key_weights = ['x_embedder.weight', 't_embedder.freqs', 'cond_projection.weight', 'transformer.layers.0.attention.wq.weight']
            for weight_name in key_weights:
                if weight_name in pytorch_dit_weights and weight_name in mlx_dit_weights:
                    pt_weight = pytorch_dit_weights[weight_name]
                    mx_weight = mlx_dit_weights[weight_name]
                    
                    if hasattr(pt_weight, 'shape') and hasattr(mx_weight, 'shape'):
                        print(f"   {weight_name}: PyTorch={pt_weight.shape}, MLX={mx_weight.shape}")
                    else:
                        print(f"   {weight_name}: PyTorch={type(pt_weight)}, MLX={type(mx_weight)}")
            
        except Exception as e:
            print(f"❌ DiT 权重加载分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def analyze_dit_numerical_differences(self):
        """分析 DiT 数值计算差异"""
        print(f"\n{'='*60}")
        print(f"🔍 DiT 数值计算差异分析")
        print(f"{'='*60}")
        
        # 加载实际的 DiT 输入输出数据
        self._analyze_dit_io_differences()
        
        # 分析关键计算步骤
        self._analyze_key_computations()
    
    def _analyze_dit_io_differences(self):
        """分析 DiT 输入输出差异"""
        print(f"\n📊 DiT 输入输出差异分析:")
        
        # 查找最新的 DiT 缓存文件
        dit_input_files = glob.glob(os.path.join(self.cache_dir, "*_dit_input_*.pkl"))
        dit_output_files = glob.glob(os.path.join(self.cache_dir, "*_dit_output_*.pkl"))
        
        if not dit_input_files or not dit_output_files:
            print("   ⚠️  未找到 DiT 缓存文件")
            return
        
        # 加载最新的文件
        latest_input = max(dit_input_files, key=os.path.getctime)
        latest_output = max(dit_output_files, key=os.path.getctime)
        
        try:
            with open(latest_input, 'rb') as f:
                input_data = pickle.load(f)
            
            with open(latest_output, 'rb') as f:
                output_data = pickle.load(f)
            
            print(f"   输入文件: {os.path.basename(latest_input)}")
            print(f"   输出文件: {os.path.basename(latest_output)}")
            
            # 分析输入数据
            print(f"\n   输入数据:")
            for key, value in input_data.items():
                if hasattr(value, 'shape'):
                    print(f"     {key}: shape={value.shape}")
                else:
                    print(f"     {key}: {type(value)}")
            
            # 分析输出数据
            print(f"\n   输出数据:")
            for key, value in output_data.items():
                if hasattr(value, 'shape'):
                    print(f"     {key}: shape={value.shape}")
                else:
                    print(f"     {key}: {type(value)}")
            
        except Exception as e:
            print(f"   ❌ 加载 DiT 缓存文件失败: {e}")
    
    def _analyze_key_computations(self):
        """分析关键计算步骤"""
        print(f"\n📊 关键计算步骤分析:")
        
        # 分析转置操作
        self._analyze_transpose_operations()
        
        # 分析拼接操作
        self._analyze_concatenation_operations()
        
        # 分析矩阵乘法
        self._analyze_matrix_operations()
    
    def _analyze_transpose_operations(self):
        """分析转置操作"""
        print(f"\n   转置操作分析:")
        
        # 创建测试数据
        test_data = torch.randn(2, 80, 415)
        
        # PyTorch 转置
        pytorch_transposed = test_data.transpose(1, 2)
        
        # MLX 转置
        test_data_mlx = mx.array(test_data.numpy())
        mlx_transposed = test_data_mlx.transpose(0, 2, 1)
        
        print(f"     原始形状: {test_data.shape}")
        print(f"     PyTorch 转置: {pytorch_transposed.shape}")
        print(f"     MLX 转置: {mlx_transposed.shape}")
        
        # 比较结果
        pytorch_np = pytorch_transposed.numpy()
        mlx_np = np.array(mlx_transposed)
        
        if pytorch_np.shape == mlx_np.shape:
            diff = np.abs(pytorch_np - mlx_np)
            max_diff = np.max(diff)
            print(f"     转置差异: max={max_diff:.10f}")
            if max_diff < 1e-10:
                print(f"     ✅ 转置操作一致")
            else:
                print(f"     ❌ 转置操作不一致")
        else:
            print(f"     ❌ 转置结果形状不匹配")
    
    def _analyze_concatenation_operations(self):
        """分析拼接操作"""
        print(f"\n   拼接操作分析:")
        
        # 创建测试数据
        a = torch.randn(2, 415, 80)
        b = torch.randn(2, 415, 80)
        c = torch.randn(2, 415, 512)
        
        # PyTorch 拼接
        pytorch_concat = torch.cat([a, b, c], dim=-1)
        
        # MLX 拼接
        a_mlx = mx.array(a.numpy())
        b_mlx = mx.array(b.numpy())
        c_mlx = mx.array(c.numpy())
        mlx_concat = mx.concatenate([a_mlx, b_mlx, c_mlx], -1)
        
        print(f"     输入形状: a={a.shape}, b={b.shape}, c={c.shape}")
        print(f"     PyTorch 拼接: {pytorch_concat.shape}")
        print(f"     MLX 拼接: {mlx_concat.shape}")
        
        # 比较结果
        pytorch_np = pytorch_concat.numpy()
        mlx_np = np.array(mlx_concat)
        
        if pytorch_np.shape == mlx_np.shape:
            diff = np.abs(pytorch_np - mlx_np)
            max_diff = np.max(diff)
            print(f"     拼接差异: max={max_diff:.10f}")
            if max_diff < 1e-10:
                print(f"     ✅ 拼接操作一致")
            else:
                print(f"     ❌ 拼接操作不一致")
        else:
            print(f"     ❌ 拼接结果形状不匹配")
    
    def _analyze_matrix_operations(self):
        """分析矩阵操作"""
        print(f"\n   矩阵操作分析:")
        
        # 创建测试数据
        a = torch.randn(2, 415, 80)
        b = torch.randn(80, 512)
        
        # PyTorch 矩阵乘法
        pytorch_matmul = torch.matmul(a, b)
        
        # MLX 矩阵乘法
        a_mlx = mx.array(a.numpy())
        b_mlx = mx.array(b.numpy())
        mlx_matmul = mx.matmul(a_mlx, b_mlx)
        
        print(f"     输入形状: a={a.shape}, b={b.shape}")
        print(f"     PyTorch 矩阵乘法: {pytorch_matmul.shape}")
        print(f"     MLX 矩阵乘法: {mlx_matmul.shape}")
        
        # 比较结果
        pytorch_np = pytorch_matmul.numpy()
        mlx_np = np.array(mlx_matmul)
        
        if pytorch_np.shape == mlx_np.shape:
            diff = np.abs(pytorch_np - mlx_np)
            max_diff = np.max(diff)
            print(f"     矩阵乘法差异: max={max_diff:.10f}")
            if max_diff < 1e-10:
                print(f"     ✅ 矩阵乘法一致")
            else:
                print(f"     ❌ 矩阵乘法不一致")
        else:
            print(f"     ❌ 矩阵乘法结果形状不匹配")

def main():
    """主函数"""
    analyzer = DiTLayerImplementationAnalyzer()
    analyzer.analyze_dit_implementation_differences()

if __name__ == "__main__":
    main()
