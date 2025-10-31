#!/usr/bin/env python3
"""
DiT 数据流分析工具
分析 PyTorch 和 MLX DiT 的数据流和形状处理差异
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

class DiTDataflowAnalyzer:
    def __init__(self, cache_dir: str = "cfm_production_cache"):
        self.cache_dir = cache_dir
        
    def analyze_dit_dataflow(self):
        """分析 DiT 数据流差异"""
        print("🔍 DiT 数据流分析工具")
        print("="*60)
        
        # 1. 加载生产环境模型
        pytorch_dit, mlx_dit = self._load_production_models()
        if pytorch_dit is None or mlx_dit is None:
            return
        
        # 2. 准备测试数据
        test_data = self._prepare_test_data()
        if test_data is None:
            return
        
        # 3. 分析数据流差异
        self._analyze_input_preprocessing(pytorch_dit, mlx_dit, test_data)
        self._analyze_embedding_layers(pytorch_dit, mlx_dit, test_data)
        self._analyze_transformer_input(pytorch_dit, mlx_dit, test_data)
        
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
    
    def _analyze_input_preprocessing(self, pytorch_dit, mlx_dit, test_data):
        """分析输入预处理差异"""
        print(f"\n{'='*60}")
        print(f"🔍 DiT 输入预处理分析")
        print(f"{'='*60}")
        
        # 原始输入
        x = test_data['x']
        prompt_x = test_data['prompt_x']
        mu = test_data['mu']
        style = test_data['style']
        t = test_data['t']
        x_lens = test_data['x_lens']
        
        print(f"原始输入形状:")
        print(f"  x: {x.shape}")
        print(f"  prompt_x: {prompt_x.shape}")
        print(f"  mu: {mu.shape}")
        print(f"  style: {style.shape}")
        print(f"  t: {t.shape}")
        print(f"  x_lens: {x_lens.shape}")
        
        # 分析转置操作
        self._analyze_transpose_operations(x, prompt_x)
        
        # 分析拼接操作
        self._analyze_concatenation_operations(x, prompt_x, mu)
        
        # 分析风格条件处理
        self._analyze_style_conditioning(style, x.shape[0], x.shape[2])
    
    def _analyze_transpose_operations(self, x, prompt_x):
        """分析转置操作"""
        print(f"\n📊 转置操作分析:")
        
        # PyTorch 转置
        x_t_pt = torch.tensor(x).transpose(1, 2)  # (B, C, T) -> (B, T, C)
        prompt_x_t_pt = torch.tensor(prompt_x).transpose(1, 2)
        
        # MLX 转置
        x_t_mx = mx.array(x).transpose(0, 2, 1)  # (B, C, T) -> (B, T, C)
        prompt_x_t_mx = mx.array(prompt_x).transpose(0, 2, 1)
        
        print(f"  x 转置:")
        print(f"    PyTorch: {x_t_pt.shape}")
        print(f"    MLX:     {x_t_mx.shape}")
        
        print(f"  prompt_x 转置:")
        print(f"    PyTorch: {prompt_x_t_pt.shape}")
        print(f"    MLX:     {prompt_x_t_mx.shape}")
        
        # 比较转置结果
        self._compare_tensors("x_transpose", x_t_pt, x_t_mx)
        self._compare_tensors("prompt_x_transpose", prompt_x_t_pt, prompt_x_t_mx)
    
    def _analyze_concatenation_operations(self, x, prompt_x, mu):
        """分析拼接操作"""
        print(f"\n📊 拼接操作分析:")
        
        # 转置后的数据
        x_t_pt = torch.tensor(x).transpose(1, 2)
        prompt_x_t_pt = torch.tensor(prompt_x).transpose(1, 2)
        mu_pt = torch.tensor(mu)
        
        x_t_mx = mx.array(x).transpose(0, 2, 1)
        prompt_x_t_mx = mx.array(prompt_x).transpose(0, 2, 1)
        mu_mx = mx.array(mu)
        
        # PyTorch 拼接
        x_in_pt = torch.cat([x_t_pt, prompt_x_t_pt, mu_pt], dim=-1)
        
        # MLX 拼接
        x_in_mx = mx.concatenate([x_t_mx, prompt_x_t_mx, mu_mx], -1)
        
        print(f"  拼接结果:")
        print(f"    PyTorch: {x_in_pt.shape}")
        print(f"    MLX:     {x_in_mx.shape}")
        
        # 比较拼接结果
        self._compare_tensors("concatenation", x_in_pt, x_in_mx)
    
    def _analyze_style_conditioning(self, style, batch_size, seq_len):
        """分析风格条件处理"""
        print(f"\n📊 风格条件处理分析:")
        
        # PyTorch 风格条件处理
        style_pt = torch.tensor(style)
        style_broadcast_pt = style_pt.unsqueeze(1).expand(batch_size, seq_len, -1)
        
        # MLX 风格条件处理
        style_mx = mx.array(style)
        style_broadcast_mx = mx.broadcast_to(style_mx.reshape(batch_size, 1, -1), (batch_size, seq_len, style.shape[-1]))
        
        print(f"  风格条件广播:")
        print(f"    PyTorch: {style_broadcast_pt.shape}")
        print(f"    MLX:     {style_broadcast_mx.shape}")
        
        # 比较风格条件处理
        self._compare_tensors("style_broadcast", style_broadcast_pt, style_broadcast_mx)
    
    def _analyze_embedding_layers(self, pytorch_dit, mlx_dit, test_data):
        """分析嵌入层差异"""
        print(f"\n{'='*60}")
        print(f"🔍 DiT 嵌入层分析")
        print(f"{'='*60}")
        
        # 准备输入数据
        x = test_data['x']
        prompt_x = test_data['prompt_x']
        mu = test_data['mu']
        style = test_data['style']
        t = test_data['t']
        x_lens = test_data['x_lens']
        
        # 转换为 PyTorch 格式
        device = next(pytorch_dit.parameters()).device
        x_pt = torch.as_tensor(x).to(device)
        prompt_x_pt = torch.as_tensor(prompt_x).to(device)
        mu_pt = torch.as_tensor(mu).to(device)
        style_pt = torch.as_tensor(style).to(device)
        t_pt = torch.as_tensor(t).to(device)
        x_lens_pt = torch.as_tensor(x_lens).to(device)
        
        # 转换为 MLX 格式
        x_mx = mx.array(x)
        prompt_x_mx = mx.array(prompt_x)
        mu_mx = mx.array(mu)
        style_mx = mx.array(style)
        t_mx = mx.array(t)
        x_lens_mx = mx.array(x_lens)
        
        # 分析 x_embedder 的输入形状问题
        self._analyze_x_embedder_input_shape(pytorch_dit, mlx_dit, x_pt, x_mx)
        
        # 分析 t_embedder
        self._analyze_t_embedder_detailed(pytorch_dit, mlx_dit, t_pt, t_mx)
        
        # 分析 cond_projection
        self._analyze_cond_projection_detailed(pytorch_dit, mlx_dit, mu_pt, mu_mx)
        
        # 分析 cond_embedder 的输入问题
        self._analyze_cond_embedder_input_issue(pytorch_dit, mlx_dit, mu_pt, mu_mx)
    
    def _analyze_x_embedder_input_shape(self, pytorch_dit, mlx_dit, x_pt, x_mx):
        """分析 x_embedder 输入形状问题"""
        print(f"\n📊 x_embedder 输入形状分析:")
        
        print(f"  x 输入形状: {x_pt.shape}")
        print(f"  x_embedder 期望输入形状: {pytorch_dit.x_embedder.in_features}")
        
        # 检查是否需要转置
        if x_pt.shape[1] != pytorch_dit.x_embedder.in_features:
            print(f"  ❌ 输入形状不匹配: 需要转置")
            print(f"    当前: {x_pt.shape}")
            print(f"    期望: (B, {pytorch_dit.x_embedder.in_features}, T)")
            
            # 尝试转置
            x_t_pt = x_pt.transpose(1, 2)  # (B, C, T) -> (B, T, C)
            print(f"    转置后: {x_t_pt.shape}")
            
            if x_t_pt.shape[2] == pytorch_dit.x_embedder.in_features:
                print(f"    ✅ 转置后形状匹配")
                
                # 测试转置后的 x_embedder
                try:
                    with torch.no_grad():
                        x_embed_pt = pytorch_dit.x_embedder(x_t_pt)
                    print(f"    PyTorch x_embedder 输出: {x_embed_pt.shape}")
                except Exception as e:
                    print(f"    ❌ PyTorch x_embedder 失败: {e}")
        else:
            print(f"  ✅ 输入形状匹配")
    
    def _analyze_t_embedder_detailed(self, pytorch_dit, mlx_dit, t_pt, t_mx):
        """详细分析 t_embedder"""
        print(f"\n📊 t_embedder 详细分析:")
        
        try:
            # PyTorch t_embedder
            with torch.no_grad():
                t_embed_pt = pytorch_dit.t_embedder(t_pt)
            
            # MLX t_embedder
            t_embed_mx = mlx_dit.t_embedder(t_mx)
            
            print(f"  t_embedder 输出:")
            print(f"    PyTorch: {t_embed_pt.shape}")
            print(f"    MLX:     {t_embed_mx.shape}")
            
            # 比较结果
            self._compare_tensors("t_embedder", t_embed_pt, t_embed_mx)
            
        except Exception as e:
            print(f"  ❌ t_embedder 分析失败: {e}")
    
    def _analyze_cond_projection_detailed(self, pytorch_dit, mlx_dit, mu_pt, mu_mx):
        """详细分析 cond_projection"""
        print(f"\n📊 cond_projection 详细分析:")
        
        try:
            # PyTorch cond_projection
            with torch.no_grad():
                cond_proj_pt = pytorch_dit.cond_projection(mu_pt)
            
            # MLX cond_projection
            cond_proj_mx = mlx_dit.cond_projection(mu_mx)
            
            print(f"  cond_projection 输出:")
            print(f"    PyTorch: {cond_proj_pt.shape}")
            print(f"    MLX:     {cond_proj_mx.shape}")
            
            # 比较结果
            self._compare_tensors("cond_projection", cond_proj_pt, cond_proj_mx)
            
        except Exception as e:
            print(f"  ❌ cond_projection 分析失败: {e}")
    
    def _analyze_cond_embedder_input_issue(self, pytorch_dit, mlx_dit, mu_pt, mu_mx):
        """分析 cond_embedder 输入问题"""
        print(f"\n📊 cond_embedder 输入问题分析:")
        
        print(f"  mu 输入形状: {mu_pt.shape}")
        print(f"  mu 数据类型: {mu_pt.dtype}")
        
        # 检查 cond_embedder 的实现
        print(f"  cond_embedder 类型: {type(pytorch_dit.cond_embedder)}")
        
        # 检查是否需要整数索引
        if hasattr(pytorch_dit.cond_embedder, 'weight'):
            print(f"  cond_embedder 权重形状: {pytorch_dit.cond_embedder.weight.shape}")
            print(f"  cond_embedder 是嵌入层，需要整数索引")
            
            # 尝试转换为整数
            try:
                mu_int_pt = mu_pt.long()
                print(f"  转换为整数后: {mu_int_pt.shape}, {mu_int_pt.dtype}")
                
                with torch.no_grad():
                    cond_embed_pt = pytorch_dit.cond_embedder(mu_int_pt)
                print(f"  PyTorch cond_embedder 输出: {cond_embed_pt.shape}")
                
            except Exception as e:
                print(f"  ❌ 转换为整数后仍然失败: {e}")
        else:
            print(f"  cond_embedder 不是嵌入层")
    
    def _analyze_transformer_input(self, pytorch_dit, mlx_dit, test_data):
        """分析 Transformer 输入"""
        print(f"\n{'='*60}")
        print(f"🔍 DiT Transformer 输入分析")
        print(f"{'='*60}")
        
        # 准备输入数据
        x = test_data['x']
        prompt_x = test_data['prompt_x']
        mu = test_data['mu']
        style = test_data['style']
        t = test_data['t']
        x_lens = test_data['x_lens']
        
        # 转换为 PyTorch 格式
        device = next(pytorch_dit.parameters()).device
        x_pt = torch.as_tensor(x).to(device)
        prompt_x_pt = torch.as_tensor(prompt_x).to(device)
        mu_pt = torch.as_tensor(mu).to(device)
        style_pt = torch.as_tensor(style).to(device)
        t_pt = torch.as_tensor(t).to(device)
        x_lens_pt = torch.as_tensor(x_lens).to(device)
        
        # 转换为 MLX 格式
        x_mx = mx.array(x)
        prompt_x_mx = mx.array(prompt_x)
        mu_mx = mx.array(mu)
        style_mx = mx.array(style)
        t_mx = mx.array(t)
        x_lens_mx = mx.array(x_lens)
        
        # 模拟 DiT 的完整前向传播过程
        self._simulate_dit_forward_process(pytorch_dit, mlx_dit, x_pt, prompt_x_pt, x_lens_pt, t_pt, style_pt, mu_pt, x_mx, prompt_x_mx, x_lens_mx, t_mx, style_mx, mu_mx)
    
    def _simulate_dit_forward_process(self, pytorch_dit, mlx_dit, x_pt, prompt_x_pt, x_lens_pt, t_pt, style_pt, mu_pt, x_mx, prompt_x_mx, x_lens_mx, t_mx, style_mx, mu_mx):
        """模拟 DiT 的完整前向传播过程"""
        print(f"\n📊 DiT 完整前向传播过程模拟:")
        
        try:
            # 步骤1: 转置 x 和 prompt_x
            print(f"  步骤1: 转置 x 和 prompt_x")
            x_t_pt = x_pt.transpose(1, 2)  # (B, C, T) -> (B, T, C)
            prompt_x_t_pt = prompt_x_pt.transpose(1, 2)
            x_t_mx = x_mx.transpose(0, 2, 1)  # (B, C, T) -> (B, T, C)
            prompt_x_t_mx = prompt_x_mx.transpose(0, 2, 1)
            
            print(f"    x_t: PyTorch={x_t_pt.shape}, MLX={x_t_mx.shape}")
            print(f"    prompt_x_t: PyTorch={prompt_x_t_pt.shape}, MLX={prompt_x_t_mx.shape}")
            
            # 步骤2: 拼接输入
            print(f"  步骤2: 拼接输入")
            x_in_pt = torch.cat([x_t_pt, prompt_x_t_pt, mu_pt], dim=-1)
            x_in_mx = mx.concatenate([x_t_mx, prompt_x_t_mx, mu_mx], -1)
            
            print(f"    x_in: PyTorch={x_in_pt.shape}, MLX={x_in_mx.shape}")
            
            # 步骤3: 添加风格条件
            print(f"  步骤3: 添加风格条件")
            B, T, _ = x_in_pt.shape
            style_broadcast_pt = style_pt.unsqueeze(1).expand(B, T, -1)
            style_broadcast_mx = mx.broadcast_to(style_mx.reshape(B, 1, -1), (B, T, style_pt.shape[-1]))
            
            x_in_pt = torch.cat([x_in_pt, style_broadcast_pt], dim=-1)
            x_in_mx = mx.concatenate([x_in_mx, style_broadcast_mx], -1)
            
            print(f"    x_in (with style): PyTorch={x_in_pt.shape}, MLX={x_in_mx.shape}")
            
            # 步骤4: 通过 cond_x_merge_linear
            print(f"  步骤4: 通过 cond_x_merge_linear")
            print(f"    cond_x_merge_linear 输入形状: {x_in_pt.shape}")
            print(f"    cond_x_merge_linear 权重形状: {pytorch_dit.cond_x_merge_linear.weight.shape}")
            
            if x_in_pt.shape[-1] != pytorch_dit.cond_x_merge_linear.in_features:
                print(f"    ❌ 输入特征维度不匹配:")
                print(f"      输入: {x_in_pt.shape[-1]}")
                print(f"      期望: {pytorch_dit.cond_x_merge_linear.in_features}")
            else:
                print(f"    ✅ 输入特征维度匹配")
                
                # 测试 cond_x_merge_linear
                try:
                    with torch.no_grad():
                        merged_pt = pytorch_dit.cond_x_merge_linear(x_in_pt)
                    print(f"    PyTorch cond_x_merge_linear 输出: {merged_pt.shape}")
                except Exception as e:
                    print(f"    ❌ PyTorch cond_x_merge_linear 失败: {e}")
                
                try:
                    merged_mx = mlx_dit.cond_x_merge_linear(x_in_mx)
                    print(f"    MLX cond_x_merge_linear 输出: {merged_mx.shape}")
                except Exception as e:
                    print(f"    ❌ MLX cond_x_merge_linear 失败: {e}")
            
        except Exception as e:
            print(f"  ❌ 模拟 DiT 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _compare_tensors(self, name: str, pytorch_tensor, mlx_tensor):
        """比较两个张量"""
        try:
            # 转换为 numpy
            pytorch_np = pytorch_tensor.detach().cpu().numpy()
            mlx_np = np.array(mlx_tensor)
            
            print(f"    {name}:")
            print(f"      PyTorch: shape={pytorch_np.shape}, min={np.min(pytorch_np):.6f}, max={np.max(pytorch_np):.6f}, avg={np.mean(pytorch_np):.6f}")
            print(f"      MLX:     shape={mlx_np.shape}, min={np.min(mlx_np):.6f}, max={np.max(mlx_np):.6f}, avg={np.mean(mlx_np):.6f}")
            
            if pytorch_np.shape == mlx_np.shape:
                diff = np.abs(pytorch_np - mlx_np)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                print(f"      差异: max={max_diff:.6f}, mean={mean_diff:.6f}")
                
                if max_diff < 1e-6:
                    print(f"      ✅ {name} 完全一致")
                elif max_diff < 0.1:
                    print(f"      ⚠️  {name} 存在小幅差异")
                else:
                    print(f"      ❌ {name} 存在显著差异")
            else:
                print(f"      ❌ {name} 形状不匹配")
                
        except Exception as e:
            print(f"      ❌ {name} 比较失败: {e}")

def main():
    """主函数"""
    analyzer = DiTDataflowAnalyzer()
    analyzer.analyze_dit_dataflow()

if __name__ == "__main__":
    main()







