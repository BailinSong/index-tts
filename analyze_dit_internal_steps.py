#!/usr/bin/env python3
"""
DiT 内部步骤分析工具
逐步分析 PyTorch 和 MLX DiT 内部的关键计算步骤，找出差异根源
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

class DiTInternalStepsAnalyzer:
    def __init__(self, cache_dir: str = "cfm_production_cache"):
        self.cache_dir = cache_dir
        
    def analyze_dit_internal_steps(self):
        """分析 DiT 内部步骤差异"""
        print("🔍 DiT 内部步骤分析工具")
        print("="*60)
        
        # 1. 加载生产环境模型
        pytorch_dit, mlx_dit = self._load_production_models()
        if pytorch_dit is None or mlx_dit is None:
            return
        
        # 2. 准备测试数据
        test_data = self._prepare_test_data()
        if test_data is None:
            return
        
        # 3. 逐步分析 DiT 内部计算
        self._analyze_dit_embedders(pytorch_dit, mlx_dit, test_data)
        self._analyze_dit_transformer_layers(pytorch_dit, mlx_dit, test_data)
        self._analyze_dit_final_layers(pytorch_dit, mlx_dit, test_data)
        
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
    
    def _analyze_dit_embedders(self, pytorch_dit, mlx_dit, test_data):
        """分析 DiT 嵌入层"""
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
        
        # 分析 x_embedder
        self._analyze_x_embedder(pytorch_dit, mlx_dit, x_pt, x_mx)
        
        # 分析 t_embedder
        self._analyze_t_embedder(pytorch_dit, mlx_dit, t_pt, t_mx)
        
        # 分析 cond_projection
        self._analyze_cond_projection(pytorch_dit, mlx_dit, mu_pt, mu_mx)
        
        # 分析 cond_embedder
        self._analyze_cond_embedder(pytorch_dit, mlx_dit, mu_pt, mu_mx)
        
        # 分析 cond_x_merge_linear
        self._analyze_cond_x_merge_linear(pytorch_dit, mlx_dit, x_pt, prompt_x_pt, mu_pt, x_mx, prompt_x_mx, mu_mx)
    
    def _analyze_x_embedder(self, pytorch_dit, mlx_dit, x_pt, x_mx):
        """分析 x_embedder"""
        print(f"\n📊 x_embedder 分析:")
        
        try:
            # PyTorch x_embedder
            with torch.no_grad():
                x_embed_pt = pytorch_dit.x_embedder(x_pt)
            
            # MLX x_embedder
            x_embed_mx = mlx_dit.x_embedder(x_mx)
            
            # 比较结果
            self._compare_tensors("x_embedder", x_embed_pt, x_embed_mx)
            
        except Exception as e:
            print(f"   ❌ x_embedder 分析失败: {e}")
    
    def _analyze_t_embedder(self, pytorch_dit, mlx_dit, t_pt, t_mx):
        """分析 t_embedder"""
        print(f"\n📊 t_embedder 分析:")
        
        try:
            # PyTorch t_embedder
            with torch.no_grad():
                t_embed_pt = pytorch_dit.t_embedder(t_pt)
            
            # MLX t_embedder
            t_embed_mx = mlx_dit.t_embedder(t_mx)
            
            # 比较结果
            self._compare_tensors("t_embedder", t_embed_pt, t_embed_mx)
            
        except Exception as e:
            print(f"   ❌ t_embedder 分析失败: {e}")
    
    def _analyze_cond_projection(self, pytorch_dit, mlx_dit, mu_pt, mu_mx):
        """分析 cond_projection"""
        print(f"\n📊 cond_projection 分析:")
        
        try:
            # PyTorch cond_projection
            with torch.no_grad():
                cond_proj_pt = pytorch_dit.cond_projection(mu_pt)
            
            # MLX cond_projection
            cond_proj_mx = mlx_dit.cond_projection(mu_mx)
            
            # 比较结果
            self._compare_tensors("cond_projection", cond_proj_pt, cond_proj_mx)
            
        except Exception as e:
            print(f"   ❌ cond_projection 分析失败: {e}")
    
    def _analyze_cond_embedder(self, pytorch_dit, mlx_dit, mu_pt, mu_mx):
        """分析 cond_embedder"""
        print(f"\n📊 cond_embedder 分析:")
        
        try:
            # PyTorch cond_embedder
            with torch.no_grad():
                cond_embed_pt = pytorch_dit.cond_embedder(mu_pt)
            
            # MLX cond_embedder
            cond_embed_mx = mlx_dit.cond_embedder(mu_mx)
            
            # 比较结果
            self._compare_tensors("cond_embedder", cond_embed_pt, cond_embed_mx)
            
        except Exception as e:
            print(f"   ❌ cond_embedder 分析失败: {e}")
    
    def _analyze_cond_x_merge_linear(self, pytorch_dit, mlx_dit, x_pt, prompt_x_pt, mu_pt, x_mx, prompt_x_mx, mu_mx):
        """分析 cond_x_merge_linear"""
        print(f"\n📊 cond_x_merge_linear 分析:")
        
        try:
            # 准备输入：转置 x 和 prompt_x
            x_t_pt = x_pt.transpose(1, 2)  # (B, C, T) -> (B, T, C)
            prompt_x_t_pt = prompt_x_pt.transpose(1, 2)
            x_t_mx = x_mx.transpose(0, 2, 1)  # (B, C, T) -> (B, T, C)
            prompt_x_t_mx = prompt_x_mx.transpose(0, 2, 1)
            
            # 拼接输入
            x_in_pt = torch.cat([x_t_pt, prompt_x_t_pt, mu_pt], dim=-1)
            x_in_mx = mx.concatenate([x_t_mx, prompt_x_t_mx, mu_mx], -1)
            
            # PyTorch cond_x_merge_linear
            with torch.no_grad():
                merged_pt = pytorch_dit.cond_x_merge_linear(x_in_pt)
            
            # MLX cond_x_merge_linear
            merged_mx = mlx_dit.cond_x_merge_linear(x_in_mx)
            
            # 比较结果
            self._compare_tensors("cond_x_merge_linear", merged_pt, merged_mx)
            
        except Exception as e:
            print(f"   ❌ cond_x_merge_linear 分析失败: {e}")
    
    def _analyze_dit_transformer_layers(self, pytorch_dit, mlx_dit, test_data):
        """分析 DiT Transformer 层"""
        print(f"\n{'='*60}")
        print(f"🔍 DiT Transformer 层分析")
        print(f"{'='*60}")
        
        # 准备完整的输入
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
        
        # 分析前几层 Transformer
        for i in range(min(3, len(pytorch_dit.transformer.layers))):
            print(f"\n📊 Transformer Layer {i} 分析:")
            
            try:
                # 准备输入到 Transformer 层
                # 这里需要模拟 DiT 的完整前向传播到第 i 层
                # 为了简化，我们直接调用完整的 DiT 并分析中间结果
                self._analyze_transformer_layer(pytorch_dit, mlx_dit, i, x_pt, prompt_x_pt, x_lens_pt, t_pt, style_pt, mu_pt, x_mx, prompt_x_mx, x_lens_mx, t_mx, style_mx, mu_mx)
                
            except Exception as e:
                print(f"   ❌ Transformer Layer {i} 分析失败: {e}")
    
    def _analyze_transformer_layer(self, pytorch_dit, mlx_dit, layer_idx, x_pt, prompt_x_pt, x_lens_pt, t_pt, style_pt, mu_pt, x_mx, prompt_x_mx, x_lens_mx, t_mx, style_mx, mu_mx):
        """分析单个 Transformer 层"""
        print(f"   🔍 分析 Layer {layer_idx} 的注意力机制和前馈网络...")
        
        # 这里需要更深入的实现来获取中间结果
        # 由于 DiT 的 forward 方法可能没有暴露中间结果，我们需要修改模型或使用 hook
        print(f"   ⚠️  需要实现中间结果提取机制")
    
    def _analyze_dit_final_layers(self, pytorch_dit, mlx_dit, test_data):
        """分析 DiT 最终层"""
        print(f"\n{'='*60}")
        print(f"🔍 DiT 最终层分析")
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
        
        # 分析 final_layer
        self._analyze_final_layer(pytorch_dit, mlx_dit, x_pt, prompt_x_pt, x_lens_pt, t_pt, style_pt, mu_pt, x_mx, prompt_x_mx, x_lens_mx, t_mx, style_mx, mu_mx)
    
    def _analyze_final_layer(self, pytorch_dit, mlx_dit, x_pt, prompt_x_pt, x_lens_pt, t_pt, style_pt, mu_pt, x_mx, prompt_x_mx, x_lens_mx, t_mx, style_mx, mu_mx):
        """分析 final_layer"""
        print(f"\n📊 final_layer 分析:")
        
        try:
            # 这里需要模拟 DiT 的完整前向传播到 final_layer
            # 由于 final_layer 的输入依赖于前面的计算，我们需要完整的 forward 过程
            print(f"   ⚠️  需要实现完整的 DiT forward 过程分析")
            
        except Exception as e:
            print(f"   ❌ final_layer 分析失败: {e}")
    
    def _compare_tensors(self, name: str, pytorch_tensor, mlx_tensor):
        """比较两个张量"""
        try:
            # 转换为 numpy
            pytorch_np = pytorch_tensor.detach().cpu().numpy()
            mlx_np = np.array(mlx_tensor)
            
            print(f"   {name}:")
            print(f"     PyTorch: shape={pytorch_np.shape}, min={np.min(pytorch_np):.6f}, max={np.max(pytorch_np):.6f}, avg={np.mean(pytorch_np):.6f}")
            print(f"     MLX:     shape={mlx_np.shape}, min={np.min(mlx_np):.6f}, max={np.max(mlx_np):.6f}, avg={np.mean(mlx_np):.6f}")
            
            if pytorch_np.shape == mlx_np.shape:
                diff = np.abs(pytorch_np - mlx_np)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                print(f"     差异: max={max_diff:.6f}, mean={mean_diff:.6f}")
                
                if max_diff < 1e-6:
                    print(f"     ✅ {name} 完全一致")
                elif max_diff < 0.1:
                    print(f"     ⚠️  {name} 存在小幅差异")
                else:
                    print(f"     ❌ {name} 存在显著差异")
            else:
                print(f"     ❌ {name} 形状不匹配")
                
        except Exception as e:
            print(f"     ❌ {name} 比较失败: {e}")

def main():
    """主函数"""
    analyzer = DiTInternalStepsAnalyzer()
    analyzer.analyze_dit_internal_steps()

if __name__ == "__main__":
    main()







