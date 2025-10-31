#!/usr/bin/env python3
"""
Transformer 层差异分析工具
深入分析 PyTorch 和 MLX DiT 中 Transformer 层的实现差异
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

class TransformerLayersAnalyzer:
    def __init__(self, cache_dir: str = "cfm_production_cache"):
        self.cache_dir = cache_dir
        
    def analyze_transformer_layers(self):
        """分析 Transformer 层差异"""
        print("🔍 Transformer 层差异分析工具")
        print("="*60)
        
        # 1. 加载生产环境模型
        pytorch_dit, mlx_dit = self._load_production_models()
        if pytorch_dit is None or mlx_dit is None:
            return
        
        # 2. 准备测试数据
        test_data = self._prepare_test_data()
        if test_data is None:
            return
        
        # 3. 分析 Transformer 层差异
        self._analyze_transformer_structure(pytorch_dit, mlx_dit)
        self._analyze_transformer_weights(pytorch_dit, mlx_dit)
        self._analyze_transformer_forward(pytorch_dit, mlx_dit, test_data)
        
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
    
    def _analyze_transformer_structure(self, pytorch_dit, mlx_dit):
        """分析 Transformer 结构差异"""
        print(f"\n{'='*60}")
        print(f"🔍 Transformer 结构分析")
        print(f"{'='*60}")
        
        # PyTorch Transformer
        pytorch_transformer = pytorch_dit.transformer
        print(f"PyTorch Transformer:")
        print(f"  类型: {type(pytorch_transformer)}")
        print(f"  层数: {len(pytorch_transformer.layers)}")
        
        # 分析第一层
        if len(pytorch_transformer.layers) > 0:
            layer0 = pytorch_transformer.layers[0]
            print(f"  第一层类型: {type(layer0)}")
            print(f"  第一层属性: {[attr for attr in dir(layer0) if not attr.startswith('_')]}")
        
        # MLX Transformer
        mlx_transformer = mlx_dit.transformer
        print(f"\nMLX Transformer:")
        print(f"  类型: {type(mlx_transformer)}")
        print(f"  层数: {len(mlx_transformer.layers)}")
        
        # 分析第一层
        if len(mlx_transformer.layers) > 0:
            layer0 = mlx_transformer.layers[0]
            print(f"  第一层类型: {type(layer0)}")
            print(f"  第一层属性: {[attr for attr in dir(layer0) if not attr.startswith('_')]}")
    
    def _analyze_transformer_weights(self, pytorch_dit, mlx_dit):
        """分析 Transformer 权重差异"""
        print(f"\n{'='*60}")
        print(f"🔍 Transformer 权重分析")
        print(f"{'='*60}")
        
        # 分析 PyTorch Transformer 权重
        print(f"PyTorch Transformer 权重:")
        pytorch_weights = {}
        for name, param in pytorch_dit.transformer.named_parameters():
            pytorch_weights[name] = param.detach().cpu().numpy()
            print(f"  {name}: {param.shape}, range=[{param.min().item():.6f}, {param.max().item():.6f}]")
        
        # 分析 MLX Transformer 权重
        print(f"\nMLX Transformer 权重:")
        mlx_weights = {}
        try:
            for name, param in mlx_dit.transformer.named_parameters():
                mlx_weights[name] = np.array(param)
                print(f"  {name}: {param.shape}, range=[{np.min(param):.6f}, {np.max(param):.6f}]")
        except AttributeError:
            print(f"  MLX Transformer 不支持 named_parameters，使用 parameters() 方法")
            for name, param in mlx_dit.transformer.parameters().items():
                mlx_weights[name] = np.array(param)
                print(f"  {name}: {param.shape}, range=[{np.min(param):.6f}, {np.max(param):.6f}]")
        
        # 比较权重
        print(f"\n权重比较:")
        common_keys = set(pytorch_weights.keys()) & set(mlx_weights.keys())
        print(f"  共同权重数量: {len(common_keys)}")
        
        for key in sorted(common_keys):
            pt_weight = pytorch_weights[key]
            mlx_weight = mlx_weights[key]
            
            if pt_weight.shape == mlx_weight.shape:
                diff = np.abs(pt_weight - mlx_weight)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                
                if max_diff < 1e-6:
                    print(f"  ✅ {key}: 完全一致")
                elif max_diff < 0.1:
                    print(f"  ⚠️  {key}: 小幅差异 (max={max_diff:.6f}, mean={mean_diff:.6f})")
                else:
                    print(f"  ❌ {key}: 显著差异 (max={max_diff:.6f}, mean={mean_diff:.6f})")
            else:
                print(f"  ❌ {key}: 形状不匹配 (PyTorch={pt_weight.shape}, MLX={mlx_weight.shape})")
    
    def _analyze_transformer_forward(self, pytorch_dit, mlx_dit, test_data):
        """分析 Transformer 前向传播差异"""
        print(f"\n{'='*60}")
        print(f"🔍 Transformer 前向传播分析")
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
        
        # 模拟 DiT 的完整前向传播到 Transformer 输入
        print(f"准备 Transformer 输入...")
        
        # PyTorch 路径
        try:
            # 转置
            x_t_pt = x_pt.transpose(1, 2)
            prompt_x_t_pt = prompt_x_pt.transpose(1, 2)
            
            # 条件投影
            cond_proj_pt = pytorch_dit.cond_projection(mu_pt)
            
            # 拼接
            x_in_pt = torch.cat([x_t_pt, prompt_x_t_pt, cond_proj_pt], dim=-1)
            
            # 添加风格条件
            B, T, _ = x_in_pt.shape
            style_broadcast_pt = style_pt.unsqueeze(1).expand(B, T, -1)
            x_in_pt = torch.cat([x_in_pt, style_broadcast_pt], dim=-1)
            
            # 通过 cond_x_merge_linear
            x_in_pt = pytorch_dit.cond_x_merge_linear(x_in_pt)
            
            # 时间嵌入
            t_emb_pt = pytorch_dit.t_embedder(t_pt)
            
            print(f"  PyTorch Transformer 输入: {x_in_pt.shape}")
            print(f"  PyTorch 时间嵌入: {t_emb_pt.shape}")
            
        except Exception as e:
            print(f"  ❌ PyTorch 准备失败: {e}")
            return
        
        # MLX 路径
        try:
            # 转置
            x_t_mx = x_mx.transpose(0, 2, 1)
            prompt_x_t_mx = prompt_x_mx.transpose(0, 2, 1)
            
            # 条件投影
            cond_proj_mx = mlx_dit.cond_projection(mu_mx)
            
            # 拼接
            x_in_mx = mx.concatenate([x_t_mx, prompt_x_t_mx, cond_proj_mx], -1)
            
            # 添加风格条件
            B, T, _ = x_in_mx.shape
            style_broadcast_mx = mx.broadcast_to(style_mx.reshape(B, 1, -1), (B, T, style_mx.shape[-1]))
            x_in_mx = mx.concatenate([x_in_mx, style_broadcast_mx], -1)
            
            # 通过 cond_x_merge_linear
            x_in_mx = mlx_dit.cond_x_merge_linear(x_in_mx)
            
            # 时间嵌入
            t_emb_mx = mlx_dit.t_embedder(t_mx)
            
            print(f"  MLX Transformer 输入: {x_in_mx.shape}")
            print(f"  MLX 时间嵌入: {t_emb_mx.shape}")
            
        except Exception as e:
            print(f"  ❌ MLX 准备失败: {e}")
            return
        
        # 比较 Transformer 输入
        print(f"\nTransformer 输入比较:")
        self._compare_tensors("transformer_input", x_in_pt, x_in_mx)
        self._compare_tensors("timestep_embedding", t_emb_pt, t_emb_mx)
        
        # 分析 Transformer 层
        self._analyze_transformer_layer_by_layer(pytorch_dit, mlx_dit, x_in_pt, t_emb_pt, x_in_mx, t_emb_mx)
    
    def _analyze_transformer_layer_by_layer(self, pytorch_dit, mlx_dit, x_in_pt, t_emb_pt, x_in_mx, t_emb_mx):
        """逐层分析 Transformer"""
        print(f"\n{'='*60}")
        print(f"🔍 Transformer 逐层分析")
        print(f"{'='*60}")
        
        # 分析前几层
        for i in range(min(3, len(pytorch_dit.transformer.layers))):
            print(f"\n📊 Layer {i} 分析:")
            
            try:
                # PyTorch 层
                pytorch_layer = pytorch_dit.transformer.layers[i]
                print(f"  PyTorch Layer {i}: {type(pytorch_layer)}")
                
                # MLX 层
                mlx_layer = mlx_dit.transformer.layers[i]
                print(f"  MLX Layer {i}: {type(mlx_layer)}")
                
                # 比较层结构
                self._compare_layer_structure(pytorch_layer, mlx_layer, i)
                
            except Exception as e:
                print(f"  ❌ Layer {i} 分析失败: {e}")
    
    def _compare_layer_structure(self, pytorch_layer, mlx_layer, layer_idx):
        """比较层结构"""
        print(f"    层结构比较:")
        
        # 比较属性
        pytorch_attrs = [attr for attr in dir(pytorch_layer) if not attr.startswith('_')]
        mlx_attrs = [attr for attr in dir(mlx_layer) if not attr.startswith('_')]
        
        common_attrs = set(pytorch_attrs) & set(mlx_attrs)
        pytorch_only = set(pytorch_attrs) - set(mlx_attrs)
        mlx_only = set(mlx_attrs) - set(pytorch_attrs)
        
        print(f"      共同属性: {len(common_attrs)}")
        print(f"      PyTorch 独有: {len(pytorch_only)}")
        print(f"      MLX 独有: {len(mlx_only)}")
        
        if pytorch_only:
            print(f"      PyTorch 独有属性: {list(pytorch_only)[:5]}...")
        if mlx_only:
            print(f"      MLX 独有属性: {list(mlx_only)[:5]}...")
    
    def _compare_tensors(self, name: str, pytorch_tensor, mlx_tensor):
        """比较两个张量"""
        try:
            # 转换为 numpy
            pytorch_np = pytorch_tensor.detach().cpu().numpy()
            mlx_np = np.array(mlx_tensor)
            
            print(f"  {name}:")
            print(f"    PyTorch: shape={pytorch_np.shape}, min={np.min(pytorch_np):.6f}, max={np.max(pytorch_np):.6f}, avg={np.mean(pytorch_np):.6f}")
            print(f"    MLX:     shape={mlx_np.shape}, min={np.min(mlx_np):.6f}, max={np.max(mlx_np):.6f}, avg={np.mean(mlx_np):.6f}")
            
            if pytorch_np.shape == mlx_np.shape:
                diff = np.abs(pytorch_np - mlx_np)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                print(f"    差异: max={max_diff:.6f}, mean={mean_diff:.6f}")
                
                if max_diff < 1e-6:
                    print(f"    ✅ {name} 完全一致")
                elif max_diff < 0.1:
                    print(f"    ⚠️  {name} 存在小幅差异")
                else:
                    print(f"    ❌ {name} 存在显著差异")
            else:
                print(f"    ❌ {name} 形状不匹配")
                
        except Exception as e:
            print(f"    ❌ {name} 比较失败: {e}")

def main():
    """主函数"""
    analyzer = TransformerLayersAnalyzer()
    analyzer.analyze_transformer_layers()

if __name__ == "__main__":
    main()
