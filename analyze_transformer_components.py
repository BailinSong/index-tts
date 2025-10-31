#!/usr/bin/env python3
"""
Transformer 组件差异分析工具
深入分析 PyTorch 和 MLX DiT 中 Transformer 层的各个组件实现差异
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

class TransformerComponentsAnalyzer:
    def __init__(self, cache_dir: str = "cfm_production_cache"):
        self.cache_dir = cache_dir
        
    def analyze_transformer_components(self):
        """分析 Transformer 组件差异"""
        print("🔍 Transformer 组件差异分析工具")
        print("="*60)
        
        # 1. 加载生产环境模型
        pytorch_dit, mlx_dit = self._load_production_models()
        if pytorch_dit is None or mlx_dit is None:
            return
        
        # 2. 准备测试数据
        test_data = self._prepare_test_data()
        if test_data is None:
            return
        
        # 3. 分析 Transformer 组件差异
        self._analyze_attention_mechanism(pytorch_dit, mlx_dit, test_data)
        self._analyze_feed_forward_network(pytorch_dit, mlx_dit, test_data)
        self._analyze_layer_normalization(pytorch_dit, mlx_dit, test_data)
        self._analyze_residual_connections(pytorch_dit, mlx_dit, test_data)
        
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
    
    def _prepare_transformer_input(self, pytorch_dit, mlx_dit, test_data):
        """准备 Transformer 输入"""
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
        print("🔧 准备 Transformer 输入...")
        
        # PyTorch 路径
        x_t_pt = x_pt.transpose(1, 2)
        prompt_x_t_pt = prompt_x_pt.transpose(1, 2)
        cond_proj_pt = pytorch_dit.cond_projection(mu_pt)
        x_in_pt = torch.cat([x_t_pt, prompt_x_t_pt, cond_proj_pt], dim=-1)
        B, T, _ = x_in_pt.shape
        style_broadcast_pt = style_pt.unsqueeze(1).expand(B, T, -1)
        x_in_pt = torch.cat([x_in_pt, style_broadcast_pt], dim=-1)
        x_in_pt = pytorch_dit.cond_x_merge_linear(x_in_pt)
        t_emb_pt = pytorch_dit.t_embedder(t_pt)
        
        # MLX 路径
        x_t_mx = x_mx.transpose(0, 2, 1)
        prompt_x_t_mx = prompt_x_mx.transpose(0, 2, 1)
        cond_proj_mx = mlx_dit.cond_projection(mu_mx)
        x_in_mx = mx.concatenate([x_t_mx, prompt_x_t_mx, cond_proj_mx], -1)
        B, T, _ = x_in_mx.shape
        style_broadcast_mx = mx.broadcast_to(style_mx.reshape(B, 1, -1), (B, T, style_mx.shape[-1]))
        x_in_mx = mx.concatenate([x_in_mx, style_broadcast_mx], -1)
        x_in_mx = mlx_dit.cond_x_merge_linear(x_in_mx)
        t_emb_mx = mlx_dit.t_embedder(t_mx)
        
        return x_in_pt, t_emb_pt, x_in_mx, t_emb_mx
    
    def _analyze_attention_mechanism(self, pytorch_dit, mlx_dit, test_data):
        """分析注意力机制差异"""
        print(f"\n{'='*60}")
        print(f"🔍 注意力机制分析")
        print(f"{'='*60}")
        
        # 准备 Transformer 输入
        x_in_pt, t_emb_pt, x_in_mx, t_emb_mx = self._prepare_transformer_input(pytorch_dit, mlx_dit, test_data)
        
        # 分析第一层的注意力机制
        print(f"\n📊 Layer 0 注意力机制分析:")
        
        try:
            # 获取第一层
            pytorch_layer = pytorch_dit.transformer.layers[0]
            mlx_layer = mlx_dit.transformer.layers[0]
            
            print(f"  PyTorch Layer 0: {type(pytorch_layer)}")
            print(f"  MLX Layer 0: {type(mlx_layer)}")
            
            # 分析注意力权重
            self._analyze_attention_weights(pytorch_layer, mlx_layer)
            
            # 分析注意力计算
            self._analyze_attention_computation(pytorch_layer, mlx_layer, x_in_pt, t_emb_pt, x_in_mx, t_emb_mx)
            
        except Exception as e:
            print(f"  ❌ 注意力机制分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_attention_weights(self, pytorch_layer, mlx_layer):
        """分析注意力权重"""
        print(f"\n  📊 注意力权重分析:")
        
        try:
            # PyTorch 注意力权重
            pytorch_attn = pytorch_layer.attention
            print(f"    PyTorch 注意力模块: {type(pytorch_attn)}")
            
            # 获取 QKV 权重
            if hasattr(pytorch_attn, 'wqkv'):
                wqkv_pt = pytorch_attn.wqkv.weight
                print(f"    PyTorch wqkv 权重: {wqkv_pt.shape}, range=[{wqkv_pt.min().item():.6f}, {wqkv_pt.max().item():.6f}]")
            
            if hasattr(pytorch_attn, 'wo'):
                wo_pt = pytorch_attn.wo.weight
                print(f"    PyTorch wo 权重: {wo_pt.shape}, range=[{wo_pt.min().item():.6f}, {wo_pt.max().item():.6f}]")
            
            # MLX 注意力权重
            mlx_attn = mlx_layer.attention
            print(f"    MLX 注意力模块: {type(mlx_attn)}")
            
            # 获取 MLX 权重
            if hasattr(mlx_attn, 'wqkv'):
                wqkv_mx = mlx_attn.wqkv.weight
                wqkv_np = np.array(wqkv_mx)
                print(f"    MLX wqkv 权重: {wqkv_mx.shape}, range=[{np.min(wqkv_np):.6f}, {np.max(wqkv_np):.6f}]")
            
            if hasattr(mlx_attn, 'wo'):
                wo_mx = mlx_attn.wo.weight
                wo_np = np.array(wo_mx)
                print(f"    MLX wo 权重: {wo_mx.shape}, range=[{np.min(wo_np):.6f}, {np.max(wo_np):.6f}]")
            
            # 比较权重
            if hasattr(pytorch_attn, 'wqkv') and hasattr(mlx_attn, 'wqkv'):
                self._compare_tensors("wqkv_weights", wqkv_pt, wqkv_mx)
            
            if hasattr(pytorch_attn, 'wo') and hasattr(mlx_attn, 'wo'):
                self._compare_tensors("wo_weights", wo_pt, wo_mx)
                
        except Exception as e:
            print(f"    ❌ 注意力权重分析失败: {e}")
    
    def _analyze_attention_computation(self, pytorch_layer, mlx_layer, x_in_pt, t_emb_pt, x_in_mx, t_emb_mx):
        """分析注意力计算"""
        print(f"\n  📊 注意力计算分析:")
        
        try:
            # 分析注意力计算过程
            print(f"    分析注意力计算过程...")
            
            # 这里需要更深入的实现来获取中间结果
            # 由于注意力计算可能没有暴露中间结果，我们需要修改模型或使用 hook
            print(f"    ⚠️  需要实现注意力计算中间结果提取机制")
            
        except Exception as e:
            print(f"    ❌ 注意力计算分析失败: {e}")
    
    def _analyze_feed_forward_network(self, pytorch_dit, mlx_dit, test_data):
        """分析前馈网络差异"""
        print(f"\n{'='*60}")
        print(f"🔍 前馈网络分析")
        print(f"{'='*60}")
        
        # 准备 Transformer 输入
        x_in_pt, t_emb_pt, x_in_mx, t_emb_mx = self._prepare_transformer_input(pytorch_dit, mlx_dit, test_data)
        
        # 分析第一层的前馈网络
        print(f"\n📊 Layer 0 前馈网络分析:")
        
        try:
            # 获取第一层
            pytorch_layer = pytorch_dit.transformer.layers[0]
            mlx_layer = mlx_dit.transformer.layers[0]
            
            # 分析前馈网络权重
            self._analyze_feed_forward_weights(pytorch_layer, mlx_layer)
            
        except Exception as e:
            print(f"  ❌ 前馈网络分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_feed_forward_weights(self, pytorch_layer, mlx_layer):
        """分析前馈网络权重"""
        print(f"\n  📊 前馈网络权重分析:")
        
        try:
            # PyTorch 前馈网络权重
            pytorch_ffn = pytorch_layer.feed_forward
            print(f"    PyTorch 前馈网络: {type(pytorch_ffn)}")
            
            # 获取前馈网络权重
            if hasattr(pytorch_ffn, 'w1'):
                w1_pt = pytorch_ffn.w1.weight
                print(f"    PyTorch w1 权重: {w1_pt.shape}, range=[{w1_pt.min().item():.6f}, {w1_pt.max().item():.6f}]")
            
            if hasattr(pytorch_ffn, 'w2'):
                w2_pt = pytorch_ffn.w2.weight
                print(f"    PyTorch w2 权重: {w2_pt.shape}, range=[{w2_pt.min().item():.6f}, {w2_pt.max().item():.6f}]")
            
            if hasattr(pytorch_ffn, 'w3'):
                w3_pt = pytorch_ffn.w3.weight
                print(f"    PyTorch w3 权重: {w3_pt.shape}, range=[{w3_pt.min().item():.6f}, {w3_pt.max().item():.6f}]")
            
            # MLX 前馈网络权重
            mlx_ffn = mlx_layer.feed_forward
            print(f"    MLX 前馈网络: {type(mlx_ffn)}")
            
            # 获取 MLX 权重
            if hasattr(mlx_ffn, 'w1'):
                w1_mx = mlx_ffn.w1.weight
                w1_np = np.array(w1_mx)
                print(f"    MLX w1 权重: {w1_mx.shape}, range=[{np.min(w1_np):.6f}, {np.max(w1_np):.6f}]")
            
            if hasattr(mlx_ffn, 'w2'):
                w2_mx = mlx_ffn.w2.weight
                w2_np = np.array(w2_mx)
                print(f"    MLX w2 权重: {w2_mx.shape}, range=[{np.min(w2_np):.6f}, {np.max(w2_np):.6f}]")
            
            if hasattr(mlx_ffn, 'w3'):
                w3_mx = mlx_ffn.w3.weight
                w3_np = np.array(w3_mx)
                print(f"    MLX w3 权重: {w3_mx.shape}, range=[{np.min(w3_np):.6f}, {np.max(w3_np):.6f}]")
            
            # 比较权重
            if hasattr(pytorch_ffn, 'w1') and hasattr(mlx_ffn, 'w1'):
                self._compare_tensors("w1_weights", w1_pt, w1_mx)
            
            if hasattr(pytorch_ffn, 'w2') and hasattr(mlx_ffn, 'w2'):
                self._compare_tensors("w2_weights", w2_pt, w2_mx)
            
            if hasattr(pytorch_ffn, 'w3') and hasattr(mlx_ffn, 'w3'):
                self._compare_tensors("w3_weights", w3_pt, w3_mx)
                
        except Exception as e:
            print(f"    ❌ 前馈网络权重分析失败: {e}")
    
    def _analyze_layer_normalization(self, pytorch_dit, mlx_dit, test_data):
        """分析层归一化差异"""
        print(f"\n{'='*60}")
        print(f"🔍 层归一化分析")
        print(f"{'='*60}")
        
        # 准备 Transformer 输入
        x_in_pt, t_emb_pt, x_in_mx, t_emb_mx = self._prepare_transformer_input(pytorch_dit, mlx_dit, test_data)
        
        # 分析第一层的层归一化
        print(f"\n📊 Layer 0 层归一化分析:")
        
        try:
            # 获取第一层
            pytorch_layer = pytorch_dit.transformer.layers[0]
            mlx_layer = mlx_dit.transformer.layers[0]
            
            # 分析注意力归一化
            self._analyze_attention_norm(pytorch_layer, mlx_layer)
            
            # 分析前馈网络归一化
            self._analyze_ffn_norm(pytorch_layer, mlx_layer)
            
        except Exception as e:
            print(f"  ❌ 层归一化分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_attention_norm(self, pytorch_layer, mlx_layer):
        """分析注意力归一化"""
        print(f"\n  📊 注意力归一化分析:")
        
        try:
            # PyTorch 注意力归一化
            pytorch_attn_norm = pytorch_layer.attention_norm
            print(f"    PyTorch 注意力归一化: {type(pytorch_attn_norm)}")
            
            # 获取归一化权重
            if hasattr(pytorch_attn_norm, 'project_layer'):
                proj_pt = pytorch_attn_norm.project_layer.weight
                print(f"    PyTorch 投影层权重: {proj_pt.shape}, range=[{proj_pt.min().item():.6f}, {proj_pt.max().item():.6f}]")
            
            if hasattr(pytorch_attn_norm, 'norm'):
                norm_pt = pytorch_attn_norm.norm.weight
                print(f"    PyTorch 归一化权重: {norm_pt.shape}, range=[{norm_pt.min().item():.6f}, {norm_pt.max().item():.6f}]")
            
            # MLX 注意力归一化
            mlx_attn_norm = mlx_layer.attention_norm
            print(f"    MLX 注意力归一化: {type(mlx_attn_norm)}")
            
            # 获取 MLX 权重
            if hasattr(mlx_attn_norm, 'project_layer'):
                proj_mx = mlx_attn_norm.project_layer.weight
                proj_np = np.array(proj_mx)
                print(f"    MLX 投影层权重: {proj_mx.shape}, range=[{np.min(proj_np):.6f}, {np.max(proj_np):.6f}]")
            
            if hasattr(mlx_attn_norm, 'norm'):
                norm_mx = mlx_attn_norm.norm.weight
                norm_np = np.array(norm_mx)
                print(f"    MLX 归一化权重: {norm_mx.shape}, range=[{np.min(norm_np):.6f}, {np.max(norm_np):.6f}]")
            
            # 比较权重
            if hasattr(pytorch_attn_norm, 'project_layer') and hasattr(mlx_attn_norm, 'project_layer'):
                self._compare_tensors("attn_norm_proj_weights", proj_pt, proj_mx)
            
            if hasattr(pytorch_attn_norm, 'norm') and hasattr(mlx_attn_norm, 'norm'):
                self._compare_tensors("attn_norm_weights", norm_pt, norm_mx)
                
        except Exception as e:
            print(f"    ❌ 注意力归一化分析失败: {e}")
    
    def _analyze_ffn_norm(self, pytorch_layer, mlx_layer):
        """分析前馈网络归一化"""
        print(f"\n  📊 前馈网络归一化分析:")
        
        try:
            # PyTorch 前馈网络归一化
            pytorch_ffn_norm = pytorch_layer.ffn_norm
            print(f"    PyTorch 前馈网络归一化: {type(pytorch_ffn_norm)}")
            
            # 获取归一化权重
            if hasattr(pytorch_ffn_norm, 'project_layer'):
                proj_pt = pytorch_ffn_norm.project_layer.weight
                print(f"    PyTorch 投影层权重: {proj_pt.shape}, range=[{proj_pt.min().item():.6f}, {proj_pt.max().item():.6f}]")
            
            if hasattr(pytorch_ffn_norm, 'norm'):
                norm_pt = pytorch_ffn_norm.norm.weight
                print(f"    PyTorch 归一化权重: {norm_pt.shape}, range=[{norm_pt.min().item():.6f}, {norm_pt.max().item():.6f}]")
            
            # MLX 前馈网络归一化
            mlx_ffn_norm = mlx_layer.ffn_norm
            print(f"    MLX 前馈网络归一化: {type(mlx_ffn_norm)}")
            
            # 获取 MLX 权重
            if hasattr(mlx_ffn_norm, 'project_layer'):
                proj_mx = mlx_ffn_norm.project_layer.weight
                proj_np = np.array(proj_mx)
                print(f"    MLX 投影层权重: {proj_mx.shape}, range=[{np.min(proj_np):.6f}, {np.max(proj_np):.6f}]")
            
            if hasattr(mlx_ffn_norm, 'norm'):
                norm_mx = mlx_ffn_norm.norm.weight
                norm_np = np.array(norm_mx)
                print(f"    MLX 归一化权重: {norm_mx.shape}, range=[{np.min(norm_np):.6f}, {np.max(norm_np):.6f}]")
            
            # 比较权重
            if hasattr(pytorch_ffn_norm, 'project_layer') and hasattr(mlx_ffn_norm, 'project_layer'):
                self._compare_tensors("ffn_norm_proj_weights", proj_pt, proj_mx)
            
            if hasattr(pytorch_ffn_norm, 'norm') and hasattr(mlx_ffn_norm, 'norm'):
                self._compare_tensors("ffn_norm_weights", norm_pt, norm_mx)
                
        except Exception as e:
            print(f"    ❌ 前馈网络归一化分析失败: {e}")
    
    def _analyze_residual_connections(self, pytorch_dit, mlx_dit, test_data):
        """分析残差连接差异"""
        print(f"\n{'='*60}")
        print(f"🔍 残差连接分析")
        print(f"{'='*60}")
        
        # 准备 Transformer 输入
        x_in_pt, t_emb_pt, x_in_mx, t_emb_mx = self._prepare_transformer_input(pytorch_dit, mlx_dit, test_data)
        
        # 分析第一层的残差连接
        print(f"\n📊 Layer 0 残差连接分析:")
        
        try:
            # 获取第一层
            pytorch_layer = pytorch_dit.transformer.layers[0]
            mlx_layer = mlx_dit.transformer.layers[0]
            
            # 分析 skip connection
            self._analyze_skip_connection(pytorch_layer, mlx_layer)
            
        except Exception as e:
            print(f"  ❌ 残差连接分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_skip_connection(self, pytorch_layer, mlx_layer):
        """分析 skip connection"""
        print(f"\n  📊 Skip Connection 分析:")
        
        try:
            # PyTorch skip connection
            if hasattr(pytorch_layer, 'skip_in_linear'):
                pytorch_skip = pytorch_layer.skip_in_linear
                print(f"    PyTorch skip connection: {type(pytorch_skip)}")
                
                if hasattr(pytorch_skip, 'weight'):
                    skip_weight_pt = pytorch_skip.weight
                    print(f"    PyTorch skip 权重: {skip_weight_pt.shape}, range=[{skip_weight_pt.min().item():.6f}, {skip_weight_pt.max().item():.6f}]")
                
                if hasattr(pytorch_skip, 'bias'):
                    skip_bias_pt = pytorch_skip.bias
                    print(f"    PyTorch skip 偏置: {skip_bias_pt.shape}, range=[{skip_bias_pt.min().item():.6f}, {skip_bias_pt.max().item():.6f}]")
            
            # MLX skip connection
            if hasattr(mlx_layer, 'skip_in_linear'):
                mlx_skip = mlx_layer.skip_in_linear
                print(f"    MLX skip connection: {type(mlx_skip)}")
                
                if hasattr(mlx_skip, 'weight'):
                    skip_weight_mx = mlx_skip.weight
                    skip_weight_np = np.array(skip_weight_mx)
                    print(f"    MLX skip 权重: {skip_weight_mx.shape}, range=[{np.min(skip_weight_np):.6f}, {np.max(skip_weight_np):.6f}]")
                
                if hasattr(mlx_skip, 'bias'):
                    skip_bias_mx = mlx_skip.bias
                    skip_bias_np = np.array(skip_bias_mx)
                    print(f"    MLX skip 偏置: {skip_bias_mx.shape}, range=[{np.min(skip_bias_np):.6f}, {np.max(skip_bias_np):.6f}]")
            
            # 比较权重
            if (hasattr(pytorch_layer, 'skip_in_linear') and hasattr(mlx_layer, 'skip_in_linear') and
                hasattr(pytorch_layer.skip_in_linear, 'weight') and hasattr(mlx_layer.skip_in_linear, 'weight')):
                self._compare_tensors("skip_weights", pytorch_layer.skip_in_linear.weight, mlx_layer.skip_in_linear.weight)
            
            if (hasattr(pytorch_layer, 'skip_in_linear') and hasattr(mlx_layer, 'skip_in_linear') and
                hasattr(pytorch_layer.skip_in_linear, 'bias') and hasattr(mlx_layer.skip_in_linear, 'bias')):
                self._compare_tensors("skip_bias", pytorch_layer.skip_in_linear.bias, mlx_layer.skip_in_linear.bias)
                
        except Exception as e:
            print(f"    ❌ Skip Connection 分析失败: {e}")
    
    def _compare_tensors(self, name: str, pytorch_tensor, mlx_tensor):
        """比较两个张量"""
        try:
            # 转换为 numpy
            pytorch_np = pytorch_tensor.detach().cpu().numpy()
            mlx_np = np.array(mlx_tensor)
            
            print(f"      {name}:")
            print(f"        PyTorch: shape={pytorch_np.shape}, min={np.min(pytorch_np):.6f}, max={np.max(pytorch_np):.6f}, avg={np.mean(pytorch_np):.6f}")
            print(f"        MLX:     shape={mlx_np.shape}, min={np.min(mlx_np):.6f}, max={np.max(mlx_np):.6f}, avg={np.mean(mlx_np):.6f}")
            
            if pytorch_np.shape == mlx_np.shape:
                diff = np.abs(pytorch_np - mlx_np)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                print(f"        差异: max={max_diff:.6f}, mean={mean_diff:.6f}")
                
                if max_diff < 1e-6:
                    print(f"        ✅ {name} 完全一致")
                elif max_diff < 0.1:
                    print(f"        ⚠️  {name} 存在小幅差异")
                else:
                    print(f"        ❌ {name} 存在显著差异")
            else:
                print(f"        ❌ {name} 形状不匹配")
                
        except Exception as e:
            print(f"        ❌ {name} 比较失败: {e}")

def main():
    """主函数"""
    analyzer = TransformerComponentsAnalyzer()
    analyzer.analyze_transformer_components()

if __name__ == "__main__":
    main()
