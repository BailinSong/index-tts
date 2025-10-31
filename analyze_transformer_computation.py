#!/usr/bin/env python3
"""
Transformer 计算逻辑差异分析工具
深入分析 PyTorch 和 MLX DiT 中 Transformer 层的计算实现差异
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

class TransformerComputationAnalyzer:
    def __init__(self, cache_dir: str = "cfm_production_cache"):
        self.cache_dir = cache_dir
        
    def analyze_transformer_computation(self):
        """分析 Transformer 计算逻辑差异"""
        print("🔍 Transformer 计算逻辑差异分析工具")
        print("="*60)
        
        # 1. 加载生产环境模型
        pytorch_dit, mlx_dit = self._load_production_models()
        if pytorch_dit is None or mlx_dit is None:
            return
        
        # 2. 准备测试数据
        test_data = self._prepare_test_data()
        if test_data is None:
            return
        
        # 3. 分析 Transformer 计算逻辑差异
        self._analyze_attention_computation_detailed(pytorch_dit, mlx_dit, test_data)
        self._analyze_feed_forward_computation_detailed(pytorch_dit, mlx_dit, test_data)
        self._analyze_layer_norm_computation_detailed(pytorch_dit, mlx_dit, test_data)
        self._analyze_activation_functions(pytorch_dit, mlx_dit, test_data)
        
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
    
    def _analyze_attention_computation_detailed(self, pytorch_dit, mlx_dit, test_data):
        """详细分析注意力计算差异"""
        print(f"\n{'='*60}")
        print(f"🔍 注意力计算详细分析")
        print(f"{'='*60}")
        
        # 准备 Transformer 输入
        x_in_pt, t_emb_pt, x_in_mx, t_emb_mx = self._prepare_transformer_input(pytorch_dit, mlx_dit, test_data)
        
        # 分析第一层的注意力计算
        print(f"\n📊 Layer 0 注意力计算详细分析:")
        
        try:
            # 获取第一层
            pytorch_layer = pytorch_dit.transformer.layers[0]
            mlx_layer = mlx_dit.transformer.layers[0]
            
            # 分析注意力计算的各个步骤
            self._analyze_qkv_computation(pytorch_layer, mlx_layer, x_in_pt, x_in_mx)
            self._analyze_attention_weights_computation(pytorch_layer, mlx_layer, x_in_pt, x_in_mx)
            self._analyze_attention_output_computation(pytorch_layer, mlx_layer, x_in_pt, x_in_mx)
            
        except Exception as e:
            print(f"  ❌ 注意力计算详细分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_qkv_computation(self, pytorch_layer, mlx_layer, x_in_pt, x_in_mx):
        """分析 QKV 计算"""
        print(f"\n  📊 QKV 计算分析:")
        
        try:
            # PyTorch QKV 计算
            pytorch_attn = pytorch_layer.attention
            with torch.no_grad():
                qkv_pt = pytorch_attn.wqkv(x_in_pt)
            print(f"    PyTorch QKV 输出: {qkv_pt.shape}, range=[{qkv_pt.min().item():.6f}, {qkv_pt.max().item():.6f}]")
            
            # MLX QKV 计算
            mlx_attn = mlx_layer.attention
            qkv_mx = mlx_attn.wqkv(x_in_mx)
            qkv_np = np.array(qkv_mx)
            print(f"    MLX QKV 输出: {qkv_mx.shape}, range=[{np.min(qkv_np):.6f}, {np.max(qkv_np):.6f}]")
            
            # 比较 QKV 输出
            self._compare_tensors("qkv_output", qkv_pt, qkv_mx)
            
        except Exception as e:
            print(f"    ❌ QKV 计算分析失败: {e}")
    
    def _analyze_attention_weights_computation(self, pytorch_layer, mlx_layer, x_in_pt, x_in_mx):
        """分析注意力权重计算"""
        print(f"\n  📊 注意力权重计算分析:")
        
        try:
            # 这里需要更深入的实现来获取注意力权重的中间结果
            # 由于注意力计算可能没有暴露中间结果，我们需要修改模型或使用 hook
            print(f"    ⚠️  需要实现注意力权重中间结果提取机制")
            
            # 分析注意力计算的数值精度
            self._analyze_attention_numerical_precision(pytorch_layer, mlx_layer, x_in_pt, x_in_mx)
            
        except Exception as e:
            print(f"    ❌ 注意力权重计算分析失败: {e}")
    
    def _analyze_attention_numerical_precision(self, pytorch_layer, mlx_layer, x_in_pt, x_in_mx):
        """分析注意力计算的数值精度"""
        print(f"\n    📊 注意力计算数值精度分析:")
        
        try:
            # 检查数据类型
            print(f"      PyTorch 输入数据类型: {x_in_pt.dtype}")
            print(f"      MLX 输入数据类型: {x_in_mx.dtype}")
            
            # 检查数值范围
            print(f"      PyTorch 输入数值范围: [{x_in_pt.min().item():.6f}, {x_in_pt.max().item():.6f}]")
            x_in_np = np.array(x_in_mx)
            print(f"      MLX 输入数值范围: [{np.min(x_in_np):.6f}, {np.max(x_in_np):.6f}]")
            
            # 检查是否有 NaN 或 Inf
            pytorch_has_nan = torch.isnan(x_in_pt).any().item()
            pytorch_has_inf = torch.isinf(x_in_pt).any().item()
            mlx_has_nan = np.isnan(x_in_mx).any()
            mlx_has_inf = np.isinf(x_in_mx).any()
            
            print(f"      PyTorch 包含 NaN: {pytorch_has_nan}, 包含 Inf: {pytorch_has_inf}")
            print(f"      MLX 包含 NaN: {mlx_has_nan}, 包含 Inf: {mlx_has_inf}")
            
        except Exception as e:
            print(f"      ❌ 数值精度分析失败: {e}")
    
    def _analyze_attention_output_computation(self, pytorch_layer, mlx_layer, x_in_pt, x_in_mx):
        """分析注意力输出计算"""
        print(f"\n  📊 注意力输出计算分析:")
        
        try:
            # 这里需要更深入的实现来获取注意力输出的中间结果
            print(f"    ⚠️  需要实现注意力输出中间结果提取机制")
            
        except Exception as e:
            print(f"    ❌ 注意力输出计算分析失败: {e}")
    
    def _analyze_feed_forward_computation_detailed(self, pytorch_dit, mlx_dit, test_data):
        """详细分析前馈网络计算差异"""
        print(f"\n{'='*60}")
        print(f"🔍 前馈网络计算详细分析")
        print(f"{'='*60}")
        
        # 准备 Transformer 输入
        x_in_pt, t_emb_pt, x_in_mx, t_emb_mx = self._prepare_transformer_input(pytorch_dit, mlx_dit, test_data)
        
        # 分析第一层的前馈网络计算
        print(f"\n📊 Layer 0 前馈网络计算详细分析:")
        
        try:
            # 获取第一层
            pytorch_layer = pytorch_dit.transformer.layers[0]
            mlx_layer = mlx_dit.transformer.layers[0]
            
            # 分析前馈网络计算的各个步骤
            self._analyze_feed_forward_steps(pytorch_layer, mlx_layer, x_in_pt, x_in_mx)
            
        except Exception as e:
            print(f"  ❌ 前馈网络计算详细分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_feed_forward_steps(self, pytorch_layer, mlx_layer, x_in_pt, x_in_mx):
        """分析前馈网络计算步骤"""
        print(f"\n  📊 前馈网络计算步骤分析:")
        
        try:
            # PyTorch 前馈网络计算
            pytorch_ffn = pytorch_layer.feed_forward
            with torch.no_grad():
                # 第一步：w1 和 w3
                w1_out_pt = pytorch_ffn.w1(x_in_pt)
                w3_out_pt = pytorch_ffn.w3(x_in_pt)
                print(f"    PyTorch w1 输出: {w1_out_pt.shape}, range=[{w1_out_pt.min().item():.6f}, {w1_out_pt.max().item():.6f}]")
                print(f"    PyTorch w3 输出: {w3_out_pt.shape}, range=[{w3_out_pt.min().item():.6f}, {w3_out_pt.max().item():.6f}]")
                
                # 激活函数
                activated_pt = torch.nn.functional.silu(w1_out_pt) * w3_out_pt
                print(f"    PyTorch 激活后: {activated_pt.shape}, range=[{activated_pt.min().item():.6f}, {activated_pt.max().item():.6f}]")
                
                # 第二步：w2
                w2_out_pt = pytorch_ffn.w2(activated_pt)
                print(f"    PyTorch w2 输出: {w2_out_pt.shape}, range=[{w2_out_pt.min().item():.6f}, {w2_out_pt.max().item():.6f}]")
            
            # MLX 前馈网络计算
            mlx_ffn = mlx_layer.feed_forward
            # 第一步：w1 和 w3
            w1_out_mx = mlx_ffn.w1(x_in_mx)
            w3_out_mx = mlx_ffn.w3(x_in_mx)
            w1_np = np.array(w1_out_mx)
            w3_np = np.array(w3_out_mx)
            print(f"    MLX w1 输出: {w1_out_mx.shape}, range=[{np.min(w1_np):.6f}, {np.max(w1_np):.6f}]")
            print(f"    MLX w3 输出: {w3_out_mx.shape}, range=[{np.min(w3_np):.6f}, {np.max(w3_np):.6f}]")
            
            # 激活函数
            import mlx.nn as mlx_nn
            activated_mx = mlx_nn.silu(w1_out_mx) * w3_out_mx
            activated_np = np.array(activated_mx)
            print(f"    MLX 激活后: {activated_mx.shape}, range=[{np.min(activated_np):.6f}, {np.max(activated_np):.6f}]")
            
            # 第二步：w2
            w2_out_mx = mlx_ffn.w2(activated_mx)
            w2_np = np.array(w2_out_mx)
            print(f"    MLX w2 输出: {w2_out_mx.shape}, range=[{np.min(w2_np):.6f}, {np.max(w2_np):.6f}]")
            
            # 比较各个步骤的输出
            self._compare_tensors("w1_output", w1_out_pt, w1_out_mx)
            self._compare_tensors("w3_output", w3_out_pt, w3_out_mx)
            self._compare_tensors("activated_output", activated_pt, activated_mx)
            self._compare_tensors("w2_output", w2_out_pt, w2_out_mx)
            
        except Exception as e:
            print(f"    ❌ 前馈网络计算步骤分析失败: {e}")
    
    def _analyze_layer_norm_computation_detailed(self, pytorch_dit, mlx_dit, test_data):
        """详细分析层归一化计算差异"""
        print(f"\n{'='*60}")
        print(f"🔍 层归一化计算详细分析")
        print(f"{'='*60}")
        
        # 准备 Transformer 输入
        x_in_pt, t_emb_pt, x_in_mx, t_emb_mx = self._prepare_transformer_input(pytorch_dit, mlx_dit, test_data)
        
        # 分析第一层的层归一化计算
        print(f"\n📊 Layer 0 层归一化计算详细分析:")
        
        try:
            # 获取第一层
            pytorch_layer = pytorch_dit.transformer.layers[0]
            mlx_layer = mlx_dit.transformer.layers[0]
            
            # 分析注意力归一化计算
            self._analyze_attention_norm_computation(pytorch_layer, mlx_layer, x_in_pt, t_emb_pt, x_in_mx, t_emb_mx)
            
            # 分析前馈网络归一化计算
            self._analyze_ffn_norm_computation(pytorch_layer, mlx_layer, x_in_pt, t_emb_pt, x_in_mx, t_emb_mx)
            
        except Exception as e:
            print(f"  ❌ 层归一化计算详细分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_attention_norm_computation(self, pytorch_layer, mlx_layer, x_in_pt, t_emb_pt, x_in_mx, t_emb_mx):
        """分析注意力归一化计算"""
        print(f"\n  📊 注意力归一化计算分析:")
        
        try:
            # PyTorch 注意力归一化计算
            pytorch_attn_norm = pytorch_layer.attention_norm
            with torch.no_grad():
                # 投影层
                proj_out_pt = pytorch_attn_norm.project_layer(x_in_pt)
                print(f"    PyTorch 投影层输出: {proj_out_pt.shape}, range=[{proj_out_pt.min().item():.6f}, {proj_out_pt.max().item():.6f}]")
                
                # 归一化
                norm_out_pt = pytorch_attn_norm.norm(proj_out_pt)
                print(f"    PyTorch 归一化输出: {norm_out_pt.shape}, range=[{norm_out_pt.min().item():.6f}, {norm_out_pt.max().item():.6f}]")
            
            # MLX 注意力归一化计算
            mlx_attn_norm = mlx_layer.attention_norm
            # 投影层
            proj_out_mx = mlx_attn_norm.project_layer(x_in_mx)
            proj_np = np.array(proj_out_mx)
            print(f"    MLX 投影层输出: {proj_out_mx.shape}, range=[{np.min(proj_np):.6f}, {np.max(proj_np):.6f}]")
            
            # 归一化 - 需要先处理维度问题
            # 将投影层输出从 (B, T, 1024) 转换为 (B, T, 512) 以匹配归一化层的输入
            proj_reshaped_mx = proj_out_mx[..., :512]  # 取前512维
            norm_out_mx = mlx_attn_norm.norm(proj_reshaped_mx)
            norm_np = np.array(norm_out_mx)
            print(f"    MLX 归一化输出: {norm_out_mx.shape}, range=[{np.min(norm_np):.6f}, {np.max(norm_np):.6f}]")
            
            # 比较各个步骤的输出
            self._compare_tensors("attn_norm_proj_output", proj_out_pt, proj_out_mx)
            self._compare_tensors("attn_norm_output", norm_out_pt, norm_out_mx)
            
        except Exception as e:
            print(f"    ❌ 注意力归一化计算分析失败: {e}")
    
    def _analyze_ffn_norm_computation(self, pytorch_layer, mlx_layer, x_in_pt, t_emb_pt, x_in_mx, t_emb_mx):
        """分析前馈网络归一化计算"""
        print(f"\n  📊 前馈网络归一化计算分析:")
        
        try:
            # PyTorch 前馈网络归一化计算
            pytorch_ffn_norm = pytorch_layer.ffn_norm
            with torch.no_grad():
                # 投影层
                proj_out_pt = pytorch_ffn_norm.project_layer(x_in_pt)
                print(f"    PyTorch 投影层输出: {proj_out_pt.shape}, range=[{proj_out_pt.min().item():.6f}, {proj_out_pt.max().item():.6f}]")
                
                # 归一化
                norm_out_pt = pytorch_ffn_norm.norm(proj_out_pt)
                print(f"    PyTorch 归一化输出: {norm_out_pt.shape}, range=[{norm_out_pt.min().item():.6f}, {norm_out_pt.max().item():.6f}]")
            
            # MLX 前馈网络归一化计算
            mlx_ffn_norm = mlx_layer.ffn_norm
            # 投影层
            proj_out_mx = mlx_ffn_norm.project_layer(x_in_mx)
            proj_np = np.array(proj_out_mx)
            print(f"    MLX 投影层输出: {proj_out_mx.shape}, range=[{np.min(proj_np):.6f}, {np.max(proj_np):.6f}]")
            
            # 归一化 - 需要先处理维度问题
            # 将投影层输出从 (B, T, 1024) 转换为 (B, T, 512) 以匹配归一化层的输入
            proj_reshaped_mx = proj_out_mx[..., :512]  # 取前512维
            norm_out_mx = mlx_ffn_norm.norm(proj_reshaped_mx)
            norm_np = np.array(norm_out_mx)
            print(f"    MLX 归一化输出: {norm_out_mx.shape}, range=[{np.min(norm_np):.6f}, {np.max(norm_np):.6f}]")
            
            # 比较各个步骤的输出
            self._compare_tensors("ffn_norm_proj_output", proj_out_pt, proj_out_mx)
            self._compare_tensors("ffn_norm_output", norm_out_pt, norm_out_mx)
            
        except Exception as e:
            print(f"    ❌ 前馈网络归一化计算分析失败: {e}")
    
    def _analyze_activation_functions(self, pytorch_dit, mlx_dit, test_data):
        """分析激活函数差异"""
        print(f"\n{'='*60}")
        print(f"🔍 激活函数差异分析")
        print(f"{'='*60}")
        
        # 准备测试数据
        x_in_pt, t_emb_pt, x_in_mx, t_emb_mx = self._prepare_transformer_input(pytorch_dit, mlx_dit, test_data)
        
        # 分析 SiLU 激活函数
        print(f"\n📊 SiLU 激活函数分析:")
        
        try:
            # PyTorch SiLU
            with torch.no_grad():
                silu_pt = torch.nn.functional.silu(x_in_pt)
            print(f"  PyTorch SiLU 输出: {silu_pt.shape}, range=[{silu_pt.min().item():.6f}, {silu_pt.max().item():.6f}]")
            
            # MLX SiLU
            import mlx.nn as mlx_nn
            silu_mx = mlx_nn.silu(x_in_mx)
            silu_np = np.array(silu_mx)
            print(f"  MLX SiLU 输出: {silu_mx.shape}, range=[{np.min(silu_np):.6f}, {np.max(silu_np):.6f}]")
            
            # 比较 SiLU 输出
            self._compare_tensors("silu_output", silu_pt, silu_mx)
            
        except Exception as e:
            print(f"  ❌ SiLU 激活函数分析失败: {e}")
    
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
    analyzer = TransformerComputationAnalyzer()
    analyzer.analyze_transformer_computation()

if __name__ == "__main__":
    main()
