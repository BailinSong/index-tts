#!/usr/bin/env python3
"""
DiT 最终输出层差异分析工具
深入分析 PyTorch 和 MLX DiT 中最终输出层的实现差异
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

class DiTFinalLayersAnalyzer:
    def __init__(self, cache_dir: str = "cfm_production_cache"):
        self.cache_dir = cache_dir
        
    def analyze_dit_final_layers(self):
        """分析 DiT 最终输出层差异"""
        print("🔍 DiT 最终输出层差异分析工具")
        print("="*60)
        
        # 1. 加载生产环境模型
        pytorch_dit, mlx_dit = self._load_production_models()
        if pytorch_dit is None or mlx_dit is None:
            return
        
        # 2. 准备测试数据
        test_data = self._prepare_test_data()
        if test_data is None:
            return
        
        # 3. 分析 DiT 最终输出层差异
        self._analyze_final_layer_implementation(pytorch_dit, mlx_dit, test_data)
        self._analyze_final_norm_implementation(pytorch_dit, mlx_dit, test_data)
        self._analyze_skip_connection_implementation(pytorch_dit, mlx_dit, test_data)
        self._analyze_final_transpose_implementation(pytorch_dit, mlx_dit, test_data)
        
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
    
    def _prepare_transformer_output(self, pytorch_dit, mlx_dit, test_data):
        """准备 Transformer 输出（使用完整的 DiT 前向传播）"""
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
        
        print("🔧 使用完整 DiT 前向传播...")
        
        # 使用完整的 DiT 前向传播
        with torch.no_grad():
            pytorch_output = pytorch_dit(x_pt, prompt_x_pt, x_lens_pt, t_pt, style_pt, mu_pt)
        
        mlx_output = mlx_dit(x_mx, prompt_x_mx, x_lens_mx, t_mx, style_mx, mu_mx)
        
        # 为了分析，我们需要模拟 Transformer 输出
        # 这里我们使用一个简化的方法：假设 Transformer 输出就是最终输出的转置
        x_t_pt = x_pt.transpose(1, 2)
        x_t_mx = x_mx.transpose(0, 2, 1)
        
        # 模拟 Transformer 输出（从最终输出反推）
        transformer_output_pt = pytorch_output.transpose(1, 2)  # (B, T, C)
        transformer_output_mx = mlx_output.transpose(0, 2, 1)   # (B, T, C)
        
        # 时间嵌入
        t_emb_pt = pytorch_dit.t_embedder(t_pt)
        t_emb_mx = mlx_dit.t_embedder(t_mx)
        
        return transformer_output_pt, t_emb_pt, transformer_output_mx, t_emb_mx, x_t_pt, x_t_mx
    
    def _analyze_final_layer_implementation(self, pytorch_dit, mlx_dit, test_data):
        """分析最终输出层实现差异"""
        print(f"\n{'='*60}")
        print(f"🔍 最终输出层实现分析")
        print(f"{'='*60}")
        
        # 准备 Transformer 输出
        transformer_output_pt, t_emb_pt, transformer_output_mx, t_emb_mx, x_t_pt, x_t_mx = self._prepare_transformer_output(pytorch_dit, mlx_dit, test_data)
        
        # 分析最终输出层
        print(f"\n📊 最终输出层分析:")
        
        try:
            # PyTorch 最终输出层
            pytorch_final_layer = pytorch_dit.final_layer
            print(f"  PyTorch 最终输出层: {type(pytorch_final_layer)}")
            
            # 分析最终输出层的权重
            self._analyze_final_layer_weights(pytorch_final_layer, mlx_dit.final_layer)
            
            # 分析最终输出层的计算
            self._analyze_final_layer_computation(pytorch_final_layer, mlx_dit.final_layer, transformer_output_pt, t_emb_pt, transformer_output_mx, t_emb_mx)
            
        except Exception as e:
            print(f"  ❌ 最终输出层分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_final_layer_weights(self, pytorch_final_layer, mlx_final_layer):
        """分析最终输出层权重"""
        print(f"\n  📊 最终输出层权重分析:")
        
        try:
            # PyTorch 最终输出层权重
            print(f"    PyTorch 最终输出层权重:")
            for name, param in pytorch_final_layer.named_parameters():
                print(f"      {name}: {param.shape}, range=[{param.min().item():.6f}, {param.max().item():.6f}]")
            
            # MLX 最终输出层权重
            print(f"    MLX 最终输出层权重:")
            for name, param in mlx_final_layer.parameters().items():
                param_np = np.array(param)
                print(f"      {name}: {param.shape}, range=[{np.min(param_np):.6f}, {np.max(param_np):.6f}]")
            
            # 比较权重
            self._compare_final_layer_weights(pytorch_final_layer, mlx_final_layer)
            
        except Exception as e:
            print(f"    ❌ 最终输出层权重分析失败: {e}")
    
    def _compare_final_layer_weights(self, pytorch_final_layer, mlx_final_layer):
        """比较最终输出层权重"""
        print(f"\n    📊 最终输出层权重比较:")
        
        try:
            # 获取 PyTorch 权重
            pytorch_weights = {}
            for name, param in pytorch_final_layer.named_parameters():
                pytorch_weights[name] = param.detach().cpu().numpy()
            
            # 获取 MLX 权重
            mlx_weights = {}
            for name, param in mlx_final_layer.parameters().items():
                mlx_weights[name] = np.array(param)
            
            # 比较权重
            common_keys = set(pytorch_weights.keys()) & set(mlx_weights.keys())
            print(f"      共同权重数量: {len(common_keys)}")
            
            for key in sorted(common_keys):
                pt_weight = pytorch_weights[key]
                mlx_weight = mlx_weights[key]
                
                if pt_weight.shape == mlx_weight.shape:
                    diff = np.abs(pt_weight - mlx_weight)
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)
                    
                    if max_diff < 1e-6:
                        print(f"      ✅ {key}: 完全一致")
                    elif max_diff < 0.1:
                        print(f"      ⚠️  {key}: 小幅差异 (max={max_diff:.6f}, mean={mean_diff:.6f})")
                    else:
                        print(f"      ❌ {key}: 显著差异 (max={max_diff:.6f}, mean={mean_diff:.6f})")
                else:
                    print(f"      ❌ {key}: 形状不匹配 (PyTorch={pt_weight.shape}, MLX={mlx_weight.shape})")
                    
        except Exception as e:
            print(f"      ❌ 最终输出层权重比较失败: {e}")
    
    def _analyze_final_layer_computation(self, pytorch_final_layer, mlx_final_layer, transformer_output_pt, t_emb_pt, transformer_output_mx, t_emb_mx):
        """分析最终输出层计算"""
        print(f"\n  📊 最终输出层计算分析:")
        
        try:
            # PyTorch 最终输出层计算
            with torch.no_grad():
                pytorch_final_output = pytorch_final_layer(transformer_output_pt, t_emb_pt)
            print(f"    PyTorch 最终输出: {pytorch_final_output.shape}, range=[{pytorch_final_output.min().item():.6f}, {pytorch_final_output.max().item():.6f}]")
            
            # MLX 最终输出层计算
            mlx_final_output = mlx_final_layer(transformer_output_mx, t_emb_mx)
            mlx_final_np = np.array(mlx_final_output)
            print(f"    MLX 最终输出: {mlx_final_output.shape}, range=[{np.min(mlx_final_np):.6f}, {np.max(mlx_final_np):.6f}]")
            
            # 比较最终输出
            self._compare_tensors("final_layer_output", pytorch_final_output, mlx_final_output)
            
        except Exception as e:
            print(f"    ❌ 最终输出层计算分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_final_norm_implementation(self, pytorch_dit, mlx_dit, test_data):
        """分析最终归一化实现差异"""
        print(f"\n{'='*60}")
        print(f"🔍 最终归一化实现分析")
        print(f"{'='*60}")
        
        # 准备 Transformer 输出
        transformer_output_pt, t_emb_pt, transformer_output_mx, t_emb_mx, x_t_pt, x_t_mx = self._prepare_transformer_output(pytorch_dit, mlx_dit, test_data)
        
        # 分析最终归一化
        print(f"\n📊 最终归一化分析:")
        
        try:
            # PyTorch 最终归一化
            if hasattr(pytorch_dit, 'norm'):
                pytorch_final_norm = pytorch_dit.norm
                print(f"  PyTorch 最终归一化: {type(pytorch_final_norm)}")
                
                # 分析最终归一化权重
                self._analyze_final_norm_weights(pytorch_final_norm, mlx_dit.norm if hasattr(mlx_dit, 'norm') else None)
                
                # 分析最终归一化计算
                self._analyze_final_norm_computation(pytorch_final_norm, mlx_dit.norm if hasattr(mlx_dit, 'norm') else None, transformer_output_pt, transformer_output_mx)
            else:
                print(f"  PyTorch 没有最终归一化层")
            
        except Exception as e:
            print(f"  ❌ 最终归一化分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_final_norm_weights(self, pytorch_final_norm, mlx_final_norm):
        """分析最终归一化权重"""
        print(f"\n  📊 最终归一化权重分析:")
        
        try:
            if pytorch_final_norm is None or mlx_final_norm is None:
                print(f"    跳过最终归一化权重分析（不存在）")
                return
            
            # PyTorch 最终归一化权重
            print(f"    PyTorch 最终归一化权重:")
            for name, param in pytorch_final_norm.named_parameters():
                print(f"      {name}: {param.shape}, range=[{param.min().item():.6f}, {param.max().item():.6f}]")
            
            # MLX 最终归一化权重
            print(f"    MLX 最终归一化权重:")
            for name, param in mlx_final_norm.parameters().items():
                param_np = np.array(param)
                print(f"      {name}: {param.shape}, range=[{np.min(param_np):.6f}, {np.max(param_np):.6f}]")
            
            # 比较权重
            self._compare_final_norm_weights(pytorch_final_norm, mlx_final_norm)
            
        except Exception as e:
            print(f"    ❌ 最终归一化权重分析失败: {e}")
    
    def _compare_final_norm_weights(self, pytorch_final_norm, mlx_final_norm):
        """比较最终归一化权重"""
        print(f"\n    📊 最终归一化权重比较:")
        
        try:
            # 获取 PyTorch 权重
            pytorch_weights = {}
            for name, param in pytorch_final_norm.named_parameters():
                pytorch_weights[name] = param.detach().cpu().numpy()
            
            # 获取 MLX 权重
            mlx_weights = {}
            for name, param in mlx_final_norm.parameters().items():
                mlx_weights[name] = np.array(param)
            
            # 比较权重
            common_keys = set(pytorch_weights.keys()) & set(mlx_weights.keys())
            print(f"      共同权重数量: {len(common_keys)}")
            
            for key in sorted(common_keys):
                pt_weight = pytorch_weights[key]
                mlx_weight = mlx_weights[key]
                
                if pt_weight.shape == mlx_weight.shape:
                    diff = np.abs(pt_weight - mlx_weight)
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)
                    
                    if max_diff < 1e-6:
                        print(f"      ✅ {key}: 完全一致")
                    elif max_diff < 0.1:
                        print(f"      ⚠️  {key}: 小幅差异 (max={max_diff:.6f}, mean={mean_diff:.6f})")
                    else:
                        print(f"      ❌ {key}: 显著差异 (max={max_diff:.6f}, mean={mean_diff:.6f})")
                else:
                    print(f"      ❌ {key}: 形状不匹配 (PyTorch={pt_weight.shape}, MLX={mlx_weight.shape})")
                    
        except Exception as e:
            print(f"      ❌ 最终归一化权重比较失败: {e}")
    
    def _analyze_final_norm_computation(self, pytorch_final_norm, mlx_final_norm, transformer_output_pt, transformer_output_mx):
        """分析最终归一化计算"""
        print(f"\n  📊 最终归一化计算分析:")
        
        try:
            if pytorch_final_norm is None or mlx_final_norm is None:
                print(f"    跳过最终归一化计算分析（不存在）")
                return
            
            # PyTorch 最终归一化计算
            with torch.no_grad():
                pytorch_norm_output = pytorch_final_norm(transformer_output_pt)
            print(f"    PyTorch 归一化输出: {pytorch_norm_output.shape}, range=[{pytorch_norm_output.min().item():.6f}, {pytorch_norm_output.max().item():.6f}]")
            
            # MLX 最终归一化计算
            mlx_norm_output = mlx_final_norm(transformer_output_mx)
            mlx_norm_np = np.array(mlx_norm_output)
            print(f"    MLX 归一化输出: {mlx_norm_output.shape}, range=[{np.min(mlx_norm_np):.6f}, {np.max(mlx_norm_np):.6f}]")
            
            # 比较归一化输出
            self._compare_tensors("final_norm_output", pytorch_norm_output, mlx_norm_output)
            
        except Exception as e:
            print(f"    ❌ 最终归一化计算分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_skip_connection_implementation(self, pytorch_dit, mlx_dit, test_data):
        """分析 Skip Connection 实现差异"""
        print(f"\n{'='*60}")
        print(f"🔍 Skip Connection 实现分析")
        print(f"{'='*60}")
        
        # 准备 Transformer 输出
        transformer_output_pt, t_emb_pt, transformer_output_mx, t_emb_mx, x_t_pt, x_t_mx = self._prepare_transformer_output(pytorch_dit, mlx_dit, test_data)
        
        # 分析 Skip Connection
        print(f"\n📊 Skip Connection 分析:")
        
        try:
            # 检查是否有 skip connection
            if hasattr(pytorch_dit, 'long_skip_connection') and pytorch_dit.long_skip_connection:
                print(f"  PyTorch 启用 Skip Connection: {pytorch_dit.long_skip_connection}")
                
                # 分析 skip connection 计算
                self._analyze_skip_connection_computation(pytorch_dit, mlx_dit, transformer_output_pt, x_t_pt, transformer_output_mx, x_t_mx)
            else:
                print(f"  PyTorch 未启用 Skip Connection")
            
        except Exception as e:
            print(f"  ❌ Skip Connection 分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_skip_connection_computation(self, pytorch_dit, mlx_dit, transformer_output_pt, x_t_pt, transformer_output_mx, x_t_mx):
        """分析 Skip Connection 计算"""
        print(f"\n  📊 Skip Connection 计算分析:")
        
        try:
            # PyTorch Skip Connection 计算
            if hasattr(pytorch_dit, 'skip_linear'):
                with torch.no_grad():
                    skip_input_pt = torch.cat([transformer_output_pt, x_t_pt], dim=-1)
                    pytorch_skip_output = pytorch_dit.skip_linear(skip_input_pt)
                print(f"    PyTorch Skip 输入: {skip_input_pt.shape}, range=[{skip_input_pt.min().item():.6f}, {skip_input_pt.max().item():.6f}]")
                print(f"    PyTorch Skip 输出: {pytorch_skip_output.shape}, range=[{pytorch_skip_output.min().item():.6f}, {pytorch_skip_output.max().item():.6f}]")
            
            # MLX Skip Connection 计算
            if hasattr(mlx_dit, 'skip_linear'):
                skip_input_mx = mx.concatenate([transformer_output_mx, x_t_mx], -1)
                mlx_skip_output = mlx_dit.skip_linear(skip_input_mx)
                skip_input_np = np.array(skip_input_mx)
                mlx_skip_np = np.array(mlx_skip_output)
                print(f"    MLX Skip 输入: {skip_input_mx.shape}, range=[{np.min(skip_input_np):.6f}, {np.max(skip_input_np):.6f}]")
                print(f"    MLX Skip 输出: {mlx_skip_output.shape}, range=[{np.min(mlx_skip_np):.6f}, {np.max(mlx_skip_np):.6f}]")
            
            # 比较 Skip Connection 输出
            if hasattr(pytorch_dit, 'skip_linear') and hasattr(mlx_dit, 'skip_linear'):
                self._compare_tensors("skip_connection_output", pytorch_skip_output, mlx_skip_output)
            
        except Exception as e:
            print(f"    ❌ Skip Connection 计算分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_final_transpose_implementation(self, pytorch_dit, mlx_dit, test_data):
        """分析最终转置实现差异"""
        print(f"\n{'='*60}")
        print(f"🔍 最终转置实现分析")
        print(f"{'='*60}")
        
        # 准备 Transformer 输出
        transformer_output_pt, t_emb_pt, transformer_output_mx, t_emb_mx, x_t_pt, x_t_mx = self._prepare_transformer_output(pytorch_dit, mlx_dit, test_data)
        
        # 分析最终转置
        print(f"\n📊 最终转置分析:")
        
        try:
            # 模拟最终转置操作
            # PyTorch: (B, T, C) -> (B, C, T)
            pytorch_final_transpose = transformer_output_pt.transpose(1, 2)
            print(f"  PyTorch 最终转置: {pytorch_final_transpose.shape}, range=[{pytorch_final_transpose.min().item():.6f}, {pytorch_final_transpose.max().item():.6f}]")
            
            # MLX: (B, T, C) -> (B, C, T)
            mlx_final_transpose = transformer_output_mx.transpose(0, 2, 1)
            mlx_final_np = np.array(mlx_final_transpose)
            print(f"  MLX 最终转置: {mlx_final_transpose.shape}, range=[{np.min(mlx_final_np):.6f}, {np.max(mlx_final_np):.6f}]")
            
            # 比较最终转置
            self._compare_tensors("final_transpose", pytorch_final_transpose, mlx_final_transpose)
            
        except Exception as e:
            print(f"  ❌ 最终转置分析失败: {e}")
            import traceback
            traceback.print_exc()
    
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
    analyzer = DiTFinalLayersAnalyzer()
    analyzer.analyze_dit_final_layers()

if __name__ == "__main__":
    main()
