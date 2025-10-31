#!/usr/bin/env python3
"""
DiT 完整前向传播差异分析工具
重新分析 DiT 输出差异的真正原因
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

class DiTCompleteForwardAnalyzer:
    def __init__(self, cache_dir: str = "cfm_production_cache"):
        self.cache_dir = cache_dir
        
    def analyze_dit_complete_forward(self):
        """分析 DiT 完整前向传播差异"""
        print("🔍 DiT 完整前向传播差异分析工具")
        print("="*60)
        
        # 1. 加载生产环境模型
        pytorch_dit, mlx_dit = self._load_production_models()
        if pytorch_dit is None or mlx_dit is None:
            return
        
        # 2. 准备测试数据
        test_data = self._prepare_test_data()
        if test_data is None:
            return
        
        # 3. 分析 DiT 完整前向传播
        self._analyze_dit_forward_step_by_step(pytorch_dit, mlx_dit, test_data)
        self._analyze_skip_connection_differences(pytorch_dit, mlx_dit, test_data)
        self._analyze_final_transpose_differences(pytorch_dit, mlx_dit, test_data)
        self._analyze_dit_architecture_differences(pytorch_dit, mlx_dit)
        
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
    
    def _analyze_dit_forward_step_by_step(self, pytorch_dit, mlx_dit, test_data):
        """分析 DiT 前向传播步骤差异"""
        print(f"\n{'='*60}")
        print(f"🔍 DiT 前向传播步骤分析")
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
        
        print("🔧 分析 DiT 前向传播步骤...")
        
        try:
            # 步骤 1: 转置操作
            print(f"\n📊 步骤 1: 转置操作")
            x_t_pt = x_pt.transpose(1, 2)
            prompt_x_t_pt = prompt_x_pt.transpose(1, 2)
            x_t_mx = x_mx.transpose(0, 2, 1)
            prompt_x_t_mx = prompt_x_mx.transpose(0, 2, 1)
            
            print(f"  PyTorch x_t: {x_t_pt.shape}, range=[{x_t_pt.min().item():.6f}, {x_t_pt.max().item():.6f}]")
            print(f"  MLX x_t: {x_t_mx.shape}, range=[{np.min(np.array(x_t_mx)):.6f}, {np.max(np.array(x_t_mx)):.6f}]")
            self._compare_tensors("x_t", x_t_pt, x_t_mx)
            
            # 步骤 2: 条件投影
            print(f"\n📊 步骤 2: 条件投影")
            with torch.no_grad():
                cond_proj_pt = pytorch_dit.cond_projection(mu_pt)
            cond_proj_mx = mlx_dit.cond_projection(mu_mx)
            
            print(f"  PyTorch cond_proj: {cond_proj_pt.shape}, range=[{cond_proj_pt.min().item():.6f}, {cond_proj_pt.max().item():.6f}]")
            print(f"  MLX cond_proj: {cond_proj_mx.shape}, range=[{np.min(np.array(cond_proj_mx)):.6f}, {np.max(np.array(cond_proj_mx)):.6f}]")
            self._compare_tensors("cond_proj", cond_proj_pt, cond_proj_mx)
            
            # 步骤 3: 拼接输入
            print(f"\n📊 步骤 3: 拼接输入")
            with torch.no_grad():
                x_in_pt = torch.cat([x_t_pt, prompt_x_t_pt, cond_proj_pt], dim=-1)
            x_in_mx = mx.concatenate([x_t_mx, prompt_x_t_mx, cond_proj_mx], -1)
            
            print(f"  PyTorch x_in: {x_in_pt.shape}, range=[{x_in_pt.min().item():.6f}, {x_in_pt.max().item():.6f}]")
            print(f"  MLX x_in: {x_in_mx.shape}, range=[{np.min(np.array(x_in_mx)):.6f}, {np.max(np.array(x_in_mx)):.6f}]")
            self._compare_tensors("x_in", x_in_pt, x_in_mx)
            
            # 步骤 4: 风格条件
            print(f"\n📊 步骤 4: 风格条件")
            B, T, _ = x_in_pt.shape
            with torch.no_grad():
                style_broadcast_pt = style_pt.unsqueeze(1).expand(B, T, -1)
                x_in_pt = torch.cat([x_in_pt, style_broadcast_pt], dim=-1)
            
            style_broadcast_mx = mx.broadcast_to(style_mx.reshape(B, 1, -1), (B, T, style_mx.shape[-1]))
            x_in_mx = mx.concatenate([x_in_mx, style_broadcast_mx], -1)
            
            print(f"  PyTorch x_in_with_style: {x_in_pt.shape}, range=[{x_in_pt.min().item():.6f}, {x_in_pt.max().item():.6f}]")
            print(f"  MLX x_in_with_style: {x_in_mx.shape}, range=[{np.min(np.array(x_in_mx)):.6f}, {np.max(np.array(x_in_mx)):.6f}]")
            self._compare_tensors("x_in_with_style", x_in_pt, x_in_mx)
            
            # 步骤 5: 条件合并线性层
            print(f"\n📊 步骤 5: 条件合并线性层")
            with torch.no_grad():
                x_in_pt = pytorch_dit.cond_x_merge_linear(x_in_pt)
            x_in_mx = mlx_dit.cond_x_merge_linear(x_in_mx)
            
            print(f"  PyTorch x_in_merged: {x_in_pt.shape}, range=[{x_in_pt.min().item():.6f}, {x_in_pt.max().item():.6f}]")
            print(f"  MLX x_in_merged: {x_in_mx.shape}, range=[{np.min(np.array(x_in_mx)):.6f}, {np.max(np.array(x_in_mx)):.6f}]")
            self._compare_tensors("x_in_merged", x_in_pt, x_in_mx)
            
            # 步骤 6: 时间嵌入
            print(f"\n📊 步骤 6: 时间嵌入")
            with torch.no_grad():
                t_emb_pt = pytorch_dit.t_embedder(t_pt)
            t_emb_mx = mlx_dit.t_embedder(t_mx)
            
            print(f"  PyTorch t_emb: {t_emb_pt.shape}, range=[{t_emb_pt.min().item():.6f}, {t_emb_pt.max().item():.6f}]")
            print(f"  MLX t_emb: {t_emb_mx.shape}, range=[{np.min(np.array(t_emb_mx)):.6f}, {np.max(np.array(t_emb_mx)):.6f}]")
            self._compare_tensors("t_emb", t_emb_pt, t_emb_mx)
            
            # 步骤 7: Transformer
            print(f"\n📊 步骤 7: Transformer")
            with torch.no_grad():
                # 准备 Transformer 输入
                x_mask = torch.ones(B, 1, T, device=device, dtype=torch.bool)
                input_pos = torch.arange(T, device=device)
                x_mask_expanded = x_mask[:, None, :].unsqueeze(1).repeat(1, 1, T, 1)
                transformer_output_pt = pytorch_dit.transformer(x_in_pt, t_emb_pt.unsqueeze(1), input_pos, x_mask_expanded)
            
            # MLX Transformer
            x_mask_mx = mx.ones((B, 1, T), dtype=mx.bool)
            input_pos_mx = mx.arange(T)
            x_mask_expanded_mx = mx.broadcast_to(x_mask_mx.reshape(B, 1, 1, T), (B, 1, T, T))
            transformer_output_mx = mlx_dit.transformer(x_in_mx, t_emb_mx.reshape(B, 1, -1), input_pos_mx, x_mask_expanded_mx)
            
            print(f"  PyTorch transformer_output: {transformer_output_pt.shape}, range=[{transformer_output_pt.min().item():.6f}, {transformer_output_pt.max().item():.6f}]")
            print(f"  MLX transformer_output: {transformer_output_mx.shape}, range=[{np.min(np.array(transformer_output_mx)):.6f}, {np.max(np.array(transformer_output_mx)):.6f}]")
            self._compare_tensors("transformer_output", transformer_output_pt, transformer_output_mx)
            
            # 步骤 8: FinalLayer
            print(f"\n📊 步骤 8: FinalLayer")
            with torch.no_grad():
                final_output_pt = pytorch_dit.final_layer(transformer_output_pt, t_emb_pt)
            final_output_mx = mlx_dit.final_layer(transformer_output_mx, t_emb_mx)
            
            print(f"  PyTorch final_output: {final_output_pt.shape}, range=[{final_output_pt.min().item():.6f}, {final_output_pt.max().item():.6f}]")
            print(f"  MLX final_output: {final_output_mx.shape}, range=[{np.min(np.array(final_output_mx)):.6f}, {np.max(np.array(final_output_mx)):.6f}]")
            self._compare_tensors("final_output", final_output_pt, final_output_mx)
            
            # 步骤 9: Skip Connection
            print(f"\n📊 步骤 9: Skip Connection")
            if hasattr(pytorch_dit, 'long_skip_connection') and pytorch_dit.long_skip_connection:
                with torch.no_grad():
                    skip_input_pt = torch.cat([transformer_output_pt, x_t_pt], dim=-1)
                    skip_output_pt = pytorch_dit.skip_linear(skip_input_pt)
                    final_output_pt = final_output_pt + skip_output_pt
                
                skip_input_mx = mx.concatenate([transformer_output_mx, x_t_mx], -1)
                skip_output_mx = mlx_dit.skip_linear(skip_input_mx)
                final_output_mx = final_output_mx + skip_output_mx
                
                print(f"  PyTorch skip_output: {skip_output_pt.shape}, range=[{skip_output_pt.min().item():.6f}, {skip_output_pt.max().item():.6f}]")
                print(f"  MLX skip_output: {skip_output_mx.shape}, range=[{np.min(np.array(skip_output_mx)):.6f}, {np.max(np.array(skip_output_mx)):.6f}]")
                self._compare_tensors("skip_output", skip_output_pt, skip_output_mx)
                
                print(f"  PyTorch final_with_skip: {final_output_pt.shape}, range=[{final_output_pt.min().item():.6f}, {final_output_pt.max().item():.6f}]")
                print(f"  MLX final_with_skip: {final_output_mx.shape}, range=[{np.min(final_output_mx):.6f}, {np.max(final_output_mx):.6f}]")
                self._compare_tensors("final_with_skip", final_output_pt, final_output_mx)
            else:
                print(f"  Skip Connection 未启用")
            
            # 步骤 10: 最终转置
            print(f"\n📊 步骤 10: 最终转置")
            with torch.no_grad():
                final_transpose_pt = final_output_pt.transpose(1, 2)
            final_transpose_mx = final_output_mx.transpose(0, 2, 1)
            
            print(f"  PyTorch final_transpose: {final_transpose_pt.shape}, range=[{final_transpose_pt.min().item():.6f}, {final_transpose_pt.max().item():.6f}]")
            print(f"  MLX final_transpose: {final_transpose_mx.shape}, range=[{np.min(final_transpose_mx):.6f}, {np.max(final_transpose_mx):.6f}]")
            self._compare_tensors("final_transpose", final_transpose_pt, final_transpose_mx)
            
        except Exception as e:
            print(f"❌ DiT 前向传播步骤分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_skip_connection_differences(self, pytorch_dit, mlx_dit, test_data):
        """分析 Skip Connection 差异"""
        print(f"\n{'='*60}")
        print(f"🔍 Skip Connection 差异分析")
        print(f"{'='*60}")
        
        try:
            # 检查 Skip Connection 配置
            print(f"\n📊 Skip Connection 配置:")
            print(f"  PyTorch long_skip_connection: {getattr(pytorch_dit, 'long_skip_connection', False)}")
            print(f"  MLX long_skip_connection: {getattr(mlx_dit, 'long_skip_connection', False)}")
            
            # 检查 Skip Connection 模块
            print(f"\n📊 Skip Connection 模块:")
            if hasattr(pytorch_dit, 'skip_linear'):
                print(f"  PyTorch skip_linear: {type(pytorch_dit.skip_linear)}")
                for name, param in pytorch_dit.skip_linear.named_parameters():
                    print(f"    {name}: {param.shape}, range=[{param.min().item():.6f}, {param.max().item():.6f}]")
            else:
                print(f"  PyTorch skip_linear: 不存在")
            
            if hasattr(mlx_dit, 'skip_linear'):
                print(f"  MLX skip_linear: {type(mlx_dit.skip_linear)}")
                for name, param in mlx_dit.skip_linear.parameters().items():
                    param_np = np.array(param)
                    print(f"    {name}: {param.shape}, range=[{np.min(param_np):.6f}, {np.max(param_np):.6f}]")
            else:
                print(f"  MLX skip_linear: 不存在")
            
        except Exception as e:
            print(f"❌ Skip Connection 差异分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_final_transpose_differences(self, pytorch_dit, mlx_dit, test_data):
        """分析最终转置差异"""
        print(f"\n{'='*60}")
        print(f"🔍 最终转置差异分析")
        print(f"{'='*60}")
        
        try:
            # 准备测试数据
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
            
            print("🔧 分析最终转置差异...")
            
            # 执行完整的 DiT 前向传播
            with torch.no_grad():
                pytorch_output = pytorch_dit(x_pt, prompt_x_pt, x_lens_pt, t_pt, style_pt, mu_pt)
            
            mlx_output = mlx_dit(x_mx, prompt_x_mx, x_lens_mx, t_mx, style_mx, mu_mx)
            
            print(f"\n📊 最终输出比较:")
            print(f"  PyTorch 输出: {pytorch_output.shape}, range=[{pytorch_output.min().item():.6f}, {pytorch_output.max().item():.6f}]")
            print(f"  MLX 输出: {mlx_output.shape}, range=[{np.min(mlx_output):.6f}, {np.max(mlx_output):.6f}]")
            self._compare_tensors("final_output", pytorch_output, mlx_output)
            
        except Exception as e:
            print(f"❌ 最终转置差异分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_dit_architecture_differences(self, pytorch_dit, mlx_dit):
        """分析 DiT 架构差异"""
        print(f"\n{'='*60}")
        print(f"🔍 DiT 架构差异分析")
        print(f"{'='*60}")
        
        try:
            print(f"\n📊 PyTorch DiT 架构:")
            print(f"  类型: {type(pytorch_dit)}")
            print(f"  模块列表:")
            for name, module in pytorch_dit.named_children():
                print(f"    {name}: {type(module)}")
            
            print(f"\n📊 MLX DiT 架构:")
            print(f"  类型: {type(mlx_dit)}")
            print(f"  模块列表:")
            for name, module in mlx_dit.named_children():
                print(f"    {name}: {type(module)}")
            
            # 比较关键配置
            print(f"\n📊 关键配置比较:")
            pytorch_config = {
                'time_as_token': getattr(pytorch_dit, 'time_as_token', None),
                'style_as_token': getattr(pytorch_dit, 'style_as_token', None),
                'uvit_skip_connection': getattr(pytorch_dit, 'uvit_skip_connection', None),
                'long_skip_connection': getattr(pytorch_dit, 'long_skip_connection', None),
            }
            
            mlx_config = {
                'time_as_token': getattr(mlx_dit, 'time_as_token', None),
                'style_as_token': getattr(mlx_dit, 'style_as_token', None),
                'uvit_skip_connection': getattr(mlx_dit, 'uvit_skip_connection', None),
                'long_skip_connection': getattr(mlx_dit, 'long_skip_connection', None),
            }
            
            for key in pytorch_config:
                pytorch_val = pytorch_config[key]
                mlx_val = mlx_config[key]
                if pytorch_val == mlx_val:
                    print(f"    ✅ {key}: {pytorch_val}")
                else:
                    print(f"    ❌ {key}: PyTorch={pytorch_val}, MLX={mlx_val}")
            
        except Exception as e:
            print(f"❌ DiT 架构差异分析失败: {e}")
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
    analyzer = DiTCompleteForwardAnalyzer()
    analyzer.analyze_dit_complete_forward()

if __name__ == "__main__":
    main()
