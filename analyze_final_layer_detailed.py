#!/usr/bin/env python3
"""
FinalLayer 详细差异分析工具
深入分析 PyTorch 和 MLX FinalLayer 的具体实现差异
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

class FinalLayerDetailedAnalyzer:
    def __init__(self, cache_dir: str = "cfm_production_cache"):
        self.cache_dir = cache_dir
        
    def analyze_final_layer_detailed(self):
        """分析 FinalLayer 详细差异"""
        print("🔍 FinalLayer 详细差异分析工具")
        print("="*60)
        
        # 1. 加载生产环境模型
        pytorch_dit, mlx_dit = self._load_production_models()
        if pytorch_dit is None or mlx_dit is None:
            return
        
        # 2. 准备测试数据
        test_data = self._prepare_test_data()
        if test_data is None:
            return
        
        # 3. 分析 FinalLayer 详细差异
        self._analyze_final_layer_structure(pytorch_dit, mlx_dit)
        self._analyze_final_layer_weights_detailed(pytorch_dit, mlx_dit)
        self._analyze_final_layer_computation_step_by_step(pytorch_dit, mlx_dit, test_data)
        self._analyze_modulate_function_differences(pytorch_dit, mlx_dit, test_data)
        
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
    
    def _analyze_final_layer_structure(self, pytorch_dit, mlx_dit):
        """分析 FinalLayer 结构差异"""
        print(f"\n{'='*60}")
        print(f"🔍 FinalLayer 结构分析")
        print(f"{'='*60}")
        
        try:
            # PyTorch FinalLayer
            pytorch_final_layer = pytorch_dit.final_layer
            print(f"\n📊 PyTorch FinalLayer 结构:")
            print(f"  类型: {type(pytorch_final_layer)}")
            print(f"  模块列表:")
            for name, module in pytorch_final_layer.named_children():
                print(f"    {name}: {type(module)}")
                if hasattr(module, 'weight'):
                    print(f"      权重形状: {module.weight.shape}")
                if hasattr(module, 'bias'):
                    print(f"      偏置形状: {module.bias.shape}")
            
            # MLX FinalLayer
            mlx_final_layer = mlx_dit.final_layer
            print(f"\n📊 MLX FinalLayer 结构:")
            print(f"  类型: {type(mlx_final_layer)}")
            print(f"  模块列表:")
            for name, module in mlx_final_layer.named_children():
                print(f"    {name}: {type(module)}")
                if hasattr(module, 'weight'):
                    print(f"      权重形状: {module.weight.shape}")
                if hasattr(module, 'bias'):
                    print(f"      偏置形状: {module.bias.shape}")
            
        except Exception as e:
            print(f"❌ FinalLayer 结构分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_final_layer_weights_detailed(self, pytorch_dit, mlx_dit):
        """分析 FinalLayer 权重详细差异"""
        print(f"\n{'='*60}")
        print(f"🔍 FinalLayer 权重详细分析")
        print(f"{'='*60}")
        
        try:
            pytorch_final_layer = pytorch_dit.final_layer
            mlx_final_layer = mlx_dit.final_layer
            
            # 分析 norm_final
            print(f"\n📊 norm_final 权重分析:")
            self._compare_layer_weights("norm_final", 
                                     pytorch_final_layer.norm_final, 
                                     mlx_final_layer.norm_final)
            
            # 分析 linear
            print(f"\n📊 linear 权重分析:")
            self._compare_layer_weights("linear", 
                                     pytorch_final_layer.linear, 
                                     mlx_final_layer.linear)
            
            # 分析 adaLN_modulation
            print(f"\n📊 adaLN_modulation 权重分析:")
            self._compare_layer_weights("adaLN_modulation", 
                                     pytorch_final_layer.adaLN_modulation, 
                                     mlx_final_layer.adaLN_modulation)
            
        except Exception as e:
            print(f"❌ FinalLayer 权重详细分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _compare_layer_weights(self, name, pytorch_layer, mlx_layer):
        """比较层权重"""
        try:
            print(f"  {name} 权重比较:")
            
            # PyTorch 权重
            pytorch_weights = {}
            for param_name, param in pytorch_layer.named_parameters():
                pytorch_weights[param_name] = param.detach().cpu().numpy()
                print(f"    PyTorch {param_name}: {param.shape}, range=[{param.min().item():.6f}, {param.max().item():.6f}]")
            
            # MLX 权重
            mlx_weights = {}
            try:
                for param_name, param in mlx_layer.named_parameters():
                    mlx_weights[param_name] = np.array(param)
                    param_np = np.array(param)
                    print(f"    MLX {param_name}: {param.shape}, range=[{np.min(param_np):.6f}, {np.max(param_np):.6f}]")
            except AttributeError:
                # MLX 可能不支持 named_parameters
                for param_name, param in mlx_layer.parameters().items():
                    mlx_weights[param_name] = np.array(param)
                    param_np = np.array(param)
                    print(f"    MLX {param_name}: {param.shape}, range=[{np.min(param_np):.6f}, {np.max(param_np):.6f}]")
            
            # 比较权重
            common_keys = set(pytorch_weights.keys()) & set(mlx_weights.keys())
            print(f"    共同权重数量: {len(common_keys)}")
            
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
            print(f"    ❌ {name} 权重比较失败: {e}")
    
    def _analyze_final_layer_computation_step_by_step(self, pytorch_dit, mlx_dit, test_data):
        """分析 FinalLayer 计算步骤差异"""
        print(f"\n{'='*60}")
        print(f"🔍 FinalLayer 计算步骤分析")
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
        
        print("🔧 准备 FinalLayer 输入...")
        
        try:
            # 模拟到 FinalLayer 的输入
            # 这里我们使用一个简化的方法：直接使用 DiT 的完整前向传播
            with torch.no_grad():
                pytorch_output = pytorch_dit(x_pt, prompt_x_pt, x_lens_pt, t_pt, style_pt, mu_pt)
            
            mlx_output = mlx_dit(x_mx, prompt_x_mx, x_lens_mx, t_mx, style_mx, mu_mx)
            
            # 反推 FinalLayer 的输入（假设是最终输出的转置）
            transformer_output_pt = pytorch_output.transpose(1, 2)  # (B, T, C)
            transformer_output_mx = mlx_output.transpose(0, 2, 1)   # (B, T, C)
            
            # 时间嵌入
            t_emb_pt = pytorch_dit.t_embedder(t_pt)
            t_emb_mx = mlx_dit.t_embedder(t_mx)
            
            print(f"  Transformer 输出形状: PyTorch={transformer_output_pt.shape}, MLX={transformer_output_mx.shape}")
            print(f"  时间嵌入形状: PyTorch={t_emb_pt.shape}, MLX={t_emb_mx.shape}")
            
            # 分析 FinalLayer 的各个计算步骤
            self._analyze_final_layer_steps(pytorch_dit.final_layer, mlx_dit.final_layer, 
                                          transformer_output_pt, t_emb_pt, 
                                          transformer_output_mx, t_emb_mx)
            
        except Exception as e:
            print(f"❌ FinalLayer 计算步骤分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_final_layer_steps(self, pytorch_final_layer, mlx_final_layer, 
                                 transformer_output_pt, t_emb_pt, 
                                 transformer_output_mx, t_emb_mx):
        """分析 FinalLayer 的各个计算步骤"""
        print(f"\n📊 FinalLayer 计算步骤详细分析:")
        
        try:
            # 步骤 1: adaLN_modulation
            print(f"\n  步骤 1: adaLN_modulation")
            with torch.no_grad():
                pytorch_c_emb = pytorch_final_layer.adaLN_modulation(t_emb_pt)
            mlx_c_emb = mlx_final_layer.adaLN_modulation(t_emb_mx)
            
            print(f"    PyTorch c_emb: {pytorch_c_emb.shape}, range=[{pytorch_c_emb.min().item():.6f}, {pytorch_c_emb.max().item():.6f}]")
            mlx_c_emb_np = np.array(mlx_c_emb)
            print(f"    MLX c_emb: {mlx_c_emb.shape}, range=[{np.min(mlx_c_emb_np):.6f}, {np.max(mlx_c_emb_np):.6f}]")
            
            # 比较 c_emb
            self._compare_tensors("c_emb", pytorch_c_emb, mlx_c_emb)
            
            # 步骤 2: chunk/split 操作
            print(f"\n  步骤 2: chunk/split 操作")
            with torch.no_grad():
                pytorch_shift, pytorch_scale = pytorch_c_emb.chunk(2, dim=1)
            mlx_shift, mlx_scale = mx.split(mlx_c_emb, 2, axis=-1)
            
            print(f"    PyTorch shift: {pytorch_shift.shape}, range=[{pytorch_shift.min().item():.6f}, {pytorch_shift.max().item():.6f}]")
            print(f"    PyTorch scale: {pytorch_scale.shape}, range=[{pytorch_scale.min().item():.6f}, {pytorch_scale.max().item():.6f}]")
            
            mlx_shift_np = np.array(mlx_shift)
            mlx_scale_np = np.array(mlx_scale)
            print(f"    MLX shift: {mlx_shift.shape}, range=[{np.min(mlx_shift_np):.6f}, {np.max(mlx_shift_np):.6f}]")
            print(f"    MLX scale: {mlx_scale.shape}, range=[{np.min(mlx_scale_np):.6f}, {np.max(mlx_scale_np):.6f}]")
            
            # 比较 shift 和 scale
            self._compare_tensors("shift", pytorch_shift, mlx_shift)
            self._compare_tensors("scale", pytorch_scale, mlx_scale)
            
            # 步骤 3: norm_final
            print(f"\n  步骤 3: norm_final")
            with torch.no_grad():
                pytorch_x_norm = pytorch_final_layer.norm_final(transformer_output_pt)
            mlx_x_norm = mlx_final_layer.norm_final(transformer_output_mx)
            
            print(f"    PyTorch x_norm: {pytorch_x_norm.shape}, range=[{pytorch_x_norm.min().item():.6f}, {pytorch_x_norm.max().item():.6f}]")
            mlx_x_norm_np = np.array(mlx_x_norm)
            print(f"    MLX x_norm: {mlx_x_norm.shape}, range=[{np.min(mlx_x_norm_np):.6f}, {np.max(mlx_x_norm_np):.6f}]")
            
            # 比较 x_norm
            self._compare_tensors("x_norm", pytorch_x_norm, mlx_x_norm)
            
            # 步骤 4: modulate 操作
            print(f"\n  步骤 4: modulate 操作")
            with torch.no_grad():
                pytorch_x_modulated = pytorch_x_norm * (1 + pytorch_scale.unsqueeze(1)) + pytorch_shift.unsqueeze(1)
            
            # MLX modulate 操作
            batch_size = mlx_scale.shape[0]
            mlx_scale_reshaped = mlx_scale.reshape(batch_size, 1, -1)
            mlx_shift_reshaped = mlx_shift.reshape(batch_size, 1, -1)
            mlx_x_modulated = mlx_x_norm * (1 + mlx_scale_reshaped) + mlx_shift_reshaped
            
            print(f"    PyTorch x_modulated: {pytorch_x_modulated.shape}, range=[{pytorch_x_modulated.min().item():.6f}, {pytorch_x_modulated.max().item():.6f}]")
            mlx_x_modulated_np = np.array(mlx_x_modulated)
            print(f"    MLX x_modulated: {mlx_x_modulated.shape}, range=[{np.min(mlx_x_modulated_np):.6f}, {np.max(mlx_x_modulated_np):.6f}]")
            
            # 比较 x_modulated
            self._compare_tensors("x_modulated", pytorch_x_modulated, mlx_x_modulated)
            
            # 步骤 5: linear 层
            print(f"\n  步骤 5: linear 层")
            with torch.no_grad():
                pytorch_final_output = pytorch_final_layer.linear(pytorch_x_modulated)
            mlx_final_output = mlx_final_layer.linear(mlx_x_modulated)
            
            print(f"    PyTorch final_output: {pytorch_final_output.shape}, range=[{pytorch_final_output.min().item():.6f}, {pytorch_final_output.max().item():.6f}]")
            mlx_final_output_np = np.array(mlx_final_output)
            print(f"    MLX final_output: {mlx_final_output.shape}, range=[{np.min(mlx_final_output_np):.6f}, {np.max(mlx_final_output_np):.6f}]")
            
            # 比较 final_output
            self._compare_tensors("final_output", pytorch_final_output, mlx_final_output)
            
        except Exception as e:
            print(f"    ❌ FinalLayer 计算步骤分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_modulate_function_differences(self, pytorch_dit, mlx_dit, test_data):
        """分析 modulate 函数差异"""
        print(f"\n{'='*60}")
        print(f"🔍 modulate 函数差异分析")
        print(f"{'='*60}")
        
        try:
            # 创建测试数据
            batch_size = 2
            seq_len = 415
            hidden_size = 512
            
            # 创建测试张量
            x_pt = torch.randn(batch_size, seq_len, hidden_size)
            shift_pt = torch.randn(batch_size, hidden_size)
            scale_pt = torch.randn(batch_size, hidden_size)
            
            x_mx = mx.array(x_pt.numpy())
            shift_mx = mx.array(shift_pt.numpy())
            scale_mx = mx.array(scale_pt.numpy())
            
            print(f"测试数据形状: x={x_pt.shape}, shift={shift_pt.shape}, scale={scale_pt.shape}")
            
            # PyTorch modulate
            pytorch_result = x_pt * (1 + scale_pt.unsqueeze(1)) + shift_pt.unsqueeze(1)
            
            # MLX modulate
            batch_size = scale_mx.shape[0]
            scale_reshaped = scale_mx.reshape(batch_size, 1, -1)
            shift_reshaped = shift_mx.reshape(batch_size, 1, -1)
            mlx_result = x_mx * (1 + scale_reshaped) + shift_reshaped
            
            print(f"PyTorch modulate 结果: {pytorch_result.shape}, range=[{pytorch_result.min().item():.6f}, {pytorch_result.max().item():.6f}]")
            mlx_result_np = np.array(mlx_result)
            print(f"MLX modulate 结果: {mlx_result.shape}, range=[{np.min(mlx_result_np):.6f}, {np.max(mlx_result_np):.6f}]")
            
            # 比较结果
            self._compare_tensors("modulate_result", pytorch_result, mlx_result)
            
        except Exception as e:
            print(f"❌ modulate 函数差异分析失败: {e}")
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
    analyzer = FinalLayerDetailedAnalyzer()
    analyzer.analyze_final_layer_detailed()

if __name__ == "__main__":
    main()







