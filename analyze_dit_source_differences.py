#!/usr/bin/env python3
"""
DiT 源码差异分析工具
对比 PyTorch 和 MLX DiT 的源码实现差异
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

class DiTSourceDifferenceAnalyzer:
    def __init__(self, cache_dir: str = "cfm_production_cache"):
        self.cache_dir = cache_dir
        
    def analyze_dit_source_differences(self):
        """分析 DiT 源码差异"""
        print("🔍 DiT 源码差异分析工具")
        print("="*60)
        
        # 1. 分析关键实现差异
        self._analyze_key_implementation_differences()
        
        # 2. 分析数值精度差异
        self._analyze_numerical_precision_differences()
        
        # 3. 分析具体计算步骤差异
        self._analyze_specific_calculation_differences()
        
    def _analyze_key_implementation_differences(self):
        """分析关键实现差异"""
        print(f"\n{'='*60}")
        print(f"🔍 关键实现差异分析")
        print(f"{'='*60}")
        
        print(f"\n📊 PyTorch DiT 关键实现:")
        print(f"  1. 转置操作: x.transpose(1, 2)")
        print(f"  2. 拼接操作: torch.cat([x_t, prompt_x_t, cond_proj], dim=-1)")
        print(f"  3. 风格条件: style[:, None, :].repeat(1, T, 1)")
        print(f"  4. 条件合并: self.cond_x_merge_linear(x_in)")
        print(f"  5. 时间嵌入: self.t_embedder(t)")
        print(f"  6. Transformer: self.transformer(x_in, t1.unsqueeze(1), input_pos, x_mask_expanded)")
        print(f"  7. Skip Connection: self.skip_linear(torch.cat([x_res, x_t], dim=-1))")
        print(f"  8. Final Layer: self.final_layer(x, t1).transpose(1, 2)")
        print(f"  9. 最终转置: x.transpose(1, 2)")
        
        print(f"\n📊 MLX DiT 关键实现:")
        print(f"  1. 转置操作: x.transpose(0, 2, 1)")
        print(f"  2. 拼接操作: mx.concatenate([x_t, prompt_x_t, cond_proj], axis=-1)")
        print(f"  3. 风格条件: mx.broadcast_to(style.reshape(batch, 1, -1), (batch, seq_len, style.shape[-1]))")
        print(f"  4. 条件合并: self.cond_x_merge_linear(x_in)")
        print(f"  5. 时间嵌入: self.t_embedder(t)")
        print(f"  6. Transformer: self.transformer(x_in, t_emb, input_pos=input_pos, mask=mask_expanded)")
        print(f"  7. Skip Connection: self.skip_linear(mx.concatenate([x_res, x_t], axis=-1))")
        print(f"  8. Final Layer: self.final_layer(x_out, t_emb)")
        print(f"  9. 最终转置: x_out.transpose(0, 2, 1)")
        
        print(f"\n📊 关键差异分析:")
        print(f"  1. 转置操作: PyTorch 使用 (1, 2)，MLX 使用 (0, 2, 1)")
        print(f"  2. 拼接操作: PyTorch 使用 dim=-1，MLX 使用 axis=-1")
        print(f"  3. 风格条件: PyTorch 使用 repeat，MLX 使用 broadcast_to")
        print(f"  4. Transformer 调用: PyTorch 使用位置参数，MLX 使用关键字参数")
        print(f"  5. 最终转置: PyTorch 在 FinalLayer 内部转置，MLX 在外部转置")
        
    def _analyze_numerical_precision_differences(self):
        """分析数值精度差异"""
        print(f"\n{'='*60}")
        print(f"🔍 数值精度差异分析")
        print(f"{'='*60}")
        
        print(f"\n📊 可能的数值精度差异:")
        print(f"  1. 数据类型: PyTorch 默认 float32，MLX 可能使用不同精度")
        print(f"  2. 计算顺序: 不同框架的计算顺序可能导致精度累积误差")
        print(f"  3. 优化策略: 不同框架的优化策略可能影响数值稳定性")
        print(f"  4. 内存布局: 不同框架的内存布局可能影响计算精度")
        
        # 检查实际的数据类型
        try:
            from indextts.infer_v2 import IndexTTS2
            
            # 加载模型
            pytorch_tts = IndexTTS2(use_mlx=False)
            mlx_tts = IndexTTS2(use_mlx=True)
            
            pytorch_dit = pytorch_tts.s2mel.models['cfm'].estimator
            mlx_dit = mlx_tts.mlx_s2mel_cfm.estimator
            
            print(f"\n📊 实际数据类型检查:")
            
            # 检查 PyTorch 数据类型
            pytorch_params = list(pytorch_dit.parameters())
            if pytorch_params:
                pytorch_dtype = pytorch_params[0].dtype
                print(f"  PyTorch 参数数据类型: {pytorch_dtype}")
            
            # 检查 MLX 数据类型
            mlx_params = list(mlx_dit.parameters().values())
            if mlx_params:
                mlx_dtype = mlx_params[0].dtype
                print(f"  MLX 参数数据类型: {mlx_dtype}")
            
            # 检查设备
            pytorch_device = pytorch_params[0].device if pytorch_params else "unknown"
            print(f"  PyTorch 设备: {pytorch_device}")
            print(f"  MLX 设备: CPU (MLX 默认)")
            
        except Exception as e:
            print(f"❌ 数据类型检查失败: {e}")
    
    def _analyze_specific_calculation_differences(self):
        """分析具体计算步骤差异"""
        print(f"\n{'='*60}")
        print(f"🔍 具体计算步骤差异分析")
        print(f"{'='*60}")
        
        print(f"\n📊 潜在差异点分析:")
        
        print(f"\n  1. 转置操作差异:")
        print(f"     PyTorch: x.transpose(1, 2)  # 交换维度 1 和 2")
        print(f"     MLX:     x.transpose(0, 2, 1)  # 交换维度 0 和 2")
        print(f"     影响: 如果输入形状不同，转置结果可能不同")
        
        print(f"\n  2. 拼接操作差异:")
        print(f"     PyTorch: torch.cat([x_t, prompt_x_t, cond_proj], dim=-1)")
        print(f"     MLX:     mx.concatenate([x_t, prompt_x_t, cond_proj], axis=-1)")
        print(f"     影响: 参数名称不同，但功能相同")
        
        print(f"\n  3. 风格条件差异:")
        print(f"     PyTorch: style[:, None, :].repeat(1, T, 1)")
        print(f"     MLX:     mx.broadcast_to(style.reshape(batch, 1, -1), (batch, seq_len, style.shape[-1]))")
        print(f"     影响: 实现方式不同，但结果应该相同")
        
        print(f"\n  4. Transformer 调用差异:")
        print(f"     PyTorch: self.transformer(x_in, t1.unsqueeze(1), input_pos, x_mask_expanded)")
        print(f"     MLX:     self.transformer(x_in, t_emb, input_pos=input_pos, mask=mask_expanded)")
        print(f"     影响: 参数传递方式不同，可能影响内部计算")
        
        print(f"\n  5. 最终转置差异:")
        print(f"     PyTorch: 在 FinalLayer 内部进行 transpose(1, 2)")
        print(f"     MLX:     在外部进行 transpose(0, 2, 1)")
        print(f"     影响: 转置时机不同，可能影响最终结果")
        
        print(f"\n📊 最可能的差异原因:")
        print(f"  1. 转置操作: PyTorch 和 MLX 的转置操作可能处理不同形状的输入")
        print(f"  2. Transformer 调用: 参数传递方式不同可能导致内部计算差异")
        print(f"  3. 最终转置: 转置时机不同可能影响最终结果")
        print(f"  4. 数值精度: 不同框架的数值精度处理可能不同")
        
    def _test_specific_differences(self):
        """测试具体差异"""
        print(f"\n{'='*60}")
        print(f"🔍 具体差异测试")
        print(f"{'='*60}")
        
        try:
            # 创建测试数据
            batch_size = 2
            seq_len = 415
            in_channels = 80
            hidden_dim = 512
            
            # 创建测试张量
            x_pt = torch.randn(batch_size, in_channels, seq_len)
            x_mx = mx.array(x_pt.numpy())
            
            print(f"\n📊 转置操作测试:")
            print(f"  输入形状: {x_pt.shape}")
            
            # PyTorch 转置
            x_t_pt = x_pt.transpose(1, 2)
            print(f"  PyTorch transpose(1, 2): {x_t_pt.shape}")
            
            # MLX 转置
            x_t_mx = x_mx.transpose(0, 2, 1)
            print(f"  MLX transpose(0, 2, 1): {x_t_mx.shape}")
            
            # 比较结果
            x_t_pt_np = x_t_pt.detach().cpu().numpy()
            x_t_mx_np = np.array(x_t_mx)
            
            if x_t_pt_np.shape == x_t_mx_np.shape:
                diff = np.abs(x_t_pt_np - x_t_mx_np)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                print(f"  转置差异: max={max_diff:.6f}, mean={mean_diff:.6f}")
                
                if max_diff < 1e-6:
                    print(f"  ✅ 转置操作完全一致")
                else:
                    print(f"  ❌ 转置操作存在差异")
            else:
                print(f"  ❌ 转置结果形状不匹配")
            
        except Exception as e:
            print(f"❌ 具体差异测试失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    analyzer = DiTSourceDifferenceAnalyzer()
    analyzer.analyze_dit_source_differences()

if __name__ == "__main__":
    main()







