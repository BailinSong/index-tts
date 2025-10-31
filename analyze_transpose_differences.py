#!/usr/bin/env python3
"""
转置操作差异分析工具
专门测试 PyTorch 和 MLX 转置操作的差异
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

class TransposeDifferenceAnalyzer:
    def __init__(self, cache_dir: str = "cfm_production_cache"):
        self.cache_dir = cache_dir
        
    def analyze_transpose_differences(self):
        """分析转置操作差异"""
        print("🔍 转置操作差异分析工具")
        print("="*60)
        
        # 1. 测试基本转置操作
        self._test_basic_transpose_operations()
        
        # 2. 测试实际 DiT 转置操作
        self._test_dit_transpose_operations()
        
        # 3. 分析转置时机差异
        self._analyze_transpose_timing_differences()
        
    def _test_basic_transpose_operations(self):
        """测试基本转置操作"""
        print(f"\n{'='*60}")
        print(f"🔍 基本转置操作测试")
        print(f"{'='*60}")
        
        # 创建测试数据
        batch_size = 2
        in_channels = 80
        seq_len = 415
        
        print(f"\n📊 测试数据形状: ({batch_size}, {in_channels}, {seq_len})")
        
        # 创建 PyTorch 张量
        x_pt = torch.randn(batch_size, in_channels, seq_len)
        print(f"  PyTorch 输入: {x_pt.shape}")
        
        # 创建 MLX 张量
        x_mx = mx.array(x_pt.numpy())
        print(f"  MLX 输入: {x_mx.shape}")
        
        # PyTorch 转置操作
        print(f"\n📊 PyTorch 转置操作:")
        x_t_pt = x_pt.transpose(1, 2)
        print(f"  x.transpose(1, 2): {x_t_pt.shape}")
        
        # MLX 转置操作
        print(f"\n📊 MLX 转置操作:")
        x_t_mx = x_mx.transpose(0, 2, 1)
        print(f"  x.transpose(0, 2, 1): {x_t_mx.shape}")
        
        # 比较结果
        print(f"\n📊 转置结果比较:")
        x_t_pt_np = x_t_pt.detach().cpu().numpy()
        x_t_mx_np = np.array(x_t_mx)
        
        print(f"  PyTorch 结果: {x_t_pt_np.shape}")
        print(f"  MLX 结果: {x_t_mx_np.shape}")
        
        if x_t_pt_np.shape == x_t_mx_np.shape:
            diff = np.abs(x_t_pt_np - x_t_mx_np)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            print(f"  差异: max={max_diff:.6f}, mean={mean_diff:.6f}")
            
            if max_diff < 1e-6:
                print(f"  ✅ 转置操作完全一致")
            else:
                print(f"  ❌ 转置操作存在差异")
        else:
            print(f"  ❌ 转置结果形状不匹配")
            
        # 测试反向转置
        print(f"\n📊 反向转置测试:")
        
        # PyTorch 反向转置
        x_back_pt = x_t_pt.transpose(1, 2)
        print(f"  PyTorch 反向转置: {x_back_pt.shape}")
        
        # MLX 反向转置
        x_back_mx = x_t_mx.transpose(0, 2, 1)
        print(f"  MLX 反向转置: {x_back_mx.shape}")
        
        # 比较反向转置结果
        x_back_pt_np = x_back_pt.detach().cpu().numpy()
        x_back_mx_np = np.array(x_back_mx)
        
        if x_back_pt_np.shape == x_back_mx_np.shape:
            diff_back = np.abs(x_back_pt_np - x_back_mx_np)
            max_diff_back = np.max(diff_back)
            mean_diff_back = np.mean(diff_back)
            print(f"  反向转置差异: max={max_diff_back:.6f}, mean={mean_diff_back:.6f}")
            
            if max_diff_back < 1e-6:
                print(f"  ✅ 反向转置操作完全一致")
            else:
                print(f"  ❌ 反向转置操作存在差异")
        else:
            print(f"  ❌ 反向转置结果形状不匹配")
    
    def _test_dit_transpose_operations(self):
        """测试实际 DiT 转置操作"""
        print(f"\n{'='*60}")
        print(f"🔍 DiT 转置操作测试")
        print(f"{'='*60}")
        
        try:
            from indextts.infer_v2 import IndexTTS2
            
            # 加载模型
            pytorch_tts = IndexTTS2(use_mlx=False)
            mlx_tts = IndexTTS2(use_mlx=True)
            
            pytorch_dit = pytorch_tts.s2mel.models['cfm'].estimator
            mlx_dit = mlx_tts.mlx_s2mel_cfm.estimator
            
            # 加载测试数据
            dit_input_files = glob.glob(os.path.join(self.cache_dir, "*_dit_input_*.pkl"))
            if not dit_input_files:
                print("❌ 未找到 DiT 输入缓存文件")
                return
            
            latest_file = max(dit_input_files, key=os.path.getctime)
            with open(latest_file, 'rb') as f:
                test_data = pickle.load(f)
            
            print(f"✅ 加载测试数据: {os.path.basename(latest_file)}")
            
            # 准备输入数据
            x = test_data['x']
            prompt_x = test_data['prompt_x']
            mu = test_data['mu']
            style = test_data['style']
            t = test_data['t']
            x_lens = test_data['x_lens']
            
            print(f"\n📊 输入数据形状:")
            print(f"  x: {x.shape}")
            print(f"  prompt_x: {prompt_x.shape}")
            print(f"  mu: {mu.shape}")
            print(f"  style: {style.shape}")
            print(f"  t: {t.shape}")
            print(f"  x_lens: {x_lens.shape}")
            
            # 转换为 PyTorch 格式
            device = next(pytorch_dit.parameters()).device
            x_pt = torch.as_tensor(x).to(device)
            prompt_x_pt = torch.as_tensor(prompt_x).to(device)
            
            # 转换为 MLX 格式
            x_mx = mx.array(x)
            prompt_x_mx = mx.array(prompt_x)
            
            print(f"\n📊 DiT 转置操作测试:")
            
            # 测试 x 转置
            print(f"\n  1. x 转置测试:")
            x_t_pt = x_pt.transpose(1, 2)
            x_t_mx = x_mx.transpose(0, 2, 1)
            
            print(f"    PyTorch x.transpose(1, 2): {x_t_pt.shape}")
            print(f"    MLX x.transpose(0, 2, 1): {x_t_mx.shape}")
            
            # 比较 x 转置结果
            x_t_pt_np = x_t_pt.detach().cpu().numpy()
            x_t_mx_np = np.array(x_t_mx)
            
            if x_t_pt_np.shape == x_t_mx_np.shape:
                diff_x = np.abs(x_t_pt_np - x_t_mx_np)
                max_diff_x = np.max(diff_x)
                mean_diff_x = np.mean(diff_x)
                print(f"    x 转置差异: max={max_diff_x:.6f}, mean={mean_diff_x:.6f}")
                
                if max_diff_x < 1e-6:
                    print(f"    ✅ x 转置操作完全一致")
                else:
                    print(f"    ❌ x 转置操作存在差异")
            else:
                print(f"    ❌ x 转置结果形状不匹配")
            
            # 测试 prompt_x 转置
            print(f"\n  2. prompt_x 转置测试:")
            prompt_x_t_pt = prompt_x_pt.transpose(1, 2)
            prompt_x_t_mx = prompt_x_mx.transpose(0, 2, 1)
            
            print(f"    PyTorch prompt_x.transpose(1, 2): {prompt_x_t_pt.shape}")
            print(f"    MLX prompt_x.transpose(0, 2, 1): {prompt_x_t_mx.shape}")
            
            # 比较 prompt_x 转置结果
            prompt_x_t_pt_np = prompt_x_t_pt.detach().cpu().numpy()
            prompt_x_t_mx_np = np.array(prompt_x_t_mx)
            
            if prompt_x_t_pt_np.shape == prompt_x_t_mx_np.shape:
                diff_prompt_x = np.abs(prompt_x_t_pt_np - prompt_x_t_mx_np)
                max_diff_prompt_x = np.max(diff_prompt_x)
                mean_diff_prompt_x = np.mean(diff_prompt_x)
                print(f"    prompt_x 转置差异: max={max_diff_prompt_x:.6f}, mean={mean_diff_prompt_x:.6f}")
                
                if max_diff_prompt_x < 1e-6:
                    print(f"    ✅ prompt_x 转置操作完全一致")
                else:
                    print(f"    ❌ prompt_x 转置操作存在差异")
            else:
                print(f"    ❌ prompt_x 转置结果形状不匹配")
            
        except Exception as e:
            print(f"❌ DiT 转置操作测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_transpose_timing_differences(self):
        """分析转置时机差异"""
        print(f"\n{'='*60}")
        print(f"🔍 转置时机差异分析")
        print(f"{'='*60}")
        
        print(f"\n📊 PyTorch DiT 转置时机:")
        print(f"  1. 输入转置: x.transpose(1, 2) 和 prompt_x.transpose(1, 2)")
        print(f"  2. 中间转置: 在 FinalLayer 内部进行 transpose(1, 2)")
        print(f"  3. 最终转置: 在 DiT 外部进行 transpose(1, 2)")
        
        print(f"\n📊 MLX DiT 转置时机:")
        print(f"  1. 输入转置: x.transpose(0, 2, 1) 和 prompt_x.transpose(0, 2, 1)")
        print(f"  2. 中间转置: 在 FinalLayer 内部不进行转置")
        print(f"  3. 最终转置: 在 DiT 外部进行 transpose(0, 2, 1)")
        
        print(f"\n📊 转置时机差异分析:")
        print(f"  1. 输入转置: 两个版本都在开始时进行转置，但转置方式不同")
        print(f"  2. 中间转置: PyTorch 在 FinalLayer 内部转置，MLX 不转置")
        print(f"  3. 最终转置: 两个版本都在最后进行转置，但转置方式不同")
        
        print(f"\n📊 潜在问题:")
        print(f"  1. 转置方式不同: PyTorch 使用 (1, 2)，MLX 使用 (0, 2, 1)")
        print(f"  2. 转置时机不同: PyTorch 在 FinalLayer 内部转置，MLX 在外部转置")
        print(f"  3. 形状处理不同: 不同框架可能处理不同形状的输入")
        
        print(f"\n📊 建议:")
        print(f"  1. 检查转置操作的具体实现")
        print(f"  2. 验证转置时机的影响")
        print(f"  3. 测试不同形状输入的转置结果")

def main():
    """主函数"""
    analyzer = TransposeDifferenceAnalyzer()
    analyzer.analyze_transpose_differences()

if __name__ == "__main__":
    main()







