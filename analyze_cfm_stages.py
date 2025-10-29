#!/usr/bin/env python3
"""
CFM 阶段分析工具
逐个阶段分析 PyTorch 和 MLX CFM 的输入输出数据
"""

import os
import pickle
import glob
import torch
import mlx.core as mx
import numpy as np
from typing import Dict, Any, List, Tuple

class CFMStageAnalyzer:
    def __init__(self, cache_dir: str = "cfm_production_cache", consistent_test_file: str = "cfm_consistent_test/consistent_test_data.pkl"):
        self.cache_dir = cache_dir
        self.consistent_test_file = consistent_test_file
        
    def find_latest_cache_files(self) -> Tuple[str, str, str, str]:
        """找到最新的缓存文件"""
        pytorch_inputs = self._find_latest_file("cfm_pytorch_inputs_*.pkl")
        pytorch_outputs = self._find_latest_file("cfm_pytorch_output_*.pkl")
        mlx_inputs = self._find_latest_file("cfm_mlx_inputs_*.pkl")
        mlx_outputs = self._find_latest_file("cfm_mlx_output_*.pkl")
        
        return pytorch_inputs, pytorch_outputs, mlx_inputs, mlx_outputs
    
    def _find_latest_file(self, pattern: str) -> str:
        """找到最新的匹配文件"""
        files = glob.glob(os.path.join(self.cache_dir, pattern))
        if not files:
            return None
        return max(files, key=os.path.getctime)
    
    def load_cache_file(self, filepath: str) -> Dict[str, Any]:
        """加载缓存文件"""
        if not filepath or not os.path.exists(filepath):
            return None
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def load_consistent_test_data(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """加载一致的测试数据"""
        if not os.path.exists(self.consistent_test_file):
            print(f"❌ 一致测试数据文件不存在: {self.consistent_test_file}")
            print("💡 请先运行 analyze_cfm_consistency.py 生成一致的测试数据")
            return None, None
        
        with open(self.consistent_test_file, 'rb') as f:
            data = pickle.load(f)
        
        pytorch_inputs = data['pytorch_inputs']
        mlx_inputs = data['mlx_inputs']
        
        print(f"✅ 加载一致的测试数据")
        pytorch_shapes = [f'{k}: {v.shape if hasattr(v, "shape") else v}' for k, v in pytorch_inputs.items()]
        mlx_shapes = [f'{k}: {v.shape if hasattr(v, "shape") else v}' for k, v in mlx_inputs.items()]
        print(f"   PyTorch 输入形状: {pytorch_shapes}")
        print(f"   MLX 输入形状: {mlx_shapes}")
        
        return pytorch_inputs, mlx_inputs
    
    def analyze_tensor_stats(self, tensor, name: str) -> Dict[str, float]:
        """分析张量统计信息"""
        if tensor is None:
            return {"min": 0, "max": 0, "avg": 0, "std": 0}
        
        if isinstance(tensor, torch.Tensor):
            # 检查是否为整数类型
            if tensor.dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
                return {
                    "min": float(tensor.min().item()),
                    "max": float(tensor.max().item()),
                    "avg": float(tensor.float().mean().item()),
                    "std": float(tensor.float().std().item())
                }
            else:
                return {
                    "min": float(tensor.min().item()),
                    "max": float(tensor.max().item()),
                    "avg": float(tensor.mean().item()),
                    "std": float(tensor.std().item())
                }
        elif isinstance(tensor, mx.array):
            # 检查是否为整数类型
            if tensor.dtype in [mx.int64, mx.int32, mx.int16, mx.int8]:
                return {
                    "min": float(mx.min(tensor)),
                    "max": float(mx.max(tensor)),
                    "avg": float(mx.mean(tensor.astype(mx.float32))),
                    "std": float(mx.std(tensor.astype(mx.float32)))
                }
            else:
                return {
                    "min": float(mx.min(tensor)),
                    "max": float(mx.max(tensor)),
                    "avg": float(mx.mean(tensor)),
                    "std": float(mx.std(tensor))
                }
        elif isinstance(tensor, np.ndarray):
            # 检查是否为整数类型
            if np.issubdtype(tensor.dtype, np.integer):
                return {
                    "min": float(np.min(tensor)),
                    "max": float(np.max(tensor)),
                    "avg": float(np.mean(tensor.astype(np.float32))),
                    "std": float(np.std(tensor.astype(np.float32)))
                }
            else:
                return {
                    "min": float(np.min(tensor)),
                    "max": float(np.max(tensor)),
                    "avg": float(np.mean(tensor)),
                    "std": float(np.std(tensor))
                }
        else:
            return {"min": 0, "max": 0, "avg": 0, "std": 0}
    
    def print_stage_stats(self, stage_name: str, pytorch_data: Dict, mlx_data: Dict):
        """打印阶段统计信息"""
        print(f"\n{'='*60}")
        print(f"🔍 {stage_name} 阶段分析")
        print(f"{'='*60}")
        
        # PyTorch 输入分析
        if pytorch_data:
            print(f"\n📊 PyTorch {stage_name} Input:")
            for key, value in pytorch_data.items():
                if isinstance(value, (torch.Tensor, mx.array, np.ndarray)):
                    stats = self.analyze_tensor_stats(value, key)
                    print(f"   {key}: shape={value.shape}, min={stats['min']:.6f}, max={stats['max']:.6f}, avg={stats['avg']:.6f}")
                else:
                    print(f"   {key}: {value}")
        
        # MLX 输入分析
        if mlx_data:
            print(f"\n📊 MLX {stage_name} Input:")
            for key, value in mlx_data.items():
                if isinstance(value, (torch.Tensor, mx.array, np.ndarray)):
                    stats = self.analyze_tensor_stats(value, key)
                    print(f"   {key}: shape={value.shape}, min={stats['min']:.6f}, max={stats['max']:.6f}, avg={stats['avg']:.6f}")
                else:
                    print(f"   {key}: {value}")
    
    def analyze_initialization_noise(self):
        """分析初始化噪声阶段"""
        print("🔍 分析初始化噪声阶段...")
        
        # 加载一致的测试数据
        pytorch_data, mlx_data = self.load_consistent_test_data()
        
        if pytorch_data is None or mlx_data is None:
            print("❌ 无法加载一致的测试数据")
            return
        
        # 分析输入参数一致性
        print(f"\n{'='*60}")
        print(f"🔍 输入参数一致性分析 (使用同一个 PyTorch CFM 前级缓存)")
        print(f"{'='*60}")
        
        # 分析各个输入参数
        params_to_analyze = ['mu', 'prompt', 'style', 'x_lens']
        
        for param in params_to_analyze:
            pytorch_param = pytorch_data.get(param)
            mlx_param = mlx_data.get(param)
            
            if pytorch_param is not None and mlx_param is not None:
                print(f"\n📊 {param} 参数:")
                
                pytorch_stats = self.analyze_tensor_stats(pytorch_param, param)
                mlx_stats = self.analyze_tensor_stats(mlx_param, param)
                
                print(f"   PyTorch: shape={pytorch_param.shape}, min={pytorch_stats['min']:.6f}, max={pytorch_stats['max']:.6f}, avg={pytorch_stats['avg']:.6f}")
                print(f"   MLX:     shape={mlx_param.shape}, min={mlx_stats['min']:.6f}, max={mlx_stats['max']:.6f}, avg={mlx_stats['avg']:.6f}")
                
                # 比较差异
                if pytorch_param.shape == mlx_param.shape:
                    # 转换为相同类型进行比较
                    if isinstance(pytorch_param, torch.Tensor):
                        pytorch_np = pytorch_param.detach().cpu().numpy()
                    else:
                        pytorch_np = pytorch_param
                    
                    if isinstance(mlx_param, mx.array):
                        mlx_np = np.array(mlx_param)
                    else:
                        mlx_np = mlx_param
                    
                    diff = np.abs(pytorch_np - mlx_np)
                    max_diff = np.max(diff)
                    avg_diff = np.mean(diff)
                    
                    print(f"   差异: max={max_diff:.6f}, avg={avg_diff:.6f}")
                    
                    if max_diff < 1e-6:
                        print(f"   ✅ {param} 参数完全一致")
                    else:
                        print(f"   ⚠️  {param} 参数存在差异")
                else:
                    print(f"   ❌ {param} 形状不匹配")
                    print(f"   💡 这不应该发生，因为使用了同一个 PyTorch CFM 前级缓存")
    
    def analyze_all_stages(self):
        """分析所有阶段"""
        print("🔍 CFM 阶段分析工具")
        print("="*60)
        
        # 分析初始化噪声
        self.analyze_initialization_noise()
        
        # 分析其他输入参数
        self.analyze_input_parameters()
        
        # 分析输出结果
        self.analyze_output_results()
    
    def analyze_input_parameters(self):
        """分析输入参数"""
        print(f"\n{'='*60}")
        print(f"🔍 输入参数分析 (使用一致测试数据)")
        print(f"{'='*60}")
        
        # 加载一致的测试数据
        pytorch_data, mlx_data = self.load_consistent_test_data()
        
        if pytorch_data is None or mlx_data is None:
            print("❌ 无法加载一致的测试数据")
            return
        
        # 分析各个输入参数
        params = ['mu', 'prompt', 'style', 'x_lens']
        
        for param in params:
            pytorch_param = pytorch_data.get(param)
            mlx_param = mlx_data.get(param)
            
            if pytorch_param is not None and mlx_param is not None:
                print(f"\n📊 {param} 参数:")
                
                pytorch_stats = self.analyze_tensor_stats(pytorch_param, param)
                mlx_stats = self.analyze_tensor_stats(mlx_param, param)
                
                print(f"   PyTorch: shape={pytorch_param.shape}, min={pytorch_stats['min']:.6f}, max={pytorch_stats['max']:.6f}, avg={pytorch_stats['avg']:.6f}")
                print(f"   MLX:     shape={mlx_param.shape}, min={mlx_stats['min']:.6f}, max={mlx_stats['max']:.6f}, avg={mlx_stats['avg']:.6f}")
                
                # 比较差异
                if pytorch_param.shape == mlx_param.shape:
                    if isinstance(pytorch_param, torch.Tensor):
                        pytorch_np = pytorch_param.detach().cpu().numpy()
                    else:
                        pytorch_np = pytorch_param
                    
                    if isinstance(mlx_param, mx.array):
                        mlx_np = np.array(mlx_param)
                    else:
                        mlx_np = mlx_param
                    
                    diff = np.abs(pytorch_np - mlx_np)
                    max_diff = np.max(diff)
                    avg_diff = np.mean(diff)
                    
                    print(f"   差异: max={max_diff:.6f}, avg={avg_diff:.6f}")
                    
                    if max_diff < 1e-6:
                        print(f"   ✅ {param} 参数一致")
                    else:
                        print(f"   ❌ {param} 参数不一致")
                else:
                    print(f"   ❌ {param} 形状不匹配")
    
    def analyze_mu_length_decomposition(self):
        """基于一致测试数据分解 mu 的时间长度: mu_len = prompt_len + cond_len"""
        print(f"\n{'='*60}")
        print(f"🔍 mu 长度分解 (使用一致测试数据)")
        print(f"{'='*60}")

        pytorch_data, mlx_data = self.load_consistent_test_data()
        if pytorch_data is None or mlx_data is None:
            print("❌ 无法加载一致的测试数据")
            return

        # PyTorch
        mu_pt = pytorch_data.get('mu')
        prompt_pt = pytorch_data.get('prompt')
        if mu_pt is not None and prompt_pt is not None:
            mu_len_pt = int(mu_pt.shape[1])
            prompt_len_pt = int(prompt_pt.shape[2]) if len(prompt_pt.shape) == 3 else None
            cond_len_pt = mu_len_pt - (prompt_len_pt or 0)
            print(f"   PyTorch: mu_len={mu_len_pt}, prompt_len={prompt_len_pt}, cond_len={cond_len_pt}")

        # MLX
        mu_mx = mlx_data.get('mu')
        prompt_mx = mlx_data.get('prompt')
        if mu_mx is not None and prompt_mx is not None:
            mu_len_mx = int(mu_mx.shape[1])
            prompt_len_mx = int(prompt_mx.shape[2]) if len(prompt_mx.shape) == 3 else None
            cond_len_mx = mu_len_mx - (prompt_len_mx or 0)
            print(f"   MLX:     mu_len={mu_len_mx}, prompt_len={prompt_len_mx}, cond_len={cond_len_mx}")

        if (mu_pt is not None and mu_mx is not None):
            if mu_pt.shape[1] == mu_mx.shape[1]:
                print("   ✅ mu 时间长度一致")
            else:
                print("   ❌ mu 时间长度不一致")
                print(f"      PyTorch mu_len={mu_pt.shape[1]} vs MLX mu_len={mu_mx.shape[1]}")
                if prompt_pt is not None and prompt_mx is not None:
                    print(f"      提示: prompt_len(pt)={prompt_pt.shape[2]} vs prompt_len(mx)={prompt_mx.shape[2]}")
                print("      请检查 length_regulator 与 target_lengths 的一致性")

    def analyze_output_results(self):
        """分析输出结果"""
        print(f"\n{'='*60}")
        print(f"🔍 输出结果分析")
        print(f"{'='*60}")
        
        # 找到最新的缓存文件（输出只能来自实际推理缓存）
        _, pytorch_outputs, _, mlx_outputs = self.find_latest_cache_files()
        
        if not pytorch_outputs or not mlx_outputs:
            print("❌ 未找到输出缓存文件")
            return
        
        # 加载输出数据
        pytorch_data = self.load_cache_file(pytorch_outputs)
        mlx_data = self.load_cache_file(mlx_outputs)
        
        if not pytorch_data or not mlx_data:
            print("❌ 无法加载缓存数据")
            return
        
        # 分析输出结果
        pytorch_output = pytorch_data.get('output')
        mlx_output = mlx_data.get('output')
        
        if pytorch_output is not None and mlx_output is not None:
            print(f"\n📊 CFM 输出结果:")
            
            pytorch_stats = self.analyze_tensor_stats(pytorch_output, 'output')
            mlx_stats = self.analyze_tensor_stats(mlx_output, 'output')
            
            print(f"   PyTorch: shape={pytorch_output.shape}, min={pytorch_stats['min']:.6f}, max={pytorch_stats['max']:.6f}, avg={pytorch_stats['avg']:.6f}")
            print(f"   MLX:     shape={mlx_output.shape}, min={mlx_stats['min']:.6f}, max={mlx_stats['max']:.6f}, avg={mlx_stats['avg']:.6f}")
            
            # 比较差异
            if pytorch_output.shape == mlx_output.shape:
                if isinstance(pytorch_output, torch.Tensor):
                    pytorch_np = pytorch_output.detach().cpu().numpy()
                else:
                    pytorch_np = pytorch_output
                
                if isinstance(mlx_output, mx.array):
                    mlx_np = np.array(mlx_output)
                else:
                    mlx_np = mlx_output
                
                diff = np.abs(pytorch_np - mlx_np)
                max_diff = np.max(diff)
                avg_diff = np.mean(diff)
                
                print(f"   差异: max={max_diff:.6f}, avg={avg_diff:.6f}")
                
                if max_diff < 1e-6:
                    print(f"   ✅ 输出结果一致")
                else:
                    print(f"   ❌ 输出结果不一致")
            else:
                print(f"   ❌ 输出形状不匹配")

    def analyze_cfm_euler_steps(self):
        """分析 CFM Euler 求解步骤"""
        print(f"\n{'='*60}")
        print(f"🔍 CFM Euler 求解步骤分析")
        print(f"{'='*60}")
        
        # 输入改为使用一致测试数据，避免两端各自生成造成的长度差
        pytorch_input_data, mlx_input_data = self.load_consistent_test_data()
        
        # 输出仍使用生产缓存（若存在）
        _, pytorch_outputs, _, mlx_outputs = self.find_latest_cache_files()
        pytorch_output_data = self.load_cache_file(pytorch_outputs) if pytorch_outputs else {}
        mlx_output_data = self.load_cache_file(mlx_outputs) if mlx_outputs else {}
        
        # 分析第一步的详细输入输出
        self.analyze_first_euler_step(pytorch_input_data, mlx_input_data, pytorch_output_data, mlx_output_data)
        
        if pytorch_input_data is None or mlx_input_data is None:
            print("❌ 无法加载一致的输入数据")
            return
        
        # 分析输入数据
        print(f"\n📊 CFM 输入数据:")
        
        # PyTorch 输入
        pytorch_x = pytorch_input_data.get('x')
        pytorch_mu = pytorch_input_data.get('mu')
        
        if pytorch_x is not None:
            pytorch_x_stats = self.analyze_tensor_stats(pytorch_x, 'x')
            print(f"   PyTorch x: shape={pytorch_x.shape}, min={pytorch_x_stats['min']:.6f}, max={pytorch_x_stats['max']:.6f}, avg={pytorch_x_stats['avg']:.6f}")
        
        if pytorch_mu is not None:
            pytorch_mu_stats = self.analyze_tensor_stats(pytorch_mu, 'mu')
            print(f"   PyTorch mu: shape={pytorch_mu.shape}, min={pytorch_mu_stats['min']:.6f}, max={pytorch_mu_stats['max']:.6f}, avg={pytorch_mu_stats['avg']:.6f}")
        
        # MLX 输入
        mlx_x = mlx_input_data.get('x')
        mlx_mu = mlx_input_data.get('mu')
        
        if mlx_x is not None:
            mlx_x_stats = self.analyze_tensor_stats(mlx_x, 'x')
            print(f"   MLX x:     shape={mlx_x.shape}, min={mlx_x_stats['min']:.6f}, max={mlx_x_stats['max']:.6f}, avg={mlx_x_stats['avg']:.6f}")
        
        if mlx_mu is not None:
            mlx_mu_stats = self.analyze_tensor_stats(mlx_mu, 'mu')
            print(f"   MLX mu:    shape={mlx_mu.shape}, min={mlx_mu_stats['min']:.6f}, max={mlx_mu_stats['max']:.6f}, avg={mlx_mu_stats['avg']:.6f}")
        
        # 分析输出数据
        print(f"\n📊 CFM 输出数据:")
        
        pytorch_output = pytorch_output_data.get('output')
        mlx_output = mlx_output_data.get('output')
        
        if pytorch_output is not None:
            pytorch_output_stats = self.analyze_tensor_stats(pytorch_output, 'output')
            print(f"   PyTorch output: shape={pytorch_output.shape}, min={pytorch_output_stats['min']:.6f}, max={pytorch_output_stats['max']:.6f}, avg={pytorch_output_stats['avg']:.6f}")
        
        if mlx_output is not None:
            mlx_output_stats = self.analyze_tensor_stats(mlx_output, 'output')
            print(f"   MLX output:     shape={mlx_output.shape}, min={mlx_output_stats['min']:.6f}, max={mlx_output_stats['max']:.6f}, avg={mlx_output_stats['avg']:.6f}")
        
        # 比较输入输出变化
        if pytorch_x is not None and pytorch_output is not None and mlx_x is not None and mlx_output is not None:
            print(f"\n📊 输入输出变化分析:")
            
            # 转换为 numpy 进行比较
            pytorch_x_np = pytorch_x.detach().cpu().numpy() if isinstance(pytorch_x, torch.Tensor) else pytorch_x
            pytorch_output_np = pytorch_output.detach().cpu().numpy() if isinstance(pytorch_output, torch.Tensor) else pytorch_output
            mlx_x_np = np.array(mlx_x) if isinstance(mlx_x, mx.array) else mlx_x
            mlx_output_np = np.array(mlx_output) if isinstance(mlx_output, mx.array) else mlx_output
            
            # 计算变化
            pytorch_change = pytorch_output_np - pytorch_x_np
            mlx_change = mlx_output_np - mlx_x_np
            
            pytorch_change_stats = {
                'min': np.min(pytorch_change),
                'max': np.max(pytorch_change),
                'avg': np.mean(pytorch_change),
                'std': np.std(pytorch_change)
            }
            
            mlx_change_stats = {
                'min': np.min(mlx_change),
                'max': np.max(mlx_change),
                'avg': np.mean(mlx_change),
                'std': np.std(mlx_change)
            }
            
            print(f"   PyTorch 变化: min={pytorch_change_stats['min']:.6f}, max={pytorch_change_stats['max']:.6f}, avg={pytorch_change_stats['avg']:.6f}, std={pytorch_change_stats['std']:.6f}")
            print(f"   MLX 变化:     min={mlx_change_stats['min']:.6f}, max={mlx_change_stats['max']:.6f}, avg={mlx_change_stats['avg']:.6f}, std={mlx_change_stats['std']:.6f}")
            
            # 比较变化差异（处理形状不匹配）
            if pytorch_change.shape == mlx_change.shape:
                change_diff = np.abs(pytorch_change - mlx_change)
                change_max_diff = np.max(change_diff)
                change_mean_diff = np.mean(change_diff)
                
                print(f"   变化差异: max={change_max_diff:.6f}, mean={change_mean_diff:.6f}")
                
                if change_max_diff < 0.1:
                    print(f"   ✅ CFM 推理过程基本一致")
                else:
                    print(f"   ❌ CFM 推理过程存在差异")
            else:
                print(f"   ❌ 输入输出形状不匹配，无法直接比较")
                print(f"   PyTorch 变化形状: {pytorch_change.shape}")
                print(f"   MLX 变化形状: {mlx_change.shape}")
                print(f"   💡 这表明 PyTorch 和 MLX 生成了不同长度的序列")
    
    def analyze_dit_weights(self):
        """分析 DiT 权重差异"""
        print(f"\n{'='*60}")
        print(f"🔍 DiT 权重分析")
        print(f"{'='*60}")
        
        try:
            # 加载 PyTorch DiT 模型
            from indextts.infer_v2 import IndexTTS2
            pytorch_tts = IndexTTS2(use_mlx=False)
            pytorch_dit = pytorch_tts.s2mel.models['cfm'].estimator
            
            # 加载 MLX DiT 模型
            mlx_tts = IndexTTS2(use_mlx=True)
            mlx_dit = mlx_tts.mlx_s2mel_cfm.estimator
            
            print("✅ 成功加载 PyTorch 和 MLX DiT 模型")
            
            # 分析关键权重
            self._analyze_weight_layer("cond_projection", pytorch_dit.cond_projection, mlx_dit.cond_projection)
            self._analyze_weight_layer("t_embedder", pytorch_dit.t_embedder, mlx_dit.t_embedder)
            self._analyze_weight_layer("cond_embedder", pytorch_dit.cond_embedder, mlx_dit.cond_embedder)
            self._analyze_weight_layer("x_embedder", pytorch_dit.x_embedder, mlx_dit.x_embedder)
            self._analyze_weight_layer("cond_x_merge_linear", pytorch_dit.cond_x_merge_linear, mlx_dit.cond_x_merge_linear)
            
            # 分析 Transformer 权重
            self._analyze_transformer_weights(pytorch_dit.transformer, mlx_dit.transformer)
            
            # 分析 Final Layer 权重
            if hasattr(pytorch_dit, 'final_layer') and hasattr(mlx_dit, 'final_layer'):
                self._analyze_weight_layer("final_layer", pytorch_dit.final_layer, mlx_dit.final_layer)
            
        except Exception as e:
            print(f"❌ DiT 权重分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_weight_layer(self, layer_name: str, pytorch_layer, mlx_layer):
        """分析单个权重层"""
        print(f"\n📊 {layer_name} 权重分析:")
        
        try:
            # PyTorch 权重
            pytorch_weight = None
            pytorch_bias = None
            
            if hasattr(pytorch_layer, 'weight'):
                pytorch_weight = pytorch_layer.weight
            if hasattr(pytorch_layer, 'bias') and pytorch_layer.bias is not None:
                pytorch_bias = pytorch_layer.bias
            
            # MLX 权重
            mlx_weight = None
            mlx_bias = None
            
            if hasattr(mlx_layer, 'weight'):
                mlx_weight = mlx_layer.weight
            if hasattr(mlx_layer, 'bias') and mlx_layer.bias is not None:
                mlx_bias = mlx_layer.bias
            
            # 分析权重
            if pytorch_weight is not None and mlx_weight is not None:
                # 对 x_embedder 特殊处理：PyTorch 使用 weight_norm，.weight 不是直接可比
                if layer_name == "x_embedder" and hasattr(pytorch_layer, 'weight_g') and hasattr(pytorch_layer, 'weight_v'):
                    # 用 PyTorch 的 g、v 重构权重（Linear 按输入维度归一化）
                    g = pytorch_layer.weight_g.detach().cpu().numpy()
                    v = pytorch_layer.weight_v.detach().cpu().numpy()
                    norm_v = np.sqrt(np.sum(v**2, axis=1, keepdims=True))
                    pt_w = g * v / (norm_v + 1e-8)  # (out, in) = (512, 80)
                    mlx_w = np.array(mlx_weight)
                    # MLX Linear 权重为 (in, out) = (80, 512)，需转置
                    if pt_w.shape == mlx_w.shape[::-1]:
                        diff = np.abs(pt_w - mlx_w.T)
                        print("   🔧 使用 g/v 重构 PyTorch 权重，并与 MLX(转置后)比较")
                        max_diff = float(np.max(diff))
                        avg_diff = float(np.mean(diff))
                        print(f"   权重差异: max={max_diff:.6f}, avg={avg_diff:.6f}")
                        if max_diff < 1e-6:
                            print("   ✅ x_embedder.weight 一致")
                        else:
                            print("   ❌ x_embedder.weight 不一致")
                    else:
                        print("   ❌ x_embedder.weight 形状不匹配(重构后)")
                else:
                    pytorch_weight_stats = self.analyze_tensor_stats(pytorch_weight, f"{layer_name}.weight")
                    mlx_weight_stats = self.analyze_tensor_stats(mlx_weight, f"{layer_name}.weight")
                    print(f"   PyTorch {layer_name}.weight: shape={pytorch_weight.shape}, min={pytorch_weight_stats['min']:.6f}, max={pytorch_weight_stats['max']:.6f}, avg={pytorch_weight_stats['avg']:.6f}")
                    print(f"   MLX {layer_name}.weight:     shape={mlx_weight.shape}, min={mlx_weight_stats['min']:.6f}, max={mlx_weight_stats['max']:.6f}, avg={mlx_weight_stats['avg']:.6f}")
                    # 常规差异比较（处理转置）
                    pytorch_np = pytorch_weight.detach().cpu().numpy()
                    mlx_np = np.array(mlx_weight)
                    if pytorch_weight.shape == mlx_weight.shape:
                        diff = np.abs(pytorch_np - mlx_np)
                    elif pytorch_weight.shape == mlx_weight.shape[::-1]:
                        diff = np.abs(pytorch_np - mlx_np.T)
                        print(f"   🔧 MLX 权重已转置进行比较")
                    else:
                        print(f"   ❌ {layer_name}.weight 形状不匹配且无法转置比较")
                        return
                    max_diff = np.max(diff)
                    avg_diff = np.mean(diff)
                    print(f"   权重差异: max={max_diff:.6f}, avg={avg_diff:.6f}")
                    if max_diff < 1e-6:
                        print(f"   ✅ {layer_name}.weight 一致")
                    else:
                        print(f"   ❌ {layer_name}.weight 不一致")
            
            # 分析偏置
            if pytorch_bias is not None and mlx_bias is not None:
                pytorch_bias_stats = self.analyze_tensor_stats(pytorch_bias, f"{layer_name}.bias")
                mlx_bias_stats = self.analyze_tensor_stats(mlx_bias, f"{layer_name}.bias")
                
                print(f"   PyTorch {layer_name}.bias: shape={pytorch_bias.shape}, min={pytorch_bias_stats['min']:.6f}, max={pytorch_bias_stats['max']:.6f}, avg={pytorch_bias_stats['avg']:.6f}")
                print(f"   MLX {layer_name}.bias:     shape={mlx_bias.shape}, min={mlx_bias_stats['min']:.6f}, max={mlx_bias_stats['max']:.6f}, avg={mlx_bias_stats['avg']:.6f}")
                
                # 比较偏置差异
                if pytorch_bias.shape == mlx_bias.shape:
                    pytorch_np = pytorch_bias.detach().cpu().numpy()
                    mlx_np = np.array(mlx_bias)
                    diff = np.abs(pytorch_np - mlx_np)
                    max_diff = np.max(diff)
                    avg_diff = np.mean(diff)
                    
                    print(f"   偏置差异: max={max_diff:.6f}, avg={avg_diff:.6f}")
                    
                    if max_diff < 1e-6:
                        print(f"   ✅ {layer_name}.bias 一致")
                    else:
                        print(f"   ❌ {layer_name}.bias 不一致")
                else:
                    print(f"   ❌ {layer_name}.bias 形状不匹配")
            
        except Exception as e:
            print(f"   ❌ {layer_name} 权重分析失败: {e}")
    
    def _analyze_transformer_weights(self, pytorch_transformer, mlx_transformer):
        """分析 Transformer 权重"""
        print(f"\n📊 Transformer 权重分析:")
        
        try:
            # 分析 Transformer 层
            if hasattr(pytorch_transformer, 'layers') and hasattr(mlx_transformer, 'layers'):
                pytorch_layers = pytorch_transformer.layers
                mlx_layers = mlx_transformer.layers
                
                print(f"   PyTorch layers: {len(pytorch_layers)}")
                print(f"   MLX layers:     {len(mlx_layers)}")
                
                if len(pytorch_layers) == len(mlx_layers):
                    # 分析前几层
                    for i in range(min(3, len(pytorch_layers))):
                        print(f"\n   📊 Layer {i}:")
                        
                        # 分析 attention 权重
                        if hasattr(pytorch_layers[i], 'attention') and hasattr(mlx_layers[i], 'attention'):
                            pytorch_attn = pytorch_layers[i].attention
                            mlx_attn = mlx_layers[i].attention
                            
                            if hasattr(pytorch_attn, 'wq') and hasattr(mlx_attn, 'wq'):
                                self._analyze_weight_layer(f"layer_{i}.attention.wq", pytorch_attn.wq, mlx_attn.wq)
                            
                            if hasattr(pytorch_attn, 'wk') and hasattr(mlx_attn, 'wk'):
                                self._analyze_weight_layer(f"layer_{i}.attention.wk", pytorch_attn.wk, mlx_attn.wk)
                            
                            if hasattr(pytorch_attn, 'wv') and hasattr(mlx_attn, 'wv'):
                                self._analyze_weight_layer(f"layer_{i}.attention.wv", pytorch_attn.wv, mlx_attn.wv)
                            
                            if hasattr(pytorch_attn, 'wo') and hasattr(mlx_attn, 'wo'):
                                self._analyze_weight_layer(f"layer_{i}.attention.wo", pytorch_attn.wo, mlx_attn.wo)
                        
                        # 分析 feed forward 权重
                        if hasattr(pytorch_layers[i], 'feed_forward') and hasattr(mlx_layers[i], 'feed_forward'):
                            pytorch_ff = pytorch_layers[i].feed_forward
                            mlx_ff = mlx_layers[i].feed_forward
                            
                            if hasattr(pytorch_ff, 'w1') and hasattr(mlx_ff, 'w1'):
                                self._analyze_weight_layer(f"layer_{i}.feed_forward.w1", pytorch_ff.w1, mlx_ff.w1)
                            
                            if hasattr(pytorch_ff, 'w2') and hasattr(mlx_ff, 'w2'):
                                self._analyze_weight_layer(f"layer_{i}.feed_forward.w2", pytorch_ff.w2, mlx_ff.w2)
                            
                            if hasattr(pytorch_ff, 'w3') and hasattr(mlx_ff, 'w3'):
                                self._analyze_weight_layer(f"layer_{i}.feed_forward.w3", pytorch_ff.w3, mlx_ff.w3)
                else:
                    print(f"   ❌ Transformer 层数不匹配")
            
        except Exception as e:
            print(f"   ❌ Transformer 权重分析失败: {e}")
    
    def analyze_mu_length_decomposition(self):
        """分析 mu 长度分解：mu_len = prompt_len + cond_len"""
        print(f"\n{'='*60}")
        print(f"🔍 mu 长度分解分析 (使用一致测试数据)")
        print(f"{'='*60}")
        
        # 加载一致的测试数据
        pytorch_data, mlx_data = self.load_consistent_test_data()
        
        if pytorch_data is None or mlx_data is None:
            print("❌ 无法加载一致的测试数据")
            return
        
        # PyTorch 分析
        mu_pt = pytorch_data.get('mu')
        prompt_pt = pytorch_data.get('prompt')
        
        if mu_pt is not None and prompt_pt is not None:
            mu_len_pt = int(mu_pt.shape[1])
            prompt_len_pt = int(prompt_pt.shape[2]) if len(prompt_pt.shape) == 3 else None
            cond_len_pt = mu_len_pt - (prompt_len_pt or 0)
            print(f"   PyTorch: mu_len={mu_len_pt}, prompt_len={prompt_len_pt}, cond_len={cond_len_pt}")
        
        # MLX 分析
        mu_mx = mlx_data.get('mu')
        prompt_mx = mlx_data.get('prompt')
        
        if mu_mx is not None and prompt_mx is not None:
            mu_len_mx = int(mu_mx.shape[1])
            prompt_len_mx = int(prompt_mx.shape[2]) if len(prompt_mx.shape) == 3 else None
            cond_len_mx = mu_len_mx - (prompt_len_mx or 0)
            print(f"   MLX:     mu_len={mu_len_mx}, prompt_len={prompt_len_mx}, cond_len={cond_len_mx}")
        
        # 比较分析
        if mu_pt is not None and mu_mx is not None:
            if mu_pt.shape[1] == mu_mx.shape[1]:
                print("   ✅ mu 时间长度一致")
            else:
                print("   ❌ mu 时间长度不一致")
                print(f"      PyTorch mu_len={mu_pt.shape[1]} vs MLX mu_len={mu_mx.shape[1]}")
                if prompt_pt is not None and prompt_mx is not None:
                    print(f"      提示: prompt_len(pt)={prompt_pt.shape[2]} vs prompt_len(mx)={prompt_mx.shape[2]}")
                print("      💡 问题根源：length_regulator 或 target_lengths 计算不一致")
                print("      💡 建议：检查 PyTorch 和 MLX 的 GPT 输出 codes 和 code_lens 是否一致")
    
    def analyze_gpt_outputs(self):
        """分析 GPT 输出一致性"""
        print(f"\n{'='*60}")
        print(f"🔍 GPT 输出一致性分析 (使用一致测试数据)")
        print(f"{'='*60}")
        
        # 加载一致的测试数据
        pytorch_data, mlx_data = self.load_consistent_test_data()
        
        if pytorch_data is None or mlx_data is None:
            print("❌ 无法加载一致的测试数据")
            return
        
        # 分析 codes
        pytorch_codes = pytorch_data.get('codes')
        mlx_codes = mlx_data.get('codes')
        
        if pytorch_codes is not None and mlx_codes is not None:
            print(f"\n📊 codes 分析:")
            print(f"   PyTorch: shape={pytorch_codes.shape}, min={pytorch_codes.min().item()}, max={pytorch_codes.max().item()}")
            print(f"   MLX:     shape={mlx_codes.shape}, min={mx.min(mlx_codes).item()}, max={mx.max(mlx_codes).item()}")
            
            if pytorch_codes.shape == mlx_codes.shape:
                # 转换为相同类型进行比较
                pytorch_np = pytorch_codes.detach().cpu().numpy()
                mlx_np = np.array(mlx_codes)
                
                diff = np.abs(pytorch_np - mlx_np)
                max_diff = np.max(diff)
                avg_diff = np.mean(diff)
                
                print(f"   差异: max={max_diff:.6f}, avg={avg_diff:.6f}")
                
                if max_diff < 1e-6:
                    print(f"   ✅ codes 完全一致")
                else:
                    print(f"   ⚠️  codes 存在差异")
            else:
                print(f"   ❌ codes 形状不匹配")
        
        # 分析 speech_conditioning_latent
        pytorch_scl = pytorch_data.get('speech_conditioning_latent')
        mlx_scl = mlx_data.get('speech_conditioning_latent')
        
        if pytorch_scl is not None and mlx_scl is not None:
            print(f"\n📊 speech_conditioning_latent 分析:")
            print(f"   PyTorch: shape={pytorch_scl.shape}, min={pytorch_scl.min().item():.6f}, max={pytorch_scl.max().item():.6f}, avg={pytorch_scl.mean().item():.6f}")
            print(f"   MLX:     shape={mlx_scl.shape}, min={mx.min(mlx_scl).item():.6f}, max={mx.max(mlx_scl).item():.6f}, avg={mx.mean(mlx_scl).item():.6f}")
            
            if pytorch_scl.shape == mlx_scl.shape:
                # 转换为相同类型进行比较
                pytorch_np = pytorch_scl.detach().cpu().numpy()
                mlx_np = np.array(mlx_scl)
                
                diff = np.abs(pytorch_np - mlx_np)
                max_diff = np.max(diff)
                avg_diff = np.mean(diff)
                
                print(f"   差异: max={max_diff:.6f}, avg={avg_diff:.6f}")
                
                if max_diff < 1e-6:
                    print(f"   ✅ speech_conditioning_latent 完全一致")
                else:
                    print(f"   ⚠️  speech_conditioning_latent 存在差异")
            else:
                print(f"   ❌ speech_conditioning_latent 形状不匹配")
        
        # 分析 code_lens (从 codes 长度推导)
        if pytorch_codes is not None and mlx_codes is not None:
            pytorch_code_lens = pytorch_codes.shape[1]
            mlx_code_lens = mlx_codes.shape[1]
            
            print(f"\n📊 code_lens 分析:")
            print(f"   PyTorch: {pytorch_code_lens}")
            print(f"   MLX:     {mlx_code_lens}")
            
            if pytorch_code_lens == mlx_code_lens:
                print(f"   ✅ code_lens 一致")
            else:
                print(f"   ❌ code_lens 不一致")
                print(f"   💡 这解释了为什么 mu 长度不同")

    def analyze_all_stages(self):
        """分析所有阶段"""
        print("🔍 CFM 阶段分析工具")
        print("="*60)
        
        # 分析初始化噪声
        self.analyze_initialization_noise()
        
        # 分析其他输入参数
        self.analyze_input_parameters()
        
        # 分析 mu 长度分解
        self.analyze_mu_length_decomposition()
        
        # 分析 GPT 输出一致性
        self.analyze_gpt_outputs()
        
        # 执行第一步推理并生成步骤子过程缓存
        self.execute_first_step_inference()
        
        # 分析输出结果
        self.analyze_output_results()
        
        # 分析 CFM Euler 求解步骤
        self.analyze_cfm_euler_steps()
        
        # 分析 DiT 权重
        self.analyze_dit_weights()
        
        # 分析 DiT 层实现差异
        self.analyze_dit_layer_implementation()
        
        # 保存调试数据
        self.save_debug_data()

    def execute_first_step_inference(self):
        """执行第一步推理并生成步骤子过程缓存"""
        print(f"\n{'='*60}")
        print(f"🔍 执行第一步推理并生成步骤子过程缓存")
        print(f"{'='*60}")
        
        # 加载一致的测试数据
        pytorch_data, mlx_data = self.load_consistent_test_data()
        if pytorch_data is None or mlx_data is None:
            print("❌ 无法加载一致的测试数据，跳过第一步推理")
            return
        
        print("✅ 加载一致的测试数据")
        pytorch_shapes = [f'{k}: {v.shape if hasattr(v, "shape") else type(v).__name__}' for k, v in pytorch_data.items() if k in ['mu', 'x_lens', 'prompt', 'style', 'f0', 'n_timesteps', 'temperature', 'inference_cfg_rate', 'unified_random_seed', 'timestamp', 'model_type', 'codes', 'speech_conditioning_latent']]
        mlx_shapes = [f'{k}: {v.shape if hasattr(v, "shape") else type(v).__name__}' for k, v in mlx_data.items() if k in ['mu', 'x_lens', 'prompt', 'style', 'f0', 'n_timesteps', 'temperature', 'inference_cfg_rate', 'unified_random_seed', 'timestamp', 'model_type', 'codes', 'speech_conditioning_latent']]
        print(f"   PyTorch 输入形状: {pytorch_shapes}")
        print(f"   MLX 输入形状: {mlx_shapes}")
        
        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 导入必要的模块（MLX 直接使用 MLXCFM；PyTorch 使用 IndexTTS2 提供的生产加载）
        try:
            import numpy as np
            from omegaconf import OmegaConf
            from unified_random_generator import UnifiedRandomGenerator
            from indextts.s2mel.modules.mlx_flow_matching import MLXCFM
            from indextts.infer_v2 import IndexTTS2 as TorchIndexTTS2
        except ImportError as e:
            print(f"❌ 导入模块失败: {e}")
            return

        # 仅初始化 MLX 模型（按基准脚本风格加载权重）
        print("\n🔧 初始化 MLX 模型...")
        try:
            args = OmegaConf.load("checkpoints/config.yaml")
            mlx_cfm = MLXCFM(args.s2mel)
            # 从 checkpoints/mlx/s2mel.npz 加载权重
            npz_path = "checkpoints/mlx/s2mel.npz"
            if not os.path.exists(npz_path):
                raise FileNotFoundError(f"未找到 MLX 权重缓存: {npz_path}")
            with np.load(npz_path, allow_pickle=True) as data:
                cache_dict = {k: data[k] for k in data.files}
            mlx_cfm.load_from_cache(cache_dict)
            
            # 启用 MLX DiT 调试模式
            print("   🔧 启用 MLX DiT 调试模式...")
            mlx_dit = mlx_cfm.estimator
            mlx_dit._debug_layers = True
            print("   ✅ MLX DiT 调试模式已启用")
            
            print("✅ 模型初始化成功并加载 MLX 权重")
        except Exception as e:
            print(f"❌ 模型初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 准备输入数据
        print("\n🔧 准备输入数据...")
        try:
            # PyTorch 输入
            mu_pt = pytorch_data['mu']
            x_lens_pt = pytorch_data['x_lens']
            prompt_pt = pytorch_data['prompt']
            style_pt = pytorch_data['style']

            # MLX 输入
            mu_mlx = mlx_data['mu']
            x_lens_mlx = mlx_data['x_lens']
            prompt_mlx = mlx_data['prompt']
            style_mlx = mlx_data['style']
            
            print("✅ 输入数据准备完成")
        except Exception as e:
            print(f"❌ 输入数据准备失败: {e}")
            return
        
        # 执行第一步推理（PyTorch 与 MLX）
        print("\n🔧 执行第一步推理（PyTorch 与 MLX）...")
        try:
            # 设置随机种子
            unified_random = UnifiedRandomGenerator(seed=42)

            # 初始化 PyTorch CFM（按生产链路）
            print("   🔧 初始化 PyTorch CFM...")
            pytorch_tts = TorchIndexTTS2(use_mlx=False)
            pytorch_cfm = pytorch_tts.s2mel.models['cfm']
            # 设备迁移
            device = next(pytorch_cfm.parameters()).device
            mu_pt = mu_pt.to(device)
            x_lens_pt = x_lens_pt.to(device)
            prompt_pt = prompt_pt.to(device)
            style_pt = style_pt.to(device)
            # 统一缓存目录
            pytorch_cfm._cfm_cache_enabled = True
            pytorch_cfm._cfm_cache_dir = os.path.abspath(self.cache_dir)
            os.makedirs(pytorch_cfm._cfm_cache_dir, exist_ok=True)
            
            # 启用 DiT 调试模式
            print("   🔧 启用 PyTorch DiT 调试模式...")
            pytorch_dit = pytorch_cfm.estimator
            pytorch_dit._debug_layers = True
            print("   ✅ PyTorch DiT 调试模式已启用")

            # 执行 PyTorch 第一步推理（会生成 cfm_pytorch_step_001_*）
            print("   🔧 执行 PyTorch 第一步推理...")
            pytorch_output = pytorch_cfm.inference(
                mu=mu_pt,
                x_lens=x_lens_pt,
                prompt=prompt_pt,
                style=style_pt,
                f0=None,
                n_timesteps=1,
                temperature=1.0,
                inference_cfg_rate=0.7,
                unified_random=unified_random
            )

            # 统一 MLX 缓存目录
            mlx_cfm._cfm_cache_enabled = True
            mlx_cfm._cfm_cache_dir = os.path.abspath(self.cache_dir)
            os.makedirs(mlx_cfm._cfm_cache_dir, exist_ok=True)

            # 重新设置随机种子，确保 MLX 使用相同的噪声
            print("   🔧 重新设置随机种子确保噪声一致性...")
            unified_random.reset_seed(42)

            # 执行 MLX 第一步推理（会生成 cfm_mlx_step_001_*）
            print("   🔧 执行 MLX 第一步推理...")
            mlx_output = mlx_cfm.inference(
                mu=mu_mlx,
                x_lens=x_lens_mlx,
                prompt=prompt_mlx,
                style=style_mlx,
                f0=None,
                n_timesteps=1,  # 只执行第一步
                temperature=1.0,
                inference_cfg_rate=0.7,
                unified_random=unified_random
            )
            
            print("✅ 第一步推理完成")
            
            # 保存输出结果
            pytorch_output_file = os.path.join(self.cache_dir, "pytorch_first_step_output.pkl")
            mlx_output_file = os.path.join(self.cache_dir, "mlx_first_step_output.pkl")

            with open(pytorch_output_file, 'wb') as f:
                pickle.dump(pytorch_output, f)
            with open(mlx_output_file, 'wb') as f:
                pickle.dump(mlx_output, f)

            print(f"✅ 输出结果已保存:")
            print(f"   PyTorch: {pytorch_output_file}")
            print(f"   MLX: {mlx_output_file}")
            
        except Exception as e:
            print(f"❌ 第一步推理失败: {e}")
            import traceback
            traceback.print_exc()

    def analyze_first_euler_step(self, pytorch_input_data, mlx_input_data, pytorch_output_data, mlx_output_data):
        """分析第一步 Euler 求解的详细输入输出"""
        print(f"\n{'='*60}")
        print(f"🔍 第一步 Euler 求解详细分析")
        print(f"{'='*60}")
        
        # 分析输入数据
        print(f"\n📊 第一步输入数据:")
        pytorch_mu = pytorch_input_data.get('mu')
        mlx_mu = mlx_input_data.get('mu')
        
        if pytorch_mu is not None and mlx_mu is not None:
            pytorch_stats = self.analyze_tensor_stats(pytorch_mu, "mu")
            mlx_stats = self.analyze_tensor_stats(mlx_mu, "mu")
            
            print(f"   PyTorch mu: shape={pytorch_mu.shape}, min={pytorch_stats['min']:.6f}, max={pytorch_stats['max']:.6f}, avg={pytorch_stats['avg']:.6f}")
            print(f"   MLX mu:     shape={mlx_mu.shape}, min={mlx_stats['min']:.6f}, max={mlx_stats['max']:.6f}, avg={mlx_stats['avg']:.6f}")
            
            # 检查 mu 一致性
            if pytorch_mu.shape == mlx_mu.shape:
                pytorch_np = pytorch_mu.detach().cpu().numpy()
                mlx_np = np.array(mlx_mu)
                diff = np.abs(pytorch_np - mlx_np)
                max_diff = np.max(diff)
                avg_diff = np.mean(diff)
                print(f"   mu 差异: max={max_diff:.6f}, avg={avg_diff:.6f}")
                if max_diff < 1e-6:
                    print(f"   ✅ mu 输入完全一致")
                else:
                    print(f"   ❌ mu 输入存在差异")
            else:
                print(f"   ❌ mu 形状不匹配")
        
        # 分析输出数据
        print(f"\n📊 第一步输出数据:")
        pytorch_output = pytorch_output_data.get('output')
        mlx_output = mlx_output_data.get('output')
        
        if pytorch_output is not None and mlx_output is not None:
            pytorch_stats = self.analyze_tensor_stats(pytorch_output, "output")
            mlx_stats = self.analyze_tensor_stats(mlx_output, "output")
            
            print(f"   PyTorch output: shape={pytorch_output.shape}, min={pytorch_stats['min']:.6f}, max={pytorch_stats['max']:.6f}, avg={pytorch_stats['avg']:.6f}")
            print(f"   MLX output:     shape={mlx_output.shape}, min={mlx_stats['min']:.6f}, max={mlx_stats['max']:.6f}, avg={mlx_stats['avg']:.6f}")
            
            # 检查输出一致性
            if pytorch_output.shape == mlx_output.shape:
                pytorch_np = pytorch_output.detach().cpu().numpy()
                mlx_np = np.array(mlx_output)
                diff = np.abs(pytorch_np - mlx_np)
                max_diff = np.max(diff)
                avg_diff = np.mean(diff)
                print(f"   输出差异: max={max_diff:.6f}, avg={avg_diff:.6f}")
                if max_diff < 1e-6:
                    print(f"   ✅ 第一步输出完全一致")
                else:
                    print(f"   ❌ 第一步输出存在差异")
            else:
                print(f"   ❌ 输出形状不匹配")
        
        # 分析第一步的中间步骤（如果缓存中有详细步骤数据）
        self.analyze_first_step_subprocesses(pytorch_output_data, mlx_output_data)
    
    def analyze_first_step_subprocesses(self, pytorch_output_data, mlx_output_data):
        """分析第一步的子过程（条件编码、风格融合、参考融合、DiT估计、Euler步）"""
        print(f"\n📊 第一步子过程分析:")
        
        # 尝试加载实际的步骤缓存文件
        self.load_and_analyze_step_cache_files()

        # 使用生产 DiT 模型在缓存的 DiT 输入上直接验证
        self.verify_dit_with_production_on_cached_inputs()
    
    def load_and_analyze_step_cache_files(self):
        """加载并分析实际的步骤缓存文件"""
        import os
        import glob
        import pickle
        
        # 查找最新的第一步缓存文件
        pytorch_step_files = glob.glob(os.path.join(self.cache_dir, "cfm_pytorch_step_001_*_*.pkl"))
        mlx_step_files = glob.glob(os.path.join(self.cache_dir, "cfm_mlx_step_001_*_*.pkl"))
        
        if not pytorch_step_files or not mlx_step_files:
            print("   ⚠️  未找到第一步的详细缓存文件")
            return
        
        # 按时间戳排序，获取最新的文件
        pytorch_step_files.sort(key=os.path.getmtime, reverse=True)
        mlx_step_files.sort(key=os.path.getmtime, reverse=True)
        
        # 加载最新的文件
        latest_pytorch_files = pytorch_step_files[:5]  # 取最新的5个文件
        latest_mlx_files = mlx_step_files[:5]
        
        print(f"   📁 找到 {len(pytorch_step_files)} 个 PyTorch 缓存文件")
        print(f"   📁 找到 {len(mlx_step_files)} 个 MLX 缓存文件")
        
        # 分析输入和输出文件
        self.analyze_step_files(latest_pytorch_files, latest_mlx_files, "input")
        self.analyze_step_files(latest_pytorch_files, latest_mlx_files, "output")
        
        # 分析DiT输入和输出文件
        self.analyze_step_files(latest_pytorch_files, latest_mlx_files, "dit_input")
        self.analyze_step_files(latest_pytorch_files, latest_mlx_files, "dit_output")

        # 进一步：专门对 DiT 子流程做严格对齐与差异报告
        self.analyze_dit_subflow_diffs()

    def verify_dit_with_production_on_cached_inputs(self):
        """使用生产 PyTorch/MLX DiT 在最近的缓存 DiT 输入上直接跑一遍并对比输出。"""
        print(f"\n   🔍 使用生产 DiT 在缓存输入上直接验证:")
        import glob
        import os
        import pickle
        import numpy as np
        try:
            from indextts.infer_v2 import IndexTTS2
        except Exception as e:
            print(f"   ❌ 无法导入生产入口 IndexTTS2: {e}")
            return

        # 定位最近的一组 DiT 输入缓存（任选一侧作为输入源，这里优先用 PyTorch 的）
        pt_in_files = glob.glob(os.path.join(self.cache_dir, "cfm_pytorch_step_*_dit_input_*.pkl"))
        mx_in_files = glob.glob(os.path.join(self.cache_dir, "cfm_mlx_step_*_dit_input_*.pkl"))
        if not pt_in_files and not mx_in_files:
            print("   ⚠️  未找到任何 DiT 输入缓存文件，跳过验证")
            return
        input_file = max(pt_in_files or mx_in_files, key=os.path.getmtime)
        try:
            with open(input_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"   ❌ 加载缓存输入失败: {e}")
            return

        # 规范输入到 numpy/torch/mx
        x_np = data.get('x')
        prompt_x_np = data.get('prompt_x')
        mu_np = data.get('mu')
        style_np = data.get('style')
        t_np = data.get('t')
        x_lens_np = data.get('x_lens')
        if any(v is None for v in [x_np, prompt_x_np, mu_np, style_np, t_np, x_lens_np]):
            print("   ❌ DiT 输入缓存键不完整，跳过验证")
            return

        # 加载生产 PyTorch 与 MLX 模型（只取 DiT 估计器）
        try:
            torch_tts = IndexTTS2(use_mlx=False)
            torch_dit = torch_tts.s2mel.models['cfm'].estimator
        except Exception as e:
            print(f"   ❌ 加载 PyTorch DiT 失败: {e}")
            return
        try:
            mlx_tts = IndexTTS2(use_mlx=True)
            mlx_dit = mlx_tts.mlx_s2mel_cfm.estimator
        except Exception as e:
            print(f"   ❌ 加载 MLX DiT 失败: {e}")
            return

        # 准备并运行 PyTorch 前向
        try:
            device = next(torch_dit.parameters()).device
            x_pt = torch.as_tensor(x_np).to(device)
            prompt_x_pt = torch.as_tensor(prompt_x_np).to(device)
            mu_pt = torch.as_tensor(mu_np).to(device)
            style_pt = torch.as_tensor(style_np).to(device)
            t_pt = torch.as_tensor(t_np).to(device)
            x_lens_pt = torch.as_tensor(x_lens_np).to(device)
            with torch.no_grad():
                y_pt = torch_dit(x_pt, prompt_x_pt, x_lens_pt, t_pt, style_pt, mu_pt)
        except Exception as e:
            print(f"   ❌ PyTorch DiT 前向失败: {e}")
            return

        # 准备并运行 MLX 前向
        try:
            x_mx = mx.array(x_np)
            prompt_x_mx = mx.array(prompt_x_np)
            mu_mx = mx.array(mu_np)
            style_mx = mx.array(style_np)
            t_mx = mx.array(t_np)
            x_lens_mx = mx.array(x_lens_np)
            y_mx = mlx_dit(x_mx, prompt_x_mx, x_lens_mx, t_mx, style_mx, mu_mx)
        except Exception as e:
            print(f"   ❌ MLX DiT 前向失败: {e}")
            return

        # 对比输出
        try:
            y_pt_np = y_pt.detach().cpu().numpy()
            y_mx_np = np.array(y_mx)
            print(f"   PyTorch 输出: shape={y_pt_np.shape}, min={np.min(y_pt_np):.6f}, max={np.max(y_pt_np):.6f}, avg={np.mean(y_pt_np):.6f}")
            print(f"   MLX 输出:     shape={y_mx_np.shape}, min={np.min(y_mx_np):.6f}, max={np.max(y_mx_np):.6f}, avg={np.mean(y_mx_np):.6f}")
            if y_pt_np.shape == y_mx_np.shape:
                diff = np.abs(y_pt_np - y_mx_np)
                print(f"   差异: max={float(np.max(diff)):.6f}, mean={float(np.mean(diff)):.6f}")
            else:
                print(f"   ❌ 输出形状不一致: PT={y_pt_np.shape}, MLX={y_mx_np.shape}")
        except Exception as e:
            print(f"   ❌ 输出对比失败: {e}")

    def analyze_dit_subflow_diffs(self):
        """严格对齐并比较 DiT 子流程 torch 与 MLX 的输入/输出差异。

        - 只比较同一步(step)中最近的一组 `dit_input` 和 `dit_output` 文件
        - 针对关键键: x, prompt_x, mu, style, t, x_lens, dphi_dt
        - 处理常见形状差异: t 的标量/一维, x_lens 的标量/一维
        - 提供转置建议: 若 (B, C, T) 与 (B, T, C) 错置则提示
        """
        import os
        import glob
        import pickle
        import numpy as np

        def _latest_by_pattern(pattern: str):
            files = glob.glob(os.path.join(self.cache_dir, pattern))
            return max(files, key=os.path.getmtime) if files else None

        # 定位最近的一组 torch/mlx dit_input 与 dit_output
        pt_in = _latest_by_pattern("cfm_pytorch_step_*_dit_input_*.pkl")
        mx_in = _latest_by_pattern("cfm_mlx_step_*_dit_input_*.pkl")
        pt_out = _latest_by_pattern("cfm_pytorch_step_*_dit_output_*.pkl")
        mx_out = _latest_by_pattern("cfm_mlx_step_*_dit_output_*.pkl")

        if not (pt_in and mx_in):
            print("   ⚠️  未找到完整的 DiT 输入缓存对 (pytorch/mlx)")
            return
        if not (pt_out and mx_out):
            print("   ⚠️  未找到完整的 DiT 输出缓存对 (pytorch/mlx)")
            return

        try:
            with open(pt_in, 'rb') as f: pt_in_data = pickle.load(f)
            with open(mx_in, 'rb') as f: mx_in_data = pickle.load(f)
            with open(pt_out, 'rb') as f: pt_out_data = pickle.load(f)
            with open(mx_out, 'rb') as f: mx_out_data = pickle.load(f)
        except Exception as e:
            print(f"   ❌ 载入 DiT 子流程缓存失败: {e}")
            return

        print(f"\n   🔍 DiT 子流程严格对齐差异分析:")

        def to_numpy(x):
            import torch
            import mlx.core as mx
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            if isinstance(x, mx.array):
                return np.array(x)
            if isinstance(x, np.ndarray):
                return x
            # 标量或列表
            try:
                return np.array(x)
            except Exception:
                return None

        def summarize(name, a, b):
            a_np, b_np = to_numpy(a), to_numpy(b)
            if a_np is None or b_np is None:
                print(f"     {name}: 无法转换为 numpy 进行比较")
                return
            # 统一 dtype
            try:
                a_np = a_np.astype(np.float32) if a_np.dtype.kind in 'fc' else a_np
                b_np = b_np.astype(np.float32) if b_np.dtype.kind in 'fc' else b_np
            except Exception:
                pass

            # 特殊处理: t 允许 (2,) vs (2,1) 或标量
            if name == 't':
                a_np = np.squeeze(a_np)
                b_np = np.squeeze(b_np)
                # 对齐到一维
                if a_np.ndim == 0: a_np = np.array([a_np])
                if b_np.ndim == 0: b_np = np.array([b_np])

            # 特殊处理: x_lens 允许标量/一维
            if name == 'x_lens':
                a_np = np.squeeze(a_np)
                b_np = np.squeeze(b_np)
                if a_np.ndim == 0: a_np = np.array([a_np])
                if b_np.ndim == 0: b_np = np.array([b_np])

            same_shape = a_np.shape == b_np.shape
            if not same_shape:
                print(f"     ❌ {name} 形状不匹配: PT={a_np.shape}, MLX={b_np.shape}")
                # 对 x/prompt_x 提示常见错置
                if name in ['x', 'prompt_x'] and a_np.ndim == 3 and b_np.ndim == 3:
                    if a_np.shape == (b_np.shape[0], b_np.shape[2], b_np.shape[1]):
                        print(f"        💡 {name} 可能存在 (B,C,T)<->(B,T,C) 维度错置")
                return

            # 数值差异
            try:
                diff = np.abs(a_np - b_np)
                print(f"     {name}: shape={a_np.shape}, max={float(np.max(a_np)):.6f}/{float(np.max(b_np)):.6f}, min={float(np.min(a_np)):.6f}/{float(np.min(b_np)):.6f}, avg={float(np.mean(a_np)):.6f}/{float(np.mean(b_np)):.6f}")
                print(f"       差异: max={float(np.max(diff)):.6f}, mean={float(np.mean(diff)):.6f}")
            except Exception as e:
                print(f"     ⚠️ {name} 差异计算失败: {e}")

        # 输入键对比
        print("\n     📥 DiT 输入对比:")
        for k in ['x', 'prompt_x', 'mu', 'style', 't', 'x_lens']:
            if k in pt_in_data and k in mx_in_data:
                summarize(k, pt_in_data[k], mx_in_data[k])
            else:
                print(f"     ⚠️ 缺少键: {k} (PT有? {k in pt_in_data}, MLX有? {k in mx_in_data})")

        # 输出键对比
        print("\n     📤 DiT 输出对比:")
        k = 'dphi_dt'
        if k in pt_out_data and k in mx_out_data:
            summarize(k, pt_out_data[k], mx_out_data[k])
        else:
            print(f"     ⚠️ 缺少键: {k} (PT有? {k in pt_out_data}, MLX有? {k in mx_out_data})")
    
    def analyze_step_files(self, pytorch_files, mlx_files, file_type):
        """分析特定类型的步骤文件"""
        import os
        import pickle
        
        # 查找匹配的文件
        pytorch_file = None
        mlx_file = None
        
        for f in pytorch_files:
            if f"_{file_type}_" in f:
                pytorch_file = f
                break
        
        for f in mlx_files:
            if f"_{file_type}." in f or f"_{file_type}_" in f:
                mlx_file = f
                break
        
        if pytorch_file is None or mlx_file is None:
            print(f"   ⚠️  {file_type} 文件未找到")
            return
        
        try:
            # 加载文件
            with open(pytorch_file, 'rb') as f:
                pytorch_data = pickle.load(f)
            
            with open(mlx_file, 'rb') as f:
                mlx_data = pickle.load(f)
            
            print(f"\n   🔍 {file_type} 分析:")
            
            # 分析PyTorch数据
            if isinstance(pytorch_data, dict):
                pytorch_keys = list(pytorch_data.keys())
                print(f"     PyTorch keys: {pytorch_keys}")
                
                # 分析主要张量
                for key in ['x', 't', 'mu', 'style', 'prompt_x', 'x_lens', 'dphi_dt']:
                    if key in pytorch_data:
                        tensor = pytorch_data[key]
                        if hasattr(tensor, 'shape'):
                            stats = self.analyze_tensor_stats(tensor, f"pytorch_{key}")
                            print(f"     PyTorch {key}: shape={tensor.shape}, min={stats['min']:.6f}, max={stats['max']:.6f}, avg={stats['avg']:.6f}")
            
            # 分析MLX数据
            if isinstance(mlx_data, dict):
                mlx_keys = list(mlx_data.keys())
                print(f"     MLX keys: {mlx_keys}")
                
                # 分析主要张量
                for key in ['x', 't', 'mu', 'style', 'prompt_x', 'x_lens', 'dphi_dt']:
                    if key in mlx_data:
                        tensor = mlx_data[key]
                        if hasattr(tensor, 'shape'):
                            stats = self.analyze_tensor_stats(tensor, f"mlx_{key}")
                            print(f"     MLX {key}:     shape={tensor.shape}, min={stats['min']:.6f}, max={stats['max']:.6f}, avg={stats['avg']:.6f}")
            
            # 比较相同键的数据
            if isinstance(pytorch_data, dict) and isinstance(mlx_data, dict):
                common_keys = set(pytorch_data.keys()) & set(mlx_data.keys())
                for key in common_keys:
                    if key in ['x', 't', 'mu', 'style', 'prompt_x', 'x_lens', 'dphi_dt']:
                        pytorch_tensor = pytorch_data[key]
                        mlx_tensor = mlx_data[key]
                        
                        if hasattr(pytorch_tensor, 'shape') and hasattr(mlx_tensor, 'shape'):
                            if pytorch_tensor.shape == mlx_tensor.shape:
                                pytorch_np = pytorch_tensor.detach().cpu().numpy() if hasattr(pytorch_tensor, 'detach') else np.array(pytorch_tensor)
                                mlx_np = np.array(mlx_tensor)
                                diff = np.abs(pytorch_np - mlx_np)
                                max_diff = np.max(diff)
                                avg_diff = np.mean(diff)
                                print(f"     {key} 差异: max={max_diff:.6f}, avg={avg_diff:.6f}")
                                if max_diff < 1e-6:
                                    print(f"     ✅ {key} 一致")
                                else:
                                    print(f"     ❌ {key} 不一致")
                            else:
                                print(f"     ❌ {key} 形状不匹配: PyTorch={pytorch_tensor.shape}, MLX={mlx_tensor.shape}")
        
        except Exception as e:
            print(f"   ❌ 加载 {file_type} 文件失败: {e}")
    
    def analyze_dit_layer_implementation(self):
        """分析 DiT 层实现差异"""
        print(f"\n{'='*60}")
        print(f"🔍 DiT 层实现差异分析")
        print(f"{'='*60}")
        
        try:
            # 导入 DiT 层实现分析器
            from analyze_dit_layer_implementation import DiTLayerImplementationAnalyzer
            
            # 创建分析器并执行分析
            dit_analyzer = DiTLayerImplementationAnalyzer(self.cache_dir)
            dit_analyzer.analyze_dit_implementation_differences()
            
        except Exception as e:
            print(f"❌ DiT 层实现差异分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def save_debug_data(self):
        """保存调试数据"""
        print(f"\n{'='*60}")
        print(f"🔍 保存调试数据")
        print(f"{'='*60}")
        
        try:
            from indextts.utils.cfm_debugger import get_debugger, save_cfm_debug
            
            # 获取全局调试器
            debugger = get_debugger()
            
            # 检查调试数据
            pytorch_stages = len(debugger.debug_data.get('pytorch', {}))
            mlx_stages = len(debugger.debug_data.get('mlx', {}))
            
            print(f"📊 调试数据统计:")
            print(f"   PyTorch 数据: {pytorch_stages} 个阶段")
            print(f"   MLX 数据: {mlx_stages} 个阶段")
            
            if pytorch_stages > 0 or mlx_stages > 0:
                # 保存调试数据
                debug_file = save_cfm_debug("complete_dit_debug_analysis.pkl")
                print(f"💾 调试数据已保存到: {debug_file}")
                
                # 打印详细的调试数据信息
                print(f"\n📊 PyTorch 调试数据详情:")
                for stage, data in debugger.debug_data.get('pytorch', {}).items():
                    print(f"   {stage}: {len(data)} 个张量")
                    for key, value in data.items():
                        if hasattr(value, 'shape'):
                            print(f"     {key}: {value.shape}")
                        else:
                            print(f"     {key}: {type(value)}")
                
                print(f"\n📊 MLX 调试数据详情:")
                for stage, data in debugger.debug_data.get('mlx', {}).items():
                    print(f"   {stage}: {len(data)} 个张量")
                    for key, value in data.items():
                        if hasattr(value, 'shape'):
                            print(f"     {key}: {value.shape}")
                        else:
                            print(f"     {key}: {type(value)}")
            else:
                print("❌ 没有找到调试数据")
                
        except Exception as e:
            print(f"❌ 保存调试数据失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    analyzer = CFMStageAnalyzer()
    analyzer.analyze_all_stages()

if __name__ == "__main__":
    main()
