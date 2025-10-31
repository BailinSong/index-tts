#!/usr/bin/env python3
"""
DiT 内部差异分析工具
专门分析 PyTorch 和 MLX DiT 模型的内部计算差异
"""

import os
import pickle
import torch
import mlx.core as mx
import numpy as np
from typing import Dict, Any, List, Tuple

class DiTInternalDiffAnalyzer:
    def __init__(self, cache_dir: str = "cfm_production_cache"):
        self.cache_dir = cache_dir
        
    def load_dit_models(self):
        """加载 PyTorch 和 MLX DiT 模型"""
        try:
            from indextts.infer_v2 import IndexTTS2 as TorchIndexTTS2
            from indextts.s2mel.modules.mlx_flow_matching import MLXCFM
            from omegaconf import OmegaConf
            
            # 加载 PyTorch DiT
            print("🔧 加载 PyTorch DiT 模型...")
            pytorch_tts = TorchIndexTTS2(use_mlx=False)
            pytorch_dit = pytorch_tts.s2mel.models['cfm'].estimator
            
            # 加载 MLX DiT
            print("🔧 加载 MLX DiT 模型...")
            args = OmegaConf.load("checkpoints/config.yaml")
            mlx_cfm = MLXCFM(args.s2mel)
            
            # 从缓存加载权重
            npz_path = "checkpoints/mlx/s2mel.npz"
            if os.path.exists(npz_path):
                with np.load(npz_path, allow_pickle=True) as data:
                    cache_dict = {k: data[k] for k in data.files}
                mlx_cfm.load_from_cache(cache_dict)
                print("✅ MLX 权重加载成功")
            else:
                print("❌ 未找到 MLX 权重缓存")
                return None, None
                
            mlx_dit = mlx_cfm.estimator
            
            return pytorch_dit, mlx_dit
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def load_dit_input_data(self):
        """加载 DiT 输入数据"""
        import glob
        
        # 找到最新的 DiT 输入缓存
        pytorch_files = glob.glob(os.path.join(self.cache_dir, "cfm_pytorch_step_*_dit_input_*.pkl"))
        mlx_files = glob.glob(os.path.join(self.cache_dir, "cfm_mlx_step_*_dit_input_*.pkl"))
        
        if not pytorch_files or not mlx_files:
            print("❌ 未找到 DiT 输入缓存文件")
            return None, None
            
        # 加载最新的文件
        pytorch_file = max(pytorch_files, key=os.path.getmtime)
        mlx_file = max(mlx_files, key=os.path.getmtime)
        
        with open(pytorch_file, 'rb') as f:
            pytorch_data = pickle.load(f)
        with open(mlx_file, 'rb') as f:
            mlx_data = pickle.load(f)
            
        return pytorch_data, mlx_data
    
    def analyze_weight_differences(self, pytorch_dit, mlx_dit):
        """分析 DiT 权重差异"""
        print(f"\n{'='*60}")
        print(f"🔍 DiT 权重差异分析")
        print(f"{'='*60}")
        
        # 分析关键权重层
        weight_layers = [
            ('x_embedder', 'x_embedder'),
            ('t_embedder', 't_embedder'), 
            ('cond_projection', 'cond_projection'),
            ('cond_x_merge_linear', 'cond_x_merge_linear'),
        ]
        
        for layer_name, pytorch_attr in weight_layers:
            if hasattr(pytorch_dit, pytorch_attr) and hasattr(mlx_dit, layer_name):
                self._compare_weight_layer(layer_name, getattr(pytorch_dit, pytorch_attr), getattr(mlx_dit, layer_name))
    
    def _compare_weight_layer(self, layer_name, pytorch_layer, mlx_layer):
        """比较单个权重层"""
        print(f"\n📊 {layer_name} 权重比较:")
        
        try:
            # 获取 PyTorch 权重
            pytorch_weight = None
            pytorch_bias = None
            
            if hasattr(pytorch_layer, 'weight'):
                pytorch_weight = pytorch_layer.weight
            if hasattr(pytorch_layer, 'bias') and pytorch_layer.bias is not None:
                pytorch_bias = pytorch_layer.bias
            
            # 获取 MLX 权重
            mlx_weight = None
            mlx_bias = None
            
            if hasattr(mlx_layer, 'weight'):
                mlx_weight = mlx_layer.weight
            if hasattr(mlx_layer, 'bias') and mlx_layer.bias is not None:
                mlx_bias = mlx_layer.bias
            
            # 比较权重
            if pytorch_weight is not None and mlx_weight is not None:
                pt_w = pytorch_weight.detach().cpu().numpy()
                mlx_w = np.array(mlx_weight)
                
                print(f"   PyTorch weight: {pt_w.shape}, min={pt_w.min():.6f}, max={pt_w.max():.6f}")
                print(f"   MLX weight:     {mlx_w.shape}, min={mlx_w.min():.6f}, max={mlx_w.max():.6f}")
                
                if pt_w.shape == mlx_w.shape:
                    diff = np.abs(pt_w - mlx_w)
                    print(f"   权重差异: max={diff.max():.6f}, mean={diff.mean():.6f}")
                    if diff.max() < 1e-6:
                        print(f"   ✅ {layer_name}.weight 一致")
                    else:
                        print(f"   ❌ {layer_name}.weight 不一致")
                elif pt_w.shape == mlx_w.shape[::-1]:
                    diff = np.abs(pt_w - mlx_w.T)
                    print(f"   权重差异(转置后): max={diff.max():.6f}, mean={diff.mean():.6f}")
                    if diff.max() < 1e-6:
                        print(f"   ✅ {layer_name}.weight 一致(需转置)")
                    else:
                        print(f"   ❌ {layer_name}.weight 不一致(需转置)")
                else:
                    print(f"   ❌ {layer_name}.weight 形状不匹配: {pt_w.shape} vs {mlx_w.shape}")
            
            # 比较偏置
            if pytorch_bias is not None and mlx_bias is not None:
                pt_b = pytorch_bias.detach().cpu().numpy()
                mlx_b = np.array(mlx_bias)
                
                print(f"   PyTorch bias: {pt_b.shape}, min={pt_b.min():.6f}, max={pt_b.max():.6f}")
                print(f"   MLX bias:     {mlx_b.shape}, min={mlx_b.min():.6f}, max={mlx_b.max():.6f}")
                
                if pt_b.shape == mlx_b.shape:
                    diff = np.abs(pt_b - mlx_b)
                    print(f"   偏置差异: max={diff.max():.6f}, mean={diff.mean():.6f}")
                    if diff.max() < 1e-6:
                        print(f"   ✅ {layer_name}.bias 一致")
                    else:
                        print(f"   ❌ {layer_name}.bias 不一致")
                else:
                    print(f"   ❌ {layer_name}.bias 形状不匹配: {pt_b.shape} vs {mlx_b.shape}")
                    
        except Exception as e:
            print(f"   ❌ {layer_name} 权重比较失败: {e}")
    
    def analyze_forward_pass_differences(self, pytorch_dit, mlx_dit, input_data):
        """分析前向传播差异"""
        print(f"\n{'='*60}")
        print(f"🔍 DiT 前向传播差异分析")
        print(f"{'='*60}")
        
        # 提取输入数据
        x = input_data['x']
        prompt_x = input_data['prompt_x'] 
        x_lens = input_data['x_lens']
        t = input_data['t']
        style = input_data['style']
        mu = input_data['mu']
        
        print(f"输入形状: x={x.shape}, prompt_x={prompt_x.shape}, mu={mu.shape}")
        print(f"         style={style.shape}, t={t.shape}, x_lens={x_lens.shape}")
        
        # 转换为 PyTorch 格式
        if isinstance(x, mx.array):
            from indextts.utils.mlx_utils import mlx_to_torch
            x_pt = mlx_to_torch(x, device='cpu')
            prompt_x_pt = mlx_to_torch(prompt_x, device='cpu')
            x_lens_pt = mlx_to_torch(x_lens, device='cpu')
            t_pt = mlx_to_torch(t, device='cpu')
            style_pt = mlx_to_torch(style, device='cpu')
            mu_pt = mlx_to_torch(mu, device='cpu')
        else:
            x_pt = x
            prompt_x_pt = prompt_x
            x_lens_pt = x_lens
            t_pt = t
            style_pt = style
            mu_pt = mu
        
        # 执行 PyTorch 前向传播
        print("\n🔧 执行 PyTorch 前向传播...")
        with torch.no_grad():
            pytorch_output = pytorch_dit(x_pt, prompt_x_pt, x_lens_pt, t_pt, style_pt, mu_pt)
        
        # 执行 MLX 前向传播
        print("🔧 执行 MLX 前向传播...")
        mlx_output = mlx_dit(x, prompt_x, x_lens, t, style, mu)
        
        # 转换为 numpy 进行比较
        pt_out = pytorch_output.detach().cpu().numpy()
        mlx_out = np.array(mlx_output)
        
        print(f"\n📊 输出比较:")
        print(f"   PyTorch: {pt_out.shape}, min={pt_out.min():.6f}, max={pt_out.max():.6f}, avg={pt_out.mean():.6f}")
        print(f"   MLX:     {mlx_out.shape}, min={mlx_out.min():.6f}, max={mlx_out.max():.6f}, avg={mlx_out.mean():.6f}")
        
        if pt_out.shape == mlx_out.shape:
            diff = np.abs(pt_out - mlx_out)
            print(f"   差异: max={diff.max():.6f}, mean={diff.mean():.6f}")
            
            if diff.max() < 1e-6:
                print(f"   ✅ DiT 输出一致")
            else:
                print(f"   ❌ DiT 输出不一致")
                
                # 分析差异分布
                print(f"\n📊 差异分布分析:")
                print(f"   差异 > 1.0: {np.sum(diff > 1.0)} 个元素 ({np.sum(diff > 1.0) / diff.size * 100:.2f}%)")
                print(f"   差异 > 0.1: {np.sum(diff > 0.1)} 个元素 ({np.sum(diff > 0.1) / diff.size * 100:.2f}%)")
                print(f"   差异 > 0.01: {np.sum(diff > 0.01)} 个元素 ({np.sum(diff > 0.01) / diff.size * 100:.2f}%)")
        else:
            print(f"   ❌ 输出形状不匹配: {pt_out.shape} vs {mlx_out.shape}")
    
    def analyze_all(self):
        """执行完整分析"""
        print("🔍 DiT 内部差异分析工具")
        print("="*60)
        
        # 加载模型
        pytorch_dit, mlx_dit = self.load_dit_models()
        if pytorch_dit is None or mlx_dit is None:
            return
        
        # 加载输入数据
        input_data, _ = self.load_dit_input_data()
        if input_data is None:
            return
        
        # 分析权重差异
        self.analyze_weight_differences(pytorch_dit, mlx_dit)
        
        # 分析前向传播差异
        self.analyze_forward_pass_differences(pytorch_dit, mlx_dit, input_data)

def main():
    analyzer = DiTInternalDiffAnalyzer()
    analyzer.analyze_all()

if __name__ == "__main__":
    main()


