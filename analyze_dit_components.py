#!/usr/bin/env python3
"""
深入分析 DiT 各组件的具体实现差异
比较 Transformer、LayerNorm、Attention、WaveNet 等关键组件
"""

import os
import torch
import mlx.core as mx
import numpy as np
from omegaconf import OmegaConf
from unified_random_generator import UnifiedRandomGenerator
from indextts.s2mel.modules.mlx_flow_matching import MLXCFM
from indextts.infer_v2 import IndexTTS2 as TorchIndexTTS2

class DiTComponentAnalyzer:
    """DiT 组件分析器"""
    
    def __init__(self):
        self.cache_dir = "dit_components_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def create_models(self):
        """创建 PyTorch 和 MLX 模型"""
        print("🔧 初始化模型...")
        
        # PyTorch 模型
        pytorch_tts = TorchIndexTTS2(use_mlx=False)
        pytorch_dit = pytorch_tts.s2mel.models['cfm'].estimator
        
        # MLX 模型
        args = OmegaConf.load("checkpoints/config.yaml")
        mlx_cfm = MLXCFM(args.s2mel)
        npz_path = "checkpoints/mlx/s2mel.npz"
        with np.load(npz_path, allow_pickle=True) as data:
            cache_dict = {k: data[k] for k in data.files}
        mlx_cfm.load_from_cache(cache_dict)
        mlx_dit = mlx_cfm.estimator
        
        print("✅ 模型初始化完成")
        return pytorch_dit, mlx_dit
    
    def prepare_test_data(self, pytorch_dit):
        """准备测试数据"""
        print("\n🔧 准备测试数据...")
        unified_random = UnifiedRandomGenerator(seed=42)
        
        batch_size = 2
        seq_len = 415
        in_channels = 80
        content_dim = 512
        style_dim = 192
        
        # 生成一致的测试数据
        x_pt = unified_random.generate_noise((batch_size, in_channels, seq_len), device='cpu')
        x_mlx = unified_random.generate_noise_mlx((batch_size, in_channels, seq_len))
        
        prompt_x_pt = torch.randn(batch_size, in_channels, seq_len)
        prompt_x_mlx = mx.array(prompt_x_pt.numpy())
        
        x_lens_pt = torch.tensor([415, 415])
        x_lens_mlx = mx.array([415, 415])
        
        t_pt = torch.tensor([0.0, 0.0])
        t_mlx = mx.array([0.0, 0.0])
        
        style_pt = torch.randn(batch_size, style_dim)
        style_mlx = mx.array(style_pt.numpy())
        
        mu_pt = torch.randn(batch_size, seq_len, content_dim)
        mu_mlx = mx.array(mu_pt.numpy())
        
        # 获取 PyTorch 模型的设备并移动数据
        pytorch_device = next(pytorch_dit.parameters()).device
        print(f"   📊 PyTorch 设备: {pytorch_device}")
        
        x_pt = x_pt.to(pytorch_device)
        prompt_x_pt = prompt_x_pt.to(pytorch_device)
        x_lens_pt = x_lens_pt.to(pytorch_device)
        t_pt = t_pt.to(pytorch_device)
        style_pt = style_pt.to(pytorch_device)
        mu_pt = mu_pt.to(pytorch_device)
        
        print("✅ 测试数据准备完成")
        return {
            'pytorch': (x_pt, prompt_x_pt, x_lens_pt, t_pt, style_pt, mu_pt),
            'mlx': (x_mlx, prompt_x_mlx, x_lens_mlx, t_mlx, style_mlx, mu_mlx)
        }
    
    def analyze_embedders(self, pytorch_dit, mlx_dit, data):
        """分析嵌入层差异"""
        print("\n🔍 分析嵌入层差异")
        print("="*50)
        
        x_pt, prompt_x_pt, x_lens_pt, t_pt, style_pt, mu_pt = data['pytorch']
        x_mlx, prompt_x_mlx, x_lens_mlx, t_mlx, style_mlx, mu_mlx = data['mlx']
        
        # 1. x_embedder 分析
        print("📊 x_embedder 分析...")
        with torch.no_grad():
            x_emb_pt = pytorch_dit.x_embedder(x_pt.transpose(1, 2))
        x_emb_mlx = mlx_dit.x_embedder(x_mlx.transpose(0, 2, 1))
        
        self.compare_tensors("x_embedder", x_emb_pt, x_emb_mlx)
        
        # 2. cond_projection 分析
        print("\n📊 cond_projection 分析...")
        with torch.no_grad():
            cond_proj_pt = pytorch_dit.cond_projection(mu_pt)
        cond_proj_mlx = mlx_dit.cond_projection(mu_mlx)
        
        self.compare_tensors("cond_projection", cond_proj_pt, cond_proj_mlx)
        
        # 3. t_embedder 分析
        print("\n📊 t_embedder 分析...")
        with torch.no_grad():
            t_emb_pt = pytorch_dit.t_embedder(t_pt)
        t_emb_mlx = mlx_dit.t_embedder(t_mlx)
        
        self.compare_tensors("t_embedder", t_emb_pt, t_emb_mlx)
        
        return {
            'x_embedder': (x_emb_pt, x_emb_mlx),
            'cond_projection': (cond_proj_pt, cond_proj_mlx),
            't_embedder': (t_emb_pt, t_emb_mlx)
        }
    
    def analyze_cond_x_merge(self, pytorch_dit, mlx_dit, data, embedders):
        """分析条件与输入合并层差异"""
        print("\n🔍 分析条件与输入合并层差异")
        print("="*50)
        
        x_pt, prompt_x_pt, x_lens_pt, t_pt, style_pt, mu_pt = data['pytorch']
        x_mlx, prompt_x_mlx, x_lens_mlx, t_mlx, style_mlx, mu_mlx = data['mlx']
        
        x_emb_pt, x_emb_mlx = embedders['x_embedder']
        cond_proj_pt, cond_proj_mlx = embedders['cond_projection']
        
        # 按照 DiT 的实际实现：x_t + prompt_x_t + cond_proj + style
        print("📊 构建合并输入...")
        
        # PyTorch 版本
        prompt_x_expanded_pt = torch.nn.functional.interpolate(
            prompt_x_pt, size=x_pt.shape[-1], mode='linear', align_corners=False
        )
        x_t_pt = x_pt.transpose(1, 2)  # (batch, seq_len, in_channels)
        prompt_x_t_pt = prompt_x_expanded_pt.transpose(1, 2)  # (batch, seq_len, in_channels)
        style_expanded_pt = style_pt[:, None, :].repeat(1, x_t_pt.shape[1], 1)  # (batch, seq_len, style_dim)
        
        x_cond_pt = torch.cat([x_t_pt, prompt_x_t_pt, cond_proj_pt, style_expanded_pt], dim=-1)
        
        # MLX 版本
        prompt_x_expanded_mlx = mx.interpolate(prompt_x_mlx, size=x_mlx.shape[-1], mode='linear')
        x_t_mlx = x_mlx.transpose(0, 2, 1)  # (batch, seq_len, in_channels)
        prompt_x_t_mlx = prompt_x_expanded_mlx.transpose(0, 2, 1)  # (batch, seq_len, in_channels)
        style_expanded_mlx = mx.broadcast_to(style_mlx[:, None, :], (style_mlx.shape[0], x_t_mlx.shape[1], style_mlx.shape[1]))
        
        x_cond_mlx = mx.concatenate([x_t_mlx, prompt_x_t_mlx, cond_proj_mlx, style_expanded_mlx], axis=-1)
        
        print(f"   PyTorch x_cond: shape={x_cond_pt.shape}")
        print(f"   MLX x_cond:     shape={x_cond_mlx.shape}")
        
        # 合并层分析
        print("\n📊 cond_x_merge_linear 分析...")
        with torch.no_grad():
            merged_pt = pytorch_dit.cond_x_merge_linear(x_cond_pt)
        merged_mlx = mlx_dit.cond_x_merge_linear(x_cond_mlx)
        
        self.compare_tensors("cond_x_merge", merged_pt, merged_mlx)
        
        return merged_pt, merged_mlx
    
    def analyze_transformer_layers(self, pytorch_dit, mlx_dit, merged_pt, merged_mlx, data):
        """分析 Transformer 层差异"""
        print("\n🔍 分析 Transformer 层差异")
        print("="*50)
        
        x_lens_pt = data['pytorch'][2]
        x_lens_mlx = data['mlx'][2]
        t_emb_pt = data['pytorch'][3]
        t_emb_mlx = data['mlx'][3]
        
        # 分析每一层 Transformer
        print("📊 逐层分析 Transformer...")
        
        # 获取 Transformer 层
        pytorch_transformer = pytorch_dit.transformer
        mlx_transformer = mlx_dit.transformer
        
        print(f"   PyTorch Transformer 层数: {len(pytorch_transformer.layers)}")
        print(f"   MLX Transformer 层数: {len(mlx_transformer.layers)}")
        
        # 逐层比较
        current_input_pt = merged_pt
        current_input_mlx = merged_mlx
        
        for i in range(min(len(pytorch_transformer.layers), len(mlx_transformer.layers))):
            print(f"\n📊 第 {i+1} 层 Transformer 分析...")
            
            # PyTorch 层
            with torch.no_grad():
                layer_output_pt = pytorch_transformer.layers[i](
                    current_input_pt, t_emb_pt, x_lens_pt
                )
            
            # MLX 层
            layer_output_mlx = mlx_transformer.layers[i](
                current_input_mlx, t_emb_mlx, x_lens_mlx
            )
            
            self.compare_tensors(f"transformer_layer_{i+1}", layer_output_pt, layer_output_mlx)
            
            # 更新输入用于下一层
            current_input_pt = layer_output_pt
            current_input_mlx = layer_output_mlx
        
        return current_input_pt, current_input_mlx
    
    def analyze_final_layers(self, pytorch_dit, mlx_dit, transformer_output_pt, transformer_output_mlx):
        """分析最终层差异"""
        print("\n🔍 分析最终层差异")
        print("="*50)
        
        # 分析 final_norm
        print("📊 final_norm 分析...")
        with torch.no_grad():
            norm_output_pt = pytorch_dit.final_norm(transformer_output_pt)
        norm_output_mlx = mlx_dit.final_norm(transformer_output_mlx)
        
        self.compare_tensors("final_norm", norm_output_pt, norm_output_mlx)
        
        # 分析 WaveNet 最终层
        print("\n📊 WaveNet 最终层分析...")
        with torch.no_grad():
            final_output_pt = pytorch_dit.final_layer(norm_output_pt)
        final_output_mlx = mlx_dit.final_layer(norm_output_mlx)
        
        self.compare_tensors("final_layer", final_output_pt, final_output_mlx)
        
        return final_output_pt, final_output_mlx
    
    def compare_tensors(self, name, pytorch_tensor, mlx_array):
        """比较 PyTorch 和 MLX 张量"""
        # 转换为 numpy
        if isinstance(pytorch_tensor, torch.Tensor):
            pytorch_np = pytorch_tensor.detach().cpu().numpy()
        else:
            pytorch_np = pytorch_tensor
            
        if isinstance(mlx_array, mx.array):
            mlx_np = np.array(mlx_array)
        else:
            mlx_np = mlx_array
        
        # 比较形状
        if pytorch_np.shape == mlx_np.shape:
            print(f"   ✅ {name} 形状一致: {pytorch_np.shape}")
            
            # 比较数值
            diff = np.abs(pytorch_np - mlx_np)
            max_diff = np.max(diff)
            avg_diff = np.mean(diff)
            
            print(f"   📊 PyTorch: min={np.min(pytorch_np):.6f}, max={np.max(pytorch_np):.6f}, avg={np.mean(pytorch_np):.6f}")
            print(f"   📊 MLX:     min={np.min(mlx_np):.6f}, max={np.max(mlx_np):.6f}, avg={np.mean(mlx_np):.6f}")
            print(f"   📊 差异:    max={max_diff:.6f}, avg={avg_diff:.6f}")
            
            if max_diff < 1e-6:
                print(f"   ✅ {name} 完全一致")
            else:
                print(f"   ❌ {name} 存在差异")
                # 找出差异最大的位置
                max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
                print(f"   💡 最大差异位置: {max_diff_idx}, 差异值: {max_diff:.6f}")
                
                # 分析差异分布
                print(f"   💡 差异统计:")
                print(f"      - 差异 > 1.0: {np.sum(diff > 1.0)} 个元素")
                print(f"      - 差异 > 0.1: {np.sum(diff > 0.1)} 个元素")
                print(f"      - 差异 > 0.01: {np.sum(diff > 0.01)} 个元素")
        else:
            print(f"   ❌ {name} 形状不匹配: PyTorch={pytorch_np.shape}, MLX={mlx_np.shape}")
    
    def run_analysis(self):
        """运行完整分析"""
        print("🔍 DiT 组件详细分析")
        print("="*60)
        
        # 创建模型
        pytorch_dit, mlx_dit = self.create_models()
        
        # 准备测试数据
        data = self.prepare_test_data(pytorch_dit)
        
        # 分析嵌入层
        embedders = self.analyze_embedders(pytorch_dit, mlx_dit, data)
        
        # 分析条件与输入合并层
        merged_pt, merged_mlx = self.analyze_cond_x_merge(pytorch_dit, mlx_dit, data, embedders)
        
        # 分析 Transformer 层
        transformer_output_pt, transformer_output_mlx = self.analyze_transformer_layers(
            pytorch_dit, mlx_dit, merged_pt, merged_mlx, data
        )
        
        # 分析最终层
        final_output_pt, final_output_mlx = self.analyze_final_layers(
            pytorch_dit, mlx_dit, transformer_output_pt, transformer_output_mlx
        )
        
        print("\n🎯 组件分析完成")

def main():
    analyzer = DiTComponentAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
