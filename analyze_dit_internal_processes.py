#!/usr/bin/env python3
"""
深入分析 DiT 估计器内部子过程差异
比较 PyTorch 和 MLX 在每个关键步骤的计算结果
"""

import os
import pickle
import torch
import mlx.core as mx
import numpy as np
from omegaconf import OmegaConf
from unified_random_generator import UnifiedRandomGenerator
from indextts.s2mel.modules.mlx_flow_matching import MLXCFM
from indextts.infer_v2 import IndexTTS2 as TorchIndexTTS2

class DiTInternalAnalyzer:
    """DiT 内部过程分析器"""
    
    def __init__(self):
        self.cache_dir = "dit_internal_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def create_modified_dit(self):
        """创建修改版的 DiT 估计器，添加内部步骤缓存"""
        
        # 创建 PyTorch 版本
        pytorch_tts = TorchIndexTTS2(use_mlx=False)
        pytorch_dit = pytorch_tts.s2mel.models['cfm'].estimator
        
        # 创建 MLX 版本
        args = OmegaConf.load("checkpoints/config.yaml")
        mlx_cfm = MLXCFM(args.s2mel)
        npz_path = "checkpoints/mlx/s2mel.npz"
        with np.load(npz_path, allow_pickle=True) as data:
            cache_dict = {k: data[k] for k in data.files}
        mlx_cfm.load_from_cache(cache_dict)
        mlx_dit = mlx_cfm.estimator
        
        return pytorch_dit, mlx_dit
    
    def analyze_dit_forward_pass(self):
        """分析 DiT 前向传播的每个步骤"""
        print("🔍 分析 DiT 内部子过程")
        print("="*60)
        
        # 创建修改版 DiT
        pytorch_dit, mlx_dit = self.create_modified_dit()
        
        # 准备测试数据
        print("🔧 准备测试数据...")
        unified_random = UnifiedRandomGenerator(seed=42)
        
        # 生成一致的测试数据
        batch_size = 2
        seq_len = 415
        in_channels = 80
        content_dim = 512
        style_dim = 192
        
        # 生成输入数据
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
        
        # 获取 PyTorch 模型的设备
        pytorch_device = next(pytorch_dit.parameters()).device
        print(f"   📊 PyTorch 设备: {pytorch_device}")
        
        # 将 PyTorch 数据移动到正确设备
        x_pt = x_pt.to(pytorch_device)
        prompt_x_pt = prompt_x_pt.to(pytorch_device)
        x_lens_pt = x_lens_pt.to(pytorch_device)
        t_pt = t_pt.to(pytorch_device)
        style_pt = style_pt.to(pytorch_device)
        mu_pt = mu_pt.to(pytorch_device)
        
        print("✅ 测试数据准备完成")
        
        # 分析 PyTorch DiT 前向传播
        print("\n🔧 分析 PyTorch DiT 前向传播...")
        pytorch_output = self.analyze_pytorch_dit_forward(
            pytorch_dit, x_pt, prompt_x_pt, x_lens_pt, t_pt, style_pt, mu_pt
        )
        
        # 分析 MLX DiT 前向传播
        print("\n🔧 分析 MLX DiT 前向传播...")
        mlx_output = self.analyze_mlx_dit_forward(
            mlx_dit, x_mlx, prompt_x_mlx, x_lens_mlx, t_mlx, style_mlx, mu_mlx
        )
        
        # 比较结果
        print("\n📊 比较 DiT 输出结果...")
        self.compare_dit_outputs(pytorch_output, mlx_output)
    
    def analyze_pytorch_dit_forward(self, dit, x, prompt_x, x_lens, t, style, mu):
        """分析 PyTorch DiT 前向传播的每个步骤"""
        print("   🔍 PyTorch DiT 内部步骤分析...")
        
        # 存储每个步骤的结果
        steps = {}
        
        # 1. 输入嵌入
        print("   📊 步骤1: 输入嵌入...")
        x_emb = dit.x_embedder(x.transpose(1, 2))  # (batch, seq_len, dim)
        steps['x_embedder'] = x_emb.detach().cpu()
        print(f"      x_embedder: shape={x_emb.shape}, min={x_emb.min():.6f}, max={x_emb.max():.6f}")
        
        # 2. 条件投影 (直接使用，不经过 cond_embedder)
        print("   📊 步骤2: 条件投影...")
        cond_proj = dit.cond_projection(mu)  # (batch, seq_len, dim)
        steps['cond_projection'] = cond_proj.detach().cpu()
        print(f"      cond_projection: shape={cond_proj.shape}, min={cond_proj.min():.6f}, max={cond_proj.max():.6f}")
        
        # 3. 时间嵌入
        print("   📊 步骤3: 时间嵌入...")
        t_emb = dit.t_embedder(t)  # (batch, dim)
        steps['t_embedder'] = t_emb.detach().cpu()
        print(f"      t_embedder: shape={t_emb.shape}, min={t_emb.min():.6f}, max={t_emb.max():.6f}")
        
        # 4. 条件与输入合并
        print("   📊 步骤4: 条件与输入合并...")
        # 扩展 prompt_x 到与 x 相同的长度
        prompt_x_expanded = torch.nn.functional.interpolate(
            prompt_x, size=x.shape[-1], mode='linear', align_corners=False
        )
        # 按照 DiT 的实际实现：x_t + prompt_x_t + cond_proj + style
        x_t = x.transpose(1, 2)  # (batch, seq_len, in_channels)
        prompt_x_t = prompt_x_expanded.transpose(1, 2)  # (batch, seq_len, in_channels)
        
        # 添加 style 条件
        style_expanded = style[:, None, :].repeat(1, x_t.shape[1], 1)  # (batch, seq_len, style_dim)
        
        x_cond = torch.cat([x_t, prompt_x_t, cond_proj, style_expanded], dim=-1)  # (batch, seq_len, 80+80+512+192=864)
        merged = dit.cond_x_merge_linear(x_cond)  # (batch, seq_len, dim)
        steps['cond_x_merge'] = merged.detach().cpu()
        print(f"      cond_x_merge: shape={merged.shape}, min={merged.min():.6f}, max={merged.max():.6f}")
        
        # 5. 最终输出 (直接调用 DiT 的 forward 方法)
        print("   📊 步骤5: 最终输出...")
        final_output = dit(x, prompt_x, x_lens, t, style, mu)
        steps['final_output'] = final_output.detach().cpu()
        print(f"      final_output: shape={final_output.shape}, min={final_output.min():.6f}, max={final_output.max():.6f}")
        
        # 保存步骤结果
        with open(os.path.join(self.cache_dir, "pytorch_dit_steps.pkl"), 'wb') as f:
            pickle.dump(steps, f)
        
        return steps
    
    def analyze_mlx_dit_forward(self, dit, x, prompt_x, x_lens, t, style, mu):
        """分析 MLX DiT 前向传播的每个步骤"""
        print("   🔍 MLX DiT 内部步骤分析...")
        
        # 存储每个步骤的结果
        steps = {}
        
        # 1. 输入嵌入
        print("   📊 步骤1: 输入嵌入...")
        x_emb = dit.x_embedder(x.transpose(0, 2, 1))  # MLX 版本需要转置到 (batch, seq_len, in_channels)
        steps['x_embedder'] = np.array(x_emb)
        print(f"      x_embedder: shape={x_emb.shape}, min={float(x_emb.min()):.6f}, max={float(x_emb.max()):.6f}")
        
        # 2. 条件投影 (直接使用，不经过 cond_embedder)
        print("   📊 步骤2: 条件投影...")
        cond_proj = dit.cond_projection(mu)  # (batch, seq_len, dim)
        steps['cond_projection'] = np.array(cond_proj)
        print(f"      cond_projection: shape={cond_proj.shape}, min={float(cond_proj.min()):.6f}, max={float(cond_proj.max()):.6f}")
        
        # 3. 时间嵌入
        print("   📊 步骤3: 时间嵌入...")
        t_emb = dit.t_embedder(t)  # (batch, dim)
        steps['t_embedder'] = np.array(t_emb)
        print(f"      t_embedder: shape={t_emb.shape}, min={float(t_emb.min()):.6f}, max={float(t_emb.max()):.6f}")
        
        # 4. 条件与输入合并
        print("   📊 步骤4: 条件与输入合并...")
        # 扩展 prompt_x 到与 x 相同的长度
        prompt_x_expanded = mx.interpolate(prompt_x, size=x.shape[-1], mode='linear')
        # 按照 DiT 的实际实现：x_t + prompt_x_t + cond_proj + style
        x_t = x.transpose(0, 2, 1)  # (batch, seq_len, in_channels)
        prompt_x_t = prompt_x_expanded.transpose(0, 2, 1)  # (batch, seq_len, in_channels)
        
        # 添加 style 条件
        style_expanded = mx.broadcast_to(style[:, None, :], (style.shape[0], x_t.shape[1], style.shape[1]))  # (batch, seq_len, style_dim)
        
        x_cond = mx.concatenate([x_t, prompt_x_t, cond_proj, style_expanded], axis=-1)  # (batch, seq_len, 80+80+512+192=864)
        merged = dit.cond_x_merge_linear(x_cond)  # (batch, seq_len, dim)
        steps['cond_x_merge'] = np.array(merged)
        print(f"      cond_x_merge: shape={merged.shape}, min={float(merged.min()):.6f}, max={float(merged.max()):.6f}")
        
        # 5. 最终输出 (直接调用 DiT 的 forward 方法)
        print("   📊 步骤5: 最终输出...")
        final_output = dit(x, prompt_x, x_lens, t, style, mu)
        steps['final_output'] = np.array(final_output)
        print(f"      final_output: shape={final_output.shape}, min={float(final_output.min()):.6f}, max={float(final_output.max()):.6f}")
        
        # 保存步骤结果
        with open(os.path.join(self.cache_dir, "mlx_dit_steps.pkl"), 'wb') as f:
            pickle.dump(steps, f)
        
        return steps
    
    def compare_dit_outputs(self, pytorch_steps, mlx_steps):
        """比较 DiT 各步骤的输出"""
        print("\n📊 DiT 内部步骤比较分析")
        print("="*60)
        
        # 比较每个步骤
        for step_name in pytorch_steps.keys():
            if step_name in mlx_steps:
                print(f"\n🔍 {step_name} 步骤比较:")
                
                pytorch_data = pytorch_steps[step_name]
                mlx_data = mlx_steps[step_name]
                
                # 转换为 numpy 进行比较
                if isinstance(pytorch_data, torch.Tensor):
                    pytorch_np = pytorch_data.numpy()
                else:
                    pytorch_np = pytorch_data
                
                if isinstance(mlx_data, mx.array):
                    mlx_np = np.array(mlx_data)
                else:
                    mlx_np = mlx_data
                
                # 比较形状
                if pytorch_np.shape == mlx_np.shape:
                    print(f"   ✅ 形状一致: {pytorch_np.shape}")
                    
                    # 比较数值
                    diff = np.abs(pytorch_np - mlx_np)
                    max_diff = np.max(diff)
                    avg_diff = np.mean(diff)
                    
                    print(f"   📊 PyTorch: min={np.min(pytorch_np):.6f}, max={np.max(pytorch_np):.6f}, avg={np.mean(pytorch_np):.6f}")
                    print(f"   📊 MLX:     min={np.min(mlx_np):.6f}, max={np.max(mlx_np):.6f}, avg={np.mean(mlx_np):.6f}")
                    print(f"   📊 差异:    max={max_diff:.6f}, avg={avg_diff:.6f}")
                    
                    if max_diff < 1e-6:
                        print(f"   ✅ {step_name} 完全一致")
                    else:
                        print(f"   ❌ {step_name} 存在差异")
                        # 找出差异最大的位置
                        max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
                        print(f"   💡 最大差异位置: {max_diff_idx}, 差异值: {max_diff:.6f}")
                else:
                    print(f"   ❌ 形状不匹配: PyTorch={pytorch_np.shape}, MLX={mlx_np.shape}")
            else:
                print(f"   ⚠️  {step_name} 步骤在 MLX 中未找到")

def main():
    analyzer = DiTInternalAnalyzer()
    analyzer.analyze_dit_forward_pass()

if __name__ == "__main__":
    main()
