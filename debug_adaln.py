"""
调试adaLN模块的权重加载
"""

import torch
import numpy as np
from omegaconf import OmegaConf
import os
import mlx.core as mx
import mlx.nn as nn

from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
from indextts.s2mel.modules.mlx_cfm import MLXCFM

# 加载配置
cfg = OmegaConf.load("checkpoints/config.yaml")

# 加载PyTorch模型
s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
s2mel_path = os.path.join("checkpoints", cfg.s2mel_checkpoint)
s2mel, _, _, _ = load_checkpoint2(s2mel, None, s2mel_path, load_only_params=True, ignore_modules=[], is_distributed=False)
s2mel.eval()

# 加载MLX CFM
mlx_cfm = MLXCFM(cfg.s2mel)
s2mel_state_dict = s2mel.state_dict()
s2mel_state_dict_np = {k: v.cpu().numpy() for k, v in s2mel_state_dict.items()}
mlx_cfm.load_weights_from_pytorch(s2mel_state_dict_np, prefix="models.cfm.")

# 检查权重
print("检查final_layer.adaLN权重:")
prefix = "models.cfm.estimator.final_layer"

# PyTorch
pytorch_final = s2mel.models['cfm'].estimator.final_layer
print(f"\nPyTorch adaLN_modulation:")
print(f"  Type: {type(pytorch_final.adaLN_modulation)}")
print(f"  Modules: {list(pytorch_final.adaLN_modulation.children())}")

if hasattr(pytorch_final.adaLN_modulation, '1'):
    linear_layer = pytorch_final.adaLN_modulation[1]
    print(f"\n  Linear layer (index 1):")
    print(f"    weight shape: {linear_layer.weight.shape}")
    print(f"    bias shape: {linear_layer.bias.shape}")
    print(f"    weight mean: {linear_layer.weight.mean():.6f}")
    print(f"    bias mean: {linear_layer.bias.mean():.6f}")

# MLX
mlx_final = mlx_cfm.estimator.final_layer
print(f"\nMLX final_layer.adaLN_0:")
print(f"  Type: {type(mlx_final.adaLN_0)}")
if hasattr(mlx_final.adaLN_0, 'weight'):
    print(f"  weight shape: {mlx_final.adaLN_0.weight.shape}")
    print(f"  bias shape: {mlx_final.adaLN_0.bias.shape}")
    print(f"  weight mean: {float(mx.mean(mlx_final.adaLN_0.weight)):.6f}")
    print(f"  bias mean: {float(mx.mean(mlx_final.adaLN_0.bias)):.6f}")

# 对比权重
w_torch = linear_layer.weight.data.numpy()
b_torch = linear_layer.bias.data.numpy()
w_mlx = np.array(mlx_final.adaLN_0.weight)
b_mlx = np.array(mlx_final.adaLN_0.bias)

print(f"\n权重对比:")
print(f"  weight diff: {np.abs(w_torch - w_mlx).max():.8f}")
print(f"  bias diff: {np.abs(b_torch - b_mlx).max():.8f}")

# 测试forward
t_input = torch.tensor([0.5])
print(f"\n测试forward with t={t_input}:")

with torch.no_grad():
    # PyTorch
    t_emb_t = s2mel.models['cfm'].estimator.t_embedder(t_input)
    ada_out_t = pytorch_final.adaLN_modulation(t_emb_t)
    print(f"PyTorch:")
    print(f"  t_emb: mean={t_emb_t.mean():.6f}")
    print(f"  ada_out: shape={ada_out_t.shape}, mean={ada_out_t.mean():.6f}")
    shift_t, scale_t = ada_out_t.chunk(2, dim=1)
    print(f"  shift: mean={shift_t.mean():.6f}")
    print(f"  scale: mean={scale_t.mean():.6f}")

# MLX - 新顺序: SiLU first, then Linear
t_emb_m = mlx_cfm.estimator.t_embedder(mx.array([0.5]))
mx.eval(t_emb_m)
t_emb_m_silu = nn.silu(t_emb_m)  # SiLU first
mx.eval(t_emb_m_silu)
ada_out_m = mlx_final.adaLN_0(t_emb_m_silu)  # Then Linear
mx.eval(ada_out_m)
print(f"MLX (新顺序: SiLU → Linear):")
print(f"  t_emb: mean={float(mx.mean(t_emb_m)):.6f}")
print(f"  t_emb_silu: mean={float(mx.mean(t_emb_m_silu)):.6f}")
print(f"  ada_out (after Linear): shape={ada_out_m.shape}, mean={float(mx.mean(ada_out_m)):.6f}")
shift_m = ada_out_m[:, :512]
scale_m = ada_out_m[:, 512:]
print(f"  shift: mean={float(mx.mean(shift_m)):.6f}")
print(f"  scale: mean={float(mx.mean(scale_m)):.6f}")

# 对比ada_out
ada_diff = np.abs(ada_out_t.numpy() - np.array(ada_out_m)).max()
print(f"\nada_out diff: {ada_diff:.6f}")

# 对比shift和scale
shift_diff = np.abs(shift_t.numpy() - np.array(shift_m)).max()
scale_diff = np.abs(scale_t.numpy() - np.array(scale_m)).max()
print(f"shift diff: {shift_diff:.6f}")
print(f"scale diff: {scale_diff:.6f}")

# 检查PyTorch的adaLN_modulation[0]是什么
print(f"\nPyTorch adaLN_modulation[0]: {pytorch_final.adaLN_modulation[0]}")

