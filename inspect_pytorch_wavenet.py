"""
检查PyTorch WaveNet的实际配置
"""

import torch
from omegaconf import OmegaConf
import os

from indextts.s2mel.modules.commons import MyModel, load_checkpoint2

# 加载配置
cfg = OmegaConf.load("checkpoints/config.yaml")

# 加载PyTorch模型
print("加载PyTorch WaveNet...")
s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
s2mel_path = os.path.join("checkpoints", cfg.s2mel_checkpoint)
s2mel, _, _, _ = load_checkpoint2(s2mel, None, s2mel_path, load_only_params=True, ignore_modules=[], is_distributed=False)
s2mel.eval()

wavenet = s2mel.models['cfm'].estimator.wavenet

print("\nWaveNet配置:")
print(f"  hidden_channels: {wavenet.hidden_channels}")
print(f"  kernel_size: {wavenet.kernel_size}")
print(f"  dilation_rate: {wavenet.dilation_rate}")
print(f"  n_layers: {wavenet.n_layers}")
print(f"  gin_channels: {wavenet.gin_channels}")

print("\nIn Layers详情:")
for i in range(wavenet.n_layers):
    layer = wavenet.in_layers[i]
    print(f"\nLayer {i}:")
    print(f"  Type: {type(layer).__name__}")
    print(f"  Module: {layer}")
    
    # 检查SConv1d的内部结构
    if hasattr(layer, 'conv'):
        print(f"  Has conv wrapper")
        if hasattr(layer.conv, 'conv'):
            actual_conv = layer.conv.conv
            print(f"    Actual Conv1d:")
            print(f"      kernel_size: {actual_conv.kernel_size}")
            print(f"      stride: {actual_conv.stride}")
            print(f"      dilation: {actual_conv.dilation}")
            print(f"      padding: {actual_conv.padding}")
            print(f"      groups: {actual_conv.groups}")
    
    # 检查causal和pad_mode
    if hasattr(layer, 'causal'):
        print(f"  causal: {layer.causal}")
    if hasattr(layer, 'pad_mode'):
        print(f"  pad_mode: {layer.pad_mode}")

print("\n" + "="*70)
print("关键发现:")
print("  SConv1d在forward时动态计算padding，不是固定的！")
print("  这就是为什么MLX的标准Conv1d（固定padding）会有差异")
print("="*70)

