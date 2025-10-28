#!/usr/bin/env python3
"""
简单的 MLX CFM 测试
"""

import mlx.core as mx
import numpy as np

# 创建测试数据
print("创建测试数据...")
mu = mx.ones((1, 415, 512))
x_lens = mx.array([415])
prompt = mx.ones((1, 80, 243))
style = mx.ones((1, 192))
f0 = None
n_timesteps = 25
temperature = 1.0
inference_cfg_rate = 0.7

print(f"mu: {mu.shape}")
print(f"x_lens: {x_lens.shape}, value: {x_lens}")
print(f"prompt: {prompt.shape}")
print(f"style: {style.shape}")

# 加载 MLX CFM
print("\n加载 MLX CFM...")
from indextts.infer_v2 import IndexTTS2

tts = IndexTTS2(use_mlx=True)

print("\n执行推理...")
try:
    output = tts.s2mel.models['cfm'].inference(
        mu=mu,
        x_lens=x_lens,
        prompt=prompt,
        style=style,
        f0=f0,
        n_timesteps=n_timesteps,
        temperature=temperature,
        inference_cfg_rate=inference_cfg_rate
    )
    print(f"✅ 推理成功! 输出形状: {output.shape}")
except Exception as e:
    print(f"❌ 推理失败: {e}")
    import traceback
    traceback.print_exc()

