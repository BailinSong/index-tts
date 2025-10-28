# CFM 一致测试数据

## 说明

这个测试数据包使用同一个 PyTorch CFM 前级缓存生成，确保 PyTorch 和 MLX 使用完全相同的输入数据。

## 文件说明

- `consistent_test_data.pkl`: 包含 PyTorch 和 MLX 格式的相同输入数据

## 使用方法

```python
import pickle
with open('consistent_test_data.pkl', 'rb') as f:
    data = pickle.load(f)
    pytorch_inputs = data['pytorch_inputs']
    mlx_inputs = data['mlx_inputs']
```

## 数据形状

### PyTorch 输入:
- mu: [1, 415, 512]
- x_lens: [1]
- prompt: [1, 80, 243]
- style: [1, 192]
- f0: None
- n_timesteps: 25
- temperature: 1.0
- inference_cfg_rate: 0.7
- unified_random_seed: 42
- timestamp: 1761623811.631465
- model_type: pytorch
- codes: [1, 127]
- speech_conditioning_latent: [1, 32, 1280]

### MLX 输入:
- mu: [1, 415, 512]
- x_lens: [1]
- prompt: [1, 80, 243]
- style: [1, 192]
- f0: None
- n_timesteps: 25
- temperature: 1.0
- inference_cfg_rate: 0.7
- unified_random_seed: 42
- timestamp: 1761623811.631465
- model_type: pytorch
- codes: [1, 127]
- speech_conditioning_latent: [1, 32, 1280]
