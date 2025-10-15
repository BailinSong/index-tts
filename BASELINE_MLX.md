# IndexTTS2 MLX Baseline

测试配置:
- 文本: 今天天气真不错
- 音频: examples/zh_vo_Main_Linaxita_2_4_24_6.wav
- Seed: 42 (固定)
- 运行次数: 3
- 预热次数: 1

## 性能数据

| 指标 | 平均值 | 标准差 | 最小值 | 最大值 |
|------|--------|--------|--------|--------|
| total_time | 20.61s | 2.05s | 18.38s | 22.41s |
| gpt_gen_time | 8.56s | 1.69s | 6.92s | 10.30s |
| gpt_forward_time | 0.10s | 0.02s | 0.08s | 0.11s |
| s2mel_time | 10.06s | 0.52s | 9.56s | 10.59s |
| bigvgan_time | 0.62s | 0.02s | 0.60s | 0.63s |
| rtf | 3.00s | 0.00s | 3.00s | 3.00s |

## S2MEL分解

| 模块 | 平均值 | 标准差 |
|------|--------|--------|
| s2mel_gpt_layer | 0.00s | 0.00s |
| s2mel_vq2emb | 0.01s | 0.00s |
| s2mel_prepare | 0.00s | 0.00s |
| s2mel_length_reg | 7.56s | 0.67s |
| s2mel_cfm | 2.48s | 0.16s |
