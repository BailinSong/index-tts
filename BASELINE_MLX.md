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
| total_time | 17.29s | 2.07s | 15.63s | 19.61s |
| gpt_gen_time | 9.42s | 0.95s | 8.46s | 10.36s |
| gpt_forward_time | 0.06s | 0.01s | 0.05s | 0.07s |
| s2mel_time | 5.95s | 1.12s | 5.25s | 7.24s |
| bigvgan_time | 0.63s | 0.01s | 0.62s | 0.63s |
| rtf | 3.00s | 0.00s | 3.00s | 3.00s |

## S2MEL分解

| 模块 | 平均值 | 标准差 |
|------|--------|--------|
| s2mel_gpt_layer | 0.00s | 0.00s |
| s2mel_vq2emb | 0.01s | 0.00s |
| s2mel_prepare | 0.00s | 0.00s |
| s2mel_length_reg | 3.68s | 1.18s |
| s2mel_cfm | 2.26s | 0.06s |
