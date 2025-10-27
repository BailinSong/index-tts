# 第一版优化基准（V1 Baseline）

## 优化内容

✅ Conditioning缓存（Pure MLX GPT + 缓存优化）

## 测试配置

- 文本: ['到底应该吃什么', '你为什么不愿意', '今天天气真不错']  # 后3次
- Voice: examples/zh_vo_Main_Linaxita_2_4_24_6.wav
- Seed: 42 (固定)
- 测试方法: 4次运行，忽略第1次预热，后3次平均

## V1基准数据

| 指标 | 数值 |
|------|------|
| **平均值（V1基准）** | **6.66s** |
| 中位数 | 6.93s |
| 标准差 | ±0.99s |
| 范围 | 5.57s - 7.49s |
| RTF | 2.63 |

## 详细数据

| Run | 文本 | 时间 |
|-----|------|------|
| 1 | 到底应该吃什么 | 5.57s |
| 2 | 你为什么不愿意 | 7.49s |
| 3 | 今天天气真不错 | 6.93s |

## 对比V0基准

V0基准（Pure MLX GPT，无缓存）: 16.62s (RTF=6.57)
V1基准（+ Conditioning缓存）: 6.66s (RTF=2.63)

V1 vs V0提升: 9.96s (59.9%)

## 后续优化参考

**所有后续优化都应该与V1基准(6.66s)对比，而非V0。**

测试命令:
```bash
python -m indextts.cli "今天天气真不错" \
  -v examples/zh_vo_Main_Linaxita_2_4_24_6.wav \
  --force --mlx --num-beams 1 --seed 42
```
