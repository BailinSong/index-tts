# 第一版优化基准（V1 Baseline）

## 优化内容

✅ Conditioning缓存（Pure MLX GPT + 缓存优化）

## 测试配置

- 文本: []  # 后3次
- Voice: examples/zh_vo_Main_Linaxita_2_4_24_6.wav
- Seed: 42 (固定)
- 测试方法: 4次运行，忽略第1次预热，后3次平均

## V1基准数据

| 指标 | 数值 |
|------|------|
| **平均值（V1基准）** | **10.23s** |
| 中位数 | 10.30s |
| 标准差 | ±0.36s |
| 范围 | 9.84s - 10.55s |
| RTF | 4.04 |

## 详细数据

| Run | 文本 | 时间 |
|-----|------|------|
| 1 | 今天天气真不错 | 10.30s |
| 2 | 今天天气真不错 | 10.55s |
| 3 | 今天天气真不错 | 9.84s |

## 对比V0基准

V0基准（Pure MLX GPT，无缓存）: 16.62s (RTF=6.57)
V1基准（+ Conditioning缓存）: 10.23s (RTF=4.04)

V1 vs V0提升: 6.39s (38.5%)

## 后续优化参考

**所有后续优化都应该与V1基准(10.23s)对比，而非V0。**

测试命令:
```bash
python -m indextts.cli "今天天气真不错" \
  -v examples/zh_vo_Main_Linaxita_2_4_24_6.wav \
  --force --mlx --num-beams 1 --seed 42
```
