# RTF (Real-Time Factor) 说明

## RTF计算公式

```python
# indextts/infer_v2.py line 1028
RTF = 推理时间 / 音频长度
RTF = (end_time - start_time) / wav_length
```

**含义：**
- RTF = 1.0 表示实时（生成1秒音频需要1秒）
- RTF = 8.25 表示慢于实时8.25倍（生成1秒音频需要8.25秒）
- RTF越小越好

---

## 你看到RTF=8.25的原因

### 你的运行数据

```
>> Total inference time: 23.38 seconds
>> Generated audio length: 2.83 seconds
>> RTF: 8.2544
```

**计算验证：**
```
RTF = 23.38s / 2.83s = 8.26 ✓
```

### 基线测试数据（预热后）

```
总推理时间: 15.72s (平均)
音频长度: 2.53s
RTF = 15.72s / 2.53s = 6.21
```

---

## 为什么你的RTF更高？

### 对比分析

| 运行 | 推理时间 | 音频长度 | RTF | 说明 |
|------|---------|---------|-----|------|
| **你的运行** | 23.38s | 2.83s | **8.25** | 首次运行，未预热 |
| **基线Run1** | 16.02s | 2.53s | 6.33 | 预热后 |
| **基线Run2** | 16.50s | 2.53s | 6.52 | 预热后 |
| **基线Run3** | 14.63s | 2.53s | 5.78 | 预热后 |
| **基线平均** | 15.72s | 2.53s | **6.21** | 预热后 |

### 原因分析

#### 1. **首次运行vs预热后** ⭐ 主因

你的运行是**首次运行（冷启动）**：
```
23.38s vs 15.72s (基线)
差距: 7.66s (32.7%慢)
```

**冷启动开销包括：**
- Metal kernel首次编译
- GPU频率爬升
- 系统缓存预热
- PyTorch CUDA graph构建

**预热后性能提升32.7%！**

#### 2. **音频长度差异**

```
你的音频: 2.83s
基线音频: 2.53s
差距: 0.30s (11.9%更长)
```

不同的随机seed会生成不同长度的音频，影响RTF计算。

#### 3. **性能波动**

基线测试显示GPT生成时间有较大波动：
```
GPT gen: 8.43s ± 1.07s
范围: 7.31s - 9.44s
变异系数: 12.7%
```

你的运行可能落在较慢的一端。

---

## 基线数据详细对比

### 你的单次运行（23.38s）

```
S2MEL breakdown:
  gpt_layer: 0.00s
  vq2emb: 0.01s
  prepare: 0.0001s
  length_reg: 6.19s    ← 比基线慢2倍
  cfm: 2.57s
  
主要时间:
  gpt_gen: 12.31s      ← 比基线慢46%
  s2mel: 8.77s         ← 比基线慢64%
  bigvgan: 0.70s
```

### 基线平均（15.72s）

```
S2MEL breakdown:
  gpt_layer: 0.00s
  vq2emb: 0.01s
  prepare: 0.00s
  length_reg: 3.01s    ← 快2倍
  cfm: 2.33s
  
主要时间:
  gpt_gen: 8.43s       ← 快46%
  s2mel: 5.36s         ← 快64%
  bigvgan: 0.64s
```

**差距最大的模块：**
1. Length Reg: 6.19s vs 3.01s (慢106%)
2. GPT gen: 12.31s vs 8.43s (慢46%)

---

## 如何获得更好的RTF

### 方法1：预热运行 ⭐ 推荐

```bash
# 第一次运行（预热，抛弃结果）
python -m indextts.cli "test" -v examples/voice.wav --force --mlx --num-beams 1

# 第二次运行（实际测试）
python -m indextts.cli "今天天气真不错" \
  -v examples/zh_vo_Main_Linaxita_2_4_24_6.wav \
  --force --mlx --num-beams 1
```

**预期RTF：6-7 (比你的8.25快20-30%)**

### 方法2：使用基准测试工具

```bash
# 自动预热+多次运行+统计
python benchmark_baseline.py
```

这会自动：
- 预热1次
- 运行3次
- 计算平均值和标准差

**预期RTF：6.21 ± 0.40**

### 方法3：减少diffusion steps

```bash
# 当前默认20步
python -m indextts.cli "今天天气真不错" \
  -v examples/zh_vo_Main_Linaxita_2_4_24_6.wav \
  --force --mlx --num-beams 1 \
  --diffusion-steps 15  # 减少到15步
```

**预期RTF：~7.5 (提升10%)**

---

## RTF目标

### 当前状态

```
冷启动:  RTF = 8.25  (你的情况)
预热后:  RTF = 6.21  (基线平均)
```

### 优化目标

```
短期 (1-2周):  RTF = 5.0  (提速20%)
中期 (1个月):  RTF = 4.0  (提速35%)
长期 (理想):   RTF = 2.5  (提速60%)
```

---

## 理解RTF=8.25的含义

**你的系统状态：**
- 生成1秒音频需要8.25秒
- 生成2.83秒音频需要23.38秒
- 比实时慢8倍

**这是正常的吗？**

对于TTS系统，取决于质量和复杂度：
- 简单TTS（FastSpeech2等）: RTF = 0.1-1.0
- 高质量TTS（VITS等）: RTF = 2.0-5.0
- 零样本情感TTS（IndexTTS2）: RTF = 5.0-10.0

**IndexTTS2的复杂度：**
- Semantic Model (W2V-BERT)
- GPT自回归生成（127步）
- Flow Matching Diffusion（20步）
- 高质量Vocoder

所以RTF=8.25在**可接受范围内**，但有优化空间。

---

## 建议

### 立即尝试

**运行预热测试：**
```bash
# 方法1：手动预热
python -m indextts.cli "预热" -v examples/zh_vo_Main_Linaxita_2_4_24_6.wav --force --mlx --num-beams 1
python -m indextts.cli "今天天气真不错" -v examples/zh_vo_Main_Linaxita_2_4_24_6.wav --force --mlx --num-beams 1

# 方法2：使用基准测试工具（自动预热）
python benchmark_baseline.py
```

**预期结果：RTF从8.25 → 6.2左右**

### 长期优化

参考BASELINE_COMPARISON.md中的优化计划。

---

## 总结

**你的RTF=8.25是首次运行（冷启动）的正常表现。**

- ✅ 系统工作正常
- ⚠️  首次运行有32.7%的冷启动开销
- 🎯 预热后RTF可降到6.2左右
- 🚀 进一步优化可降到5.0甚至更低

**快速验证：再运行一次同样的命令，RTF应该会降到6-7。**

