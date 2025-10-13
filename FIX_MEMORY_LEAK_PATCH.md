# 内存泄漏修复补丁

## 🚨 问题确认

从您的测试数据看，确实存在严重的内存泄漏和性能衰减：

**GPT Generation 时间持续增长：**
```
Test 1:  6.06s
Test 6: 34.51s  (5.7x 变慢！)
Test 8: 35.34s  (5.8x 变慢！)
```

**RTF 持续恶化：**
```
Test 2:  3.27x  ✅
Test 8: 16.08x  ❌ (5x 恶化！)
```

---

## 🔧 立即修复方案

### 修复 1: 在 infer() 方法添加清理（高优先级）

**文件：** `indextts/infer_v2.py`

**位置：** infer() 方法，在 line 482-491

**替换：**
```python
def infer(self, spk_audio_prompt, text, output_path,
          emo_audio_prompt=None, emo_alpha=1.0,
          emo_vector=None,
          use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
          verbose=False, max_text_tokens_per_segment=120, stream_return=False, more_segment_before=0, **generation_kwargs):
    if stream_return:
        return self.infer_generator(
            spk_audio_prompt, text, output_path,
            emo_audio_prompt, emo_alpha,
            emo_vector,
            use_emo_text, emo_text, use_random, interval_silence,
            verbose, max_text_tokens_per_segment, stream_return, more_segment_before, **generation_kwargs
        )
    else:
        try:
            return list(self.infer_generator(
                spk_audio_prompt, text, output_path,
                emo_audio_prompt, emo_alpha,
                emo_vector,
                use_emo_text, emo_text, use_random, interval_silence,
                verbose, max_text_tokens_per_segment, stream_return, more_segment_before, **generation_kwargs
            ))[0]
        except IndexError:
            return None
```

**改为：**
```python
def infer(self, spk_audio_prompt, text, output_path,
          emo_audio_prompt=None, emo_alpha=1.0,
          emo_vector=None,
          use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
          verbose=False, max_text_tokens_per_segment=120, stream_return=False, more_segment_before=0, **generation_kwargs):
    try:
        if stream_return:
            return self.infer_generator(
                spk_audio_prompt, text, output_path,
                emo_audio_prompt, emo_alpha,
                emo_vector,
                use_emo_text, emo_text, use_random, interval_silence,
                verbose, max_text_tokens_per_segment, stream_return, more_segment_before, **generation_kwargs
            )
        else:
            try:
                return list(self.infer_generator(
                    spk_audio_prompt, text, output_path,
                    emo_audio_prompt, emo_alpha,
                    emo_vector,
                    use_emo_text, emo_text, use_random, interval_silence,
                    verbose, max_text_tokens_per_segment, stream_return, more_segment_before, **generation_kwargs
                ))[0]
            except IndexError:
                return None
    finally:
        # 🔧 清理内存泄漏
        import gc
        gc.collect()
        
        # 清理 PyTorch MPS 缓存
        if self.device == 'mps' and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # 清理 MLX 缓存
        if self.use_mlx:
            import mlx.core as mx
            try:
                mx.metal.clear_cache()
            except:
                pass
```

---

### 修复 2: 清理 MLX Transformer 的 KV Cache（中优先级）

**文件：** `indextts/gpt/mlx_model.py`

**位置：** generate_with_kv_cache() 方法的最后

**添加：**
```python
def generate_with_kv_cache(self, ...):
    # ... 现有生成代码 ...
    
    # 🔧 生成完成后清理 KV cache
    # 防止 cache 累积导致内存泄漏
    try:
        for layer in self.transformer.layers:
            if hasattr(layer, 'kv_cache'):
                layer.kv_cache = None
    except:
        pass
    
    return output_ids
```

---

### 修复 3: 在 webui.py 添加定期清理（临时方案）

**文件：** `webui.py`

**位置：** 在推理调用之后

**添加：**
```python
# 在每次推理后
result = index_tts.infer(...)

# 🔧 立即清理内存
import gc
gc.collect()
torch.mps.empty_cache()
try:
    import mlx.core as mx
    mx.metal.clear_cache()
except:
    pass
```

---

## 🧪 验证修复效果

运行诊断脚本：
```bash
python experiments/diagnose_memory_leak.py
```

**预期结果：**
- ✅ 推理时间应该稳定（不应该持续增长）
- ✅ 第10次推理时间 ≈ 第2-3次（±20%）
- ✅ 内存增长 < 500 MB（10次推理）

---

## 💡 关于 CFM/BigVGAN 的建议

**结论：暂时不实现 CFM/BigVGAN MLX 化**

**理由：**
1. ❌ **内存泄漏更紧迫** - 必须先修复这个问题
2. ❌ **投入产出比低** - 14-20小时只改进 8-17%
3. ✅ **已有 Length Reg 成果** - 隔离测试证明可行（0.57s vs 5.18s）

**更好的优化方向：**
1. 🔧 修复内存泄漏（立即执行）
2. 🔧 优化 GPT Generation（可能有优化空间）
3. 🔧 考虑批处理推理（4-6h，30-50% 改进）

---

## 🚀 立即行动

我现在帮您应用修复 1 和修复 2 吗？

- **选项 A：** 立即修复（推荐，10分钟）
- **选项 B：** 您自己手动修复
- **选项 C：** 先运行诊断脚本，确认问题根因

