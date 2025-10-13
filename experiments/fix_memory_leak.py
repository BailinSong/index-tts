#!/usr/bin/env python3
"""
内存泄漏修复建议
"""

print("""
🔧 内存泄漏快速修复方案
=====================================

基于您的测试数据，主要问题在 GPT Generation 时间持续增长。

问题 1: MLX KV Cache 未清理
-----------------------------------
文件: indextts/gpt/mlx_model.py

在 UnifiedVoiceMLX 的 forward 方法中，KV cache 可能没有正确重置。

修复方法:
```python
# 在每次推理开始前清理 KV cache
if hasattr(self, 'transformer') and hasattr(self.transformer, 'clear_cache'):
    self.transformer.clear_cache()
```

问题 2: PyTorch MPS 内存碎片化
-----------------------------------
文件: indextts/infer_v2.py

在 infer() 方法的最后添加显式清理:

```python
def infer(self, ...):
    try:
        # ... 现有代码 ...
        
    finally:
        # 清理内存
        import gc
        gc.collect()
        
        if self.device == 'mps':
            torch.mps.empty_cache()
        
        # 清理 MLX cache
        import mlx.core as mx
        mx.metal.clear_cache()
```

问题 3: MLX Transformer 状态累积
-----------------------------------
文件: indextts/gpt/mlx_model.py

在 generate_with_kv_cache 方法中，确保每次生成后重置状态:

```python
def generate_with_kv_cache(self, ...):
    # ... 生成代码 ...
    
    # 生成完成后清理
    self.kv_cache = None  # 清理 KV cache
    mx.metal.clear_cache()  # 清理 MLX 显存
    
    return output_ids
```

临时解决方案: 在 webui.py 中
-----------------------------------
```python
# 在每次推理后添加:
import gc
import torch
import mlx.core as mx

gc.collect()
torch.mps.empty_cache()
mx.metal.clear_cache()
```

测试清理是否有效:
-----------------------------------
python experiments/diagnose_memory_leak.py

预期结果:
- 推理时间应该稳定 (不应该持续增长)
- 内存增长应该 < 1 GB (10次推理)

如果问题仍存在:
-----------------------------------
可能需要更深入的修复:
1. 检查 MLX Conformer 是否正确释放中间结果
2. 检查 Perceiver 是否有状态累积
3. 添加显式的 detach() 调用防止梯度累积
""")

