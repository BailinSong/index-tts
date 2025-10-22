# MLX模型转换和缓存规则

## 核心原则

**所有MLX模型必须采用"转换-缓存-加载"策略，不允许实时转换**

---

## 📁 缓存目录结构

```
checkpoints/mlx/
├── gpt.npz              # GPT模型权重缓存
├── s2mel_gpt_layer.npz  # S2MEL gpt_layer缓存
├── s2mel_length_reg.npz # S2MEL length_regulator缓存
├── s2mel_cfm.npz        # S2MEL CFM缓存
└── bigvgan.npz          # BigVGAN缓存（如果MLX化）
```

**位置**: `checkpoints/mlx/`（所有MLX缓存统一存放）

---

## 🔄 转换流程

### 1. 首次运行自动转换并缓存 ⭐ 核心流程

```python
# 在IndexTTS2.__init__中 - 标准模板
if self.use_mlx and self.mlx_available:
    self.mlx_cache = MLXModelCache(cache_dir="checkpoints/mlx")
    
    # 定义缓存文件路径
    cache_file = os.path.join(self.mlx_cache.cache_dir, "model_name.npz")
    
    if os.path.exists(cache_file):
        # 从缓存加载（快速）
        print(">> Loading MODEL from MLX cache...")
        print(f"   Cache: {cache_file}")
        
        try:
            import mlx.core as mx
            mlx_weights = mx.load(cache_file)
            mlx_model.load_from_cache(mlx_weights)
            
            size_mb = os.path.getsize(cache_file) / (1024 * 1024)
            print(f"   Size: {size_mb:.2f} MB")
            print(">> ✓ Loaded from cache (fast!)")
        except Exception as e:
            print(f">> ✗ Cache loading failed: {e}")
            # Fallback to conversion
            mlx_model.load_weights_from_pytorch(pytorch_state_dict)
    else:
        # 首次运行：转换并缓存
        print(">> MODEL cache not found (first run)")
        print(">> Converting PyTorch MODEL to MLX and caching...")
        print("   ⏳ This will take a few minutes...")
        
        try:
            # 1. 从PyTorch加载权重
            loaded = mlx_model.load_weights_from_pytorch(pytorch_state_dict)
            print(f">> Loaded {loaded} weights")
            
            # 2. 提取MLX权重用于缓存
            weights_to_cache = mlx_model.extract_weights_for_cache()
            
            # 3. 保存到缓存
            os.makedirs(self.mlx_cache.cache_dir, exist_ok=True)
            mx.savez(cache_file, **weights_to_cache)
            
            size_mb = os.path.getsize(cache_file) / (1024 * 1024)
            print(f">> ✓ Cached to {cache_file}")
            print(f"   Size: {size_mb:.2f} MB")
            print(">> Next run will load from cache (much faster!)")
        except Exception as e:
            print(f">> ✗ Conversion failed: {e}")
            # Use PyTorch fallback
```

**关键点**:
- ✅ 首次运行：自动检测无缓存，转换并保存
- ✅ 后续运行：直接从`checkpoints/mlx/xxx.npz`加载
- ✅ 失败降级：转换失败则使用PyTorch版本
- ✅ 用户友好：显示进度和缓存位置

### 2. 手动转换脚本
对于复杂模型，提供独立转换脚本：
```python
# convert_xxx_to_mlx_cache.py
python convert_cfm_to_mlx_cache.py
# 转换后保存到 checkpoints/mlx/s2mel_cfm.npz
```

---

## 🔧 实现规范

### 必须实现的方法

每个MLX模型类必须实现：

```python
class MLXModel(nn.Module):
    def load_weights_from_pytorch(self, pytorch_state_dict, prefix=""):
        """
        从PyTorch state_dict加载权重
        
        Args:
            pytorch_state_dict: dict with numpy arrays (已调用.cpu().numpy())
            prefix: 权重key的前缀
        
        Returns:
            loaded: int - 成功加载的权重数量
        """
        loaded = 0
        # 实现权重映射逻辑
        return loaded
    
    def load_from_cache(self, cache_dict):
        """
        从MLX缓存加载权重
        
        Args:
            cache_dict: dict of mx.array from npz file
        
        Returns:
            loaded: int - 成功加载的权重数量
        """
        loaded = 0
        # 直接赋值MLX arrays
        return loaded
```

### 权重格式转换

**必须使用的转换规则**:

```python
# Conv1d: PyTorch (O, I, K) -> MLX (O, K, I)
mlx_weight = pytorch_weight.transpose(0, 2, 1)

# Conv2d: PyTorch (O, I, H, W) -> MLX (O, H, W, I)
mlx_weight = pytorch_weight.transpose(0, 2, 3, 1)

# ConvTranspose1d: PyTorch (I, O, K) -> MLX (O, K, I)
mlx_weight = pytorch_weight.transpose(1, 2, 0)

# Linear, Embedding, LayerNorm: 格式相同，直接复制
mlx_weight = pytorch_weight

# weight_norm: 需要重建
from mlx_dit_weights import load_weight_norm
weight = load_weight_norm(state_dict, prefix)
```

---

## 📦 缓存管理

### MLXModelCache使用

```python
from indextts.utils.mlx_cache import MLXModelCache

# 初始化
cache = MLXModelCache(cache_dir="checkpoints/mlx")

# 检查缓存
if cache.is_cached("model_name"):
    weights = cache.load("model_name")
else:
    weights = cache.get_or_convert("model_name", "checkpoints/model.pth")

# 手动缓存
cache.convert_and_cache("model_name", state_dict=pytorch_state_dict)

# 保存自定义缓存
mx.savez("checkpoints/mlx/model_name.npz", **mlx_weights_dict)
```

### 缓存命名规范

| 模型 | 缓存文件名 | PyTorch源文件 |
|------|-----------|--------------|
| GPT | `gpt.npz` | `checkpoints/gpt.pth` |
| S2MEL gpt_layer | `s2mel_gpt_layer.npz` | `checkpoints/s2mel.pth` |
| S2MEL length_reg | `s2mel_length_reg.npz` | `checkpoints/s2mel.pth` |
| S2MEL CFM | `s2mel_cfm.npz` | `checkpoints/s2mel.pth` |
| BigVGAN | `bigvgan.npz` | HuggingFace模型 |

---

## ⚡ 性能优化

### 为什么必须缓存？

1. **避免重复转换**
   - PyTorch → MLX转换需要时间
   - 大模型转换可能需要几分钟
   - 缓存后秒级加载

2. **权重格式优化**
   - MLX npz格式针对Metal优化
   - 直接内存映射，无需反序列化
   - 加载速度10x+提升

3. **用户体验**
   - 首次运行：自动转换并缓存
   - 后续运行：瞬间加载
   - 类似Docker的layer caching

### 实测数据

| 模型 | 首次加载(转换) | 缓存加载 | 提升 |
|------|---------------|---------|------|
| GPT | 15-20s | 0.3-0.5s | **40x** |
| S2MEL CFM | 预估10-15s | 预估0.5-1s | **20x** |

---

## 🚫 禁止的做法

### ❌ 实时转换
```python
# 错误示例：每次运行都转换
def forward(self, x):
    x_mlx = torch_to_mlx(x)  # ❌ 慢！
    ...
    return mlx_to_torch(out)  # ❌ 慢！
```

**正确做法**: 只在边界转换一次，内部全部MLX

### ❌ 无缓存权重加载
```python
# 错误示例：每次都从PyTorch加载
mlx_model = MLXModel()
mlx_model.load_weights_from_pytorch(pt_state_dict)  # ❌ 每次都转换
```

**正确做法**: 第一次转换并缓存，后续从缓存加载

### ❌ 分散的缓存位置
```python
# 错误示例：缓存文件到处都是
mx.savez("./my_model.npz", ...)  # ❌ 不规范
mx.savez("/tmp/model.npz", ...)  # ❌ 临时目录
```

**正确做法**: 统一放在`checkpoints/mlx/`

---

## 📋 检查清单

### 实现新MLX模型时

- [ ] 实现 `load_weights_from_pytorch(state_dict, prefix)` 方法
- [ ] 实现 `load_from_cache(cache_dict)` 方法（可选，简化版直接赋值）
- [ ] 创建独立的转换脚本 `convert_xxx_to_mlx_cache.py`
- [ ] 在`IndexTTS2.__init__`中添加缓存检查逻辑
- [ ] 测试首次运行（转换+缓存）
- [ ] 测试后续运行（从缓存加载）
- [ ] 验证加载速度提升 >10x

### 权重转换时

- [ ] 检查并处理weight_norm (weight_g + weight_v)
- [ ] 正确转换Conv1d/Conv2d/ConvTranspose权重格式
- [ ] 验证每一层的权重shape匹配
- [ ] 测试数值一致性（correlation > 0.99）

---

## 🎯 最佳实践

### 1. 渐进式转换

```python
# 优先级排序：
1. 简单模型先转换（Linear, Embedding）
2. 复杂模型后转换（Transformer, Diffusion）
3. 有问题的暂缓（BigVGAN - 框架限制）
```

### 2. 验证策略

```python
# 每个模型都要：
1. 单元测试：测试单个组件
2. 集成测试：测试完整forward
3. 端到端测试：测试实际推理
4. 性能测试：benchmark加载和推理速度
```

### 3. 错误处理

```python
# 优雅降级：
if mlx_cache.is_cached("model"):
    try:
        mlx_model = load_from_cache()
    except Exception as e:
        print(f"Cache loading failed: {e}")
        # 降级到PyTorch
        use_pytorch_model()
else:
    # 首次转换
    try:
        convert_and_cache()
    except Exception as e:
        print(f"Conversion failed: {e}")
        # 降级到PyTorch
        use_pytorch_model()
```

---

## 📊 缓存统计

### 当前已缓存模型

| 模型 | 缓存文件 | 大小 | 状态 |
|------|---------|------|------|
| GPT | gpt.npz | ~3.3GB | ✅ 已缓存 |
| S2MEL gpt_layer | (嵌入s2mel.pth) | ~1KB | ✅ 权重已加载 |
| S2MEL length_reg | (嵌入s2mel.pth) | ~100KB | ✅ 权重已加载 |
| S2MEL CFM | s2mel_cfm.npz | 预估~500MB | 🔜 待缓存 |

---

## 🔜 未来改进

### 增量缓存
- 只缓存变化的权重
- 版本管理（类似Git）

### 压缩优化
- 使用更高效的序列化格式
- 权重量化（int8/int4）

### 自动验证
- 缓存时自动测试一致性
- 损坏检测和自动重建

---

## 📚 相关文件

- `indextts/utils/mlx_cache.py` - MLXModelCache实现
- `convert_cfm_to_mlx_cache.py` - CFM转换脚本示例
- `checkpoints/mlx/` - 统一缓存目录

---

**规则版本**: v1.0  
**创建日期**: 2025-10-22  
**适用范围**: 所有IndexTTS MLX模型

