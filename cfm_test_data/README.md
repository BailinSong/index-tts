# CFM 测试数据包

## 文件说明

- `cfm_test_package.pkl`: 完整的数据包，包含所有分析和元数据
- `pytorch_test_data.pkl`: PyTorch 格式的测试数据
- `mlx_test_data.pkl`: MLX 格式的测试数据

## 使用方法

```python
# 加载 PyTorch 测试数据
import pickle
with open('pytorch_test_data.pkl', 'rb') as f:
    pytorch_data = pickle.load(f)

# 加载 MLX 测试数据
with open('mlx_test_data.pkl', 'rb') as f:
    mlx_data = pickle.load(f)
```

## 数据对数量

- 输入数据对: 3
- 输出数据对: 3
