"""
调试 MLX 权重存储方式 - 使用 parameters() 深度展开
"""

import mlx.core as mx
import mlx.nn as nn

# 测试简单的MLX模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 10)
    
    def __call__(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

model = SimpleModel()

print("Model created")

def flatten_parameters(params, prefix=""):
    """Recursively flatten nested parameter dict"""
    flat = {}
    for name, value in params.items():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            # Recurse
            flat.update(flatten_parameters(value, full_name))
        elif isinstance(value, mx.array):
            # Leaf parameter
            flat[full_name] = value
        else:
            print(f"Unknown type for {full_name}: {type(value)}")
    return flat

# Get parameters
params = model.parameters()
print(f"\nRaw parameters:")
print(params)

# Flatten
flat_params = flatten_parameters(params)
print(f"\nFlattened parameters: {len(flat_params)}")
for name, value in flat_params.items():
    print(f"  {name}: {value.shape}")

print("\n✅ Successfully extracted parameters!")

