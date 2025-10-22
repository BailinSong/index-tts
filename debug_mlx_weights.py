"""
调试 MLX 权重存储方式
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
print("\nModel vars():")
for name, value in vars(model).items():
    print(f"  {name}: {type(value)}")
    if isinstance(value, nn.Module):
        print(f"    -> Sub-module vars:")
        for sub_name, sub_value in vars(value).items():
            print(f"       {sub_name}: {type(sub_value)}")
            if isinstance(sub_value, mx.array):
                print(f"         Shape: {sub_value.shape}")

# Test extraction
print("\n\nTesting extraction:")

def flatten_module(module, prefix=""):
    """Recursively flatten MLX module parameters"""
    weights = {}
    
    # Get module's __dict__ to access all attributes
    for name, value in vars(module).items():
        if name.startswith('_'):
            continue
            
        full_name = f"{prefix}.{name}" if prefix else name
        
        # Check if it's an MLX array (parameter)
        if isinstance(value, mx.array):
            weights[full_name] = value
            print(f"Found array: {full_name} {value.shape}")
        # Check if it's an MLX module
        elif isinstance(value, nn.Module):
            print(f"Descending into module: {full_name}")
            weights.update(flatten_module(value, full_name))
        # Check if it's a list of modules
        elif isinstance(value, list) and value and isinstance(value[0], nn.Module):
            print(f"Found list of modules: {full_name}")
            for i, item in enumerate(value):
                weights.update(flatten_module(item, f"{full_name}.{i}"))
    
    return weights

weights = flatten_module(model)
print(f"\nExtracted {len(weights)} weights")
for name in weights.keys():
    print(f"  - {name}")

