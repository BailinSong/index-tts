"""
调试 MLX 权重存储方式 - 使用 MLX 的参数方法
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

# Try MLX's tree_flatten
print("\nUsing mlx.utils.tree_flatten:")
try:
    from mlx.utils import tree_flatten
    flattened = tree_flatten(model)
    print(f"Flattened: {type(flattened)}")
    print(f"Length: {len(flattened)}")
    for item in flattened[:5]:  # Show first 5
        print(f"  {type(item)}")
except Exception as e:
    print(f"Error: {e}")

# Try accessing parameters directly
print("\nChecking parameters() method:")
if hasattr(model, 'parameters'):
    try:
        params = model.parameters()
        print(f"Parameters type: {type(params)}")
        print(f"Parameters:")
        for k, v in params.items():
            print(f"  {k}: {v.shape if isinstance(v, mx.array) else type(v)}")
    except Exception as e:
        print(f"Error: {e}")

# Try trainable_parameters
print("\nChecking trainable_parameters() method:")
if hasattr(model, 'trainable_parameters'):
    try:
        params = model.trainable_parameters()
        print(f"Trainable parameters type: {type(params)}")
        if isinstance(params, dict):
            print(f"Number of parameters: {len(params)}")
            for k, v in list(params.items())[:5]:
                print(f"  {k}: {v.shape if isinstance(v, mx.array) else type(v)}")
    except Exception as e:
        print(f"Error: {e}")

# Try children
print("\nChecking children() method:")
if hasattr(model, 'children'):
    try:
        children = model.children()
        print(f"Children type: {type(children)}")
        print(f"Children:")
        for k, v in children.items():
            print(f"  {k}: {type(v)}")
    except Exception as e:
        print(f"Error: {e}")

# Try leaf_modules
print("\nChecking leaf_modules() method:")
if hasattr(model, 'leaf_modules'):
    try:
        modules = model.leaf_modules()
        print(f"Leaf modules type: {type(modules)}")
        print(f"Leaf modules:")
        for k, v in modules.items():
            print(f"  {k}: {type(v)}")
    except Exception as e:
        print(f"Error: {e}")

# Check dir
print("\nAll methods and attributes:")
all_attrs = [a for a in dir(model) if not a.startswith('_')]
print(f"Public attributes: {all_attrs}")

