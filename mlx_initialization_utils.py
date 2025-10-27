
# MLX 初始化工具函数
import mlx.core as mx
import mlx.nn as nn
import math

def kaiming_uniform_mlx(shape, a=math.sqrt(5)):
    """
    在 MLX 中实现 Kaiming uniform 初始化
    
    Args:
        shape: 权重形状
        a: negative slope of the rectifier used after this layer
    
    Returns:
        MLX array with Kaiming uniform initialization
    """
    fan_in = shape[-1]
    bound = a / math.sqrt(fan_in) if fan_in > 0 else 0
    return mx.random.uniform(-bound, bound, shape)

def init_mlx_linear_like_pytorch(linear_layer, input_dim, output_dim, a=math.sqrt(5)):
    """
    使用与 PyTorch 相同的 Kaiming uniform 初始化 MLX Linear 层
    
    Args:
        linear_layer: MLX Linear layer
        input_dim: input dimension
        output_dim: output dimension
        a: negative slope parameter (default: sqrt(5) for PyTorch)
    """
    fan_in = input_dim
    bound = a / math.sqrt(fan_in) if fan_in > 0 else 0
    
    weight = mx.random.uniform(-bound, bound, (output_dim, input_dim))
    bias = mx.random.uniform(-bound, bound, (output_dim,))
    
    linear_layer.weight = weight
    linear_layer.bias = bias
    
    return linear_layer

def create_pytorch_compatible_mlx_linear(input_dim, output_dim, bias=True):
    """
    创建与 PyTorch 兼容的 MLX Linear 层
    
    Args:
        input_dim: input dimension
        output_dim: output dimension
        bias: whether to use bias
    
    Returns:
        MLX Linear layer with PyTorch-compatible initialization
    """
    linear = nn.Linear(input_dim, output_dim, bias=bias)
    return init_mlx_linear_like_pytorch(linear, input_dim, output_dim)
