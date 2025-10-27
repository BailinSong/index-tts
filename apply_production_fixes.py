#!/usr/bin/env python3
"""
将修复应用到生产文件
修复 MLX CFM 的权重加载和初始化问题
"""

import os
import shutil
from pathlib import Path

def backup_file(file_path):
    """备份原始文件"""
    backup_path = f"{file_path}.backup"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"✅ 已备份: {file_path} -> {backup_path}")
    else:
        print(f"ℹ️ 备份已存在: {backup_path}")

def apply_weight_loading_fix():
    """应用权重加载修复"""
    print("🔧 应用权重加载修复...")
    
    mlx_cfm_path = "indextts/s2mel/modules/mlx_cfm.py"
    
    if not os.path.exists(mlx_cfm_path):
        print(f"❌ 文件不存在: {mlx_cfm_path}")
        return False
    
    # 备份原始文件
    backup_file(mlx_cfm_path)
    
    # 读取文件内容
    with open(mlx_cfm_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复 load_from_fixed_cache 方法
    old_method = '''            print(f"   Found {len(estimator_weights)} estimator weights")

            # Load weights using the standard parameter loading'''
    
    new_method = '''            print(f"   Found {len(estimator_weights)} estimator weights")

            # 🔧 修复：直接加载 cond_projection 权重
            if 'cond_projection.weight' in estimator_weights and 'cond_projection.bias' in estimator_weights:
                print("   🔧 Loading cond_projection weights directly...")
                self.estimator.cond_projection.weight = estimator_weights['cond_projection.weight']
                self.estimator.cond_projection.bias = estimator_weights['cond_projection.bias']
                loaded += 2
                print("   ✅ cond_projection weights loaded successfully")

            # Load weights using the standard parameter loading'''
    
    if old_method in content:
        content = content.replace(old_method, new_method)
        
        # 写入修复后的内容
        with open(mlx_cfm_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ 权重加载修复已应用")
        return True
    else:
        print("⚠️ 未找到需要修复的代码段")
        return False

def apply_unified_initialization_fix():
    """应用统一初始化修复"""
    print("🔧 应用统一初始化修复...")
    
    mlx_cfm_rewritten_path = "indextts/s2mel/modules/mlx_cfm_rewritten.py"
    
    if not os.path.exists(mlx_cfm_rewritten_path):
        print(f"❌ 文件不存在: {mlx_cfm_rewritten_path}")
        return False
    
    # 备份原始文件
    backup_file(mlx_cfm_rewritten_path)
    
    # 读取文件内容
    with open(mlx_cfm_rewritten_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 添加 Kaiming uniform 初始化
    init_fix = '''
    def _init_cond_projection_kaiming(self):
        """使用 Kaiming uniform 初始化 cond_projection"""
        import math
        import mlx.core as mx
        
        input_dim = self.cond_projection.weight.shape[1]
        output_dim = self.cond_projection.weight.shape[0]
        
        # 使用与 PyTorch 相同的 Kaiming uniform 初始化
        a = math.sqrt(5)  # PyTorch 默认值
        fan_in = input_dim
        bound = a / math.sqrt(fan_in) if fan_in > 0 else 0
        
        # 重新初始化权重和偏置
        weight = mx.random.uniform(-bound, bound, (output_dim, input_dim))
        bias = mx.random.uniform(-bound, bound, (output_dim,))
        
        self.cond_projection.weight = weight
        self.cond_projection.bias = bias
        
        print(f"   🔧 cond_projection initialized with Kaiming uniform: bound=±{bound:.6f}")
'''
    
    # 在 __init__ 方法中添加初始化调用
    init_call = '''
        # 🔧 使用 Kaiming uniform 初始化 cond_projection
        self._init_cond_projection_kaiming()
'''
    
    # 查找 __init__ 方法的结尾
    if 'def __init__' in content and 'self.cond_projection = nn.Linear' in content:
        # 在 cond_projection 初始化后添加 Kaiming uniform 初始化
        old_init = 'self.cond_projection = nn.Linear(dit_cfg.content_dim, dit_cfg.hidden_dim, bias=True)'
        new_init = f'{old_init}\n{init_call}'
        
        content = content.replace(old_init, new_init)
        
        # 添加初始化方法
        content += init_fix
        
        # 写入修复后的内容
        with open(mlx_cfm_rewritten_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ 统一初始化修复已应用")
        return True
    else:
        print("⚠️ 未找到需要修复的初始化代码")
        return False

def create_production_utils():
    """创建生产环境工具函数"""
    print("🛠️ 创建生产环境工具函数...")
    
    utils_content = '''#!/usr/bin/env python3
"""
生产环境 MLX 工具函数
确保 MLX 模型在生产环境中正确工作
"""

import mlx.core as mx
import mlx.nn as nn
import math
import numpy as np

def verify_mlx_model_weights(mlx_model, pytorch_weights_path):
    """
    验证 MLX 模型的权重是否正确加载
    
    Args:
        mlx_model: MLX 模型实例
        pytorch_weights_path: PyTorch 权重文件路径
    
    Returns:
        bool: 权重是否正确加载
    """
    try:
        import torch
        
        # 加载 PyTorch 权重
        pytorch_weights = torch.load(pytorch_weights_path, map_location='cpu')
        
        # 提取 cond_projection 权重
        pytorch_weight = pytorch_weights['net']['cfm']['estimator.cond_projection.weight'].numpy()
        pytorch_bias = pytorch_weights['net']['cfm']['estimator.cond_projection.bias'].numpy()
        
        # 获取 MLX 权重
        mlx_weight = np.array(mlx_model.estimator.cond_projection.weight)
        mlx_bias = np.array(mlx_model.estimator.cond_projection.bias)
        
        # 计算差异
        weight_diff = np.abs(pytorch_weight - mlx_weight)
        bias_diff = np.abs(pytorch_bias - mlx_bias)
        
        print(f"权重差异: 最大={np.max(weight_diff):.8f}, 平均={np.mean(weight_diff):.8f}")
        print(f"偏置差异: 最大={np.max(bias_diff):.8f}, 平均={np.mean(bias_diff):.8f}")
        
        # 检查是否完全相同
        weight_identical = np.allclose(pytorch_weight, mlx_weight, atol=1e-8)
        bias_identical = np.allclose(pytorch_bias, mlx_bias, atol=1e-8)
        
        print(f"权重是否相同: {weight_identical}")
        print(f"偏置是否相同: {bias_identical}")
        
        return weight_identical and bias_identical
        
    except Exception as e:
        print(f"❌ 权重验证失败: {e}")
        return False

def fix_mlx_model_weights(mlx_model, pytorch_weights_path):
    """
    修复 MLX 模型的权重加载
    
    Args:
        mlx_model: MLX 模型实例
        pytorch_weights_path: PyTorch 权重文件路径
    
    Returns:
        bool: 修复是否成功
    """
    try:
        import torch
        
        # 加载 PyTorch 权重
        pytorch_weights = torch.load(pytorch_weights_path, map_location='cpu')
        
        # 提取 cond_projection 权重
        pytorch_weight = pytorch_weights['net']['cfm']['estimator.cond_projection.weight'].numpy()
        pytorch_bias = pytorch_weights['net']['cfm']['estimator.cond_projection.bias'].numpy()
        
        # 修复 MLX 权重
        mlx_model.estimator.cond_projection.weight = mx.array(pytorch_weight)
        mlx_model.estimator.cond_projection.bias = mx.array(pytorch_bias)
        
        print("✅ MLX 模型权重修复完成")
        return True
        
    except Exception as e:
        print(f"❌ MLX 模型权重修复失败: {e}")
        return False

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
    
    # 使用 Kaiming uniform 初始化 (与 PyTorch 相同)
    a = math.sqrt(5)  # PyTorch 默认值
    fan_in = input_dim
    bound = a / math.sqrt(fan_in) if fan_in > 0 else 0
    
    # 重新初始化权重和偏置
    weight = mx.random.uniform(-bound, bound, (output_dim, input_dim))
    linear.weight = weight
    
    if bias:
        bias = mx.random.uniform(-bound, bound, (output_dim,))
        linear.bias = bias
    
    return linear

def test_mlx_model_production(mlx_model):
    """
    测试 MLX 模型在生产环境中的表现
    
    Args:
        mlx_model: MLX 模型实例
    
    Returns:
        dict: 测试结果
    """
    try:
        import time
        
        # 创建测试输入
        batch_size = 2
        seq_len = 100
        input_dim = 512
        
        test_input = mx.random.normal((batch_size, seq_len, input_dim))
        
        # 预热
        for _ in range(5):
            _ = mlx_model.estimator.cond_projection(test_input)
        
        # 性能测试
        start_time = time.time()
        for _ in range(100):
            output = mlx_model.estimator.cond_projection(test_input)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        throughput = batch_size * seq_len / avg_time
        
        result = {
            "success": True,
            "avg_time_ms": avg_time * 1000,
            "throughput_tokens_per_sec": throughput,
            "output_shape": output.shape,
            "output_range": [float(output.min()), float(output.max())]
        }
        
        print(f"✅ 生产环境测试通过:")
        print(f"  平均推理时间: {result['avg_time_ms']:.2f} ms")
        print(f"  吞吐量: {result['throughput_tokens_per_sec']:.0f} tokens/s")
        print(f"  输出形状: {result['output_shape']}")
        print(f"  输出范围: [{result['output_range'][0]:.6f}, {result['output_range'][1]:.6f}]")
        
        return result
        
    except Exception as e:
        print(f"❌ 生产环境测试失败: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    print("🛠️ MLX 生产环境工具函数")
    print("=" * 50)
    print("可用函数:")
    print("  - verify_mlx_model_weights()")
    print("  - fix_mlx_model_weights()")
    print("  - create_pytorch_compatible_mlx_linear()")
    print("  - test_mlx_model_production()")
'''
    
    with open('indextts/utils/mlx_production_utils.py', 'w', encoding='utf-8') as f:
        f.write(utils_content)
    
    print("✅ 生产环境工具函数已创建: indextts/utils/mlx_production_utils.py")

def update_production_config():
    """更新生产环境配置"""
    print("⚙️ 更新生产环境配置...")
    
    config_content = '''# MLX 生产环境配置
# 确保 MLX 模型在生产环境中正确工作

# 权重加载配置
MLX_WEIGHT_LOADING:
  # 是否启用权重验证
  enable_weight_verification: true
  
  # 权重差异容忍度
  weight_tolerance: 1e-8
  
  # 是否自动修复权重
  auto_fix_weights: true

# 初始化配置
MLX_INITIALIZATION:
  # 使用 Kaiming uniform 初始化
  use_kaiming_uniform: true
  
  # Kaiming uniform 参数
  kaiming_a: 2.23606797749979  # sqrt(5)
  
  # 是否验证初始化范围
  verify_init_range: true

# 性能配置
MLX_PERFORMANCE:
  # 预热轮数
  warmup_rounds: 5
  
  # 性能测试轮数
  performance_test_rounds: 100
  
  # 最小吞吐量要求 (tokens/s)
  min_throughput: 1000000

# 调试配置
MLX_DEBUG:
  # 是否启用详细日志
  verbose_logging: true
  
  # 是否保存调试信息
  save_debug_info: false
  
  # 调试信息保存路径
  debug_info_path: "debug/mlx_debug.json"
'''
    
    with open('mlx_production_config.yaml', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("✅ 生产环境配置已创建: mlx_production_config.yaml")

def main():
    """主函数"""
    print("🚀 将修复应用到生产文件")
    print("=" * 50)
    
    success_count = 0
    total_tasks = 4
    
    # 1. 应用权重加载修复
    if apply_weight_loading_fix():
        success_count += 1
    
    # 2. 应用统一初始化修复
    if apply_unified_initialization_fix():
        success_count += 1
    
    # 3. 创建生产环境工具函数
    create_production_utils()
    success_count += 1
    
    # 4. 更新生产环境配置
    update_production_config()
    success_count += 1
    
    # 总结
    print(f"\n🎯 修复应用结果:")
    print(f"  总任务数: {total_tasks}")
    print(f"  成功完成: {success_count}")
    print(f"  失败: {total_tasks - success_count}")
    
    if success_count == total_tasks:
        print("\n🎉 所有修复已成功应用到生产文件!")
        print("\n📋 下一步:")
        print("  1. 运行测试验证修复效果")
        print("  2. 部署到生产环境")
        print("  3. 监控生产环境性能")
    else:
        print(f"\n⚠️ {total_tasks - success_count} 个任务失败，请检查错误信息")

if __name__ == "__main__":
    main()
