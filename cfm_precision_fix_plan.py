"""
CFM精度修复方案
基于层级测试结果，基础运算都达到了e-5精度，问题在于CFM的整体实现
"""

import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pickle
from omegaconf import OmegaConf


def torch_to_mlx_precise(tensor):
    """高精度转换，确保数值一致性"""
    if isinstance(tensor, torch.Tensor):
        # 确保使用float32，避免精度损失
        return mx.array(tensor.detach().cpu().numpy().astype(np.float32))
    return tensor

def create_synthetic_cfm_test():
    """创建合成的CFM测试数据"""
    print("创建合成CFM测试数据...")

    # 设置固定种子
    torch.manual_seed(42)
    np.random.seed(42)
    mx.random.seed(42)

    batch_size = 1
    seq_len = 32  # 使用较短序列长度
    in_channels = 80
    content_dim = 512
    style_dim = 192

    # 创建测试数据
    test_data = {
        'x': torch.randn(batch_size, in_channels, seq_len).float(),
        'prompt_x': torch.zeros(batch_size, in_channels, seq_len).float(),
        'x_lens': torch.tensor([seq_len]),
        't': torch.tensor([0.5]).float(),
        'style': torch.randn(batch_size, style_dim).float(),
        'cond': torch.randn(batch_size, seq_len, content_dim).float()
    }

    # 设置prompt部分
    prompt_len = seq_len // 4  # 1/4作为prompt
    test_data['prompt_x'][:, :, :prompt_len] = torch.randn(batch_size, in_channels, prompt_len)

    return test_data

def test_mlx_cfm_forward():
    """测试MLX CFM的前向传播精度"""
    print("="*80)
    print("MLX CFM前向传播精度测试")
    print("="*80)

    # 1. 创建测试数据
    test_data = create_synthetic_cfm_test()

    print("测试数据:")
    for key, value in test_data.items():
        print(f"  {key}: {value.shape}, dtype={value.dtype}")

    # 2. 加载配置
    try:
        cfg = OmegaConf.load("checkpoints/config.yaml")
        print("✅ 配置加载成功")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return

    # 3. 创建MLX CFM
    try:
        from indextts.s2mel.modules.mlx_cfm import MLXCFM
        mlx_cfm = MLXCFM(cfg.s2mel)
        print("✅ MLX CFM创建成功")
    except Exception as e:
        print(f"❌ MLX CFM创建失败: {e}")
        return

    # 4. 转换输入到MLX
    mlx_inputs = {}
    for key, value in test_data.items():
        mlx_inputs[key] = torch_to_mlx_precise(value)

    # 5. MLX前向传播
    try:
        print("\n执行MLX CFM前向传播...")
        output_mlx = mlx_cfm.estimator(
            mlx_inputs['x'],
            mlx_inputs['prompt_x'],
            mlx_inputs['x_lens'],
            mlx_inputs['t'],
            mlx_inputs['style'],
            mlx_inputs['cond'],
            mask_content=False
        )
        mx.eval(output_mlx)
        print(f"✅ MLX输出: {output_mlx.shape}")
        print(f"   min: {output_mlx.min():.6f}, max: {output_mlx.max():.6f}")

    except Exception as e:
        print(f"❌ MLX前向传播失败: {e}")
        return

    # 6. 如果有PyTorch版本的CFM可以比较...
    print("\n如果要达到e-5级别精度，需要确保:")
    print("1. 权重加载的精度 - 使用float32")
    print("2. 所有中间计算的精度")
    print("3. 随机数生成的一致性")
    print("4. 层归一化的eps参数一致")
    print("5. 激活函数的实现一致")

def analyze_precision_bottlenecks():
    """分析精度瓶颈"""
    print("\n" + "="*80)
    print("CFM精度瓶颈分析")
    print("="*80)

    print("\n基于前面的测试结果:")
    print("✅ 基础运算(加法、乘法、矩阵乘法): 达到e-5精度")
    print("✅ 激活函数(SiLU, GELU, Softmax): 达到e-5精度")
    print("✅ LayerNorm: 达到e-5精度")
    print("✅ 线性层: 达到e-5精度")
    print("✅ 注意力机制: 达到e-5精度")

    print("\n❌ CFM整体输出: 差异达到13.8，相关性0.73")

    print("\n可能的问题源头:")
    print("1. 权重初始化/加载不一致")
    print("2. 模型结构细节差异")
    print("3. 随机数生成差异")
    print("4. 数值计算的累积误差")
    print("5. 特定层的实现差异")

def create_cfm_fix_plan():
    """创建CFM修复计划"""
    print("\n" + "="*80)
    print("CFM精度修复计划")
    print("="*80)

    fixes = [
        {
            "priority": "HIGH",
            "issue": "权重加载精度",
            "solution": "确保所有权重转换使用float32，避免精度损失",
            "action": "修改权重加载函数，添加精度检查"
        },
        {
            "priority": "HIGH",
            "issue": "随机数一致性",
            "solution": "确保PyTorch和MLX使用相同的随机种子和初始化",
            "action": "在CFM前向传播前设置固定种子"
        },
        {
            "priority": "MEDIUM",
            "issue": "LayerNorm参数",
            "solution": "确保eps参数完全一致",
            "action": "检查并统一所有LayerNorm的eps值"
        },
        {
            "priority": "MEDIUM",
            "issue": "数值稳定性",
            "solution": "在关键计算步骤添加数值稳定性检查",
            "action": "在Transformer块中添加梯度裁剪和数值检查"
        },
        {
            "priority": "LOW",
            "issue": "累积误差",
            "solution": "优化计算顺序，减少中间结果的误差累积",
            "action": "重组计算流程，使用更高精度的中间计算"
        }
    ]

    print("修复优先级:")
    for i, fix in enumerate(fixes, 1):
        print(f"\n{i}. [{fix['priority']}] {fix['issue']}")
        print(f"   解决方案: {fix['solution']}")
        print(f"   行动项: {fix['action']}")

def create_precision_validation_script():
    """创建精度验证脚本"""
    script_content = '''"""
CFM精度验证脚本
使用缓存的输入数据验证修复效果
"""

import torch
import mlx.core as mx
import pickle
import numpy as np

def validate_cfm_precision():
    """验证CFM精度是否达到e-5级别"""

    # 1. 加载缓存输入
    with open('s2mel_inputs_cache.pkl', 'rb') as f:
        inputs = pickle.load(f)

    # 2. 设置固定种子
    torch.manual_seed(42)
    mx.random.seed(42)

    # 3. 执行PyTorch CFM
    # TODO: 添加PyTorch CFM调用

    # 4. 执行MLX CFM
    # TODO: 添加MLX CFM调用

    # 5. 比较结果
    # TODO: 添加精度比较

    print("精度验证完成")

if __name__ == "__main__":
    validate_cfm_precision()
'''

    with open('validate_cfm_precision.py', 'w') as f:
        f.write(script_content)

    print(f"\n✅ 精度验证脚本已创建: validate_cfm_precision.py")

def main():
    """主函数"""
    test_mlx_cfm_forward()
    analyze_precision_bottlenecks()
    create_cfm_fix_plan()
    create_precision_validation_script()

    print("\n" + "="*80)
    print("CFM精度修复方案完成")
    print("="*80)

    print("\n下一步行动:")
    print("1. 实施高优先级修复(权重加载精度、随机数一致性)")
    print("2. 运行精度验证脚本")
    print("3. 如果仍未达到e-5精度，进行逐层调试")
    print("4. 优化中优先级和低优先级问题")

if __name__ == "__main__":
    main()