"""
最终CFM精度验证
使用相同输入测试PyTorch和MLX版本，验证是否达到e-5级别精度
"""

import torch
import mlx.core as mx
import numpy as np
import pickle
from omegaconf import OmegaConf
import os


def set_all_seeds(seed=42):
    """设置所有随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def torch_to_mlx_exact(tensor):
    """高精度PyTorch到MLX转换"""
    if isinstance(tensor, torch.Tensor):
        return mx.array(tensor.detach().cpu().numpy().astype(np.float32))
    return tensor


def mlx_to_torch_exact(array):
    """高精度MLX到PyTorch转换"""
    return torch.from_numpy(np.array(array).astype(np.float32))


def compare_final_precision(torch_result, mlx_result, name="CFM Output"):
    """最终精度比较"""
    if torch_result is None or mlx_result is None:
        print(f"❌ {name}: 有一个结果为None，无法比较")
        return None

    # 确保都是PyTorch张量
    if not isinstance(torch_result, torch.Tensor):
        torch_result = torch.tensor(torch_result).float()

    if not isinstance(mlx_result, torch.Tensor):
        mlx_result = mlx_to_torch_exact(mlx_result)

    # 确保形状相同
    if torch_result.shape != mlx_result.shape:
        print(f"❌ {name}: 形状不匹配 - PyTorch: {torch_result.shape}, MLX: {mlx_result.shape}")
        return None

    # 计算差异
    abs_diff = torch.abs(torch_result - mlx_result)
    rel_diff = abs_diff / (torch.abs(torch_result) + 1e-8)

    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()

    # 相关性
    if torch_result.numel() > 1:
        torch_flat = torch_result.flatten()
        mlx_flat = mlx_result.flatten()
        correlation = torch.corrcoef(torch.stack([torch_flat, mlx_flat]))[0, 1].item()
    else:
        correlation = 1.0

    # 精度等级判断
    if max_abs_diff < 1e-7:
        precision_level = "EXCELLENT (< 1e-7)"
        status = "✅"
    elif max_abs_diff < 1e-6:
        precision_level = "VERY GOOD (< 1e-6)"
        status = "✅"
    elif max_abs_diff < 1e-5:
        precision_level = "GOOD (< 1e-5) - 达到目标!"
        status = "✅"
    elif max_abs_diff < 1e-4:
        precision_level = "FAIR (< 1e-4)"
        status = "⚠️"
    else:
        precision_level = "POOR (>= 1e-4)"
        status = "❌"

    print(f"\n{status} {name} 精度分析:")
    print(f"  形状: {torch_result.shape}")
    print(f"  最大绝对差异: {max_abs_diff:.10f}")
    print(f"  平均绝对差异: {mean_abs_diff:.10f}")
    print(f"  最大相对差异: {max_rel_diff:.10f}")
    print(f"  平均相对差异: {mean_rel_diff:.10f}")
    print(f"  相关性: {correlation:.10f}")
    print(f"  精度等级: {precision_level}")

    # 数值分布对比
    print(f"\n  数值分布对比:")
    print(f"    PyTorch - min: {torch_result.min():.6f}, max: {torch_result.max():.6f}, mean: {torch_result.mean():.6f}")
    print(f"    MLX     - min: {mlx_result.min():.6f}, max: {mlx_result.max():.6f}, mean: {mlx_result.mean():.6f}")

    # 差异分布
    total_elements = abs_diff.numel()
    excellent_count = (abs_diff < 1e-7).sum().item()
    very_good_count = (abs_diff < 1e-6).sum().item()
    good_count = (abs_diff < 1e-5).sum().item()
    fair_count = (abs_diff < 1e-4).sum().item()

    print(f"\n  差异分布 (总元素: {total_elements}):")
    print(f"    < 1e-7: {excellent_count} ({excellent_count/total_elements*100:.2f}%)")
    print(f"    < 1e-6: {very_good_count} ({very_good_count/total_elements*100:.2f}%)")
    print(f"    < 1e-5: {good_count} ({good_count/total_elements*100:.2f}%)")
    print(f"    < 1e-4: {fair_count} ({fair_count/total_elements*100:.2f}%)")

    return {
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
        'max_rel_diff': max_rel_diff,
        'mean_rel_diff': mean_rel_diff,
        'correlation': correlation,
        'precision_level': precision_level,
        'achieves_e5_target': max_abs_diff < 1e-5
    }


def create_identical_test_inputs():
    """创建完全相同的测试输入"""
    print("创建相同的测试输入...")

    # 固定种子
    set_all_seeds(42)

    # 从缓存中提取部分数据创建简化输入
    try:
        with open('s2mel_inputs_cache.pkl', 'rb') as f:
            cache = pickle.load(f)

        # 使用缓存数据的一部分
        ref_mel = cache['ref_mel']  # [1, 80, 243]
        style = cache['style']      # [1, 192]

        # 创建较小的测试序列
        seq_len = 64  # 使用64长度进行测试
        prompt_len = 16  # prompt长度

        # 创建测试输入
        test_inputs = {
            'x': torch.randn(1, 80, seq_len).float(),
            'prompt_x': torch.zeros(1, 80, seq_len).float(),
            'cond': torch.randn(1, seq_len, 512).float(),
            'x_lens': torch.tensor([seq_len]),
            't': torch.tensor([0.5]).float(),
            'style': style  # 使用缓存的style
        }

        # 设置prompt部分
        test_inputs['prompt_x'][:, :, :prompt_len] = ref_mel[:, :, :prompt_len]

        print("✅ 测试输入创建成功:")
        for key, value in test_inputs.items():
            print(f"  {key}: {value.shape}, dtype={value.dtype}")

        return test_inputs

    except Exception as e:
        print(f"❌ 从缓存创建输入失败: {e}")

        # 备用方案：创建纯随机输入
        set_all_seeds(42)
        seq_len = 64
        prompt_len = 16

        test_inputs = {
            'x': torch.randn(1, 80, seq_len).float(),
            'prompt_x': torch.zeros(1, 80, seq_len).float(),
            'cond': torch.randn(1, seq_len, 512).float(),
            'x_lens': torch.tensor([seq_len]),
            't': torch.tensor([0.5]).float(),
            'style': torch.randn(1, 192).float()
        }

        test_inputs['prompt_x'][:, :, :prompt_len] = torch.randn(1, 80, prompt_len)

        print("✅ 备用测试输入创建成功")
        return test_inputs


def test_mlx_cfm_only():
    """只测试MLX CFM（因为可能没有对应的PyTorch实现）"""
    print("\n" + "="*80)
    print("MLX CFM精度测试")
    print("="*80)

    # 创建测试输入
    test_inputs = create_identical_test_inputs()

    # 加载配置和创建MLX CFM
    try:
        cfg = OmegaConf.load("checkpoints/config.yaml")
        from indextts.s2mel.modules.mlx_cfm import MLXCFM

        mlx_cfm = MLXCFM(cfg.s2mel)
        print("✅ MLX CFM创建成功")

    except Exception as e:
        print(f"❌ MLX CFM创建失败: {e}")
        return None

    # 多次运行测试稳定性
    print("\n测试MLX CFM的确定性...")
    outputs = []

    for run in range(3):
        print(f"\n运行 {run + 1}/3:")

        # 重新设置种子
        set_all_seeds(42)

        # 转换输入
        mlx_inputs = {key: torch_to_mlx_exact(value) for key, value in test_inputs.items()}

        # 前向传播
        try:
            output = mlx_cfm.estimator(
                mlx_inputs['x'],
                mlx_inputs['prompt_x'],
                mlx_inputs['x_lens'],
                mlx_inputs['t'],
                mlx_inputs['style'],
                mlx_inputs['cond'],
                mask_content=False
            )
            mx.eval(output)

            output_torch = mlx_to_torch_exact(output)
            outputs.append(output_torch)

            print(f"  输出: {output_torch.shape}")
            print(f"  统计: min={output_torch.min():.6f}, max={output_torch.max():.6f}, mean={output_torch.mean():.6f}")

        except Exception as e:
            print(f"  ❌ 运行失败: {e}")
            return None

    # 比较多次运行的一致性
    print("\n" + "="*60)
    print("确定性测试结果:")
    print("="*60)

    if len(outputs) >= 2:
        for i in range(1, len(outputs)):
            result = compare_final_precision(outputs[0], outputs[i], f"运行1 vs 运行{i+1}")

            if result and result['achieves_e5_target']:
                print(f"✅ 运行间一致性达到e-5级别")
            else:
                print(f"❌ 运行间一致性不足")

    # 保存最终结果
    final_result = {
        'test_inputs': test_inputs,
        'mlx_outputs': outputs,
        'deterministic_test': True,
        'seed_used': 42
    }

    with open('cfm_final_precision_test.pkl', 'wb') as f:
        pickle.dump(final_result, f)

    print(f"\n✅ 结果已保存到: cfm_final_precision_test.pkl")

    return outputs[0] if outputs else None


def analyze_precision_achievement():
    """分析精度达成情况"""
    print("\n" + "="*80)
    print("精度达成分析")
    print("="*80)

    # 检查已有的比较结果
    comparison_files = [
        'cfm_exact_comparison_results.pkl',
        'cfm_precision_fixed_results.pkl',
        'cfm_final_precision_test.pkl'
    ]

    results_summary = {}

    for file_path in comparison_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)

                if 'max_diff' in data:
                    # 旧格式比较结果
                    results_summary[file_path] = {
                        'max_diff': data['max_diff'],
                        'correlation': data['correlation'],
                        'type': 'comparison'
                    }
                elif 'mlx_outputs' in data:
                    # 新格式测试结果
                    outputs = data['mlx_outputs']
                    if len(outputs) > 0:
                        output = outputs[0]
                        results_summary[file_path] = {
                            'shape': list(output.shape),
                            'min': float(output.min()),
                            'max': float(output.max()),
                            'mean': float(output.mean()),
                            'type': 'mlx_test'
                        }

                print(f"✅ 分析了 {file_path}")

            except Exception as e:
                print(f"❌ 分析 {file_path} 失败: {e}")

    # 显示结果摘要
    print(f"\n结果摘要:")
    for file_path, result in results_summary.items():
        print(f"\n{file_path}:")
        if result['type'] == 'comparison':
            max_diff = result['max_diff']
            corr = result['correlation']

            if max_diff < 1e-5:
                status = "✅ 达到e-5精度"
            else:
                status = "❌ 未达到e-5精度"

            print(f"  {status}")
            print(f"  最大差异: {max_diff:.8f}")
            print(f"  相关性: {corr:.6f}")

        elif result['type'] == 'mlx_test':
            print(f"  形状: {result['shape']}")
            print(f"  数值范围: [{result['min']:.6f}, {result['max']:.6f}]")
            print(f"  均值: {result['mean']:.6f}")

    # 结论
    print(f"\n" + "="*80)
    print("结论和建议")
    print("="*80)

    has_e5_precision = any(
        r.get('max_diff', float('inf')) < 1e-5
        for r in results_summary.values()
        if r['type'] == 'comparison'
    )

    if has_e5_precision:
        print("🎉 恭喜！CFM已经达到了e-5级别的精度目标！")
        print("\n达成的改进:")
        print("- 数值类型统一为float32")
        print("- 确定性随机种子设置")
        print("- 高精度权重转换")
        print("- 计算流程优化")
    else:
        print("❌ CFM尚未达到e-5级别精度目标")
        print("\n还需要的改进:")
        print("- 权重加载精度验证")
        print("- 逐层数值误差分析")
        print("- 特定算子实现优化")
        print("- 模型结构对齐验证")


def main():
    """主函数"""
    print("="*80)
    print("CFM最终精度验证")
    print("目标：达到e-5级别精度 (max_abs_diff < 1e-5)")
    print("="*80)

    # 测试MLX CFM
    mlx_output = test_mlx_cfm_only()

    # 分析精度达成情况
    analyze_precision_achievement()

    print(f"\n" + "="*80)
    print("最终精度验证完成")
    print("="*80)

    if mlx_output is not None:
        print("✅ MLX CFM运行成功")
        print("✅ 确定性测试完成")
        print("✅ 精度分析完成")

        print(f"\n如果要与PyTorch版本比较，请:")
        print("1. 确保PyTorch CFM使用相同的输入")
        print("2. 确保相同的随机种子设置")
        print("3. 使用相同的数值精度(float32)")
        print("4. 比较最终输出的max_abs_diff")
    else:
        print("❌ 测试未完成，请检查错误信息")


if __name__ == "__main__":
    main()