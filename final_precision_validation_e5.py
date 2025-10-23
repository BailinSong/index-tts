"""
最终CFM精度验证 - 比较使用正确权重的MLX和PyTorch版本
这是验证e-5精度目标的最终测试！
"""

import torch
import mlx.core as mx
import numpy as np
import pickle
from omegaconf import OmegaConf
import os


def load_pytorch_cfm_estimator():
    """加载PyTorch CFM estimator"""
    print("="*80)
    print("加载PyTorch CFM Estimator")
    print("="*80)

    try:
        # 加载配置
        cfg = OmegaConf.load("checkpoints/config.yaml")

        # 创建PyTorch DiT estimator
        from indextts.s2mel.modules.diffusion_transformer import DiT

        dit_config = cfg.s2mel.DiT
        pytorch_dit = DiT(
            in_channels=dit_config.in_channels,
            out_channels=dit_config.in_channels,  # 修复：使用in_channels作为out_channels
            hidden_size=dit_config.hidden_dim,
            num_heads=dit_config.num_heads,
            depth=dit_config.depth,
            mlp_ratio=4.0,  # 默认值
            class_dropout_prob=dit_config.class_dropout_prob,
            num_classes=dit_config.style_condition,
            learn_sigma=False,  # 默认值
            use_kv_cache=False
        )

        pytorch_dit.eval()
        print(f"✅ 创建PyTorch DiT成功")

        # 加载权重
        s2mel_path = os.path.join("checkpoints", cfg.s2mel_checkpoint)
        checkpoint = torch.load(s2mel_path, map_location='cpu')

        if 'net' in checkpoint and 'cfm' in checkpoint['net']:
            cfm_state_dict = checkpoint['net']['cfm']

            # 加载权重到PyTorch DiT
            # 需要移除'estimator.'前缀
            dit_state_dict = {}
            for key, value in cfm_state_dict.items():
                if key.startswith('estimator.'):
                    new_key = key[len('estimator.'):]
                    dit_state_dict[new_key] = value

            # 使用strict=False以防权重名称不完全匹配
            missing_keys, unexpected_keys = pytorch_dit.load_state_dict(dit_state_dict, strict=False)

            if missing_keys:
                print(f"⚠️  缺失权重: {len(missing_keys)} 个")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"     {key}")

            if unexpected_keys:
                print(f"⚠️  意外权重: {len(unexpected_keys)} 个")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"     {key}")

            loaded_keys = len(dit_state_dict) - len(missing_keys)
            print(f"✅ 成功加载 {loaded_keys} 个权重到PyTorch DiT")

            return pytorch_dit

        else:
            print("❌ 找不到CFM权重在checkpoint中")
            return None

    except Exception as e:
        print(f"❌ PyTorch CFM加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_mlx_cfm_with_correct_weights():
    """加载使用正确权重的MLX CFM"""
    print("\n" + "="*80)
    print("加载MLX CFM（正确权重）")
    print("="*80)

    try:
        # 加载配置
        cfg = OmegaConf.load("checkpoints/config.yaml")

        # 加载PyTorch权重
        s2mel_path = os.path.join("checkpoints", cfg.s2mel_checkpoint)
        checkpoint = torch.load(s2mel_path, map_location='cpu')

        if 'net' in checkpoint:
            s2mel_state_dict = {}
            for key in checkpoint['net']:
                for param_name, param_value in checkpoint['net'][key].items():
                    s2mel_state_dict[f"{key}.{param_name}"] = param_value
        else:
            s2mel_state_dict = checkpoint

        # 转换为numpy
        s2mel_state_dict_np = {}
        for key, value in s2mel_state_dict.items():
            if isinstance(value, torch.Tensor):
                s2mel_state_dict_np[key] = value.detach().cpu().numpy().astype(np.float32)
            else:
                s2mel_state_dict_np[key] = value

        # 删除缓存确保从PyTorch加载
        cache_file = "checkpoints/mlx/s2mel_cfm.npz"
        if os.path.exists(cache_file):
            os.remove(cache_file)

        # 创建MLX CFM
        from indextts.s2mel.modules.mlx_cfm import MLXCFM
        mlx_cfm = MLXCFM(cfg.s2mel)

        # 加载权重
        loaded_count = mlx_cfm.load_weights_from_pytorch(s2mel_state_dict_np, prefix="cfm.")
        print(f"✅ MLX CFM加载了 {loaded_count} 个权重")

        return mlx_cfm

    except Exception as e:
        print(f"❌ MLX CFM加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def final_precision_comparison():
    """最终精度比较"""
    print("\n" + "="*80)
    print("最终CFM精度比较 - PyTorch vs MLX")
    print("目标：max_diff < 1e-5")
    print("="*80)

    # 1. 加载两个版本
    pytorch_dit = load_pytorch_cfm_estimator()
    mlx_cfm = load_mlx_cfm_with_correct_weights()

    if pytorch_dit is None or mlx_cfm is None:
        print("❌ 模型加载失败，无法进行比较")
        return

    # 2. 设置确定性种子
    torch.manual_seed(42)
    mx.random.seed(42)
    np.random.seed(42)

    # 3. 创建相同的测试输入
    print("\n创建测试输入...")
    batch_size = 1
    seq_len = 32

    x = torch.randn(batch_size, 80, seq_len).float()
    prompt_x = torch.zeros_like(x)
    cond = torch.randn(batch_size, seq_len, 512).float()
    x_lens = torch.tensor([seq_len])
    t = torch.tensor([0.5]).float()
    style = torch.randn(batch_size, 192).float()

    print(f"输入统计:")
    print(f"  x: {x.shape}, min={x.min():.6f}, max={x.max():.6f}")
    print(f"  cond: {cond.shape}, min={cond.min():.6f}, max={cond.max():.6f}")
    print(f"  t: {t.item():.6f}")
    print(f"  style: {style.shape}, min={style.min():.6f}, max={style.max():.6f}")

    # 4. PyTorch前向传播
    print("\n执行PyTorch前向传播...")
    with torch.no_grad():
        pytorch_dit.eval()
        # 注意：PyTorch DiT的接口可能与MLX不同
        try:
            output_pytorch = pytorch_dit(x, t, style, cond)
            print(f"✅ PyTorch输出: {output_pytorch.shape}")
            print(f"   数值范围: [{output_pytorch.min():.6f}, {output_pytorch.max():.6f}]")
            print(f"   均值: {output_pytorch.mean():.6f}")
        except Exception as e:
            print(f"❌ PyTorch前向传播失败: {e}")
            # 尝试不同的接口
            try:
                output_pytorch = pytorch_dit.forward_with_cfg(x, t, style, cond)
                print(f"✅ PyTorch输出 (cfg): {output_pytorch.shape}")
            except Exception as e2:
                print(f"❌ PyTorch前向传播完全失败: {e2}")
                return

    # 5. MLX前向传播
    print("\n执行MLX前向传播...")

    def torch_to_mlx_precise(tensor):
        return mx.array(tensor.detach().cpu().numpy().astype(np.float32))

    x_mlx = torch_to_mlx_precise(x)
    prompt_x_mlx = torch_to_mlx_precise(prompt_x)
    cond_mlx = torch_to_mlx_precise(cond)
    x_lens_mlx = torch_to_mlx_precise(x_lens)
    t_mlx = torch_to_mlx_precise(t)
    style_mlx = torch_to_mlx_precise(style)

    output_mlx = mlx_cfm.estimator(
        x_mlx, prompt_x_mlx, x_lens_mlx, t_mlx, style_mlx, cond_mlx, mask_content=False
    )
    mx.eval(output_mlx)

    print(f"✅ MLX输出: {output_mlx.shape}")
    print(f"   数值范围: [{output_mlx.min():.6f}, {output_mlx.max():.6f}]")
    print(f"   均值: {output_mlx.mean():.6f}")

    # 6. 精度比较
    print("\n" + "="*80)
    print("精度比较结果")
    print("="*80)

    # 转换MLX输出为PyTorch格式
    output_mlx_torch = torch.from_numpy(np.array(output_mlx).astype(np.float32))

    # 确保形状一致
    if output_pytorch.shape != output_mlx_torch.shape:
        print(f"❌ 输出形状不匹配:")
        print(f"   PyTorch: {output_pytorch.shape}")
        print(f"   MLX: {output_mlx_torch.shape}")
        return

    # 计算差异
    abs_diff = torch.abs(output_pytorch - output_mlx_torch)
    rel_diff = abs_diff / (torch.abs(output_pytorch) + 1e-8)

    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()

    # 相关性
    pytorch_flat = output_pytorch.flatten()
    mlx_flat = output_mlx_torch.flatten()
    correlation = torch.corrcoef(torch.stack([pytorch_flat, mlx_flat]))[0, 1].item()

    print(f"精度统计:")
    print(f"  最大绝对差异: {max_abs_diff:.10f}")
    print(f"  平均绝对差异: {mean_abs_diff:.10f}")
    print(f"  最大相对差异: {max_rel_diff:.10f}")
    print(f"  平均相对差异: {mean_rel_diff:.10f}")
    print(f"  相关性: {correlation:.10f}")

    # 精度评估
    if max_abs_diff < 1e-5:
        print(f"\n🎉 成功！达到e-5级别精度目标！")
        print(f"   max_abs_diff ({max_abs_diff:.10f}) < 1e-5")
        success = True
    elif max_abs_diff < 1e-4:
        print(f"\n⚠️  接近目标，但未达到e-5精度")
        print(f"   max_abs_diff ({max_abs_diff:.10f}) < 1e-4")
        success = False
    else:
        print(f"\n❌ 精度仍不足")
        print(f"   max_abs_diff ({max_abs_diff:.10f}) >= 1e-4")
        success = False

    # 差异分布分析
    total_elements = abs_diff.numel()
    excellent_count = (abs_diff < 1e-7).sum().item()
    very_good_count = (abs_diff < 1e-6).sum().item()
    good_count = (abs_diff < 1e-5).sum().item()
    fair_count = (abs_diff < 1e-4).sum().item()

    print(f"\n差异分布 (总元素: {total_elements}):")
    print(f"  < 1e-7: {excellent_count} ({excellent_count/total_elements*100:.2f}%)")
    print(f"  < 1e-6: {very_good_count} ({very_good_count/total_elements*100:.2f}%)")
    print(f"  < 1e-5: {good_count} ({good_count/total_elements*100:.2f}%)")
    print(f"  < 1e-4: {fair_count} ({fair_count/total_elements*100:.2f}%)")

    # 保存结果
    result = {
        'pytorch_output': output_pytorch.numpy(),
        'mlx_output': np.array(output_mlx),
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
        'max_rel_diff': max_rel_diff,
        'mean_rel_diff': mean_rel_diff,
        'correlation': correlation,
        'achieves_e5_precision': success,
        'test_inputs': {
            'x': x.numpy(),
            'cond': cond.numpy(),
            't': t.numpy(),
            'style': style.numpy()
        }
    }

    with open('final_cfm_precision_results.pkl', 'wb') as f:
        pickle.dump(result, f)

    print(f"\n✅ 最终结果已保存到: final_cfm_precision_results.pkl")

    return success


def main():
    """主函数"""
    print("🎯 最终CFM精度验证")
    print("检验修复权重加载后是否达到e-5级别精度\n")

    success = final_precision_comparison()

    print("\n" + "="*80)
    print("最终结论")
    print("="*80)

    if success:
        print("🎉🎉🎉 恭喜！CFM精度问题已完全解决！")
        print("✅ 达到了e-5级别精度目标 (max_diff < 1e-5)")
        print("✅ MLX和PyTorch版本现在高度一致")
        print("\n问题解决的关键:")
        print("1. 发现并修复了权重映射不匹配问题")
        print("2. 确保MLX CFM直接从PyTorch权重加载")
        print("3. 验证了权重加载的正确性")
    else:
        print("⚠️  CFM精度有所改善，但仍未达到e-5级别")
        print("可能还需要进一步优化:")
        print("1. 检查模型结构的细微差异")
        print("2. 验证前向传播的实现一致性")
        print("3. 检查数值计算的精度")

    print(f"\n任务完成！")


if __name__ == "__main__":
    main()