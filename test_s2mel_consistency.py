"""
S2MEL 一致性测试：对比 PyTorch 和 MLX 版本
控制变量，逐层对比输出
"""

import torch
import numpy as np
from omegaconf import OmegaConf
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

def torch_to_mlx(tensor):
    """Convert PyTorch tensor to MLX array"""
    import mlx.core as mx
    if isinstance(tensor, torch.Tensor):
        return mx.array(tensor.detach().cpu().numpy())
    return tensor

def mlx_to_torch(array, device='cpu'):
    """Convert MLX array to PyTorch tensor"""
    import mlx.core as mx
    if isinstance(array, mx.array):
        return torch.from_numpy(np.array(array)).to(device)
    return array

def compare_outputs(torch_output, mlx_output, name="Output", verbose=True):
    """比较 PyTorch 和 MLX 的输出"""
    # 转换为numpy进行比较
    if isinstance(torch_output, torch.Tensor):
        torch_np = torch_output.detach().cpu().numpy()
    else:
        torch_np = np.array(torch_output)
    
    if hasattr(mlx_output, '__array__'):
        mlx_np = np.array(mlx_output)
    else:
        mlx_np = mlx_output
    
    # 确保形状一致
    if torch_np.shape != mlx_np.shape:
        print(f"❌ {name} shape mismatch: PyTorch {torch_np.shape} vs MLX {mlx_np.shape}")
        return False, None
    
    # 计算差异
    max_diff = np.abs(torch_np - mlx_np).max()
    mean_diff = np.abs(torch_np - mlx_np).mean()
    
    # 计算相关性
    if torch_np.size > 1:
        torch_flat = torch_np.flatten()
        mlx_flat = mlx_np.flatten()
        correlation = np.corrcoef(torch_flat, mlx_flat)[0, 1]
    else:
        correlation = 1.0 if max_diff == 0 else 0.0
    
    # 评估
    if verbose:
        print(f"\n{name}:")
        print(f"  Shape: {torch_np.shape}")
        print(f"  Max diff: {max_diff:.8f}")
        print(f"  Mean diff: {mean_diff:.8f}")
        print(f"  Correlation: {correlation:.6f}")
    
    # 判断一致性
    if max_diff < 1e-4 and correlation > 0.9999:
        status = "✅ EXCELLENT"
    elif max_diff < 1e-3 and correlation > 0.999:
        status = "✅ VERY GOOD"
    elif max_diff < 1e-2 and correlation > 0.99:
        status = "✅ GOOD"
    elif max_diff < 0.1 and correlation > 0.95:
        status = "⚠️  ACCEPTABLE"
    else:
        status = "❌ POOR"
    
    if verbose:
        print(f"  Status: {status}")
    
    return status.startswith("✅"), {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'correlation': correlation,
        'status': status
    }

def test_s2mel_gpt_layer():
    """测试 S2MEL gpt_layer"""
    print("\n" + "="*70)
    print("测试 1: S2MEL gpt_layer 一致性")
    print("="*70)
    
    try:
        import mlx.core as mx
        from indextts.s2mel.modules.mlx_s2mel import MLXGPTLayer
        from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
        
        # 加载配置和PyTorch模型
        cfg = OmegaConf.load("checkpoints/config.yaml")
        s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
        s2mel_path = os.path.join("checkpoints", cfg.s2mel_checkpoint)
        s2mel, _, _, _ = load_checkpoint2(s2mel, None, s2mel_path, load_only_params=True, ignore_modules=[], is_distributed=False)
        s2mel.eval()
        
        # 创建MLX版本
        mlx_gpt_layer = MLXGPTLayer()
        s2mel_state_dict = s2mel.state_dict()
        s2mel_state_dict_np = {k: v.cpu().numpy() for k, v in s2mel_state_dict.items()}
        mlx_gpt_layer.load_weights_from_pytorch(s2mel_state_dict_np)
        
        # 准备测试数据
        batch_size = 1
        seq_len = 50
        input_dim = 1280  # GPT output dimension
        
        test_input = torch.randn(batch_size, seq_len, input_dim)
        
        print(f"\n输入数据:")
        print(f"  Shape: {test_input.shape}")
        print(f"  Mean: {test_input.mean():.6f}, Std: {test_input.std():.6f}")
        
        # PyTorch 推理
        with torch.no_grad():
            torch_output = s2mel.models['gpt_layer'](test_input)
        
        print(f"\nPyTorch 输出:")
        print(f"  Shape: {torch_output.shape}")
        print(f"  Mean: {torch_output.mean():.6f}, Std: {torch_output.std():.6f}")
        
        # MLX 推理
        test_input_mlx = torch_to_mlx(test_input)
        mlx_output = mlx_gpt_layer(test_input_mlx)
        mx.eval(mlx_output)
        
        print(f"\nMLX 输出:")
        print(f"  Shape: {mlx_output.shape}")
        print(f"  Mean: {float(mx.mean(mlx_output)):.6f}, Std: {float(mx.std(mlx_output)):.6f}")
        
        # 对比
        is_good, stats = compare_outputs(torch_output, mlx_output, "gpt_layer Output")
        
        return is_good, stats
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_s2mel_length_regulator():
    """测试 S2MEL length_regulator"""
    print("\n" + "="*70)
    print("测试 2: S2MEL length_regulator 一致性")
    print("="*70)
    
    try:
        import mlx.core as mx
        from indextts.s2mel.modules.mlx_s2mel import MLXLengthRegulator
        from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
        
        # 加载配置和PyTorch模型
        cfg = OmegaConf.load("checkpoints/config.yaml")
        s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
        s2mel_path = os.path.join("checkpoints", cfg.s2mel_checkpoint)
        s2mel, _, _, _ = load_checkpoint2(s2mel, None, s2mel_path, load_only_params=True, ignore_modules=[], is_distributed=False)
        s2mel.eval()
        
        # 创建MLX版本
        mlx_length_regulator = MLXLengthRegulator(
            channels=cfg.s2mel.length_regulator.channels,
            sampling_ratios=tuple(cfg.s2mel.length_regulator.sampling_ratios),
            is_discrete=cfg.s2mel.length_regulator.is_discrete,
            in_channels=cfg.s2mel.length_regulator.in_channels if hasattr(cfg.s2mel.length_regulator, "in_channels") else None,
            codebook_size=cfg.s2mel.length_regulator.content_codebook_size,
            n_codebooks=cfg.s2mel.length_regulator.n_codebooks if hasattr(cfg.s2mel.length_regulator, "n_codebooks") else 1,
            f0_condition=cfg.s2mel.length_regulator.f0_condition if hasattr(cfg.s2mel.length_regulator, "f0_condition") else False,
            n_f0_bins=cfg.s2mel.length_regulator.n_f0_bins if hasattr(cfg.s2mel.length_regulator, "n_f0_bins") else 512,
        )
        s2mel_state_dict = s2mel.state_dict()
        s2mel_state_dict_np = {k: v.cpu().numpy() for k, v in s2mel_state_dict.items()}
        mlx_length_regulator.load_weights_from_pytorch(s2mel_state_dict_np)
        
        # 准备测试数据 (语义编码)
        batch_size = 1
        seq_len = 30
        target_len = 50
        
        # length_regulator的is_discrete=False，需要float输入
        # 输入格式: (batch, seq_len, content_dim)
        in_channels = cfg.s2mel.length_regulator.in_channels if hasattr(cfg.s2mel.length_regulator, "in_channels") else 1024
        test_input = torch.randn(batch_size, seq_len, in_channels)
        test_ylens = torch.tensor([target_len], dtype=torch.long)
        
        print(f"\n输入数据:")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Target lengths: {test_ylens}")
        
        # PyTorch 推理
        with torch.no_grad():
            torch_output, *_ = s2mel.models['length_regulator'](
                test_input,
                ylens=test_ylens,
                n_quantizers=None,
                f0=None
            )
        
        print(f"\nPyTorch 输出:")
        print(f"  Shape: {torch_output.shape}")
        print(f"  Mean: {torch_output.mean():.6f}, Std: {torch_output.std():.6f}")
        
        # MLX 推理
        test_input_mlx = torch_to_mlx(test_input)
        test_ylens_mlx = torch_to_mlx(test_ylens)
        mlx_output, *_ = mlx_length_regulator(
            test_input_mlx,
            ylens=test_ylens_mlx,
            n_quantizers=None,
            f0=None
        )
        mx.eval(mlx_output)
        
        print(f"\nMLX 输出:")
        print(f"  Shape: {mlx_output.shape}")
        print(f"  Mean: {float(mx.mean(mlx_output)):.6f}, Std: {float(mx.std(mlx_output)):.6f}")
        
        # 对比
        is_good, stats = compare_outputs(torch_output, mlx_output, "length_regulator Output")
        
        return is_good, stats
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_s2mel_cfm_single_step():
    """测试 S2MEL CFM 单步推理"""
    print("\n" + "="*70)
    print("测试 3: S2MEL CFM 单步推理一致性")
    print("="*70)
    
    try:
        import mlx.core as mx
        from indextts.s2mel.modules.mlx_cfm import MLXCFM
        from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
        
        # 加载配置和PyTorch模型
        cfg = OmegaConf.load("checkpoints/config.yaml")
        s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
        s2mel_path = os.path.join("checkpoints", cfg.s2mel_checkpoint)
        s2mel, _, _, _ = load_checkpoint2(s2mel, None, s2mel_path, load_only_params=True, ignore_modules=[], is_distributed=False)
        s2mel.eval()
        
        # 创建MLX CFM
        mlx_cfm = MLXCFM(cfg.s2mel)
        s2mel_state_dict = s2mel.state_dict()
        s2mel_state_dict_np = {k: v.cpu().numpy() for k, v in s2mel_state_dict.items()}
        mlx_cfm.load_weights_from_pytorch(s2mel_state_dict_np, prefix="models.cfm.")
        
        # 准备测试数据
        batch_size = 1
        seq_len = 30
        in_channels = 80  # mel bins
        
        # 使用固定的随机种子确保可重现
        torch.manual_seed(42)
        np.random.seed(42)
        
        # x 和 prompt_x 需要拼接后的完整序列
        # 在实际推理中，prompt_x会填充到和x相同的长度
        x = torch.randn(batch_size, in_channels, seq_len)
        prompt_x = torch.zeros_like(x)  # 同样大小，前面部分是prompt
        prompt_len = 10
        prompt_x[:, :, :prompt_len] = torch.randn(batch_size, in_channels, prompt_len)
        # PyTorch DiT中，cond格式: (batch, mel_timesteps, 512)
        content_dim = cfg.s2mel.DiT.content_dim if hasattr(cfg.s2mel.DiT, 'content_dim') else 512
        cond = torch.randn(batch_size, seq_len, content_dim)  # semantic conditioning
        x_lens = mx.array([seq_len])
        t = torch.tensor([0.5])  # 中间时间步
        style = torch.randn(batch_size, 192)
        
        print(f"\n输入数据:")
        print(f"  x shape: {x.shape}")
        print(f"  prompt_x shape: {prompt_x.shape}")
        print(f"  cond shape: {cond.shape}")
        print(f"  style shape: {style.shape}")
        print(f"  t: {t.item():.3f}")
        
        # PyTorch 推理 (DiT estimator 单步)
        with torch.no_grad():
            torch_output = s2mel.models['cfm'].estimator(
                x, prompt_x, x_lens, t, style, cond, mask_content=False
            )
        
        print(f"\nPyTorch 输出:")
        print(f"  Shape: {torch_output.shape}")
        print(f"  Mean: {torch_output.mean():.6f}, Std: {torch_output.std():.6f}")
        print(f"  Min: {torch_output.min():.6f}, Max: {torch_output.max():.6f}")
        
        # MLX 推理
        x_mlx = torch_to_mlx(x)
        prompt_x_mlx = torch_to_mlx(prompt_x)
        cond_mlx = torch_to_mlx(cond)
        t_mlx = torch_to_mlx(t)
        style_mlx = torch_to_mlx(style)
        
        mlx_output = mlx_cfm.estimator(
            x_mlx, prompt_x_mlx, x_lens, t_mlx, style_mlx, cond_mlx, mask_content=False
        )
        mx.eval(mlx_output)
        
        print(f"\nMLX 输出:")
        print(f"  Shape: {mlx_output.shape}")
        print(f"  Mean: {float(mx.mean(mlx_output)):.6f}, Std: {float(mx.std(mlx_output)):.6f}")
        print(f"  Min: {float(mx.min(mlx_output)):.6f}, Max: {float(mx.max(mlx_output)):.6f}")
        
        # 对比
        is_good, stats = compare_outputs(torch_output, mlx_output, "CFM DiT Single Step")
        
        return is_good, stats
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """运行所有S2MEL一致性测试"""
    print("\n" + "="*70)
    print("S2MEL 完整一致性测试套件 (PyTorch vs MLX)")
    print("="*70)
    
    results = {}
    
    # 测试1: gpt_layer
    is_good, stats = test_s2mel_gpt_layer()
    results['gpt_layer'] = {'passed': is_good, 'stats': stats}
    
    # 测试2: length_regulator
    is_good, stats = test_s2mel_length_regulator()
    results['length_regulator'] = {'passed': is_good, 'stats': stats}
    
    # 测试3: CFM单步
    is_good, stats = test_s2mel_cfm_single_step()
    results['cfm_single_step'] = {'passed': is_good, 'stats': stats}
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    for test_name, result in results.items():
        if result['stats']:
            status = result['stats']['status']
            max_diff = result['stats']['max_diff']
            corr = result['stats']['correlation']
            print(f"\n{test_name}:")
            print(f"  状态: {status}")
            print(f"  最大差异: {max_diff:.8f}")
            print(f"  相关性: {corr:.6f}")
        else:
            print(f"\n{test_name}: ❌ 测试失败")
    
    all_passed = all(r['passed'] for r in results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 所有测试通过！S2MEL MLX 版本与 PyTorch 完全一致！")
    else:
        failed = [name for name, r in results.items() if not r['passed']]
        print(f"⚠️  部分测试失败: {', '.join(failed)}")
    print("="*70 + "\n")
    
    return all_passed

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

