"""
调试 length_regulator 的差异
"""

import torch
import numpy as np
from omegaconf import OmegaConf
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

def torch_to_mlx(tensor):
    import mlx.core as mx
    if isinstance(tensor, torch.Tensor):
        return mx.array(tensor.detach().cpu().numpy())
    return tensor

def mlx_to_torch(array, device='cpu'):
    import mlx.core as mx
    if isinstance(array, mx.array):
        return torch.from_numpy(np.array(array)).to(device)
    return array

def main():
    import mlx.core as mx
    from indextts.s2mel.modules.mlx_s2mel import MLXLengthRegulator
    from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
    
    # 加载配置
    cfg = OmegaConf.load("checkpoints/config.yaml")
    
    # 加载PyTorch模型
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
    
    # 使用相同的随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 准备测试数据
    batch_size = 1
    seq_len = 30
    target_len = 50
    in_channels = cfg.s2mel.length_regulator.in_channels if hasattr(cfg.s2mel.length_regulator, "in_channels") else 1024
    
    test_input = torch.randn(batch_size, seq_len, in_channels)
    test_ylens = torch.tensor([target_len], dtype=torch.long)
    
    print("输入数据:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Input mean: {test_input.mean():.6f}, std: {test_input.std():.6f}")
    print(f"  Target lengths: {test_ylens}")
    
    # PyTorch 推理 - 逐层
    print("\n=== PyTorch 推理 ===")
    with torch.no_grad():
        torch_lr = s2mel.models['length_regulator']
        
        # 1. content_in_proj
        x_torch = torch_lr.content_in_proj(test_input)
        print(f"1. After content_in_proj: shape={x_torch.shape}, mean={x_torch.mean():.6f}, std={x_torch.std():.6f}")
        
        # 2. Interpolate
        if torch_lr.interpolate:
            x_torch_interp = torch.nn.functional.interpolate(
                x_torch.transpose(1, 2).contiguous(), 
                size=test_ylens.max(), 
                mode='nearest'
            )
            print(f"2. After interpolate: shape={x_torch_interp.shape}, mean={x_torch_interp.mean():.6f}, std={x_torch_interp.std():.6f}")
        else:
            x_torch_interp = x_torch.transpose(1, 2)
        
        # 3. Model layers
        x_torch_model = torch_lr.model(x_torch_interp)
        print(f"3. After model layers: shape={x_torch_model.shape}, mean={x_torch_model.mean():.6f}, std={x_torch_model.std():.6f}")
        
        # 完整forward
        torch_output, *_ = torch_lr(test_input, ylens=test_ylens, n_quantizers=None, f0=None)
        print(f"\n最终 PyTorch 输出: shape={torch_output.shape}, mean={torch_output.mean():.6f}, std={torch_output.std():.6f}")
    
    # MLX 推理 - 逐层
    print("\n=== MLX 推理 ===")
    test_input_mlx = torch_to_mlx(test_input)
    test_ylens_mlx = torch_to_mlx(test_ylens)
    
    # 1. content_in_proj
    x_mlx = mlx_length_regulator.content_in_proj(test_input_mlx)
    mx.eval(x_mlx)
    print(f"1. After content_in_proj: shape={x_mlx.shape}, mean={float(mx.mean(x_mlx)):.6f}, std={float(mx.std(x_mlx)):.6f}")
    
    # 对比第一步
    x_torch_np = x_torch.cpu().numpy()
    x_mlx_np = np.array(x_mlx)
    diff1 = np.abs(x_torch_np - x_mlx_np).max()
    corr1 = np.corrcoef(x_torch_np.flatten(), x_mlx_np.flatten())[0, 1]
    print(f"   -> Diff with PyTorch: max={diff1:.8f}, corr={corr1:.6f}")
    
    # 完整forward
    mlx_output, *_ = mlx_length_regulator(test_input_mlx, ylens=test_ylens_mlx, n_quantizers=None, f0=None)
    mx.eval(mlx_output)
    print(f"\n最终 MLX 输出: shape={mlx_output.shape}, mean={float(mx.mean(mlx_output)):.6f}, std={float(mx.std(mlx_output)):.6f}")
    
    # 最终对比
    torch_output_np = torch_output.cpu().numpy()
    mlx_output_np = np.array(mlx_output)
    max_diff = np.abs(torch_output_np - mlx_output_np).max()
    mean_diff = np.abs(torch_output_np - mlx_output_np).mean()
    correlation = np.corrcoef(torch_output_np.flatten(), mlx_output_np.flatten())[0, 1]
    
    print(f"\n最终对比:")
    print(f"  Max diff: {max_diff:.8f}")
    print(f"  Mean diff: {mean_diff:.8f}")
    print(f"  Correlation: {correlation:.6f}")
    
    if correlation < 0.99:
        print("\n⚠️  检测到显著差异，可能的原因：")
        print("  1. 插值方法不同 (nearest vs linear)")
        print("  2. GroupNorm的实现差异")
        print("  3. Upsample的实现差异")

if __name__ == "__main__":
    main()

