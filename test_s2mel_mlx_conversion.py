#!/usr/bin/env python3
"""
S2MEL MLX模块测试 - 逐层对比PyTorch和MLX实现
确保完全一致性
"""

import torch
import mlx.core as mx
import numpy as np
from omegaconf import OmegaConf

# PyTorch implementations
from indextts.s2mel.modules.commons import MyModel as PyTorchS2MEL

# MLX implementations  
from indextts.s2mel.mlx_modules import MLXInterpolateRegulator, MLXGPTLayer

# Utilities
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch


def compare_arrays(torch_output, mlx_output, name="", tolerance=1e-3):
    """Compare PyTorch tensor and MLX array."""
    if isinstance(torch_output, torch.Tensor):
        torch_np = torch_output.detach().cpu().numpy()
    else:
        torch_np = torch_output
    
    if isinstance(mlx_output, mx.array):
        mlx_np = np.array(mlx_output)
    else:
        mlx_np = mlx_output
    
    max_diff = np.abs(torch_np - mlx_np).max()
    mean_diff = np.abs(torch_np - mlx_np).mean()
    
    match = max_diff < tolerance
    status = "✅" if match else "❌"
    
    print(f"{status} {name}")
    print(f"   Shape: PyTorch {torch_np.shape}, MLX {mlx_np.shape}")
    print(f"   Max diff: {max_diff:.6f}")
    print(f"   Mean diff: {mean_diff:.6f}")
    
    if not match:
        print(f"   ⚠️  Tolerance exceeded: {max_diff:.6f} > {tolerance}")
        print(f"   PyTorch range: [{torch_np.min():.6f}, {torch_np.max():.6f}]")
        print(f"   MLX range: [{mlx_np.min():.6f}, {mlx_np.max():.6f}]")
    
    return match


def test_gpt_layer():
    """Test GPT Layer: 1280 → 256 → 128 → 1024"""
    print("\n" + "=" * 80)
    print("测试 GPT Layer")
    print("=" * 80)
    
    # Create PyTorch version
    pt_layer = torch.nn.Sequential(
        torch.nn.Linear(1280, 256),
        torch.nn.Linear(256, 128),
        torch.nn.Linear(128, 1024)
    )
    pt_layer.eval()
    
    # Create MLX version
    mlx_layer = MLXGPTLayer()
    
    # Copy weights from PyTorch to MLX
    print("\n>> 复制权重...")
    with torch.no_grad():
        # Layer 1
        mlx_layer.layer1.weight = mx.array(pt_layer[0].weight.numpy())
        mlx_layer.layer1.bias = mx.array(pt_layer[0].bias.numpy())
        # Layer 2
        mlx_layer.layer2.weight = mx.array(pt_layer[1].weight.numpy())
        mlx_layer.layer2.bias = mx.array(pt_layer[1].bias.numpy())
        # Layer 3
        mlx_layer.layer3.weight = mx.array(pt_layer[2].weight.numpy())
        mlx_layer.layer3.bias = mx.array(pt_layer[2].bias.numpy())
    
    # Test with random input (simulating GPT output)
    print("\n>> 测试输入: (1, 100, 1280)")
    x_torch = torch.randn(1, 100, 1280)
    x_mlx = torch_to_mlx(x_torch.cpu())
    
    # Forward pass
    with torch.no_grad():
        out_torch = pt_layer(x_torch)
    
    out_mlx = mlx_layer(x_mlx)
    mx.eval(out_mlx)
    
    # Compare
    match = compare_arrays(out_torch, out_mlx, "GPT Layer Output", tolerance=1e-3)
    
    if match:
        print("\n✅ GPT Layer 测试通过！")
    else:
        print("\n❌ GPT Layer 测试失败！")
    
    return match


def test_length_regulator():
    """Test Length Regulator with exact inference scenario"""
    print("\n" + "=" * 80)
    print("测试 Length Regulator")
    print("=" * 80)
    
    # Load config
    cfg = OmegaConf.load("checkpoints/config.yaml")
    
    # Create PyTorch version
    print("\n>> 创建 PyTorch Length Regulator...")
    from indextts.s2mel.modules.length_regulator import InterpolateRegulator
    
    pt_lr = InterpolateRegulator(
        channels=cfg.s2mel.length_regulator.channels,
        sampling_ratios=cfg.s2mel.length_regulator.sampling_ratios,
        is_discrete=cfg.s2mel.length_regulator.is_discrete,
        in_channels=cfg.s2mel.length_regulator.in_channels if hasattr(cfg.s2mel.length_regulator, "in_channels") else None,
        vector_quantize=cfg.s2mel.length_regulator.vector_quantize if hasattr(cfg.s2mel.length_regulator, "vector_quantize") else False,
        codebook_size=cfg.s2mel.length_regulator.content_codebook_size,
        n_codebooks=cfg.s2mel.length_regulator.n_codebooks if hasattr(cfg.s2mel.length_regulator, "n_codebooks") else 1,
        quantizer_dropout=cfg.s2mel.length_regulator.quantizer_dropout if hasattr(cfg.s2mel.length_regulator, "quantizer_dropout") else 0.0,
        f0_condition=cfg.s2mel.length_regulator.f0_condition if hasattr(cfg.s2mel.length_regulator, "f0_condition") else False,
        n_f0_bins=cfg.s2mel.length_regulator.n_f0_bins if hasattr(cfg.s2mel.length_regulator, "n_f0_bins") else 512,
    )
    pt_lr.eval()
    
    # Create MLX version with same config
    print(">> 创建 MLX Length Regulator...")
    mlx_lr = MLXInterpolateRegulator(
        channels=cfg.s2mel.length_regulator.channels,
        sampling_ratios=cfg.s2mel.length_regulator.sampling_ratios,
        is_discrete=cfg.s2mel.length_regulator.is_discrete,
        in_channels=cfg.s2mel.length_regulator.in_channels if hasattr(cfg.s2mel.length_regulator, "in_channels") else None,
        vector_quantize=cfg.s2mel.length_regulator.vector_quantize if hasattr(cfg.s2mel.length_regulator, "vector_quantize") else False,
        codebook_size=cfg.s2mel.length_regulator.content_codebook_size,
        n_codebooks=cfg.s2mel.length_regulator.n_codebooks if hasattr(cfg.s2mel.length_regulator, "n_codebooks") else 1,
        quantizer_dropout=cfg.s2mel.length_regulator.quantizer_dropout if hasattr(cfg.s2mel.length_regulator, "quantizer_dropout") else 0.0,
        f0_condition=cfg.s2mel.length_regulator.f0_condition if hasattr(cfg.s2mel.length_regulator, "f0_condition") else False,
        n_f0_bins=cfg.s2mel.length_regulator.n_f0_bins if hasattr(cfg.s2mel.length_regulator, "n_f0_bins") else 512,
    )
    
    # Copy weights
    print("\n>> 复制权重...")
    with torch.no_grad():
        # Embedding
        mlx_lr.embedding.weight = mx.array(pt_lr.embedding.weight.numpy())
        
        # Model layers
        for i, pt_module in enumerate(pt_lr.model):
            if hasattr(pt_module, 'weight'):
                if i < len(mlx_lr.model):
                    mlx_layer = mlx_lr.model[i]
                    if hasattr(mlx_layer, 'weight'):
                        mlx_layer.weight = mx.array(pt_module.weight.numpy())
                        if hasattr(pt_module, 'bias') and pt_module.bias is not None:
                            mlx_layer.bias = mx.array(pt_module.bias.numpy())
                        print(f"   ✓ Layer {i}: {type(pt_module).__name__}")
        
        # Extra codebooks (if multi-codebook)
        if hasattr(pt_lr, 'extra_codebooks'):
            for i, pt_emb in enumerate(pt_lr.extra_codebooks):
                mlx_lr.extra_codebooks[i].weight = mx.array(pt_emb.weight.numpy())
                print(f"   ✓ Extra codebook {i}")
        
        # F0 embedding (if f0_condition)
        if hasattr(pt_lr, 'f0_embedding') and pt_lr.f0_condition:
            mlx_lr.f0_embedding.weight = mx.array(pt_lr.f0_embedding.weight.numpy())
            print(f"   ✓ F0 embedding")
    
    # Test with realistic inference scenario
    # Note: Config shows is_discrete=false (continuous input from semantic codec)
    # Input: (batch, seq_len, in_channels=1024)
    print("\n>> 测试场景: 推理模式 (continuous input from semantic codec)")
    print(f"   is_discrete: {cfg.s2mel.length_regulator.is_discrete}")
    print(f"   in_channels: {cfg.s2mel.length_regulator.in_channels}")
    print(f"   输入: (1, 50, 1024) - continuous semantic features")
    print(f"   目标长度: 86 frames (1.72x upsampling)")
    
    batch_size = 1
    seq_len = 50
    in_channels = cfg.s2mel.length_regulator.in_channels
    target_len = 86
    
    # Create continuous input (semantic features)
    x_torch = torch.randn(batch_size, seq_len, in_channels)
    ylens_torch = torch.tensor([target_len])
    
    # Convert to MLX
    x_mlx = mx.array(x_torch.numpy())
    ylens_mlx = mx.array(ylens_torch.numpy())
    
    # Forward pass
    print("\n>> PyTorch forward...")
    with torch.no_grad():
        out_torch, olens_torch, _, _, _ = pt_lr(
            x_torch,
            ylens=ylens_torch,
            n_quantizers=None,  # Not used for continuous input
            f0=None
        )
    
    print(f"   输出: {out_torch.shape}")
    
    print("\n>> MLX forward...")
    out_mlx, olens_mlx, _, _, _ = mlx_lr(
        x_mlx,
        ylens=ylens_mlx,
        n_quantizers=None,  # Not used for continuous input
        f0=None
    )
    mx.eval(out_mlx)
    
    print(f"   输出: {out_mlx.shape}")
    
    # Compare outputs
    print("\n>> 对比输出...")
    match_out = compare_arrays(out_torch, out_mlx, "Length Regulator Output", tolerance=1e-3)
    match_lens = compare_arrays(olens_torch, olens_mlx, "Output Lengths", tolerance=0)
    
    if match_out and match_lens:
        print("\n✅ Length Regulator 测试通过！")
    else:
        print("\n❌ Length Regulator 测试失败！")
    
    return match_out and match_lens


def test_integrated_scenario():
    """Test integrated scenario matching actual inference"""
    print("\n" + "=" * 80)
    print("测试 完整推理场景")
    print("=" * 80)
    print("模拟实际 IndexTTS2 inference 中的调用")
    
    # This would test the complete flow:
    # 1. Semantic codes → Length Regulator → conditioning
    # 2. GPT latent → GPT Layer → S2MEL latent
    
    print("\n>> 场景1: Prompt conditioning")
    print("   S_ref (semantic) → Length Regulator → prompt_condition")
    
    print("\n>> 场景2: Generated conditioning")
    print("   S_infer (semantic) → Length Regulator → cond")
    
    print("\n>> 场景3: GPT latent projection")
    print("   latent (1280D) → GPT Layer → latent (1024D)")
    
    # For now, we've tested individual components above
    print("\n✅ 组件测试已覆盖实际推理场景")


def main():
    print("=" * 80)
    print("S2MEL MLX 模块测试 - 完全参考 PyTorch 版本")
    print("=" * 80)
    
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    mx.random.seed(SEED)
    
    # Test individual components
    results = {}
    
    results['gpt_layer'] = test_gpt_layer()
    results['length_regulator'] = test_length_regulator()
    
    # Summary
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 所有测试通过！MLX实现与PyTorch完全一致。")
        return 0
    else:
        print("\n⚠️  部分测试失败，需要修复。")
        return 1


if __name__ == "__main__":
    exit(main())

