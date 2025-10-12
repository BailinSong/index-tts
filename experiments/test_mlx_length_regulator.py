#!/usr/bin/env python3
"""
Test MLX Length Regulator implementation
Compare MLX vs PyTorch outputs
"""

import torch
import mlx.core as mx
import numpy as np
import time
import sys
sys.path.insert(0, '/Users/bailin/index-tts')

from indextts.s2mel.modules.length_regulator import InterpolateRegulator
from indextts.s2mel.mlx_modules.length_regulator import MLXInterpolateRegulator


def convert_pytorch_to_mlx_weights(pytorch_model: InterpolateRegulator, mlx_model: MLXInterpolateRegulator):
    """
    Convert PyTorch InterpolateRegulator weights to MLX format.
    """
    print("\n>> Converting PyTorch weights to MLX...")
    
    # Embedding weights
    if pytorch_model.is_discrete:
        mlx_model.embedding.weight = mx.array(pytorch_model.embedding.weight.detach().cpu().numpy())
        print(f"   Embedding: {pytorch_model.embedding.weight.shape}")
        
        if pytorch_model.n_codebooks > 1:
            for i, (pt_emb, mlx_emb) in enumerate(zip(pytorch_model.extra_codebooks, mlx_model.extra_codebooks)):
                mlx_emb.weight = mx.array(pt_emb.weight.detach().cpu().numpy())
                print(f"   Extra codebook {i}: {pt_emb.weight.shape}")
    else:
        # content_in_proj
        mlx_model.content_in_proj.weight = mx.array(pytorch_model.content_in_proj.weight.detach().cpu().numpy())
        mlx_model.content_in_proj.bias = mx.array(pytorch_model.content_in_proj.bias.detach().cpu().numpy())
    
    # Model layers
    pt_model_list = list(pytorch_model.model)
    mlx_layer_idx = 0
    
    for pt_layer in pt_model_list:
        if isinstance(pt_layer, torch.nn.Conv1d):
            # Conv1d weights
            # PyTorch: (out_channels, in_channels, kernel_size)
            # MLX: (out_channels, kernel_size, in_channels)
            pt_weight = pt_layer.weight.detach().cpu().numpy()
            # Transpose: (O, I, K) -> (O, K, I)
            mlx_weight = np.transpose(pt_weight, (0, 2, 1))
            mlx_model.model[mlx_layer_idx].weight = mx.array(mlx_weight)
            
            if pt_layer.bias is not None:
                mlx_model.model[mlx_layer_idx].bias = mx.array(pt_layer.bias.detach().cpu().numpy())
            
            print(f"   Conv1d layer: PyTorch {pt_weight.shape} -> MLX {mlx_weight.shape}")
            mlx_layer_idx += 1
            
        elif isinstance(pt_layer, torch.nn.GroupNorm):
            # GroupNorm weights
            mlx_model.model[mlx_layer_idx].weight = mx.array(pt_layer.weight.detach().cpu().numpy())
            mlx_model.model[mlx_layer_idx].bias = mx.array(pt_layer.bias.detach().cpu().numpy())
            print(f"   GroupNorm: {pt_layer.weight.shape}")
            mlx_layer_idx += 1
            
        elif isinstance(pt_layer, torch.nn.Mish):
            # Mish: no parameters
            mlx_layer_idx += 1
    
    print(">> Weight conversion complete")


def test_mlx_length_regulator():
    print("=" * 60)
    print("MLX Length Regulator Test")
    print("=" * 60)
    
    # Configuration
    channels = 1024
    codebook_size = 1024
    sampling_ratios = [2, 2]  # Example ratios
    out_channels = 512
    groups = 1
    
    # Create PyTorch model
    print("\n>> Creating PyTorch InterpolateRegulator...")
    pytorch_model = InterpolateRegulator(
        channels=channels,
        sampling_ratios=sampling_ratios,
        is_discrete=True,
        codebook_size=codebook_size,
        out_channels=out_channels,
        groups=groups,
        n_codebooks=1
    ).eval()
    
    # Create MLX model
    print(">> Creating MLX InterpolateRegulator...")
    mlx_model = MLXInterpolateRegulator(
        channels=channels,
        sampling_ratios=sampling_ratios,
        is_discrete=True,
        codebook_size=codebook_size,
        out_channels=out_channels,
        groups=groups,
        n_codebooks=1
    )
    
    # Convert weights
    convert_pytorch_to_mlx_weights(pytorch_model, mlx_model)
    
    # Prepare test inputs
    print("\n>> Preparing test inputs...")
    batch_size = 1
    seq_len = 100
    target_len = 172  # Approx seq_len * 1.72
    
    # Random discrete codes
    pytorch_input = torch.randint(0, codebook_size, (batch_size, seq_len))
    mlx_input = mx.array(pytorch_input.numpy())
    
    pytorch_ylens = torch.LongTensor([target_len])
    mlx_ylens = mx.array([target_len], dtype=mx.int32)
    
    print(f"   Input shape: {pytorch_input.shape}")
    print(f"   Target length: {target_len}")
    
    # PyTorch forward
    print("\n>> Running PyTorch forward pass...")
    t0 = time.perf_counter()
    with torch.no_grad():
        pytorch_output, _, _, _, _ = pytorch_model(
            pytorch_input,
            ylens=pytorch_ylens,
            n_quantizers=3,
            f0=None
        )
    torch.mps.synchronize() if torch.backends.mps.is_available() else None
    pytorch_time = time.perf_counter() - t0
    print(f"   PyTorch time: {pytorch_time:.4f}s")
    print(f"   PyTorch output shape: {pytorch_output.shape}")
    
    # MLX forward
    print("\n>> Running MLX forward pass...")
    t0 = time.perf_counter()
    mlx_output, _, _, _, _ = mlx_model(
        mlx_input,
        ylens=mlx_ylens,
        n_quantizers=3,
        f0=None
    )
    mx.eval(mlx_output)  # Force evaluation
    mlx_time = time.perf_counter() - t0
    print(f"   MLX time: {mlx_time:.4f}s")
    print(f"   MLX output shape: {mlx_output.shape}")
    
    # Compare outputs
    print("\n>> Comparing outputs...")
    pytorch_np = pytorch_output.detach().cpu().numpy()
    mlx_np = np.array(mlx_output)
    
    # Compute correlation
    pytorch_flat = pytorch_np.flatten()
    mlx_flat = mlx_np.flatten()
    correlation = np.corrcoef(pytorch_flat, mlx_flat)[0, 1]
    
    # Compute differences
    abs_diff = np.abs(pytorch_flat - mlx_flat)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    
    # RMS
    pytorch_rms = np.sqrt(np.mean(pytorch_flat ** 2))
    mlx_rms = np.sqrt(np.mean(mlx_flat ** 2))
    
    print(f"   Correlation: {correlation:.6f}")
    print(f"   Max absolute difference: {max_abs_diff:.6e}")
    print(f"   Mean absolute difference: {mean_abs_diff:.6e}")
    print(f"   PyTorch RMS: {pytorch_rms:.6f}")
    print(f"   MLX RMS: {mlx_rms:.6f}")
    print(f"   RMS ratio: {mlx_rms / pytorch_rms:.6f}")
    
    # Performance comparison
    print("\n>> Performance comparison:")
    print(f"   PyTorch: {pytorch_time:.4f}s")
    print(f"   MLX: {mlx_time:.4f}s")
    speedup = pytorch_time / mlx_time
    print(f"   Speedup: {speedup:.2f}x")
    
    # Verdict
    print("\n" + "=" * 60)
    if correlation > 0.99:
        print("✅ MLX Length Regulator: PASS (correlation > 0.99)")
    elif correlation > 0.95:
        print("⚠️  MLX Length Regulator: ACCEPTABLE (correlation > 0.95)")
    else:
        print("❌ MLX Length Regulator: FAIL (correlation < 0.95)")
    print("=" * 60)


if __name__ == '__main__':
    test_mlx_length_regulator()

