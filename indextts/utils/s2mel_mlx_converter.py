"""
S2MEL PyTorch → MLX 权重转换工具

将PyTorch的S2MEL模型权重转换为MLX格式。
"""

import torch
import mlx.core as mx
import numpy as np
from typing import Dict
from pathlib import Path


def torch_to_mlx_array(tensor: torch.Tensor) -> mx.array:
    """
    Convert PyTorch tensor to MLX array.
    
    Args:
        tensor: PyTorch tensor
    
    Returns:
        MLX array
    """
    return mx.array(tensor.detach().cpu().numpy())


def convert_length_regulator_weights(
    pytorch_state_dict: Dict[str, torch.Tensor],
    prefix: str = "models.length_regulator."
) -> Dict[str, mx.array]:
    """
    Convert Length Regulator weights from PyTorch to MLX format.
    
    Args:
        pytorch_state_dict: PyTorch model state dict
        prefix: Prefix for length_regulator keys (default: "models.length_regulator.")
    
    Returns:
        MLX weights dictionary
    """
    mlx_weights = {}
    
    # Mapping of PyTorch keys to MLX keys
    # PyTorch uses ModuleList/Sequential, MLX uses list
    key_mappings = {
        # Model layers (Conv1d, GroupNorm, Mish, etc.)
        # PyTorch: model.0.weight -> MLX: model.0.weight
        # Note: MLX Conv1d expects weights in (out_channels, in_channels, kernel_size)
        # same as PyTorch, no transpose needed
        
        # Embeddings
        f"{prefix}embedding.weight": "embedding.weight",
        f"{prefix}mask_token": "mask_token",
        
        # F0 conditioning (if present)
        f"{prefix}f0_embedding.weight": "f0_embedding.weight",
        f"{prefix}f0_mask": "f0_mask",
        
        # Content projection (for continuous input)
        f"{prefix}content_in_proj.weight": "content_in_proj.weight",
        f"{prefix}content_in_proj.bias": "content_in_proj.bias",
    }
    
    # Convert simple mappings
    for pt_key, mlx_key in key_mappings.items():
        if pt_key in pytorch_state_dict:
            mlx_weights[mlx_key] = torch_to_mlx_array(pytorch_state_dict[pt_key])
            print(f"  ✓ {pt_key} → {mlx_key} {mlx_weights[mlx_key].shape}")
    
    # Convert model layers (Conv1d, GroupNorm, etc.)
    model_idx = 0
    while True:
        # Check for Conv1d
        conv_weight_key = f"{prefix}model.{model_idx}.weight"
        if conv_weight_key not in pytorch_state_dict:
            break
        
        # Conv1d weight
        mlx_weights[f"model.{model_idx}.weight"] = torch_to_mlx_array(
            pytorch_state_dict[conv_weight_key]
        )
        print(f"  ✓ {conv_weight_key} → model.{model_idx}.weight")
        
        # Conv1d bias (optional)
        conv_bias_key = f"{prefix}model.{model_idx}.bias"
        if conv_bias_key in pytorch_state_dict:
            mlx_weights[f"model.{model_idx}.bias"] = torch_to_mlx_array(
                pytorch_state_dict[conv_bias_key]
            )
            print(f"  ✓ {conv_bias_key} → model.{model_idx}.bias")
        
        model_idx += 1
        
        # Check for GroupNorm (model.{idx+1})
        norm_weight_key = f"{prefix}model.{model_idx}.weight"
        if norm_weight_key in pytorch_state_dict:
            mlx_weights[f"model.{model_idx}.weight"] = torch_to_mlx_array(
                pytorch_state_dict[norm_weight_key]
            )
            mlx_weights[f"model.{model_idx}.bias"] = torch_to_mlx_array(
                pytorch_state_dict[f"{prefix}model.{model_idx}.bias"]
            )
            print(f"  ✓ {norm_weight_key} → model.{model_idx}.weight")
            model_idx += 1
        
        # Mish activation (model.{idx+2}) - no parameters
        model_idx += 1
    
    # Convert extra codebooks (if multi-codebook)
    codebook_idx = 0
    while True:
        codebook_key = f"{prefix}extra_codebooks.{codebook_idx}.weight"
        if codebook_key not in pytorch_state_dict:
            break
        mlx_weights[f"extra_codebooks.{codebook_idx}.weight"] = torch_to_mlx_array(
            pytorch_state_dict[codebook_key]
        )
        print(f"  ✓ {codebook_key} → extra_codebooks.{codebook_idx}.weight")
        codebook_idx += 1
    
    return mlx_weights


def convert_gpt_layer_weights(
    pytorch_state_dict: Dict[str, torch.Tensor],
    prefix: str = "models.gpt_layer."
) -> Dict[str, mx.array]:
    """
    Convert GPT Layer weights from PyTorch to MLX format.
    
    PyTorch Sequential(Linear, Linear, Linear) → MLX layer1, layer2, layer3
    
    Args:
        pytorch_state_dict: PyTorch model state dict
        prefix: Prefix for gpt_layer keys (default: "models.gpt_layer.")
    
    Returns:
        MLX weights dictionary
    """
    mlx_weights = {}
    
    # PyTorch: 0.weight, 0.bias, 1.weight, 1.bias, 2.weight, 2.bias
    # MLX: layer1.weight, layer1.bias, layer2.weight, layer2.bias, ...
    layer_mappings = [
        (f"{prefix}0.weight", "layer1.weight"),
        (f"{prefix}0.bias", "layer1.bias"),
        (f"{prefix}1.weight", "layer2.weight"),
        (f"{prefix}1.bias", "layer2.bias"),
        (f"{prefix}2.weight", "layer3.weight"),
        (f"{prefix}2.bias", "layer3.bias"),
    ]
    
    for pt_key, mlx_key in layer_mappings:
        if pt_key in pytorch_state_dict:
            # MLX Linear expects (out_features, in_features), same as PyTorch
            mlx_weights[mlx_key] = torch_to_mlx_array(pytorch_state_dict[pt_key])
            print(f"  ✓ {pt_key} → {mlx_key} {mlx_weights[mlx_key].shape}")
        else:
            print(f"  ⚠️  {pt_key} not found in state dict")
    
    return mlx_weights


def convert_s2mel_to_mlx(
    pytorch_checkpoint_path: str,
    output_dir: str = "checkpoints/mlx/s2mel"
) -> None:
    """
    Convert complete S2MEL model from PyTorch to MLX format.
    
    Converts:
    - Length Regulator
    - GPT Layer
    
    Args:
        pytorch_checkpoint_path: Path to PyTorch checkpoint (.pth file)
        output_dir: Output directory for MLX weights
    """
    print("=" * 80)
    print("S2MEL PyTorch → MLX 权重转换")
    print("=" * 80)
    
    # Load PyTorch checkpoint
    print(f"\n>> 加载 PyTorch checkpoint: {pytorch_checkpoint_path}")
    checkpoint = torch.load(pytorch_checkpoint_path, map_location="cpu")
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    print(f">> 找到 {len(state_dict)} 个参数")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert Length Regulator
    print("\n>> 转换 Length Regulator...")
    lr_weights = convert_length_regulator_weights(state_dict)
    lr_output_file = output_path / "length_regulator.npz"
    mx.savez(str(lr_output_file), **lr_weights)
    print(f">> ✓ Length Regulator 保存到: {lr_output_file}")
    print(f"   共 {len(lr_weights)} 个参数")
    
    # Convert GPT Layer (if present)
    gpt_layer_prefix = "models.gpt_layer."
    has_gpt_layer = any(k.startswith(gpt_layer_prefix) for k in state_dict.keys())
    
    if has_gpt_layer:
        print("\n>> 转换 GPT Layer...")
        gpt_weights = convert_gpt_layer_weights(state_dict)
        gpt_output_file = output_path / "gpt_layer.npz"
        mx.savez(str(gpt_output_file), **gpt_weights)
        print(f">> ✓ GPT Layer 保存到: {gpt_output_file}")
        print(f"   共 {len(gpt_weights)} 个参数")
    else:
        print("\n>> ⚠️  未找到 GPT Layer (可能不使用 use_gpt_latent)")
    
    print("\n" + "=" * 80)
    print("转换完成！")
    print("=" * 80)
    print(f"\nMLX 权重已保存到: {output_path}")
    print(f"- length_regulator.npz ({len(lr_weights)} 参数)")
    if has_gpt_layer:
        print(f"- gpt_layer.npz ({len(gpt_weights)} 参数)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert S2MEL PyTorch weights to MLX")
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to PyTorch checkpoint (.pth file)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/mlx/s2mel",
        help="Output directory for MLX weights (default: checkpoints/mlx/s2mel)"
    )
    
    args = parser.parse_args()
    
    convert_s2mel_to_mlx(args.checkpoint, args.output_dir)

