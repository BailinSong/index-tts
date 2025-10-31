"""
vq2emb Torch vs MLX 对比测试脚本

测试流程：
1. 使用 MLX GPT 生成 codes（或加载已有缓存）
2. 使用相同的 codes 分别调用 PyTorch 和 MLX 版本的 vq2emb
3. 对比输出差异（形状、数值范围、误差统计）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pickle
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
import safetensors.torch

from indextts.utils.maskgct_utils import build_semantic_codec
from indextts.utils.mlx_vq2emb import MLXVQ2Emb
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch
import mlx.core as mx


def load_semantic_codec(device='cpu'):
    """加载 Semantic Codec 模型"""
    cfg = OmegaConf.load('checkpoints/config.yaml')
    semantic_codec = build_semantic_codec(cfg.semantic_codec)
    semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
    safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
    semantic_codec = semantic_codec.to(device)
    semantic_codec.eval()
    print(f'>> Semantic Codec loaded from: {semantic_code_ckpt}')
    return semantic_codec


def generate_test_codes_from_cache(cache_path='gpt_outputs_mlx.pkl'):
    """
    从缓存文件加载 codes（由 MLX GPT 生成）
    
    Args:
        cache_path: 缓存文件路径
        
    Returns:
        codes: torch.Tensor (B, T) - 语义编码序列
    """
    if os.path.exists(cache_path):
        print(f">> Loading codes from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
            codes = data['codes']
            if isinstance(codes, torch.Tensor):
                codes = codes.cpu()
            else:
                codes = torch.tensor(codes)
            print(f">> Loaded codes: shape={codes.shape}, dtype={codes.dtype}")
            print(f">> Codes range: min={codes.min().item()}, max={codes.max().item()}")
            return codes
    else:
        raise FileNotFoundError(f"Cache file not found: {cache_path}")


def generate_random_codes(batch_size=1, seq_len=100, codebook_size=8194):
    """
    生成随机 codes 用于测试
    
    Args:
        batch_size: 批次大小
        seq_len: 序列长度
        codebook_size: 码本大小
        
    Returns:
        codes: torch.Tensor (B, T) - 随机语义编码
    """
    print(f">> Generating random codes: shape=({batch_size}, {seq_len}), codebook_size={codebook_size}")
    codes = torch.randint(0, codebook_size, (batch_size, seq_len), dtype=torch.long)
    print(f">> Generated codes: range=[{codes.min().item()}, {codes.max().item()}]")
    return codes


def test_pytorch_vq2emb(semantic_codec, codes):
    """
    使用 PyTorch 版本的 vq2emb
    
    Args:
        semantic_codec: Semantic Codec 模型
        codes: torch.Tensor (B, T) - 语义编码序列
        
    Returns:
        output: torch.Tensor (B, input_dim, T) - vq2emb 输出
    """
    print("\n" + "="*80)
    print("Testing PyTorch vq2emb")
    print("="*80)
    
    # PyTorch vq2emb 接收 (B, 1, T) 格式
    vq_input = codes.unsqueeze(1)  # (B, T) -> (B, 1, T)
    print(f">> Input shape: {vq_input.shape}")
    
    with torch.no_grad():
        output = semantic_codec.quantizer.vq2emb(vq_input)
    
    print(f">> Output shape: {output.shape}")
    print(f">> Output dtype: {output.dtype}")
    print(f">> Output range: min={output.min().item():.6f}, max={output.max().item():.6f}")
    print(f">> Output mean: {output.mean().item():.6f}, std={output.std().item():.6f}")
    
    return output


def test_mlx_vq2emb(semantic_codec, codes, mlx_vq2emb=None):
    """
    使用 MLX 版本的 vq2emb
    
    Args:
        semantic_codec: Semantic Codec 模型（用于创建 MLX vq2emb）
        codes: torch.Tensor (B, T) - 语义编码序列
        mlx_vq2emb: 已初始化的 MLX vq2emb（可选）
        
    Returns:
        output: torch.Tensor (B, input_dim, T) - vq2emb 输出
    """
    print("\n" + "="*80)
    print("Testing MLX vq2emb")
    print("="*80)
    
    # 创建 MLX vq2emb 如果未提供
    if mlx_vq2emb is None:
        print(">> Creating MLX vq2emb...")
        mlx_vq2emb = MLXVQ2Emb(semantic_codec.quantizer)
        print(">> ✓ MLX vq2emb created")
    
    # MLX vq2emb 接收 (B, 1, T) 格式
    vq_input = codes.unsqueeze(1)  # (B, T) -> (B, 1, T)
    print(f">> Input shape: {vq_input.shape}")
    
    # 调用 MLX vq2emb
    output_mlx = mlx_vq2emb(vq_input)
    mx.eval(output_mlx)
    
    # 转换为 PyTorch tensor 以便对比
    output = mlx_to_torch(output_mlx)
    
    print(f">> Output shape: {output.shape}")
    print(f">> Output dtype: {output.dtype}")
    print(f">> Output range: min={output.min().item():.6f}, max={output.max().item():.6f}")
    print(f">> Output mean: {output.mean().item():.6f}, std={output.std().item():.6f}")
    
    return output


def compare_outputs(pytorch_output, mlx_output, tolerance=1e-5):
    """
    对比 PyTorch 和 MLX 的输出差异
    
    Args:
        pytorch_output: PyTorch 输出 (B, input_dim, T)
        mlx_output: MLX 输出 (B, input_dim, T)
        tolerance: 数值容差
        
    Returns:
        dict: 对比结果统计
    """
    print("\n" + "="*80)
    print("Comparison Results")
    print("="*80)
    
    # 检查形状
    if pytorch_output.shape != mlx_output.shape:
        print(f"❌ Shape mismatch!")
        print(f"   PyTorch: {pytorch_output.shape}")
        print(f"   MLX:     {mlx_output.shape}")
        return {'match': False, 'error': 'Shape mismatch'}
    
    print(f"✓ Shape match: {pytorch_output.shape}")
    
    # 转换为 float32 进行对比（避免 dtype 差异）
    pytorch_output_fp32 = pytorch_output.float()
    mlx_output_fp32 = mlx_output.float()
    
    # 计算差异
    diff = pytorch_output_fp32 - mlx_output_fp32
    abs_diff = torch.abs(diff)
    
    # 统计信息
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    std_diff = abs_diff.std().item()
    
    # 相对误差（使用 PyTorch 输出的绝对值作为分母）
    abs_pytorch = torch.abs(pytorch_output_fp32)
    relative_diff = abs_diff / (abs_pytorch + 1e-8)  # 避免除零
    max_relative_diff = relative_diff.max().item()
    mean_relative_diff = relative_diff.mean().item()
    
    print(f"\nAbsolute Differences:")
    print(f"  Max:     {max_diff:.9f}")
    print(f"  Mean:    {mean_diff:.9f}")
    print(f"  Std:     {std_diff:.9f}")
    
    print(f"\nRelative Differences:")
    print(f"  Max:     {max_relative_diff:.9f} ({max_relative_diff*100:.6f}%)")
    print(f"  Mean:    {mean_relative_diff:.9f} ({mean_relative_diff*100:.6f}%)")
    
    # 检查是否在容差范围内
    within_tolerance = max_diff < tolerance
    if within_tolerance:
        print(f"\n✓ All values within tolerance ({tolerance})")
    else:
        print(f"\n⚠ Some values exceed tolerance ({tolerance})")
    
    # 统计超出容差的点
    num_outliers = (abs_diff > tolerance).sum().item()
    total_elements = abs_diff.numel()
    outlier_percentage = (num_outliers / total_elements) * 100
    
    print(f"\nOutlier Statistics:")
    print(f"  Outliers:     {num_outliers:,} / {total_elements:,}")
    print(f"  Percentage:   {outlier_percentage:.4f}%")
    
    # 详细统计（按百分位数）
    percentiles = [50, 75, 90, 95, 99, 99.9]
    print(f"\nDifference Percentiles:")
    abs_diff_flat = abs_diff.flatten()
    for p in percentiles:
        val = torch.quantile(abs_diff_flat, p/100).item()
        print(f"  {p:5.1f}%: {val:.9f}")
    
    return {
        'match': within_tolerance,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'max_relative_diff': max_relative_diff,
        'mean_relative_diff': mean_relative_diff,
        'outlier_count': num_outliers,
        'outlier_percentage': outlier_percentage,
        'shape_match': True
    }


def main():
    """主测试函数"""
    print("="*80)
    print("vq2emb Torch vs MLX Comparison Test")
    print("="*80)
    print()
    
    # 配置
    device = 'cpu'  # 使用 CPU 进行公平对比
    cache_path = 'gpt_outputs_mlx.pkl'
    tolerance = 1e-5
    
    # 1. 加载 Semantic Codec
    print("Step 1: Loading Semantic Codec...")
    semantic_codec = load_semantic_codec(device=device)
    print()
    
    # 2. 加载或生成测试 codes
    print("Step 2: Loading test codes...")
    try:
        codes = generate_test_codes_from_cache(cache_path)
    except FileNotFoundError:
        print(f">> Cache not found, generating random codes...")
        codes = generate_random_codes(batch_size=1, seq_len=162)  # 使用实际缓存中的长度
    print()
    
    # 3. 测试 PyTorch 版本
    print("Step 3: Testing PyTorch vq2emb...")
    pytorch_output = test_pytorch_vq2emb(semantic_codec, codes)
    print()
    
    # 4. 测试 MLX 版本
    print("Step 4: Testing MLX vq2emb...")
    mlx_output = test_mlx_vq2emb(semantic_codec, codes)
    print()
    
    # 5. 对比结果
    print("Step 5: Comparing outputs...")
    comparison = compare_outputs(pytorch_output, mlx_output, tolerance=tolerance)
    print()
    
    # 6. 总结
    print("="*80)
    print("Test Summary")
    print("="*80)
    if comparison['match']:
        print("✅ PASS: MLX vq2emb output matches PyTorch version within tolerance")
    else:
        print("❌ FAIL: MLX vq2emb output does not match PyTorch version")
    print()
    print(f"Maximum difference: {comparison['max_diff']:.9f}")
    print(f"Mean difference:    {comparison['mean_diff']:.9f}")
    print(f"Outlier percentage: {comparison['outlier_percentage']:.4f}%")
    print()
    
    return comparison


if __name__ == '__main__':
    comparison = main()

vq2emb Torch vs MLX 对比测试脚本

测试流程：
1. 使用 MLX GPT 生成 codes（或加载已有缓存）
2. 使用相同的 codes 分别调用 PyTorch 和 MLX 版本的 vq2emb
3. 对比输出差异（形状、数值范围、误差统计）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pickle
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
import safetensors.torch

from indextts.utils.maskgct_utils import build_semantic_codec
from indextts.utils.mlx_vq2emb import MLXVQ2Emb
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch
import mlx.core as mx


def load_semantic_codec(device='cpu'):
    """加载 Semantic Codec 模型"""
    cfg = OmegaConf.load('checkpoints/config.yaml')
    semantic_codec = build_semantic_codec(cfg.semantic_codec)
    semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
    safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
    semantic_codec = semantic_codec.to(device)
    semantic_codec.eval()
    print(f'>> Semantic Codec loaded from: {semantic_code_ckpt}')
    return semantic_codec


def generate_test_codes_from_cache(cache_path='gpt_outputs_mlx.pkl'):
    """
    从缓存文件加载 codes（由 MLX GPT 生成）
    
    Args:
        cache_path: 缓存文件路径
        
    Returns:
        codes: torch.Tensor (B, T) - 语义编码序列
    """
    if os.path.exists(cache_path):
        print(f">> Loading codes from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
            codes = data['codes']
            if isinstance(codes, torch.Tensor):
                codes = codes.cpu()
            else:
                codes = torch.tensor(codes)
            print(f">> Loaded codes: shape={codes.shape}, dtype={codes.dtype}")
            print(f">> Codes range: min={codes.min().item()}, max={codes.max().item()}")
            return codes
    else:
        raise FileNotFoundError(f"Cache file not found: {cache_path}")


def generate_random_codes(batch_size=1, seq_len=100, codebook_size=8194):
    """
    生成随机 codes 用于测试
    
    Args:
        batch_size: 批次大小
        seq_len: 序列长度
        codebook_size: 码本大小
        
    Returns:
        codes: torch.Tensor (B, T) - 随机语义编码
    """
    print(f">> Generating random codes: shape=({batch_size}, {seq_len}), codebook_size={codebook_size}")
    codes = torch.randint(0, codebook_size, (batch_size, seq_len), dtype=torch.long)
    print(f">> Generated codes: range=[{codes.min().item()}, {codes.max().item()}]")
    return codes


def test_pytorch_vq2emb(semantic_codec, codes):
    """
    使用 PyTorch 版本的 vq2emb
    
    Args:
        semantic_codec: Semantic Codec 模型
        codes: torch.Tensor (B, T) - 语义编码序列
        
    Returns:
        output: torch.Tensor (B, input_dim, T) - vq2emb 输出
    """
    print("\n" + "="*80)
    print("Testing PyTorch vq2emb")
    print("="*80)
    
    # PyTorch vq2emb 接收 (B, 1, T) 格式
    vq_input = codes.unsqueeze(1)  # (B, T) -> (B, 1, T)
    print(f">> Input shape: {vq_input.shape}")
    
    with torch.no_grad():
        output = semantic_codec.quantizer.vq2emb(vq_input)
    
    print(f">> Output shape: {output.shape}")
    print(f">> Output dtype: {output.dtype}")
    print(f">> Output range: min={output.min().item():.6f}, max={output.max().item():.6f}")
    print(f">> Output mean: {output.mean().item():.6f}, std={output.std().item():.6f}")
    
    return output


def test_mlx_vq2emb(semantic_codec, codes, mlx_vq2emb=None):
    """
    使用 MLX 版本的 vq2emb
    
    Args:
        semantic_codec: Semantic Codec 模型（用于创建 MLX vq2emb）
        codes: torch.Tensor (B, T) - 语义编码序列
        mlx_vq2emb: 已初始化的 MLX vq2emb（可选）
        
    Returns:
        output: torch.Tensor (B, input_dim, T) - vq2emb 输出
    """
    print("\n" + "="*80)
    print("Testing MLX vq2emb")
    print("="*80)
    
    # 创建 MLX vq2emb 如果未提供
    if mlx_vq2emb is None:
        print(">> Creating MLX vq2emb...")
        mlx_vq2emb = MLXVQ2Emb(semantic_codec.quantizer)
        print(">> ✓ MLX vq2emb created")
    
    # MLX vq2emb 接收 (B, 1, T) 格式
    vq_input = codes.unsqueeze(1)  # (B, T) -> (B, 1, T)
    print(f">> Input shape: {vq_input.shape}")
    
    # 调用 MLX vq2emb
    output_mlx = mlx_vq2emb(vq_input)
    mx.eval(output_mlx)
    
    # 转换为 PyTorch tensor 以便对比
    output = mlx_to_torch(output_mlx)
    
    print(f">> Output shape: {output.shape}")
    print(f">> Output dtype: {output.dtype}")
    print(f">> Output range: min={output.min().item():.6f}, max={output.max().item():.6f}")
    print(f">> Output mean: {output.mean().item():.6f}, std={output.std().item():.6f}")
    
    return output


def compare_outputs(pytorch_output, mlx_output, tolerance=1e-5):
    """
    对比 PyTorch 和 MLX 的输出差异
    
    Args:
        pytorch_output: PyTorch 输出 (B, input_dim, T)
        mlx_output: MLX 输出 (B, input_dim, T)
        tolerance: 数值容差
        
    Returns:
        dict: 对比结果统计
    """
    print("\n" + "="*80)
    print("Comparison Results")
    print("="*80)
    
    # 检查形状
    if pytorch_output.shape != mlx_output.shape:
        print(f"❌ Shape mismatch!")
        print(f"   PyTorch: {pytorch_output.shape}")
        print(f"   MLX:     {mlx_output.shape}")
        return {'match': False, 'error': 'Shape mismatch'}
    
    print(f"✓ Shape match: {pytorch_output.shape}")
    
    # 转换为 float32 进行对比（避免 dtype 差异）
    pytorch_output_fp32 = pytorch_output.float()
    mlx_output_fp32 = mlx_output.float()
    
    # 计算差异
    diff = pytorch_output_fp32 - mlx_output_fp32
    abs_diff = torch.abs(diff)
    
    # 统计信息
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    std_diff = abs_diff.std().item()
    
    # 相对误差（使用 PyTorch 输出的绝对值作为分母）
    abs_pytorch = torch.abs(pytorch_output_fp32)
    relative_diff = abs_diff / (abs_pytorch + 1e-8)  # 避免除零
    max_relative_diff = relative_diff.max().item()
    mean_relative_diff = relative_diff.mean().item()
    
    print(f"\nAbsolute Differences:")
    print(f"  Max:     {max_diff:.9f}")
    print(f"  Mean:    {mean_diff:.9f}")
    print(f"  Std:     {std_diff:.9f}")
    
    print(f"\nRelative Differences:")
    print(f"  Max:     {max_relative_diff:.9f} ({max_relative_diff*100:.6f}%)")
    print(f"  Mean:    {mean_relative_diff:.9f} ({mean_relative_diff*100:.6f}%)")
    
    # 检查是否在容差范围内
    within_tolerance = max_diff < tolerance
    if within_tolerance:
        print(f"\n✓ All values within tolerance ({tolerance})")
    else:
        print(f"\n⚠ Some values exceed tolerance ({tolerance})")
    
    # 统计超出容差的点
    num_outliers = (abs_diff > tolerance).sum().item()
    total_elements = abs_diff.numel()
    outlier_percentage = (num_outliers / total_elements) * 100
    
    print(f"\nOutlier Statistics:")
    print(f"  Outliers:     {num_outliers:,} / {total_elements:,}")
    print(f"  Percentage:   {outlier_percentage:.4f}%")
    
    # 详细统计（按百分位数）
    percentiles = [50, 75, 90, 95, 99, 99.9]
    print(f"\nDifference Percentiles:")
    abs_diff_flat = abs_diff.flatten()
    for p in percentiles:
        val = torch.quantile(abs_diff_flat, p/100).item()
        print(f"  {p:5.1f}%: {val:.9f}")
    
    return {
        'match': within_tolerance,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'max_relative_diff': max_relative_diff,
        'mean_relative_diff': mean_relative_diff,
        'outlier_count': num_outliers,
        'outlier_percentage': outlier_percentage,
        'shape_match': True
    }


def main():
    """主测试函数"""
    print("="*80)
    print("vq2emb Torch vs MLX Comparison Test")
    print("="*80)
    print()
    
    # 配置
    device = 'cpu'  # 使用 CPU 进行公平对比
    cache_path = 'gpt_outputs_mlx.pkl'
    tolerance = 1e-5
    
    # 1. 加载 Semantic Codec
    print("Step 1: Loading Semantic Codec...")
    semantic_codec = load_semantic_codec(device=device)
    print()
    
    # 2. 加载或生成测试 codes
    print("Step 2: Loading test codes...")
    try:
        codes = generate_test_codes_from_cache(cache_path)
    except FileNotFoundError:
        print(f">> Cache not found, generating random codes...")
        codes = generate_random_codes(batch_size=1, seq_len=162)  # 使用实际缓存中的长度
    print()
    
    # 3. 测试 PyTorch 版本
    print("Step 3: Testing PyTorch vq2emb...")
    pytorch_output = test_pytorch_vq2emb(semantic_codec, codes)
    print()
    
    # 4. 测试 MLX 版本
    print("Step 4: Testing MLX vq2emb...")
    mlx_output = test_mlx_vq2emb(semantic_codec, codes)
    print()
    
    # 5. 对比结果
    print("Step 5: Comparing outputs...")
    comparison = compare_outputs(pytorch_output, mlx_output, tolerance=tolerance)
    print()
    
    # 6. 总结
    print("="*80)
    print("Test Summary")
    print("="*80)
    if comparison['match']:
        print("✅ PASS: MLX vq2emb output matches PyTorch version within tolerance")
    else:
        print("❌ FAIL: MLX vq2emb output does not match PyTorch version")
    print()
    print(f"Maximum difference: {comparison['max_diff']:.9f}")
    print(f"Mean difference:    {comparison['mean_diff']:.9f}")
    print(f"Outlier percentage: {comparison['outlier_percentage']:.4f}%")
    print()
    
    return comparison


if __name__ == '__main__':
    comparison = main()

vq2emb Torch vs MLX 对比测试脚本

测试流程：
1. 使用 MLX GPT 生成 codes（或加载已有缓存）
2. 使用相同的 codes 分别调用 PyTorch 和 MLX 版本的 vq2emb
3. 对比输出差异（形状、数值范围、误差统计）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pickle
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
import safetensors.torch

from indextts.utils.maskgct_utils import build_semantic_codec
from indextts.utils.mlx_vq2emb import MLXVQ2Emb
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch
import mlx.core as mx


def load_semantic_codec(device='cpu'):
    """加载 Semantic Codec 模型"""
    cfg = OmegaConf.load('checkpoints/config.yaml')
    semantic_codec = build_semantic_codec(cfg.semantic_codec)
    semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
    safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
    semantic_codec = semantic_codec.to(device)
    semantic_codec.eval()
    print(f'>> Semantic Codec loaded from: {semantic_code_ckpt}')
    return semantic_codec


def generate_test_codes_from_cache(cache_path='gpt_outputs_mlx.pkl'):
    """
    从缓存文件加载 codes（由 MLX GPT 生成）
    
    Args:
        cache_path: 缓存文件路径
        
    Returns:
        codes: torch.Tensor (B, T) - 语义编码序列
    """
    if os.path.exists(cache_path):
        print(f">> Loading codes from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
            codes = data['codes']
            if isinstance(codes, torch.Tensor):
                codes = codes.cpu()
            else:
                codes = torch.tensor(codes)
            print(f">> Loaded codes: shape={codes.shape}, dtype={codes.dtype}")
            print(f">> Codes range: min={codes.min().item()}, max={codes.max().item()}")
            return codes
    else:
        raise FileNotFoundError(f"Cache file not found: {cache_path}")


def generate_random_codes(batch_size=1, seq_len=100, codebook_size=8194):
    """
    生成随机 codes 用于测试
    
    Args:
        batch_size: 批次大小
        seq_len: 序列长度
        codebook_size: 码本大小
        
    Returns:
        codes: torch.Tensor (B, T) - 随机语义编码
    """
    print(f">> Generating random codes: shape=({batch_size}, {seq_len}), codebook_size={codebook_size}")
    codes = torch.randint(0, codebook_size, (batch_size, seq_len), dtype=torch.long)
    print(f">> Generated codes: range=[{codes.min().item()}, {codes.max().item()}]")
    return codes


def test_pytorch_vq2emb(semantic_codec, codes):
    """
    使用 PyTorch 版本的 vq2emb
    
    Args:
        semantic_codec: Semantic Codec 模型
        codes: torch.Tensor (B, T) - 语义编码序列
        
    Returns:
        output: torch.Tensor (B, input_dim, T) - vq2emb 输出
    """
    print("\n" + "="*80)
    print("Testing PyTorch vq2emb")
    print("="*80)
    
    # PyTorch vq2emb 接收 (B, 1, T) 格式
    vq_input = codes.unsqueeze(1)  # (B, T) -> (B, 1, T)
    print(f">> Input shape: {vq_input.shape}")
    
    with torch.no_grad():
        output = semantic_codec.quantizer.vq2emb(vq_input)
    
    print(f">> Output shape: {output.shape}")
    print(f">> Output dtype: {output.dtype}")
    print(f">> Output range: min={output.min().item():.6f}, max={output.max().item():.6f}")
    print(f">> Output mean: {output.mean().item():.6f}, std={output.std().item():.6f}")
    
    return output


def test_mlx_vq2emb(semantic_codec, codes, mlx_vq2emb=None):
    """
    使用 MLX 版本的 vq2emb
    
    Args:
        semantic_codec: Semantic Codec 模型（用于创建 MLX vq2emb）
        codes: torch.Tensor (B, T) - 语义编码序列
        mlx_vq2emb: 已初始化的 MLX vq2emb（可选）
        
    Returns:
        output: torch.Tensor (B, input_dim, T) - vq2emb 输出
    """
    print("\n" + "="*80)
    print("Testing MLX vq2emb")
    print("="*80)
    
    # 创建 MLX vq2emb 如果未提供
    if mlx_vq2emb is None:
        print(">> Creating MLX vq2emb...")
        mlx_vq2emb = MLXVQ2Emb(semantic_codec.quantizer)
        print(">> ✓ MLX vq2emb created")
    
    # MLX vq2emb 接收 (B, 1, T) 格式
    vq_input = codes.unsqueeze(1)  # (B, T) -> (B, 1, T)
    print(f">> Input shape: {vq_input.shape}")
    
    # 调用 MLX vq2emb
    output_mlx = mlx_vq2emb(vq_input)
    mx.eval(output_mlx)
    
    # 转换为 PyTorch tensor 以便对比
    output = mlx_to_torch(output_mlx)
    
    print(f">> Output shape: {output.shape}")
    print(f">> Output dtype: {output.dtype}")
    print(f">> Output range: min={output.min().item():.6f}, max={output.max().item():.6f}")
    print(f">> Output mean: {output.mean().item():.6f}, std={output.std().item():.6f}")
    
    return output


def compare_outputs(pytorch_output, mlx_output, tolerance=1e-5):
    """
    对比 PyTorch 和 MLX 的输出差异
    
    Args:
        pytorch_output: PyTorch 输出 (B, input_dim, T)
        mlx_output: MLX 输出 (B, input_dim, T)
        tolerance: 数值容差
        
    Returns:
        dict: 对比结果统计
    """
    print("\n" + "="*80)
    print("Comparison Results")
    print("="*80)
    
    # 检查形状
    if pytorch_output.shape != mlx_output.shape:
        print(f"❌ Shape mismatch!")
        print(f"   PyTorch: {pytorch_output.shape}")
        print(f"   MLX:     {mlx_output.shape}")
        return {'match': False, 'error': 'Shape mismatch'}
    
    print(f"✓ Shape match: {pytorch_output.shape}")
    
    # 转换为 float32 进行对比（避免 dtype 差异）
    pytorch_output_fp32 = pytorch_output.float()
    mlx_output_fp32 = mlx_output.float()
    
    # 计算差异
    diff = pytorch_output_fp32 - mlx_output_fp32
    abs_diff = torch.abs(diff)
    
    # 统计信息
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    std_diff = abs_diff.std().item()
    
    # 相对误差（使用 PyTorch 输出的绝对值作为分母）
    abs_pytorch = torch.abs(pytorch_output_fp32)
    relative_diff = abs_diff / (abs_pytorch + 1e-8)  # 避免除零
    max_relative_diff = relative_diff.max().item()
    mean_relative_diff = relative_diff.mean().item()
    
    print(f"\nAbsolute Differences:")
    print(f"  Max:     {max_diff:.9f}")
    print(f"  Mean:    {mean_diff:.9f}")
    print(f"  Std:     {std_diff:.9f}")
    
    print(f"\nRelative Differences:")
    print(f"  Max:     {max_relative_diff:.9f} ({max_relative_diff*100:.6f}%)")
    print(f"  Mean:    {mean_relative_diff:.9f} ({mean_relative_diff*100:.6f}%)")
    
    # 检查是否在容差范围内
    within_tolerance = max_diff < tolerance
    if within_tolerance:
        print(f"\n✓ All values within tolerance ({tolerance})")
    else:
        print(f"\n⚠ Some values exceed tolerance ({tolerance})")
    
    # 统计超出容差的点
    num_outliers = (abs_diff > tolerance).sum().item()
    total_elements = abs_diff.numel()
    outlier_percentage = (num_outliers / total_elements) * 100
    
    print(f"\nOutlier Statistics:")
    print(f"  Outliers:     {num_outliers:,} / {total_elements:,}")
    print(f"  Percentage:   {outlier_percentage:.4f}%")
    
    # 详细统计（按百分位数）
    percentiles = [50, 75, 90, 95, 99, 99.9]
    print(f"\nDifference Percentiles:")
    abs_diff_flat = abs_diff.flatten()
    for p in percentiles:
        val = torch.quantile(abs_diff_flat, p/100).item()
        print(f"  {p:5.1f}%: {val:.9f}")
    
    return {
        'match': within_tolerance,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'max_relative_diff': max_relative_diff,
        'mean_relative_diff': mean_relative_diff,
        'outlier_count': num_outliers,
        'outlier_percentage': outlier_percentage,
        'shape_match': True
    }


def main():
    """主测试函数"""
    print("="*80)
    print("vq2emb Torch vs MLX Comparison Test")
    print("="*80)
    print()
    
    # 配置
    device = 'cpu'  # 使用 CPU 进行公平对比
    cache_path = 'gpt_outputs_mlx.pkl'
    tolerance = 1e-5
    
    # 1. 加载 Semantic Codec
    print("Step 1: Loading Semantic Codec...")
    semantic_codec = load_semantic_codec(device=device)
    print()
    
    # 2. 加载或生成测试 codes
    print("Step 2: Loading test codes...")
    try:
        codes = generate_test_codes_from_cache(cache_path)
    except FileNotFoundError:
        print(f">> Cache not found, generating random codes...")
        codes = generate_random_codes(batch_size=1, seq_len=162)  # 使用实际缓存中的长度
    print()
    
    # 3. 测试 PyTorch 版本
    print("Step 3: Testing PyTorch vq2emb...")
    pytorch_output = test_pytorch_vq2emb(semantic_codec, codes)
    print()
    
    # 4. 测试 MLX 版本
    print("Step 4: Testing MLX vq2emb...")
    mlx_output = test_mlx_vq2emb(semantic_codec, codes)
    print()
    
    # 5. 对比结果
    print("Step 5: Comparing outputs...")
    comparison = compare_outputs(pytorch_output, mlx_output, tolerance=tolerance)
    print()
    
    # 6. 总结
    print("="*80)
    print("Test Summary")
    print("="*80)
    if comparison['match']:
        print("✅ PASS: MLX vq2emb output matches PyTorch version within tolerance")
    else:
        print("❌ FAIL: MLX vq2emb output does not match PyTorch version")
    print()
    print(f"Maximum difference: {comparison['max_diff']:.9f}")
    print(f"Mean difference:    {comparison['mean_diff']:.9f}")
    print(f"Outlier percentage: {comparison['outlier_percentage']:.4f}%")
    print()
    
    return comparison


if __name__ == '__main__':
    comparison = main()

vq2emb Torch vs MLX 对比测试脚本

测试流程：
1. 使用 MLX GPT 生成 codes（或加载已有缓存）
2. 使用相同的 codes 分别调用 PyTorch 和 MLX 版本的 vq2emb
3. 对比输出差异（形状、数值范围、误差统计）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pickle
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
import safetensors.torch

from indextts.utils.maskgct_utils import build_semantic_codec
from indextts.utils.mlx_vq2emb import MLXVQ2Emb
from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch
import mlx.core as mx


def load_semantic_codec(device='cpu'):
    """加载 Semantic Codec 模型"""
    cfg = OmegaConf.load('checkpoints/config.yaml')
    semantic_codec = build_semantic_codec(cfg.semantic_codec)
    semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
    safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
    semantic_codec = semantic_codec.to(device)
    semantic_codec.eval()
    print(f'>> Semantic Codec loaded from: {semantic_code_ckpt}')
    return semantic_codec


def generate_test_codes_from_cache(cache_path='gpt_outputs_mlx.pkl'):
    """
    从缓存文件加载 codes（由 MLX GPT 生成）
    
    Args:
        cache_path: 缓存文件路径
        
    Returns:
        codes: torch.Tensor (B, T) - 语义编码序列
    """
    if os.path.exists(cache_path):
        print(f">> Loading codes from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
            codes = data['codes']
            if isinstance(codes, torch.Tensor):
                codes = codes.cpu()
            else:
                codes = torch.tensor(codes)
            print(f">> Loaded codes: shape={codes.shape}, dtype={codes.dtype}")
            print(f">> Codes range: min={codes.min().item()}, max={codes.max().item()}")
            return codes
    else:
        raise FileNotFoundError(f"Cache file not found: {cache_path}")


def generate_random_codes(batch_size=1, seq_len=100, codebook_size=8194):
    """
    生成随机 codes 用于测试
    
    Args:
        batch_size: 批次大小
        seq_len: 序列长度
        codebook_size: 码本大小
        
    Returns:
        codes: torch.Tensor (B, T) - 随机语义编码
    """
    print(f">> Generating random codes: shape=({batch_size}, {seq_len}), codebook_size={codebook_size}")
    codes = torch.randint(0, codebook_size, (batch_size, seq_len), dtype=torch.long)
    print(f">> Generated codes: range=[{codes.min().item()}, {codes.max().item()}]")
    return codes


def test_pytorch_vq2emb(semantic_codec, codes):
    """
    使用 PyTorch 版本的 vq2emb
    
    Args:
        semantic_codec: Semantic Codec 模型
        codes: torch.Tensor (B, T) - 语义编码序列
        
    Returns:
        output: torch.Tensor (B, input_dim, T) - vq2emb 输出
    """
    print("\n" + "="*80)
    print("Testing PyTorch vq2emb")
    print("="*80)
    
    # PyTorch vq2emb 接收 (B, 1, T) 格式
    vq_input = codes.unsqueeze(1)  # (B, T) -> (B, 1, T)
    print(f">> Input shape: {vq_input.shape}")
    
    with torch.no_grad():
        output = semantic_codec.quantizer.vq2emb(vq_input)
    
    print(f">> Output shape: {output.shape}")
    print(f">> Output dtype: {output.dtype}")
    print(f">> Output range: min={output.min().item():.6f}, max={output.max().item():.6f}")
    print(f">> Output mean: {output.mean().item():.6f}, std={output.std().item():.6f}")
    
    return output


def test_mlx_vq2emb(semantic_codec, codes, mlx_vq2emb=None):
    """
    使用 MLX 版本的 vq2emb
    
    Args:
        semantic_codec: Semantic Codec 模型（用于创建 MLX vq2emb）
        codes: torch.Tensor (B, T) - 语义编码序列
        mlx_vq2emb: 已初始化的 MLX vq2emb（可选）
        
    Returns:
        output: torch.Tensor (B, input_dim, T) - vq2emb 输出
    """
    print("\n" + "="*80)
    print("Testing MLX vq2emb")
    print("="*80)
    
    # 创建 MLX vq2emb 如果未提供
    if mlx_vq2emb is None:
        print(">> Creating MLX vq2emb...")
        mlx_vq2emb = MLXVQ2Emb(semantic_codec.quantizer)
        print(">> ✓ MLX vq2emb created")
    
    # MLX vq2emb 接收 (B, 1, T) 格式
    vq_input = codes.unsqueeze(1)  # (B, T) -> (B, 1, T)
    print(f">> Input shape: {vq_input.shape}")
    
    # 调用 MLX vq2emb
    output_mlx = mlx_vq2emb(vq_input)
    mx.eval(output_mlx)
    
    # 转换为 PyTorch tensor 以便对比
    output = mlx_to_torch(output_mlx)
    
    print(f">> Output shape: {output.shape}")
    print(f">> Output dtype: {output.dtype}")
    print(f">> Output range: min={output.min().item():.6f}, max={output.max().item():.6f}")
    print(f">> Output mean: {output.mean().item():.6f}, std={output.std().item():.6f}")
    
    return output


def compare_outputs(pytorch_output, mlx_output, tolerance=1e-5):
    """
    对比 PyTorch 和 MLX 的输出差异
    
    Args:
        pytorch_output: PyTorch 输出 (B, input_dim, T)
        mlx_output: MLX 输出 (B, input_dim, T)
        tolerance: 数值容差
        
    Returns:
        dict: 对比结果统计
    """
    print("\n" + "="*80)
    print("Comparison Results")
    print("="*80)
    
    # 检查形状
    if pytorch_output.shape != mlx_output.shape:
        print(f"❌ Shape mismatch!")
        print(f"   PyTorch: {pytorch_output.shape}")
        print(f"   MLX:     {mlx_output.shape}")
        return {'match': False, 'error': 'Shape mismatch'}
    
    print(f"✓ Shape match: {pytorch_output.shape}")
    
    # 转换为 float32 进行对比（避免 dtype 差异）
    pytorch_output_fp32 = pytorch_output.float()
    mlx_output_fp32 = mlx_output.float()
    
    # 计算差异
    diff = pytorch_output_fp32 - mlx_output_fp32
    abs_diff = torch.abs(diff)
    
    # 统计信息
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    std_diff = abs_diff.std().item()
    
    # 相对误差（使用 PyTorch 输出的绝对值作为分母）
    abs_pytorch = torch.abs(pytorch_output_fp32)
    relative_diff = abs_diff / (abs_pytorch + 1e-8)  # 避免除零
    max_relative_diff = relative_diff.max().item()
    mean_relative_diff = relative_diff.mean().item()
    
    print(f"\nAbsolute Differences:")
    print(f"  Max:     {max_diff:.9f}")
    print(f"  Mean:    {mean_diff:.9f}")
    print(f"  Std:     {std_diff:.9f}")
    
    print(f"\nRelative Differences:")
    print(f"  Max:     {max_relative_diff:.9f} ({max_relative_diff*100:.6f}%)")
    print(f"  Mean:    {mean_relative_diff:.9f} ({mean_relative_diff*100:.6f}%)")
    
    # 检查是否在容差范围内
    within_tolerance = max_diff < tolerance
    if within_tolerance:
        print(f"\n✓ All values within tolerance ({tolerance})")
    else:
        print(f"\n⚠ Some values exceed tolerance ({tolerance})")
    
    # 统计超出容差的点
    num_outliers = (abs_diff > tolerance).sum().item()
    total_elements = abs_diff.numel()
    outlier_percentage = (num_outliers / total_elements) * 100
    
    print(f"\nOutlier Statistics:")
    print(f"  Outliers:     {num_outliers:,} / {total_elements:,}")
    print(f"  Percentage:   {outlier_percentage:.4f}%")
    
    # 详细统计（按百分位数）
    percentiles = [50, 75, 90, 95, 99, 99.9]
    print(f"\nDifference Percentiles:")
    abs_diff_flat = abs_diff.flatten()
    for p in percentiles:
        val = torch.quantile(abs_diff_flat, p/100).item()
        print(f"  {p:5.1f}%: {val:.9f}")
    
    return {
        'match': within_tolerance,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'max_relative_diff': max_relative_diff,
        'mean_relative_diff': mean_relative_diff,
        'outlier_count': num_outliers,
        'outlier_percentage': outlier_percentage,
        'shape_match': True
    }


def main():
    """主测试函数"""
    print("="*80)
    print("vq2emb Torch vs MLX Comparison Test")
    print("="*80)
    print()
    
    # 配置
    device = 'cpu'  # 使用 CPU 进行公平对比
    cache_path = 'gpt_outputs_mlx.pkl'
    tolerance = 1e-5
    
    # 1. 加载 Semantic Codec
    print("Step 1: Loading Semantic Codec...")
    semantic_codec = load_semantic_codec(device=device)
    print()
    
    # 2. 加载或生成测试 codes
    print("Step 2: Loading test codes...")
    try:
        codes = generate_test_codes_from_cache(cache_path)
    except FileNotFoundError:
        print(f">> Cache not found, generating random codes...")
        codes = generate_random_codes(batch_size=1, seq_len=162)  # 使用实际缓存中的长度
    print()
    
    # 3. 测试 PyTorch 版本
    print("Step 3: Testing PyTorch vq2emb...")
    pytorch_output = test_pytorch_vq2emb(semantic_codec, codes)
    print()
    
    # 4. 测试 MLX 版本
    print("Step 4: Testing MLX vq2emb...")
    mlx_output = test_mlx_vq2emb(semantic_codec, codes)
    print()
    
    # 5. 对比结果
    print("Step 5: Comparing outputs...")
    comparison = compare_outputs(pytorch_output, mlx_output, tolerance=tolerance)
    print()
    
    # 6. 总结
    print("="*80)
    print("Test Summary")
    print("="*80)
    if comparison['match']:
        print("✅ PASS: MLX vq2emb output matches PyTorch version within tolerance")
    else:
        print("❌ FAIL: MLX vq2emb output does not match PyTorch version")
    print()
    print(f"Maximum difference: {comparison['max_diff']:.9f}")
    print(f"Mean difference:    {comparison['mean_diff']:.9f}")
    print(f"Outlier percentage: {comparison['outlier_percentage']:.4f}%")
    print()
    
    return comparison


if __name__ == '__main__':
    comparison = main()

