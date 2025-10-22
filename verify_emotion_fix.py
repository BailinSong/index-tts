#!/usr/bin/env python3
"""
验证Emotion Conditioning修复效果的脚本
"""

import os
import sys
import torch
import numpy as np

# 设置环境
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'

def main():
    print("🧪 Testing Emotion Conditioning Fix")
    print("=" * 50)
    
    try:
        # 导入必要的模块
        from omegaconf import OmegaConf
        from indextts.gpt.model_v2 import UnifiedVoice
        from indextts.gpt.mlx_model import UnifiedVoiceMLX
        from indextts.utils.checkpoint import load_checkpoint
        from indextts.utils.mlx_utils import check_mlx_available
        from indextts.utils.mlx_cache import MLXModelCache
        
        if not check_mlx_available():
            print("❌ MLX not available")
            return 1
        
        # 加载配置
        cfg = OmegaConf.load('checkpoints/config.yaml')
        gpt_path = os.path.join('checkpoints', cfg.gpt_checkpoint)
        
        print("📦 Loading models...")
        
        # 加载PyTorch版本
        gpt_pytorch = UnifiedVoice(**cfg.gpt)
        load_checkpoint(gpt_pytorch, gpt_path)
        gpt_pytorch = gpt_pytorch.to('mps')
        gpt_pytorch.eval()
        
        # 加载MLX版本
        mlx_cache = MLXModelCache(cache_dir='checkpoints/mlx')
        mlx_gpt_weights = mlx_cache.get_or_convert('gpt', gpt_path)
        gpt_mlx = UnifiedVoiceMLX(use_mlx_conditioning=True, **cfg.gpt)
        gpt_mlx.load_weights_from_dict(mlx_gpt_weights)
        
        print("✅ Models loaded successfully")
        
        # 生成测试输入
        torch.manual_seed(42)
        np.random.seed(42)
        
        batch_size = 1
        time_steps = 50
        feature_dim = 1024
        
        speech_input = torch.randn(batch_size, time_steps, feature_dim, device='mps')
        cond_lengths = torch.tensor([time_steps], device='mps')
        
        print(f"🎯 Testing with input shape: {speech_input.shape}")
        
        # 测试PyTorch版本
        print("\n🔄 Testing PyTorch get_emo_conditioning...")
        pytorch_input = speech_input.transpose(1, 2)  # (b, t, 1024) -> (b, 1024, t)
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        with torch.no_grad():
            pytorch_result = gpt_pytorch.get_emo_conditioning(pytorch_input, cond_lengths)
        
        print(f"  ✅ PyTorch result shape: {pytorch_result.shape}")
        
        # 测试MLX版本
        print("\n🔄 Testing MLX get_emo_conditioning...")
        mlx_input = speech_input  # (b, t, 1024)
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        mlx_result = gpt_mlx.get_emo_conditioning(mlx_input, cond_lengths)
        
        print(f"  ✅ MLX result shape: {mlx_result.shape}")
        
        # 计算差异
        print("\n📊 Comparing results...")
        
        diff = torch.abs(pytorch_result - mlx_result)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        # 计算相关系数
        pytorch_flat = pytorch_result.cpu().numpy().flatten()
        mlx_flat = mlx_result.cpu().numpy().flatten()
        correlation = np.corrcoef(pytorch_flat, mlx_flat)[0, 1]
        
        print(f"  Max difference: {max_diff:.8f}")
        print(f"  Mean difference: {mean_diff:.8f}")
        print(f"  Correlation: {correlation:.6f}")
        
        # 评估结果
        print("\n🎯 Result Assessment:")
        if max_diff < 1e-4:
            print("  ✅ EXCELLENT: Differences are minimal (< 1e-4)")
            status = "EXCELLENT"
        elif max_diff < 0.01:
            print("  ✅ GOOD: Differences are small (< 0.01)")
            status = "GOOD"
        elif max_diff < 0.1:
            print("  ⚠️  MODERATE: Differences are moderate (< 0.1)")
            status = "MODERATE"
        else:
            print("  ❌ LARGE: Differences are large (>= 0.1)")
            status = "LARGE"
        
        if correlation > 0.99:
            print("  ✅ EXCELLENT: Very high correlation (> 0.99)")
        elif correlation > 0.95:
            print("  ✅ GOOD: High correlation (> 0.95)")
        elif correlation > 0.9:
            print("  ⚠️  MODERATE: Moderate correlation (> 0.9)")
        else:
            print("  ❌ LOW: Low correlation (<= 0.9)")
        
        print(f"\n📋 Summary: {status}")
        
        if status in ["EXCELLENT", "GOOD"]:
            print("🎉 Emotion conditioning fix appears to be successful!")
            return 0
        else:
            print("⚠️  Emotion conditioning still needs further improvement")
            return 1
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
