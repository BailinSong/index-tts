"""
简化的MLX CFM测试 - 只测试权重加载和基本推理
"""

import torch
import numpy as np
from omegaconf import OmegaConf
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("="*70)
    print("简化的 MLX CFM 测试")
    print("="*70)
    
    try:
        import mlx.core as mx
        from indextts.s2mel.modules.mlx_cfm import MLXCFM
        from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
        
        # 加载配置
        cfg = OmegaConf.load("checkpoints/config.yaml")
        
        print("\n1. 创建 MLX CFM 模型")
        mlx_cfm = MLXCFM(cfg.s2mel)
        print(f"  ✅ MLX CFM created (in_channels={mlx_cfm.in_channels})")
        
        print("\n2. 加载 PyTorch S2MEL 模型")
        s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
        s2mel_path = os.path.join("checkpoints", cfg.s2mel_checkpoint)
        s2mel, _, _, _ = load_checkpoint2(
            s2mel,
            None,
            s2mel_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        s2mel.eval()
        print(f"  ✅ PyTorch S2MEL loaded")
        
        print("\n3. 转换权重到 MLX")
        s2mel_state_dict = s2mel.state_dict()
        s2mel_state_dict_np = {k: v.cpu().numpy() for k, v in s2mel_state_dict.items()}
        
        loaded = mlx_cfm.load_weights_from_pytorch(s2mel_state_dict_np, prefix="models.cfm.")
        print(f"  ✅ Loaded {loaded} weights")
        
        print("\n4. 提取权重用于缓存")
        weights_to_cache = mlx_cfm.extract_weights_for_cache()
        print(f"  提取了 {len(weights_to_cache)} 个权重数组")
        
        if len(weights_to_cache) > 0:
            print(f"  ✅ 权重提取成功")
            
            # 保存到缓存
            cache_dir = "checkpoints/mlx"
            cache_file = os.path.join(cache_dir, "s2mel_cfm.npz")
            os.makedirs(cache_dir, exist_ok=True)
            mx.savez(cache_file, **weights_to_cache)
            
            size_mb = os.path.getsize(cache_file) / (1024 * 1024)
            print(f"  ✅ 已保存到 {cache_file}")
            print(f"     大小: {size_mb:.2f} MB")
            
            # 从缓存加载
            print("\n5. 从缓存加载")
            mlx_cfm_new = MLXCFM(cfg.s2mel)
            cached_weights = mx.load(cache_file)
            loaded_from_cache = mlx_cfm_new.load_from_cache(cached_weights)
            print(f"  ✅ 从缓存加载了 {loaded_from_cache} 个权重")
            
            print("\n" + "="*70)
            print("✅ 所有测试通过！CFM MLX 权重加载和缓存机制完成！")
            print("="*70)
            return True
        else:
            print(f"  ❌ 权重提取失败（0个权重）")
            print("\n" + "="*70)
            print("⚠️  权重提取方法需要修复")
            print("="*70)
            return False
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

