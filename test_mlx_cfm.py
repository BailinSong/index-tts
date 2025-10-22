"""
测试 MLX CFM 实现的完整性和一致性
对比 PyTorch CFM 和 MLX CFM 的输出
"""

import torch
import numpy as np
from omegaconf import OmegaConf
import os
import sys

# 确保可以导入项目模块
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

def compare_outputs(torch_output, mlx_output, name="Output"):
    """比较 PyTorch 和 MLX 的输出"""
    # 转换为numpy进行比较
    torch_np = torch_output.detach().cpu().numpy()
    mlx_np = np.array(mlx_output)
    
    # 确保形状一致
    if torch_np.shape != mlx_np.shape:
        print(f"❌ {name} shape mismatch: PyTorch {torch_np.shape} vs MLX {mlx_np.shape}")
        return False
    
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
    print(f"\n{name}:")
    print(f"  Shape: {torch_np.shape}")
    print(f"  Max diff: {max_diff:.8f}")
    print(f"  Mean diff: {mean_diff:.8f}")
    print(f"  Correlation: {correlation:.6f}")
    
    # 判断一致性
    if max_diff < 1e-3 and correlation > 0.99:
        status = "✅ EXCELLENT"
    elif max_diff < 1e-2 and correlation > 0.95:
        status = "✅ GOOD"
    elif max_diff < 0.1 and correlation > 0.9:
        status = "⚠️  ACCEPTABLE"
    else:
        status = "❌ POOR"
    
    print(f"  Status: {status}")
    return status.startswith("✅")

def test_cfm_components():
    """测试CFM各个组件"""
    print("="*70)
    print("测试 1: CFM 组件单独测试")
    print("="*70)
    
    try:
        import mlx.core as mx
        from indextts.s2mel.modules.mlx_cfm import (
            MLXTimestepEmbedder,
            MLXStyleEmbedder,
            MLXFinalLayer,
            MLXDiT
        )
        
        # 创建配置
        config = OmegaConf.create({
            'DiT': {
                'in_channels': 80,
                'hidden_dim': 512,
                'depth': 13,
                'num_heads': 8,
                'max_seq_len': 8192,
                'final_layer_type': 'wavenet',
                'use_rope': True,
                'rope_theta': 10000.0,
                'zero_prompt_speech_token': False,
                'time_as_token': False,
                'style_as_token': False,
                'long_skip_connection': True,
                'style_condition': True,
                'is_causal': False,
                'content_type': 'discrete',
                'content_codebook_size': 1024,
                'content_dim': 1024,
                'class_dropout_prob': 0.1,
            },
            'style_encoder': {
                'dim': 192
            },
            'wavenet': {
                'hidden_dim': 512,
                'kernel_size': 5,
                'dilation_rate': 1,
                'num_layers': 8,
                'p_dropout': 0.0,
                'style_condition': True
            }
        })
        
        print("\n1.1 测试 TimestepEmbedder")
        t_embedder = MLXTimestepEmbedder(hidden_size=1024)
        t = mx.array([0.5])
        t_emb = t_embedder(t)
        print(f"  Input shape: {t.shape}")
        print(f"  Output shape: {t_emb.shape}")
        print(f"  ✅ TimestepEmbedder works")
        
        print("\n1.2 测试 StyleEmbedder")
        style_embedder = MLXStyleEmbedder(input_size=192, hidden_size=1024)
        style = mx.random.normal((1, 192))
        style_emb = style_embedder(style)
        print(f"  Input shape: {style.shape}")
        print(f"  Output shape: {style_emb.shape}")
        print(f"  ✅ StyleEmbedder works")
        
        print("\n1.3 测试 FinalLayer")
        final_layer = MLXFinalLayer(
            hidden_size=1024,
            patch_size=1,
            out_channels=1024
        )
        x = mx.random.normal((1, 100, 1024))
        c = mx.random.normal((1, 1024))
        output = final_layer(x, c)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  ✅ FinalLayer works")
        
        print("\n1.4 测试 DiT (完整模型)")
        dit = MLXDiT(config)
        x = mx.random.normal((1, 1024, 100))
        cond = mx.random.normal((1, 1024, 50))
        mask = mx.ones((1, 150), dtype=mx.bool_)
        timestep = mx.array([0.5])
        style = mx.random.normal((1, 192))
        f0 = None
        
        output = dit(x, cond, mask, timestep, style, f0)
        print(f"  Input x shape: {x.shape}")
        print(f"  Condition shape: {cond.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  ✅ DiT (complete model) works")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cfm_inference():
    """测试CFM完整推理流程"""
    print("\n" + "="*70)
    print("测试 2: CFM 完整推理流程")
    print("="*70)
    
    try:
        import mlx.core as mx
        from indextts.s2mel.modules.mlx_cfm import MLXCFM
        
        # 加载配置
        cfg = OmegaConf.load("checkpoints/config.yaml")
        
        print("\n2.1 创建 MLX CFM 模型")
        mlx_cfm = MLXCFM(cfg.s2mel)
        print(f"  ✅ MLX CFM created")
        print(f"  in_channels: {mlx_cfm.in_channels}")
        
        print("\n2.2 加载 PyTorch S2MEL 模型")
        from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
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
        
        print("\n2.3 转换权重到 MLX")
        s2mel_state_dict = s2mel.state_dict()
        s2mel_state_dict_np = {k: v.cpu().numpy() for k, v in s2mel_state_dict.items()}
        
        loaded = mlx_cfm.load_weights_from_pytorch(s2mel_state_dict_np, prefix="models.cfm.")
        print(f"  ✅ Loaded {loaded} weights")
        
        print("\n2.4 准备测试数据")
        batch_size = 1
        seq_len = 100
        prompt_len = 20
        in_channels = mlx_cfm.in_channels  # 80
        
        # 创建测试输入（应该是 mel spectrogram 格式）
        mu = torch.randn(batch_size, in_channels, seq_len)
        x_lens = torch.tensor([seq_len])
        prompt = torch.randn(batch_size, in_channels, prompt_len)
        style = torch.randn(batch_size, 192)
        f0 = None
        
        print(f"  mu shape: {mu.shape}")
        print(f"  prompt shape: {prompt.shape}")
        print(f"  style shape: {style.shape}")
        
        print("\n2.5 运行 PyTorch CFM 推理")
        with torch.no_grad():
            torch_output = s2mel.models['cfm'].inference(
                mu, x_lens, prompt, style, f0,
                n_timesteps=5,  # 少量步数用于快速测试
                inference_cfg_rate=0.5
            )
        print(f"  PyTorch output shape: {torch_output.shape}")
        
        print("\n2.6 运行 MLX CFM 推理")
        mlx_output = mlx_cfm.inference(
            mu, x_lens, prompt, style, f0,
            n_timesteps=5,
            inference_cfg_rate=0.5
        )
        print(f"  MLX output shape: {mlx_output.shape}")
        
        print("\n2.7 对比输出")
        is_good = compare_outputs(torch_output, mlx_output, "CFM Inference Output")
        
        return is_good
        
    except Exception as e:
        print(f"\n❌ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cfm_caching():
    """测试CFM缓存机制"""
    print("\n" + "="*70)
    print("测试 3: CFM 缓存机制")
    print("="*70)
    
    try:
        import mlx.core as mx
        from indextts.s2mel.modules.mlx_cfm import MLXCFM
        from indextts.utils.mlx_cache import MLXModelCache
        
        # 加载配置
        cfg = OmegaConf.load("checkpoints/config.yaml")
        cache_dir = "checkpoints/mlx"
        cache_file = os.path.join(cache_dir, "s2mel_cfm.npz")
        
        print("\n3.1 创建 MLX CFM 模型")
        mlx_cfm = MLXCFM(cfg.s2mel)
        
        print("\n3.2 加载 PyTorch 权重")
        from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
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
        s2mel_state_dict = s2mel.state_dict()
        s2mel_state_dict_np = {k: v.cpu().numpy() for k, v in s2mel_state_dict.items()}
        
        print("\n3.3 加载权重到 MLX CFM")
        loaded = mlx_cfm.load_weights_from_pytorch(s2mel_state_dict_np, prefix="models.cfm.")
        print(f"  ✅ Loaded {loaded} weights")
        
        print("\n3.4 提取权重用于缓存")
        weights_to_cache = mlx_cfm.extract_weights_for_cache()
        print(f"  ✅ Extracted {len(weights_to_cache)} weight arrays")
        
        print("\n3.5 保存到缓存")
        os.makedirs(cache_dir, exist_ok=True)
        mx.savez(cache_file, **weights_to_cache)
        size_mb = os.path.getsize(cache_file) / (1024 * 1024)
        print(f"  ✅ Saved to {cache_file}")
        print(f"  Cache size: {size_mb:.2f} MB")
        
        print("\n3.6 从缓存加载")
        mlx_cfm_new = MLXCFM(cfg.s2mel)
        cached_weights = mx.load(cache_file)
        loaded_from_cache = mlx_cfm_new.load_from_cache(cached_weights)
        print(f"  ✅ Loaded {loaded_from_cache} weights from cache")
        
        print("\n3.7 验证缓存加载的正确性")
        # 准备测试数据
        batch_size = 1
        seq_len = 50
        prompt_len = 10
        in_channels = mlx_cfm.in_channels  # 80
        
        mu = torch.randn(batch_size, in_channels, seq_len)
        x_lens = torch.tensor([seq_len])
        prompt = torch.randn(batch_size, in_channels, prompt_len)
        style = torch.randn(batch_size, 192)
        f0 = None
        
        # 原始模型推理
        output1 = mlx_cfm.inference(mu, x_lens, prompt, style, f0, n_timesteps=3, inference_cfg_rate=0.5)
        
        # 缓存加载的模型推理
        output2 = mlx_cfm_new.inference(mu, x_lens, prompt, style, f0, n_timesteps=3, inference_cfg_rate=0.5)
        
        # 对比
        is_same = compare_outputs(output1, output2, "Cached vs Original")
        
        if is_same:
            print("\n✅ 缓存机制验证成功！")
        else:
            print("\n⚠️  缓存加载的模型输出与原始模型有差异")
        
        return is_same
        
    except Exception as e:
        print(f"\n❌ Caching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("MLX CFM 完整测试套件")
    print("="*70)
    
    results = {}
    
    # 测试1: 组件测试
    results['components'] = test_cfm_components()
    
    # 测试2: 推理测试
    results['inference'] = test_cfm_inference()
    
    # 测试3: 缓存测试
    results['caching'] = test_cfm_caching()
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name.capitalize():15s}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*70)
    if all_passed:
        print("🎉 所有测试通过！CFM MLX 实现完成！")
    else:
        print("⚠️  部分测试失败，请检查实现")
    print("="*70 + "\n")
    
    return all_passed

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

