"""
最终CFM对比测试
"""

import torch
import numpy as np
import os
import sys
import mlx.core as mx
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(__file__))

def torch_to_mlx(tensor):
    if isinstance(tensor, torch.Tensor):
        return mx.array(tensor.detach().cpu().numpy())
    return tensor

def final_cfm_comparison():
    """最终CFM对比测试"""
    
    print("="*80)
    print("最终CFM对比测试")
    print("="*80)
    
    # 加载缓存的S2MEL输入
    if not os.path.exists('s2mel_inputs_cache.pkl'):
        print("❌ 未找到s2mel_inputs_cache.pkl")
        return
    
    print("加载缓存的S2MEL输入...")
    import pickle
    with open('s2mel_inputs_cache.pkl', 'rb') as f:
        s2mel_inputs = pickle.load(f)
    
    # 加载PyTorch模型
    print("加载PyTorch S2MEL模型...")
    from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
    
    cfg = OmegaConf.load("checkpoints/config.yaml")
    s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
    s2mel_path = os.path.join("checkpoints", cfg.s2mel_checkpoint)
    s2mel, _, _, _ = load_checkpoint2(s2mel, None, s2mel_path, load_only_params=True, ignore_modules=[], is_distributed=False)
    s2mel.eval()
    s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
    
    # 加载MLX模型
    print("加载MLX S2MEL模型...")
    from indextts.s2mel.modules.mlx_cfm import MLXCFM
    
    mlx_cfm = MLXCFM(cfg.s2mel)
    s2mel_state = {k: v.cpu().numpy() for k, v in s2mel.state_dict().items()}
    mlx_cfm.load_weights_from_pytorch(s2mel_state, prefix="models.cfm.")
    
    # 提取输入数据
    device = torch.device('cpu')
    
    cat_condition = s2mel_inputs['cat_condition'].to(device)
    x_lens = s2mel_inputs['x_lens'].to(device)
    ref_mel = s2mel_inputs['ref_mel'].to(device)
    style = s2mel_inputs['style'].to(device)
    diffusion_steps = s2mel_inputs['diffusion_steps']
    inference_cfg_rate = s2mel_inputs['inference_cfg_rate']
    
    # 将模型移动到CPU
    s2mel = s2mel.to(device)
    s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
    
    # 转换到MLX
    cat_condition_mlx = torch_to_mlx(cat_condition)
    x_lens_mlx = torch_to_mlx(x_lens)
    ref_mel_mlx = torch_to_mlx(ref_mel)
    style_mlx = torch_to_mlx(style)
    
    # 设置相同的随机种子
    torch.manual_seed(42)
    mx.random.seed(42)
    
    print(f"\n输入数据:")
    print(f"  cat_condition: {cat_condition.shape}")
    print(f"  x_lens: {x_lens.item()}")
    print(f"  ref_mel: {ref_mel.shape}")
    print(f"  style: {style.shape}")
    print(f"  diffusion_steps: {diffusion_steps}")
    print(f"  inference_cfg_rate: {inference_cfg_rate}")
    
    # 调用PyTorch CFM
    print(f"\n调用PyTorch CFM...")
    try:
        with torch.no_grad():
            result_pytorch = s2mel.models['cfm'].inference(
                cat_condition, x_lens, ref_mel, style, None, 
                diffusion_steps, inference_cfg_rate=inference_cfg_rate
            )
            print(f"PyTorch CFM结果: {result_pytorch.shape}")
            print(f"  min={result_pytorch.min():.6f}")
            print(f"  max={result_pytorch.max():.6f}")
            print(f"  mean={result_pytorch.mean():.6f}")
            print(f"  std={result_pytorch.std():.6f}")
    except Exception as e:
        print(f"❌ PyTorch CFM失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 调用MLX CFM
    print(f"\n调用MLX CFM...")
    try:
        result_mlx = mlx_cfm.inference(
            cat_condition_mlx, x_lens_mlx, ref_mel_mlx, style_mlx, None, 
            diffusion_steps, inference_cfg_rate=inference_cfg_rate
        )
        mx.eval(result_mlx)
        
        print(f"MLX CFM结果: {result_mlx.shape}")
        print(f"  min={float(result_mlx.min()):.6f}")
        print(f"  max={float(result_mlx.max()):.6f}")
        print(f"  mean={float(result_mlx.mean()):.6f}")
        print(f"  std={float(result_mlx.std()):.6f}")
    except Exception as e:
        print(f"❌ MLX CFM失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 对比结果
    print(f"\nCFM输出对比:")
    torch_np = result_pytorch.detach().cpu().numpy()
    mlx_np = np.array(result_mlx)
    
    diff = np.abs(torch_np - mlx_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    corr = np.corrcoef(torch_np.flatten(), mlx_np.flatten())[0, 1]
    
    print(f"  max_diff={max_diff:.6f}")
    print(f"  mean_diff={mean_diff:.6f}")
    print(f"  corr={corr:.6f}")
    
    if max_diff < 1e-6:
        print(f"  ✅ EXCELLENT一致性")
    elif max_diff < 1e-5:
        print(f"  ✅ GOOD一致性")
    elif max_diff < 1e-4:
        print(f"  ⚠️  FAIR一致性")
    else:
        print(f"  ❌ POOR一致性")
    
    # 检查数值范围
    pytorch_min, pytorch_max = result_pytorch.min().item(), result_pytorch.max().item()
    mlx_min, mlx_max = float(result_mlx.min()), float(result_mlx.max())
    
    print(f"\n数值范围对比:")
    print(f"  PyTorch: min={pytorch_min:.6f}, max={pytorch_max:.6f}")
    print(f"  MLX: min={mlx_min:.6f}, max={mlx_max:.6f}")
    
    # 判断是否在正常范围内
    normal_min, normal_max = -15.0, 5.0
    pytorch_normal = normal_min <= pytorch_min <= normal_max and normal_min <= pytorch_max <= normal_max
    mlx_normal = normal_min <= mlx_min <= normal_max and normal_min <= mlx_max <= normal_max
    
    print(f"\n数值范围判断:")
    if pytorch_normal:
        print(f"  ✅ PyTorch CFM输出范围正常")
    else:
        print(f"  ❌ PyTorch CFM输出范围异常")
        
    if mlx_normal:
        print(f"  ✅ MLX CFM输出范围正常")
    else:
        print(f"  ❌ MLX CFM输出范围异常")
    
    print(f"\n{'='*80}")
    print("最终CFM对比测试完成")
    print(f"{'='*80}")

if __name__ == "__main__":
    final_cfm_comparison()
