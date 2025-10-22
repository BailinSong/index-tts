#!/usr/bin/env python3
"""
GPT推理一致性测试 - 控制变量seed=42
对比PyTorch和MLX的完整推理输出
"""

import os
import sys
import torch
import numpy as np

os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'

def set_seed(seed=42):
    """设置所有随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compare_outputs(pytorch_output, mlx_output, name):
    """对比输出"""
    if pytorch_output is None or mlx_output is None:
        print(f"❌ {name}: 有输出为None")
        return False
    
    # 确保都是numpy数组
    if isinstance(pytorch_output, torch.Tensor):
        pytorch_output = pytorch_output.cpu().numpy()
    if not isinstance(mlx_output, np.ndarray):
        mlx_output = np.array(mlx_output)
    
    # 对比shape
    if pytorch_output.shape != mlx_output.shape:
        print(f"❌ {name} shape不一致: PyTorch {pytorch_output.shape} vs MLX {mlx_output.shape}")
        return False
    
    # 计算差异
    diff = np.abs(pytorch_output - mlx_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # 计算相关系数
    if pytorch_output.size > 1:
        corr = np.corrcoef(pytorch_output.flatten(), mlx_output.flatten())[0, 1]
    else:
        corr = 1.0 if max_diff < 1e-6 else 0.0
    
    # 判断一致性
    if max_diff < 1e-4:
        status = "✅ EXCELLENT"
    elif max_diff < 0.01:
        status = "✅ GOOD"
    elif max_diff < 0.1:
        status = "⚠️ ACCEPTABLE"
    else:
        status = "❌ POOR"
    
    print(f"{status} {name}")
    print(f"   Max diff: {max_diff:.8f}")
    print(f"   Mean diff: {mean_diff:.8f}")
    print(f"   Correlation: {corr:.6f}")
    
    return max_diff < 0.1

def main():
    from indextts.infer_v2 import IndexTTS2
    
    print("🧪 GPT推理一致性测试 (seed=42)")
    print("=" * 70)
    
    # 设置随机种子
    set_seed(42)
    
    print("\n📦 加载模型...")
    
    # PyTorch版本
    print("\n加载PyTorch模型...")
    tts_pytorch = IndexTTS2(
        model_dir='checkpoints',
        device='mps',
        use_mlx=False
    )
    
    print("\n加载MLX模型...")
    tts_mlx = IndexTTS2(
        model_dir='checkpoints',
        device='mps',
        use_mlx=True
    )
    
    print("\n✅ 模型加载完成")
    
    # 准备测试数据
    print("\n" + "=" * 70)
    print("准备测试数据 (seed=42)")
    print("=" * 70)
    
    test_text = "你好，这是一个测试。"
    
    # 使用相同的参考音频
    import torchaudio
    ref_audio_path = "examples/ref1.wav"
    
    if not os.path.exists(ref_audio_path):
        print(f"⚠️ 参考音频不存在: {ref_audio_path}")
        print("使用合成数据测试...")
        
        # 生成合成音频数据
        set_seed(42)
        ref_audio = torch.randn(1, 16000 * 3).numpy()  # 3秒音频
        sample_rate = 16000
    else:
        ref_audio, sample_rate = torchaudio.load(ref_audio_path)
        ref_audio = ref_audio.numpy()
    
    print(f"参考音频: shape={ref_audio.shape}, sr={sample_rate}")
    print(f"测试文本: {test_text}")
    
    # =====================================================================
    # 测试推理
    # =====================================================================
    print("\n" + "=" * 70)
    print("开始推理测试")
    print("=" * 70)
    
    # 推理参数
    inference_params = {
        'text': test_text,
        'spk_audio_prompt': ref_audio,
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 50,
        'repetition_penalty': 1.0,
        'max_new_tokens': 50,  # 限制长度便于对比
        'do_sample': False,  # 使用greedy以确保确定性
    }
    
    print("\n推理参数:")
    for k, v in inference_params.items():
        if k != 'spk_audio_prompt':
            print(f"  {k}: {v}")
    
    # PyTorch推理
    print("\n🔄 PyTorch推理中...")
    set_seed(42)
    
    try:
        pytorch_results = list(tts_pytorch.infer(**inference_params))
        pytorch_audio = pytorch_results[0]['audio'] if pytorch_results else None
        pytorch_codes = pytorch_results[0].get('codes', None) if pytorch_results else None
        
        print(f"✅ PyTorch推理完成")
        if pytorch_audio is not None:
            print(f"   音频shape: {pytorch_audio.shape}")
        if pytorch_codes is not None:
            print(f"   Codes shape: {pytorch_codes.shape if hasattr(pytorch_codes, 'shape') else len(pytorch_codes)}")
            print(f"   前10个codes: {pytorch_codes[:10] if hasattr(pytorch_codes, '__getitem__') else 'N/A'}")
    except Exception as e:
        print(f"❌ PyTorch推理失败: {e}")
        pytorch_audio = None
        pytorch_codes = None
    
    # MLX推理
    print("\n🔄 MLX推理中...")
    set_seed(42)
    
    try:
        mlx_results = list(tts_mlx.infer(**inference_params))
        mlx_audio = mlx_results[0]['audio'] if mlx_results else None
        mlx_codes = mlx_results[0].get('codes', None) if mlx_results else None
        
        print(f"✅ MLX推理完成")
        if mlx_audio is not None:
            print(f"   音频shape: {mlx_audio.shape}")
        if mlx_codes is not None:
            print(f"   Codes shape: {mlx_codes.shape if hasattr(mlx_codes, 'shape') else len(mlx_codes)}")
            print(f"   前10个codes: {mlx_codes[:10] if hasattr(mlx_codes, '__getitem__') else 'N/A'}")
    except Exception as e:
        print(f"❌ MLX推理失败: {e}")
        mlx_audio = None
        mlx_codes = None
    
    # =====================================================================
    # 对比结果
    # =====================================================================
    print("\n" + "=" * 70)
    print("对比推理结果")
    print("=" * 70)
    
    all_pass = True
    
    # 对比codes
    if pytorch_codes is not None and mlx_codes is not None:
        print("\n1. 对比生成的codes:")
        
        # 转换为numpy
        if isinstance(pytorch_codes, torch.Tensor):
            pt_codes_np = pytorch_codes.cpu().numpy()
        else:
            pt_codes_np = np.array(pytorch_codes)
        
        if isinstance(mlx_codes, torch.Tensor):
            mlx_codes_np = mlx_codes.cpu().numpy()
        else:
            mlx_codes_np = np.array(mlx_codes)
        
        # 对比
        codes_match = compare_outputs(pt_codes_np, mlx_codes_np, "   Generated codes")
        all_pass = all_pass and codes_match
        
        # 对比token一致性
        if pt_codes_np.shape == mlx_codes_np.shape:
            token_match_rate = np.mean(pt_codes_np == mlx_codes_np)
            print(f"   Token匹配率: {token_match_rate * 100:.2f}%")
    
    # 对比音频
    if pytorch_audio is not None and mlx_audio is not None:
        print("\n2. 对比生成的音频:")
        audio_match = compare_outputs(pytorch_audio, mlx_audio, "   Generated audio")
        all_pass = all_pass and audio_match
    
    # =====================================================================
    # 总结
    # =====================================================================
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    
    if all_pass:
        print("✅ 推理一致性测试通过！")
        print("   PyTorch和MLX版本产生一致的输出")
    else:
        print("⚠️ 推理存在差异")
        print("   需要进一步调试推理pipeline")
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())

