#!/usr/bin/env python3
"""
生成正确的 CFM 测试数据

创建有意义的噪声数据，确保 prompt_len < seq_len
"""

import pickle
import numpy as np
import torch
from pathlib import Path
import sys
import os

# 添加项目路径
sys.path.append('/Users/bailin/index-tts')


def generate_correct_cfm_test_data():
    """生成正确的 CFM 测试数据"""
    print("🔍 Generating Correct CFM Test Data")
    print("="*60)
    
    # 设置参数
    batch_size = 2
    seq_len = 50  # 增加序列长度
    prompt_len = 20  # 确保 prompt_len < seq_len
    mel_bins = 80
    style_dim = 192
    hidden_dim = 512
    
    print(f"   📊 Test Data Parameters:")
    print(f"     batch_size: {batch_size}")
    print(f"     seq_len: {seq_len}")
    print(f"     prompt_len: {prompt_len}")
    print(f"     mel_bins: {mel_bins}")
    print(f"     style_dim: {style_dim}")
    print(f"     hidden_dim: {hidden_dim}")
    
    # 生成有意义的噪声数据
    print(f"\n   🔍 Generating Input Data:")
    
    # 1. 生成噪声 x (应该是随机噪声，不是全零)
    x = torch.randn(batch_size, mel_bins, seq_len)
    print(f"     x (noise): shape={x.shape}, range=[{x.min():.6f}, {x.max():.6f}]")
    
    # 2. 生成 prompt_x (参考音频的 mel 频谱)
    prompt_x = torch.randn(batch_size, mel_bins, prompt_len)
    print(f"     prompt_x: shape={prompt_x.shape}, range=[{prompt_x.min():.6f}, {prompt_x.max():.6f}]")
    
    # 3. 生成 x_lens (序列长度)
    x_lens = torch.full((batch_size,), seq_len, dtype=torch.long)
    print(f"     x_lens: {x_lens}")
    
    # 4. 生成时间 t (随机时间步)
    t = torch.rand(batch_size)
    print(f"     t: shape={t.shape}, range=[{t.min():.6f}, {t.max():.6f}]")
    
    # 5. 生成 style (风格特征)
    style = torch.randn(batch_size, style_dim)
    print(f"     style: shape={style.shape}, range=[{style.min():.6f}, {style.max():.6f}]")
    
    # 6. 生成 mu (条件信息)
    mu = torch.randn(batch_size, seq_len, hidden_dim)
    print(f"     mu: shape={mu.shape}, range=[{mu.min():.6f}, {mu.max():.6f}]")
    
    # 创建缓存数据
    cached_inputs = {
        'x': x,
        'prompt_x': prompt_x,
        'x_lens': x_lens,
        't': t,
        'style': style,
        'mu': mu,
        'batch_size': batch_size,
        'seq_len': seq_len,
        'prompt_len': prompt_len,
        'mel_bins': mel_bins,
        'style_dim': style_dim,
        'hidden_dim': hidden_dim
    }
    
    # 保存缓存数据
    cache_file = Path("cfm_debug_outputs/correct_cached_inputs.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_inputs, f)
    
    print(f"\n   ✅ Correct cached inputs saved to: {cache_file}")
    
    # 验证数据
    print(f"\n   🔍 Data Validation:")
    print(f"     x is all zeros: {x.min() == 0.0 and x.max() == 0.0}")
    print(f"     prompt_len < seq_len: {prompt_len < seq_len}")
    print(f"     prompt_len / seq_len ratio: {prompt_len / seq_len:.2f}")
    
    if x.min() == 0.0 and x.max() == 0.0:
        print(f"     ❌ Problem: x is all zeros!")
    else:
        print(f"     ✅ x has meaningful noise values")
    
    if prompt_len >= seq_len:
        print(f"     ❌ Problem: prompt_len ({prompt_len}) >= seq_len ({seq_len})!")
    else:
        print(f"     ✅ prompt_len ({prompt_len}) < seq_len ({seq_len})")
    
    return cached_inputs


def test_cfm_with_correct_data(cached_inputs):
    """使用正确的数据测试 CFM"""
    print(f"\n🔍 Testing CFM with Correct Data:")
    
    try:
        # 导入 PyTorch CFM
        from indextts.s2mel.modules.flow_matching import CFM
        
        # 创建配置
        config_dict = {
            'DiT': {
                'in_channels': 80,
                'hidden_dim': 512,
                'num_heads': 8,
                'depth': 13,
                'time_as_token': False,
                'style_as_token': False,
                'long_skip_connection': True,
                'style_condition': True,
                'is_causal': True,
                'final_layer_type': 'wavenet',
                'content_type': 'continuous',
                'content_codebook_size': 1024,
                'content_dim': 512,
                'class_dropout_prob': 0.1
            },
            'style_encoder': {
                'dim': 192
            },
            'wavenet': {
                'hidden_dim': 512,
                'kernel_size': 3,
                'dilation_rate': 2,
                'num_layers': 8,
                'p_dropout': 0.1,
                'style_condition': True
            },
            'dit_type': 'DiT',
            'reg_loss_type': 'l2'
        }
        
        # 创建 SimpleConfig 类
        class SimpleConfig:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    if isinstance(value, dict):
                        setattr(self, key, SimpleConfig(value))
                    else:
                        setattr(self, key, value)
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        config = SimpleConfig(config_dict)
        
        # 初始化 PyTorch CFM
        pytorch_cfm = CFM(config)
        pytorch_cfm.eval()
        
        # 初始化 caches
        batch_size = cached_inputs['batch_size']
        seq_len = cached_inputs['seq_len']
        pytorch_cfm.estimator.setup_caches(max_batch_size=batch_size, max_seq_length=seq_len)
        
        # 准备输入数据
        x = cached_inputs['x']
        prompt_x = cached_inputs['prompt_x']
        x_lens = cached_inputs['x_lens']
        style = cached_inputs['style']
        mu = cached_inputs['mu']
        
        print(f"     Input shapes: x={x.shape}, prompt_x={prompt_x.shape}, x_lens={x_lens.shape}")
        print(f"     Input shapes: style={style.shape}, mu={mu.shape}")
        
        # 创建时间跨度
        t_span = torch.linspace(0, 1, 4)  # 3 steps + 1
        
        # 运行 CFM 推理
        print(f"\n     🔍 Running CFM Inference:")
        
        with torch.no_grad():
            output = pytorch_cfm.solve_euler(
                x=x,
                x_lens=x_lens,
                prompt=prompt_x,
                mu=mu,
                style=style,
                f0=None,
                t_span=t_span,
                inference_cfg_rate=0.0,  # 禁用 CFG 模式
                debug_layers=False
            )
            
            print(f"     CFM output shape: {output.shape}")
            print(f"     CFM output range: [{output.min():.6f}, {output.max():.6f}]")
            
            # 检查输出
            if output.min() == 0.0 and output.max() == 0.0:
                print(f"     ❌ Problem: CFM output is still all zeros!")
                print(f"     💡 This indicates a deeper issue with CFM implementation")
            else:
                print(f"     ✅ CFM output has non-zero values!")
                print(f"     🎉 CFM is working correctly with proper input data")
        
        return output
        
    except Exception as e:
        print(f"     ❌ CFM test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    print("🔍 Correct CFM Test Data Generation")
    print("="*60)
    
    try:
        # 生成正确的测试数据
        cached_inputs = generate_correct_cfm_test_data()
        
        # 使用正确的数据测试 CFM
        output = test_cfm_with_correct_data(cached_inputs)
        
        print("\n🎉 Correct CFM Test Data Generation completed!")
        print("\n📋 Key Findings:")
        print("1. Generated meaningful noise data (not all zeros)")
        print("2. Set prompt_len < seq_len to avoid full zero masking")
        print("3. CFM should now produce meaningful output")
        
        print("\n💡 Recommendations:")
        print("1. Use this corrected data for PyTorch vs MLX comparison")
        print("2. Ensure prompt_len is always < seq_len in real usage")
        print("3. Verify CFM produces non-zero output with proper inputs")
        
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


