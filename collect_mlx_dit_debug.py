#!/usr/bin/env python3
"""
收集 MLX DiT 调试数据
"""

import sys
import os
sys.path.append('/Users/bailin/index-tts')

import torch
import pickle
import mlx.core as mx
from indextts.utils.cfm_debugger import get_debugger, save_cfm_debug

def main():
    print("🔍 收集 MLX DiT 调试数据")
    print("="*50)
    
    # 获取全局调试器
    debugger = get_debugger()
    
    # 清空之前的调试数据
    debugger.debug_data = {'pytorch': {}, 'mlx': {}}
    
    print("🔧 加载测试数据...")
    
    # 加载测试数据
    test_data_file = "/Users/bailin/index-tts/cfm_production_cache/cfm_pytorch_step_001_dit_input_1761699517184.pkl"
    with open(test_data_file, 'rb') as f:
        test_data = pickle.load(f)
    
    print(f"✅ 测试数据加载完成: {test_data_file}")
    print(f"   输入形状: x={test_data['x'].shape}, prompt_x={test_data['prompt_x'].shape}")
    
    # 转换输入数据到 MLX
    x_mlx = mx.array(test_data['x'].numpy())
    prompt_x_mlx = mx.array(test_data['prompt_x'].numpy())
    x_lens_mlx = mx.array(test_data['x_lens'].numpy())
    t_mlx = mx.array(test_data['t'].numpy())
    style_mlx = mx.array(test_data['style'].numpy())
    mu_mlx = mx.array(test_data['mu'].numpy())
    
    print("🔧 加载 MLX DiT 模型...")
    
    # 使用现有的 MLX CFM 加载逻辑
    from indextts.s2mel.modules.mlx_diffusion_transformer import MLXCFMRewritten
    from indextts.s2mel.modules.mlx_gpt_fast_model import create_mlx_transformer_from_config
    from indextts.s2mel.modules.mlx_wavenet_model import MLXWaveNet
    
    # 创建配置
    class MLXConfig:
        class DiT:
            hidden_dim = 512
            in_channels = 80
            out_channels = 80
            num_layers = 13
            num_heads = 8
            head_dim = 64
            intermediate_size = 1536
            style_condition = True
            style_as_token = True
            time_as_token = True
            final_layer_type = 'wavenet'
            depth = 13
        
        class style_encoder:
            dim = 192
    
    config = MLXConfig()
    
    # 创建 MLX CFM
    mlx_cfm = MLXCFMRewritten(config)
    
    # 加载 PyTorch 权重
    print("🔧 加载 PyTorch 权重...")
    from indextts.infer_v2 import IndexTTS2
    pytorch_tts = IndexTTS2(use_mlx=False)
    pytorch_cfm = pytorch_tts.s2mel.models['cfm']
    
    # 加载权重到 MLX
    mlx_cfm.load_weights_from_pytorch(pytorch_cfm.state_dict())
    
    # 启用调试模式
    mlx_dit = mlx_cfm.estimator
    mlx_dit._debug_layers = True
    print("✅ MLX DiT 调试模式已启用")
    
    print("🔧 执行 MLX DiT 前向传播...")
    
    # 执行 MLX DiT 前向传播
    mlx_output = mlx_dit(x_mlx, prompt_x_mlx, x_lens_mlx, t_mlx, style_mlx, mu_mlx)
    
    print(f"✅ MLX DiT 输出形状: {mlx_output.shape}")
    
    # 检查调试数据
    print("\n📊 调试数据统计:")
    print(f"   PyTorch 数据: {len(debugger.debug_data.get('pytorch', {}))} 个阶段")
    print(f"   MLX 数据: {len(debugger.debug_data.get('mlx', {}))} 个阶段")
    
    # 打印 MLX 调试数据
    print(f"\n📊 MLX 调试数据:")
    for stage, data in debugger.debug_data.get('mlx', {}).items():
        print(f"   {stage}: {len(data)} 个张量")
        for key, value in data.items():
            if hasattr(value, 'shape'):
                print(f"     {key}: {value.shape}")
            else:
                print(f"     {key}: {type(value)}")
    
    # 保存调试数据
    if debugger.debug_data:
        debug_file = save_cfm_debug("mlx_dit_debug_analysis.pkl")
        print(f"\n💾 调试数据已保存到: {debug_file}")
    else:
        print(f"\n❌ 没有找到调试数据")

if __name__ == "__main__":
    main()

