#!/usr/bin/env python3
"""
简单的 DiT 调试数据收集
"""

import sys
import os
sys.path.append('/Users/bailin/index-tts')

import torch
import pickle
from indextts.utils.cfm_debugger import get_debugger, save_cfm_debug

def main():
    print("🔍 简单 DiT 调试数据收集")
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
    
    # 加载 PyTorch DiT 模型
    print("🔧 加载 PyTorch DiT 模型...")
    from indextts.infer_v2 import IndexTTS2
    pytorch_tts = IndexTTS2(use_mlx=False)
    pytorch_cfm = pytorch_tts.s2mel.models['cfm']
    pytorch_dit = pytorch_cfm.estimator
    
    # 启用调试模式
    pytorch_dit._debug_layers = True
    print("✅ PyTorch DiT 调试模式已启用")
    
    # 准备输入数据并移动到正确设备
    device = pytorch_dit.t_embedder.freqs.device
    x = test_data['x'].to(device)
    prompt_x = test_data['prompt_x'].to(device)
    x_lens = test_data['x_lens'].to(device)
    t = test_data['t'].to(device)
    style = test_data['style'].to(device)
    mu = test_data['mu'].to(device)
    
    print("🔧 执行 PyTorch DiT 前向传播...")
    
    # 执行 PyTorch DiT 前向传播
    with torch.no_grad():
        pytorch_output = pytorch_dit(x, prompt_x, x_lens, t, style, mu)
    
    print(f"✅ PyTorch DiT 输出形状: {pytorch_output.shape}")
    
    # 检查调试数据
    print("\n📊 调试数据统计:")
    print(f"   PyTorch 数据: {len(debugger.debug_data.get('pytorch', {}))} 个阶段")
    print(f"   MLX 数据: {len(debugger.debug_data.get('mlx', {}))} 个阶段")
    
    # 打印 PyTorch 调试数据
    print(f"\n📊 PyTorch 调试数据:")
    for stage, data in debugger.debug_data.get('pytorch', {}).items():
        print(f"   {stage}: {len(data)} 个张量")
        for key, value in data.items():
            if hasattr(value, 'shape'):
                print(f"     {key}: {value.shape}")
            else:
                print(f"     {key}: {type(value)}")
    
    # 保存调试数据
    if debugger.debug_data:
        debug_file = save_cfm_debug("pytorch_dit_debug_analysis.pkl")
        print(f"\n💾 调试数据已保存到: {debug_file}")
    else:
        print(f"\n❌ 没有找到调试数据")

if __name__ == "__main__":
    main()

