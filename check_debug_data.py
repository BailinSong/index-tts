#!/usr/bin/env python3
"""
检查 DiT 调试数据
"""

import sys
import os
sys.path.append('/Users/bailin/index-tts')

from indextts.utils.cfm_debugger import CFMDebugger

def main():
    print("🔍 检查 DiT 调试数据")
    print("="*50)
    
    # 创建调试器实例
    debugger = CFMDebugger()
    
    # 检查调试数据
    debug_data = debugger.debug_data
    
    print(f"📊 调试数据统计:")
    print(f"   PyTorch 数据: {len(debug_data.get('pytorch', {}))} 个阶段")
    print(f"   MLX 数据: {len(debug_data.get('mlx', {}))} 个阶段")
    
    # 打印 PyTorch 数据
    print(f"\n📊 PyTorch 调试数据:")
    for stage, data in debug_data.get('pytorch', {}).items():
        print(f"   {stage}: {len(data)} 个张量")
        for key, value in data.items():
            if hasattr(value, 'shape'):
                print(f"     {key}: {value.shape}")
            else:
                print(f"     {key}: {type(value)}")
    
    # 打印 MLX 数据
    print(f"\n📊 MLX 调试数据:")
    for stage, data in debug_data.get('mlx', {}).items():
        print(f"   {stage}: {len(data)} 个张量")
        for key, value in data.items():
            if hasattr(value, 'shape'):
                print(f"     {key}: {value.shape}")
            else:
                print(f"     {key}: {type(value)}")
    
    # 保存调试数据
    if debug_data:
        debugger.save_debug_data("dit_debug_analysis.pkl")
        print(f"\n💾 调试数据已保存到: dit_debug_analysis.pkl")
    else:
        print(f"\n❌ 没有找到调试数据")

if __name__ == "__main__":
    main()

