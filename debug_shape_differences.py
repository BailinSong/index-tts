#!/usr/bin/env python3
"""
调试形状差异问题
"""

import pickle
import numpy as np

def debug_shape_differences():
    """调试形状差异问题"""
    
    # 加载调试数据
    debug_file = "/Users/bailin/index-tts/cfm_debug_outputs/complete_dit_debug_analysis.pkl"
    
    with open(debug_file, 'rb') as f:
        debug_data = pickle.load(f)
    
    pytorch_data = debug_data.get('pytorch', {})
    mlx_data = debug_data.get('mlx', {})
    
    print("🔍 调试形状差异问题")
    print("=" * 60)
    
    # 检查 wavenet_l0_in.x_in 的形状差异
    print("\n📊 wavenet_l0_in.x_in 形状分析:")
    
    pt_x_in = pytorch_data.get('step_0_layer_0_wavenet_l0_in', {}).get('x_in', {})
    mlx_x_in = mlx_data.get('step_0_layer_0_wavenet_l0_in', {}).get('x_in', {})
    
    if pt_x_in and mlx_x_in:
        print(f"  PyTorch x_in:")
        print(f"    shape: {pt_x_in.get('shape', 'N/A')}")
        print(f"    data type: {type(pt_x_in.get('data', None))}")
        if 'data' in pt_x_in:
            data = pt_x_in['data']
            if hasattr(data, 'shape'):
                print(f"    actual shape: {data.shape}")
            if hasattr(data, 'dtype'):
                print(f"    dtype: {data.dtype}")
        
        print(f"  MLX x_in:")
        print(f"    shape: {mlx_x_in.get('shape', 'N/A')}")
        print(f"    data type: {type(mlx_x_in.get('data', None))}")
        if 'data' in mlx_x_in:
            data = mlx_x_in['data']
            if hasattr(data, 'shape'):
                print(f"    actual shape: {data.shape}")
            if hasattr(data, 'dtype'):
                print(f"    dtype: {data.dtype}")
    
    # 检查 conv2_input 的形状差异
    print("\n📊 conv2_input 形状分析:")
    
    pt_conv2_in = pytorch_data.get('step_0_layer_0_conv2_input', {}).get('conv2_in', {})
    mlx_conv2_in = mlx_data.get('step_0_layer_0_conv2_input', {}).get('conv2_in', {})
    
    if pt_conv2_in and mlx_conv2_in:
        print(f"  PyTorch conv2_in:")
        print(f"    shape: {pt_conv2_in.get('shape', 'N/A')}")
        print(f"    data type: {type(pt_conv2_in.get('data', None))}")
        if 'data' in pt_conv2_in:
            data = pt_conv2_in['data']
            if hasattr(data, 'shape'):
                print(f"    actual shape: {data.shape}")
            if hasattr(data, 'dtype'):
                print(f"    dtype: {data.dtype}")
        
        print(f"  MLX conv2_in:")
        print(f"    shape: {mlx_conv2_in.get('shape', 'N/A')}")
        print(f"    data type: {type(mlx_conv2_in.get('data', None))}")
        if 'data' in mlx_conv2_in:
            data = mlx_conv2_in['data']
            if hasattr(data, 'shape'):
                print(f"    actual shape: {data.shape}")
            if hasattr(data, 'dtype'):
                print(f"    dtype: {data.dtype}")
    
    # 检查 conv2_output 的形状差异
    print("\n📊 conv2_output 形状分析:")
    
    pt_conv2_out = pytorch_data.get('step_0_layer_0_conv2_output', {}).get('conv2_out', {})
    mlx_conv2_out = mlx_data.get('step_0_layer_0_conv2_output', {}).get('conv2_out', {})
    
    if pt_conv2_out and mlx_conv2_out:
        print(f"  PyTorch conv2_out:")
        print(f"    shape: {pt_conv2_out.get('shape', 'N/A')}")
        print(f"    data type: {type(pt_conv2_out.get('data', None))}")
        if 'data' in pt_conv2_out:
            data = pt_conv2_out['data']
            if hasattr(data, 'shape'):
                print(f"    actual shape: {data.shape}")
            if hasattr(data, 'dtype'):
                print(f"    dtype: {data.dtype}")
        
        print(f"  MLX conv2_out:")
        print(f"    shape: {mlx_conv2_out.get('shape', 'N/A')}")
        print(f"    data type: {type(mlx_conv2_out.get('data', None))}")
        if 'data' in mlx_conv2_out:
            data = mlx_conv2_out['data']
            if hasattr(data, 'shape'):
                print(f"    actual shape: {data.shape}")
            if hasattr(data, 'dtype'):
                print(f"    dtype: {data.dtype}")
    
    # 检查所有形状相关的数据
    print("\n📊 所有形状相关数据分析:")
    
    def check_shape_data(data_dict, name):
        print(f"\n  {name}:")
        for key, value in data_dict.items():
            if isinstance(value, dict) and 'shape' in value:
                print(f"    {key}: shape={value['shape']}, data_type={type(value.get('data', None))}")
                if 'data' in value and value['data'] is not None:
                    data = value['data']
                    if hasattr(data, 'shape'):
                        print(f"      actual_shape: {data.shape}")
    
    print("  PyTorch 数据:")
    for stage_name, stage_data in pytorch_data.items():
        if 'wavenet' in stage_name or 'conv2' in stage_name:
            check_shape_data(stage_data, stage_name)
    
    print("\n  MLX 数据:")
    for stage_name, stage_data in mlx_data.items():
        if 'wavenet' in stage_name or 'conv2' in stage_name:
            check_shape_data(stage_data, stage_name)

if __name__ == "__main__":
    debug_shape_differences()

