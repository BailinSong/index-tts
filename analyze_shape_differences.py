#!/usr/bin/env python3
"""
分析DiT各层形状差异的详细脚本
"""

import pickle
import numpy as np
from pathlib import Path

def load_debug_data():
    """加载调试数据"""
    debug_file = Path('/Users/bailin/index-tts/cfm_debug_outputs/complete_dit_debug_analysis.pkl')
    if not debug_file.exists():
        print(f"❌ 调试数据文件不存在: {debug_file}")
        return None
    
    with open(debug_file, 'rb') as f:
        data = pickle.load(f)
    
    return data

def analyze_shape_differences(data):
    """分析形状差异"""
    print("🔍 分析形状差异...")
    
    # 获取conv2_output数据
    pytorch_conv2 = data.get('pytorch_data', {}).get('step_0_layer_0_conv2_output', {})
    mlx_conv2 = data.get('mlx_data', {}).get('step_0_layer_0_conv2_output', {})
    
    if not pytorch_conv2 or not mlx_conv2:
        print("❌ 找不到conv2数据")
        return
    
    pytorch_conv2_out = pytorch_conv2.get('conv2_out', {})
    mlx_conv2_out = mlx_conv2.get('conv2_out', {})
    
    if not pytorch_conv2_out or not mlx_conv2_out:
        print("❌ 找不到conv2输出数据")
        return
    
    print(f"📊 Conv2输出形状分析:")
    print(f"   PyTorch: {pytorch_conv2_out.get('shape')}")
    print(f"   MLX:     {mlx_conv2_out.get('shape')}")
    
    # 分析形状差异
    pytorch_shape = pytorch_conv2_out.get('shape')
    mlx_shape = mlx_conv2_out.get('shape')
    
    if pytorch_shape and mlx_shape:
        print(f"   形状差异: PyTorch={pytorch_shape} vs MLX={mlx_shape}")
        
        # 检查形状是否匹配
        if pytorch_shape == mlx_shape:
            print("   ✅ 形状一致")
        else:
            print("   ❌ 形状不匹配")
            
            # 分析形状差异的具体原因
            if len(pytorch_shape) == len(mlx_shape):
                for i, (p, m) in enumerate(zip(pytorch_shape, mlx_shape)):
                    if p != m:
                        print(f"     维度{i}: PyTorch={p}, MLX={m}, 差异={abs(p-m)}")
            else:
                print(f"     维度数量不同: PyTorch={len(pytorch_shape)}, MLX={len(mlx_shape)}")

def analyze_final_layer_differences(data):
    """分析final_layer差异"""
    print("\n🔍 分析final_layer差异...")
    
    # 获取final_layer数据
    pytorch_final = data.get('pytorch_data', {}).get('step_0_layer_0_final_layer', {})
    mlx_final = data.get('mlx_data', {}).get('step_0_layer_0_final_layer', {})
    
    if not pytorch_final or not mlx_final:
        print("❌ 找不到final_layer数据")
        return
    
    pytorch_final_input = pytorch_final.get('final_layer_input', {})
    mlx_final_input = mlx_final.get('final_layer_input', {})
    
    pytorch_final_output = pytorch_final.get('final_layer_output', {})
    mlx_final_output = mlx_final.get('final_layer_output', {})
    
    print(f"📊 Final Layer输入分析:")
    print(f"   PyTorch: shape={pytorch_final_input.get('shape')}, min={pytorch_final_input.get('min'):.6f}, max={pytorch_final_input.get('max'):.6f}")
    print(f"   MLX:     shape={mlx_final_input.get('shape')}, min={mlx_final_input.get('min'):.6f}, max={mlx_final_input.get('max'):.6f}")
    
    print(f"📊 Final Layer输出分析:")
    print(f"   PyTorch: shape={pytorch_final_output.get('shape')}, min={pytorch_final_output.get('min'):.6f}, max={pytorch_final_output.get('max'):.6f}")
    print(f"   MLX:     shape={mlx_final_output.get('shape')}, min={mlx_final_output.get('min'):.6f}, max={mlx_final_output.get('max'):.6f}")
    
    # 分析形状差异
    pytorch_input_shape = pytorch_final_input.get('shape')
    mlx_input_shape = mlx_final_input.get('shape')
    
    pytorch_output_shape = pytorch_final_output.get('shape')
    mlx_output_shape = mlx_final_output.get('shape')
    
    print(f"   输入形状差异: PyTorch={pytorch_input_shape} vs MLX={mlx_input_shape}")
    print(f"   输出形状差异: PyTorch={pytorch_output_shape} vs MLX={mlx_output_shape}")

def analyze_transformer_differences(data):
    """分析transformer差异"""
    print("\n🔍 分析transformer差异...")
    
    # 获取transformer_output数据
    pytorch_transformer = data.get('pytorch_data', {}).get('step_0_layer_0_transformer_output', {})
    mlx_transformer = data.get('mlx_data', {}).get('step_0_layer_0_transformer_output', {})
    
    if not pytorch_transformer or not mlx_transformer:
        print("❌ 找不到transformer数据")
        return
    
    pytorch_x_res = pytorch_transformer.get('x_res', {})
    mlx_x_res = mlx_transformer.get('x_res', {})
    
    print(f"📊 Transformer输出分析:")
    print(f"   PyTorch: shape={pytorch_x_res.get('shape')}, min={pytorch_x_res.get('min'):.6f}, max={pytorch_x_res.get('max'):.6f}")
    print(f"   MLX:     shape={mlx_x_res.get('shape')}, min={mlx_x_res.get('min'):.6f}, max={mlx_x_res.get('max'):.6f}")
    
    # 分析形状差异
    pytorch_shape = pytorch_x_res.get('shape')
    mlx_shape = mlx_x_res.get('shape')
    
    print(f"   形状差异: PyTorch={pytorch_shape} vs MLX={mlx_shape}")

def main():
    """主函数"""
    print("🔍 DiT形状差异分析工具")
    print("=" * 60)
    
    # 加载调试数据
    data = load_debug_data()
    if data is None:
        return
    
    # 分析形状差异
    analyze_shape_differences(data)
    
    # 分析final_layer差异
    analyze_final_layer_differences(data)
    
    # 分析transformer差异
    analyze_transformer_differences(data)
    
    print("\n✅ 分析完成")

if __name__ == "__main__":
    main()