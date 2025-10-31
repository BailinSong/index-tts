#!/usr/bin/env python3
"""
DiT 实现对比分析工具
详细对比 PyTorch 和 MLX DiT 的实现差异
"""

import os
import inspect
from typing import Dict, Any, List

def analyze_dit_forward_signatures():
    """分析 DiT forward 方法签名"""
    print(f"\n{'='*60}")
    print(f"🔍 DiT Forward 方法签名对比")
    print(f"{'='*60}")
    
    try:
        # 导入 PyTorch DiT
        from indextts.s2mel.modules.diffusion_transformer import DiT
        pytorch_dit = DiT
        
        # 导入 MLX DiT
        from indextts.s2mel.modules.mlx_diffusion_transformer import MLXDiTRewritten
        mlx_dit = MLXDiTRewritten
        
        # 获取 forward 方法签名
        pytorch_forward = pytorch_dit.forward
        mlx_forward = mlx_dit.__call__
        
        pytorch_sig = inspect.signature(pytorch_forward)
        mlx_sig = inspect.signature(mlx_forward)
        
        print(f"PyTorch DiT.forward 签名:")
        print(f"   {pytorch_sig}")
        
        print(f"\nMLX DiT.__call__ 签名:")
        print(f"   {mlx_sig}")
        
        # 比较参数
        pytorch_params = list(pytorch_sig.parameters.keys())
        mlx_params = list(mlx_sig.parameters.keys())
        
        print(f"\n参数对比:")
        print(f"   PyTorch: {pytorch_params}")
        print(f"   MLX:     {mlx_params}")
        
        if pytorch_params == mlx_params:
            print(f"   ✅ 参数一致")
        else:
            print(f"   ❌ 参数不一致")
            
    except Exception as e:
        print(f"❌ 分析失败: {e}")

def analyze_dit_architecture():
    """分析 DiT 架构差异"""
    print(f"\n{'='*60}")
    print(f"🔍 DiT 架构对比")
    print(f"{'='*60}")
    
    try:
        # 导入 PyTorch DiT
        from indextts.s2mel.modules.diffusion_transformer import DiT
        pytorch_dit = DiT
        
        # 导入 MLX DiT
        from indextts.s2mel.modules.mlx_diffusion_transformer import MLXDiTRewritten
        mlx_dit = MLXDiTRewritten
        
        print(f"PyTorch DiT 类:")
        print(f"   模块: {pytorch_dit.__module__}")
        print(f"   基类: {[cls.__name__ for cls in pytorch_dit.__mro__]}")
        
        print(f"\nMLX DiT 类:")
        print(f"   模块: {mlx_dit.__module__}")
        print(f"   基类: {[cls.__name__ for cls in mlx_dit.__mro__]}")
        
        # 比较关键方法
        pytorch_methods = [method for method in dir(pytorch_dit) if not method.startswith('_')]
        mlx_methods = [method for method in dir(mlx_dit) if not method.startswith('_')]
        
        print(f"\n方法对比:")
        print(f"   PyTorch 方法: {len(pytorch_methods)} 个")
        print(f"   MLX 方法:     {len(mlx_methods)} 个")
        
        common_methods = set(pytorch_methods) & set(mlx_methods)
        pytorch_only = set(pytorch_methods) - set(mlx_methods)
        mlx_only = set(mlx_methods) - set(pytorch_methods)
        
        print(f"   共同方法: {len(common_methods)} 个")
        print(f"   PyTorch 独有: {len(pytorch_only)} 个")
        print(f"   MLX 独有: {len(mlx_only)} 个")
        
        if pytorch_only:
            print(f"   PyTorch 独有方法: {sorted(pytorch_only)}")
        if mlx_only:
            print(f"   MLX 独有方法: {sorted(mlx_only)}")
            
    except Exception as e:
        print(f"❌ 分析失败: {e}")

def analyze_dit_forward_implementation():
    """分析 DiT forward 实现差异"""
    print(f"\n{'='*60}")
    print(f"🔍 DiT Forward 实现对比")
    print(f"{'='*60}")
    
    try:
        # 读取 PyTorch DiT forward 实现
        pytorch_file = "indextts/s2mel/modules/diffusion_transformer.py"
        with open(pytorch_file, 'r', encoding='utf-8') as f:
            pytorch_content = f.read()
        
        # 读取 MLX DiT forward 实现
        mlx_file = "indextts/s2mel/modules/mlx_diffusion_transformer.py"
        with open(mlx_file, 'r', encoding='utf-8') as f:
            mlx_content = f.read()
        
        # 查找 forward 方法
        pytorch_forward_start = pytorch_content.find("def forward(self, x, prompt_x, x_lens, t, style, cond, mask_content=False):")
        mlx_forward_start = mlx_content.find("def __call__(self, x, prompt_x, x_lens, t, style, cond, mask_content=False):")
        
        if pytorch_forward_start == -1:
            print("❌ 未找到 PyTorch forward 方法")
            return
            
        if mlx_forward_start == -1:
            print("❌ 未找到 MLX forward 方法")
            return
        
        # 提取 forward 方法内容
        pytorch_lines = pytorch_content[pytorch_forward_start:].split('\n')
        mlx_lines = mlx_content[mlx_forward_start:].split('\n')
        
        # 找到方法结束位置
        pytorch_end = 0
        for i, line in enumerate(pytorch_lines[1:], 1):
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                pytorch_end = i
                break
        
        mlx_end = 0
        for i, line in enumerate(mlx_lines[1:], 1):
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                mlx_end = i
                break
        
        pytorch_forward = '\n'.join(pytorch_lines[:pytorch_end])
        mlx_forward = '\n'.join(mlx_lines[:mlx_end])
        
        print(f"PyTorch forward 方法 (前 50 行):")
        pytorch_preview = '\n'.join(pytorch_forward.split('\n')[:50])
        print(pytorch_preview)
        if len(pytorch_forward.split('\n')) > 50:
            print("   ... (更多内容)")
        
        print(f"\nMLX forward 方法 (前 50 行):")
        mlx_preview = '\n'.join(mlx_forward.split('\n')[:50])
        print(mlx_preview)
        if len(mlx_forward.split('\n')) > 50:
            print("   ... (更多内容)")
        
        # 比较行数
        pytorch_line_count = len(pytorch_forward.split('\n'))
        mlx_line_count = len(mlx_forward.split('\n'))
        
        print(f"\n方法长度对比:")
        print(f"   PyTorch: {pytorch_line_count} 行")
        print(f"   MLX:     {mlx_line_count} 行")
        
        if abs(pytorch_line_count - mlx_line_count) > 10:
            print(f"   ⚠️  长度差异较大，可能存在实现差异")
        else:
            print(f"   ✅ 长度相近")
            
    except Exception as e:
        print(f"❌ 分析失败: {e}")

def analyze_dit_key_differences():
    """分析 DiT 关键差异点"""
    print(f"\n{'='*60}")
    print(f"🔍 DiT 关键差异点分析")
    print(f"{'='*60}")
    
    # 基于之前的分析，我们知道问题可能在于：
    print("基于分析结果，可能的关键差异点:")
    print("1. 张量转置处理 - PyTorch 和 MLX 可能对 (B,C,T) vs (B,T,C) 处理不同")
    print("2. 注意力掩码计算 - 长度掩码的实现可能不同")
    print("3. 层归一化实现 - AdaLN 的计算可能不同")
    print("4. 激活函数 - SiLU 等激活函数的实现可能不同")
    print("5. 线性层计算 - 权重矩阵乘法的实现可能不同")
    print("6. 广播操作 - 张量广播的规则可能不同")
    
    print(f"\n建议检查的具体位置:")
    print("1. x 和 prompt_x 的转置操作")
    print("2. cond_projection 的输出处理")
    print("3. transformer 的输入准备")
    print("4. 注意力掩码的创建")
    print("5. 最终输出的转置")

def main():
    print("🔍 DiT 实现对比分析工具")
    print("="*60)
    
    analyze_dit_forward_signatures()
    analyze_dit_architecture()
    analyze_dit_forward_implementation()
    analyze_dit_key_differences()
    
    print(f"\n{'='*60}")
    print(f"📊 总结")
    print(f"{'='*60}")
    print("实现对比分析完成。")
    print("下一步建议：")
    print("1. 详细对比 PyTorch 和 MLX 的 forward 实现")
    print("2. 重点检查张量转置和形状处理")
    print("3. 验证注意力掩码和层归一化的实现")
    print("4. 逐步调试 DiT 内部计算过程")

if __name__ == "__main__":
    main()






