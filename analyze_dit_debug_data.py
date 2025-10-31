#!/usr/bin/env python3
"""
分析 DiT 调试数据，对比 PyTorch 和 MLX 在每个阶段的差异
"""

import sys
import os
sys.path.append('/Users/bailin/index-tts')

import pickle
import numpy as np
import torch
import mlx.core as mx
from indextts.utils.cfm_debugger import get_debugger

TOL = 1e-5  # 将 e-5 以内的差异视为相同

def analyze_tensor_difference(pytorch_data, mlx_data, name):
    """分析两个张量的差异"""
    if isinstance(pytorch_data, dict) and isinstance(mlx_data, dict):
        # 如果是字典，递归分析每个键
        results = {}
        for key in pytorch_data.keys():
            if key in mlx_data:
                results[key] = analyze_tensor_difference(pytorch_data[key], mlx_data[key], f"{name}.{key}")
            else:
                results[key] = f"❌ {key} 只在 PyTorch 中存在"
        for key in mlx_data.keys():
            if key not in pytorch_data:
                results[key] = f"❌ {key} 只在 MLX 中存在"
        return results
    else:
        # 如果是数值，直接比较
        try:
            # 跳过 dtype 对比，避免误导
            if str(name).endswith('.dtype'):
                return "ℹ️ 跳过 dtype 对比"
            # 尝试把字符串数值转为数字再比较
            def coerce_num(v):
                if isinstance(v, (int, float)):
                    return v
                if isinstance(v, str):
                    try:
                        if v.isdigit():
                            return int(v)
                        return float(v)
                    except Exception:
                        return v
                return v
            pytorch_data = coerce_num(pytorch_data)
            mlx_data = coerce_num(mlx_data)
            if isinstance(pytorch_data, (int, float)) and isinstance(mlx_data, (int, float)):
                diff = abs(pytorch_data - mlx_data)
                if diff <= TOL:
                    return f"✅ 完全一致: diff={diff:.2e}"
                else:
                    return f"❌ 存在差异: diff={diff:.6f}"
            elif isinstance(pytorch_data, list) and isinstance(mlx_data, list):
                # 形状字段：按完全相等比较
                if str(name).endswith('.shape'):
                    if pytorch_data == mlx_data:
                        return "✅ 形状一致"
                    # 容忍 WaveNet 内部 (B,C,T) vs (B,T,C) 的轴交换
                    try:
                        if len(pytorch_data) == 3 and len(mlx_data) == 3 and pytorch_data[0] == mlx_data[0]:
                            if sorted(pytorch_data[1:]) == sorted(mlx_data[1:]):
                                return "✅ 形状一致(轴顺序不同)"
                    except Exception:
                        pass
                    return f"❌ 形状不一致: PyTorch={pytorch_data}, MLX={mlx_data}"
                if len(pytorch_data) != len(mlx_data):
                    return f"❌ 长度不匹配: PyTorch={len(pytorch_data)}, MLX={len(mlx_data)}"
                
                # 比较列表中的数值
                diffs = []
                for i, (p, m) in enumerate(zip(pytorch_data, mlx_data)):
                    if isinstance(p, (int, float)) and isinstance(m, (int, float)):
                        diff = abs(p - m)
                        diffs.append(diff)
                    else:
                        return f"⚠️ 无法比较列表元素: PyTorch[{i}]={type(p)}, MLX[{i}]={type(m)}"
                
                max_diff = max(diffs)
                mean_diff = np.mean(diffs)
                
                if max_diff <= TOL:
                    return f"✅ 完全一致: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"
                else:
                    return f"❌ 存在差异: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
            else:
                return f"⚠️ 无法比较: PyTorch={type(pytorch_data)}, MLX={type(mlx_data)}"
        except Exception as e:
            return f"❌ 比较失败: {e}"

def main():
    print("🔍 分析 DiT 调试数据")
    print("="*60)
    
    # 获取全局调试器
    debugger = get_debugger()
    
    # 加载调试数据
    debug_file = "/Users/bailin/index-tts/cfm_debug_outputs/complete_dit_debug_analysis.pkl"
    if os.path.exists(debug_file):
        with open(debug_file, 'rb') as f:
            debug_data = pickle.load(f)
        print(f"✅ 加载调试数据: {debug_file}")
    else:
        print(f"❌ 调试数据文件不存在: {debug_file}")
        return
    
    pytorch_data = debug_data.get('pytorch', {})
    mlx_data = debug_data.get('mlx', {})
    
    print(f"\n📊 调试数据统计:")
    print(f"   PyTorch 阶段: {len(pytorch_data)}")
    print(f"   MLX 阶段: {len(mlx_data)}")
    
    # 分析每个阶段
    print(f"\n🔍 阶段对比分析:")
    print("="*60)
    
    # 获取所有阶段名称
    all_stages = set(pytorch_data.keys()) | set(mlx_data.keys())
    
    for stage in sorted(all_stages):
        print(f"\n📊 {stage}:")
        
        if stage in pytorch_data and stage in mlx_data:
            pytorch_stage = pytorch_data[stage]
            mlx_stage = mlx_data[stage]
            
            print(f"   PyTorch 张量数: {len(pytorch_stage)}")
            print(f"   MLX 张量数: {len(mlx_stage)}")
            
            # 分析每个张量
            for key in pytorch_stage.keys():
                if key in mlx_stage:
                    # WaveNet 输入中的 x_transposed 仅用于格式展示，跳过数值对比，避免误导
                    if stage.endswith('wavenet_input') and key == 'x_transposed':
                        print(f"   {key}: ℹ️ 仅格式展示(CT vs TC)，数值对齐以 x_bt_c 为准，已跳过")
                        continue
                    # 特殊处理 conv2_input/conv2_output：打印形状与头尾值
                    if (stage.endswith('conv2_output') or stage.endswith('conv2_input')) \
                        and isinstance(pytorch_stage[key], dict) and isinstance(mlx_stage[key], dict):
                        print(f"   {key}:")
                        # 形状
                        p_shape = pytorch_stage[key].get('conv2_out_shape') or pytorch_stage[key].get('conv2_in_shape')
                        m_shape = mlx_stage[key].get('conv2_out_shape') or mlx_stage[key].get('conv2_in_shape')
                        if p_shape is not None and m_shape is not None:
                            print(f"     shape: PyTorch={p_shape}, MLX={m_shape}")
                        # 头尾值
                        p_head = pytorch_stage[key].get('conv2_out_head') or pytorch_stage[key].get('conv2_in_head')
                        p_tail = pytorch_stage[key].get('conv2_out_tail') or pytorch_stage[key].get('conv2_in_tail')
                        m_head = mlx_stage[key].get('conv2_out_head') or mlx_stage[key].get('conv2_in_head')
                        m_tail = mlx_stage[key].get('conv2_out_tail') or mlx_stage[key].get('conv2_in_tail')
                        if p_head is not None and m_head is not None:
                            try:
                                p_head_np = p_head.detach().cpu().numpy() if hasattr(p_head, 'detach') else np.array(p_head)
                                m_head_np = m_head.detach().cpu().numpy() if hasattr(m_head, 'detach') else np.array(m_head)
                                print(f"     head(3): PyTorch={np.round(p_head_np,6)}, MLX={np.round(m_head_np,6)}")
                            except Exception:
                                pass
                        if p_tail is not None and m_tail is not None:
                            try:
                                p_tail_np = p_tail.detach().cpu().numpy() if hasattr(p_tail, 'detach') else np.array(p_tail)
                                m_tail_np = m_tail.detach().cpu().numpy() if hasattr(m_tail, 'detach') else np.array(m_tail)
                                print(f"     tail(3): PyTorch={np.round(p_tail_np,6)}, MLX={np.round(m_tail_np,6)}")
                            except Exception:
                                pass
                        # 常规差异
                        result = analyze_tensor_difference(pytorch_stage[key], mlx_stage[key], key)
                    else:
                        result = analyze_tensor_difference(pytorch_stage[key], mlx_stage[key], key)
                    if isinstance(result, dict):
                        print(f"   {key}:")
                        for subkey, subresult in result.items():
                            print(f"     {subkey}: {subresult}")
                    else:
                        print(f"   {key}: {result}")
                else:
                    print(f"   {key}: ❌ 只在 PyTorch 中存在")
            
            for key in mlx_stage.keys():
                if key not in pytorch_stage:
                    print(f"   {key}: ❌ 只在 MLX 中存在")
        
        elif stage in pytorch_data:
            print(f"   ❌ 只在 PyTorch 中存在")
        else:
            print(f"   ❌ 只在 MLX 中存在")
            mlx_stage = mlx_data.get(stage, {})
            # 对仅在 MLX 的阶段，直接把关键字段打印出来，便于定位（尤其是 padding 内部日志）
            if isinstance(mlx_stage, dict):
                # 优先打印 wavenet_l0_sconv_pad_internal 的关键字段
                if stage.endswith('wavenet_l0_sconv_pad_internal'):
                    keys = [
                        'left','right','extra_pad','T_in','Tp_after_extra',
                        'total_len_index','target_len_crop','len_before_crop','len_after_crop'
                    ]
                    for k in keys:
                        if k in mlx_stage:
                            print(f"   {k}: {mlx_stage[k]}")
                else:
                    # 通用打印（避免过长，仅首20项）
                    printed = 0
                    for k in sorted(mlx_stage.keys()):
                        print(f"   {k}: {mlx_stage[k]}")
                        printed += 1
                        if printed >= 20:
                            break
    
    # 总结分析
    print(f"\n📊 总结分析:")
    print("="*60)
    
    # 统计差异阶段
    diff_stages = []
    for stage in sorted(all_stages):
        if stage in pytorch_data and stage in mlx_data:
            pytorch_stage = pytorch_data[stage]
            mlx_stage = mlx_data[stage]
            
            has_diff = False
            for key in pytorch_stage.keys():
                if key in mlx_stage:
                    result = analyze_tensor_difference(pytorch_stage[key], mlx_stage[key], key)
                    if isinstance(result, str) and "❌" in result:
                        has_diff = True
                        break
                    elif isinstance(result, dict):
                        for subresult in result.values():
                            if isinstance(subresult, str) and "❌" in subresult:
                                has_diff = True
                                break
                        if has_diff:
                            break
            
            if has_diff:
                diff_stages.append(stage)
    
    print(f"   存在差异的阶段: {len(diff_stages)}")
    for stage in diff_stages:
        print(f"     - {stage}")
    
    print(f"\n✅ 分析完成")

if __name__ == "__main__":
    main()
