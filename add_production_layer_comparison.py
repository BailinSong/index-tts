#!/usr/bin/env python3
"""
在生产代码上添加输入输出逐层对比功能
使用前级缓存数据进行测试，跳过前级处理直接从CFM开始测试
"""

import os
import pickle
import torch
import mlx.core as mx
import numpy as np
from typing import Dict, Any, Optional, Tuple

def load_cached_inputs():
    """加载前级缓存的输入数据"""
    cache_files = [
        'cfm_inputs_mlx.pkl',
        'cfm_inputs_torch.pkl', 
        'cfm_outputs_mlx.pkl',
        'cfm_outputs_torch.pkl'
    ]
    
    cached_data = {}
    for file in cache_files:
        if os.path.exists(file):
            try:
                with open(file, 'rb') as f:
                    cached_data[file.replace('.pkl', '')] = pickle.load(f)
                print(f"✓ 加载缓存文件: {file}")
            except Exception as e:
                print(f"✗ 加载缓存文件失败 {file}: {e}")
        else:
            print(f"✗ 缓存文件不存在: {file}")
    
    return cached_data

def compare_tensor_values(torch_tensor, mlx_array, name, tolerance=1e-5):
    """对比PyTorch tensor和MLX array的数值"""
    # 转换MLX array为numpy
    if hasattr(mlx_array, 'numpy'):
        mlx_np = mlx_array.numpy()
    else:
        mlx_np = np.array(mlx_array)
    
    # 转换PyTorch tensor为numpy
    torch_np = torch_tensor.detach().cpu().numpy()
    
    # 确保形状一致
    if torch_np.shape != mlx_np.shape:
        print(f"❌ {name}: 形状不匹配 - PyTorch: {torch_np.shape}, MLX: {mlx_np.shape}")
        return False
    
    # 计算差异
    diff = np.abs(torch_np - mlx_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"📊 {name}:")
    print(f"   形状: {torch_np.shape}")
    print(f"   PyTorch范围: [{torch_np.min():.6f}, {torch_np.max():.6f}]")
    print(f"   MLX范围: [{mlx_np.min():.6f}, {mlx_np.max():.6f}]")
    print(f"   最大差异: {max_diff:.10f}")
    print(f"   平均差异: {mean_diff:.10f}")
    
    if max_diff < tolerance:
        print(f"   ✅ 差异在容忍范围内 (< {tolerance})")
        return True
    else:
        print(f"   ❌ 差异超出容忍范围 (>= {tolerance})")
        return False

def add_layer_comparison_to_mlx_cfm():
    """在MLX CFM模块中添加逐层对比功能"""
    
    cfm_file = 'indextts/s2mel/modules/mlx_cfm.py'
    if not os.path.exists(cfm_file):
        print(f"❌ CFM文件不存在: {cfm_file}")
        return False
    
    print(f"📝 正在修改CFM文件: {cfm_file}")
    
    # 备份原文件
    backup_file = cfm_file + '.backup_layer_comparison'
    if not os.path.exists(backup_file):
        import shutil
        shutil.copy2(cfm_file, backup_file)
        print(f"✓ 已备份原文件到: {backup_file}")
    
    # 读取文件内容
    with open(cfm_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经添加了对比功能
    if 'def compare_with_pytorch' in content:
        print("✓ 逐层对比功能已存在")
        return True
    
    # 添加逐层对比功能
    comparison_code = '''
    def compare_with_pytorch(self, pytorch_cfm, inputs, tolerance=1e-5):
        """
        与PyTorch CFM进行逐层对比
        """
        print("\\n" + "="*60)
        print("🔍 MLX CFM vs PyTorch CFM 逐层对比")
        print("="*60)
        
        # 解包输入
        cat_condition, x_lens, ref_mel, style = inputs
        
        # 转换输入格式
        from indextts.utils.mlx_utils import torch_to_mlx
        cat_condition_mlx = torch_to_mlx(cat_condition)
        x_lens_mlx = torch_to_mlx(x_lens)
        ref_mel_mlx = torch_to_mlx(ref_mel)
        style_mlx = torch_to_mlx(style)
        
        print("\\n📊 输入对比:")
        compare_tensor_values(cat_condition, cat_condition_mlx, "cat_condition", tolerance)
        compare_tensor_values(x_lens, x_lens_mlx, "x_lens", tolerance)
        compare_tensor_values(ref_mel, ref_mel_mlx, "ref_mel", tolerance)
        compare_tensor_values(style, style_mlx, "style", tolerance)
        
        # 对比estimator
        print("\\n🔍 Estimator对比:")
        if hasattr(self, 'estimator') and hasattr(pytorch_cfm, 'estimator'):
            self._compare_estimator(pytorch_cfm.estimator, tolerance)
        
        # 对比输出
        print("\\n📊 输出对比:")
        mlx_output = self.inference(cat_condition_mlx, x_lens_mlx, ref_mel_mlx, style_mlx)
        pytorch_output = pytorch_cfm.inference(cat_condition, x_lens, ref_mel, style)
        
        compare_tensor_values(pytorch_output, mlx_output, "CFM输出", tolerance)
        
        return mlx_output, pytorch_output
    
    def _compare_estimator(self, pytorch_estimator, tolerance=1e-5):
        """
        对比estimator的内部层
        """
        print("\\n🔍 Estimator内部层对比:")
        
        # 对比权重
        if hasattr(self.estimator, 'weight') and hasattr(pytorch_estimator, 'weight'):
            mlx_weight = self.estimator.weight
            pytorch_weight = pytorch_estimator.weight.detach().cpu()
            compare_tensor_values(pytorch_weight, mlx_weight, "Estimator权重", tolerance)
        
        # 对比偏置
        if hasattr(self.estimator, 'bias') and hasattr(pytorch_estimator, 'bias'):
            mlx_bias = self.estimator.bias
            pytorch_bias = pytorch_estimator.bias.detach().cpu()
            compare_tensor_values(pytorch_bias, mlx_bias, "Estimator偏置", tolerance)
        
        # 对比其他参数
        for attr_name in ['scale', 'shift']:
            if hasattr(self.estimator, attr_name) and hasattr(pytorch_estimator, attr_name):
                mlx_param = getattr(self.estimator, attr_name)
                pytorch_param = getattr(pytorch_estimator, attr_name).detach().cpu()
                compare_tensor_values(pytorch_param, mlx_param, f"Estimator {attr_name}", tolerance)
'''
    
    # 在MLXCFM类中添加对比方法
    if 'def compare_with_pytorch' not in content:
        # 找到MLXCFM类的结束位置
        class_start = content.find('class MLXCFM')
        if class_start != -1:
            # 找到类的最后一个方法
            methods = content[class_start:].split('def ')
            if len(methods) > 1:
                last_method = methods[-1]
                # 找到最后一个方法的结束位置
                last_method_end = content.rfind(last_method)
                if last_method_end != -1:
                    # 在最后一个方法后添加对比方法
                    insert_pos = content.rfind('    def ', 0, last_method_end)
                    if insert_pos != -1:
                        # 找到方法的结束位置
                        method_start = insert_pos
                        method_end = content.find('    def ', method_start + 1)
                        if method_end == -1:
                            method_end = len(content)
                        
                        # 在方法后添加对比代码
                        content = content[:method_end] + comparison_code + content[method_end:]
                        
                        # 写回文件
                        with open(cfm_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        print("✓ 已添加逐层对比功能到MLX CFM")
                        return True
    
    print("❌ 无法找到合适的插入位置")
    return False

def add_comparison_to_infer_v2():
    """在infer_v2.py中添加对比功能"""
    
    infer_file = 'indextts/infer_v2.py'
    if not os.path.exists(infer_file):
        print(f"❌ 推理文件不存在: {infer_file}")
        return False
    
    print(f"📝 正在修改推理文件: {infer_file}")
    
    # 备份原文件
    backup_file = infer_file + '.backup_layer_comparison'
    if not os.path.exists(backup_file):
        import shutil
        shutil.copy2(infer_file, backup_file)
        print(f"✓ 已备份原文件到: {backup_file}")
    
    # 读取文件内容
    with open(infer_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经添加了对比功能
    if 'def compare_mlx_pytorch_cfm' in content:
        print("✓ 对比功能已存在")
        return True
    
    # 添加对比功能
    comparison_code = '''
    def compare_mlx_pytorch_cfm(self, cached_inputs=None):
        """
        对比MLX和PyTorch CFM的输出
        """
        print("\\n" + "="*70)
        print("🔍 MLX vs PyTorch CFM 生产环境对比")
        print("="*70)
        
        # 加载缓存数据
        if cached_inputs is None:
            cached_data = load_cached_inputs()
            if not cached_data:
                print("❌ 没有找到缓存数据，无法进行对比")
                return
        else:
            cached_data = cached_inputs
        
        # 检查是否有MLX CFM
        if not hasattr(self, 'mlx_s2mel_cfm') or self.mlx_s2mel_cfm is None:
            print("❌ MLX CFM未初始化")
            return
        
        # 检查是否有PyTorch CFM
        if not hasattr(self, 's2mel') or self.s2mel is None:
            print("❌ PyTorch S2MEL未初始化")
            return
        
        try:
            # 使用MLX CFM进行对比
            if 'cfm_inputs_mlx' in cached_data:
                inputs = cached_data['cfm_inputs_mlx']
                cat_condition = inputs['cat_condition']
                x_lens = inputs['x_lens']
                ref_mel = inputs['ref_mel']
                style = inputs['style']
                
                print("\\n📊 使用缓存输入进行对比:")
                print(f"   cat_condition: {cat_condition.shape}")
                print(f"   x_lens: {x_lens.shape}")
                print(f"   ref_mel: {ref_mel.shape}")
                print(f"   style: {style.shape}")
                
                # 调用MLX CFM的对比方法
                mlx_output, pytorch_output = self.mlx_s2mel_cfm.compare_with_pytorch(
                    self.s2mel.cfm, 
                    (cat_condition, x_lens, ref_mel, style)
                )
                
                print("\\n✅ 对比完成")
                return mlx_output, pytorch_output
            else:
                print("❌ 没有找到MLX CFM输入缓存")
                return None, None
                
        except Exception as e:
            print(f"❌ 对比过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return None, None
'''
    
    # 在IndexTTS2类中添加对比方法
    if 'def compare_mlx_pytorch_cfm' not in content:
        # 找到IndexTTS2类的结束位置
        class_start = content.find('class IndexTTS2')
        if class_start != -1:
            # 找到类的最后一个方法
            methods = content[class_start:].split('def ')
            if len(methods) > 1:
                last_method = methods[-1]
                # 找到最后一个方法的结束位置
                last_method_end = content.rfind(last_method)
                if last_method_end != -1:
                    # 在最后一个方法后添加对比方法
                    insert_pos = content.rfind('    def ', 0, last_method_end)
                    if insert_pos != -1:
                        # 找到方法的结束位置
                        method_start = insert_pos
                        method_end = content.find('    def ', method_start + 1)
                        if method_end == -1:
                            method_end = len(content)
                        
                        # 在方法后添加对比代码
                        content = content[:method_end] + comparison_code + content[method_end:]
                        
                        # 写回文件
                        with open(infer_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        print("✓ 已添加对比功能到IndexTTS2")
                        return True
    
    print("❌ 无法找到合适的插入位置")
    return False

def create_test_script():
    """创建测试脚本"""
    test_script = '''#!/usr/bin/env python3
"""
测试MLX和PyTorch CFM的逐层对比
使用前级缓存数据直接从CFM开始测试
"""

import os
import sys
import pickle
import torch
import mlx.core as mx
import numpy as np

# 添加项目路径
sys.path.append('.')

from indextts.infer_v2 import IndexTTS2

def load_cached_inputs():
    """加载前级缓存的输入数据"""
    cache_files = [
        'cfm_inputs_mlx.pkl',
        'cfm_inputs_torch.pkl', 
        'cfm_outputs_mlx.pkl',
        'cfm_outputs_torch.pkl'
    ]
    
    cached_data = {}
    for file in cache_files:
        if os.path.exists(file):
            try:
                with open(file, 'rb') as f:
                    cached_data[file.replace('.pkl', '')] = pickle.load(f)
                print(f"✓ 加载缓存文件: {file}")
            except Exception as e:
                print(f"✗ 加载缓存文件失败 {file}: {e}")
        else:
            print(f"✗ 缓存文件不存在: {file}")
    
    return cached_data

def compare_tensor_values(torch_tensor, mlx_array, name, tolerance=1e-5):
    """对比PyTorch tensor和MLX array的数值"""
    # 转换MLX array为numpy
    if hasattr(mlx_array, 'numpy'):
        mlx_np = mlx_array.numpy()
    else:
        mlx_np = np.array(mlx_array)
    
    # 转换PyTorch tensor为numpy
    torch_np = torch_tensor.detach().cpu().numpy()
    
    # 确保形状一致
    if torch_np.shape != mlx_np.shape:
        print(f"❌ {name}: 形状不匹配 - PyTorch: {torch_np.shape}, MLX: {mlx_np.shape}")
        return False
    
    # 计算差异
    diff = np.abs(torch_np - mlx_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"📊 {name}:")
    print(f"   形状: {torch_np.shape}")
    print(f"   PyTorch范围: [{torch_np.min():.6f}, {torch_np.max():.6f}]")
    print(f"   MLX范围: [{mlx_np.min():.6f}, {mlx_np.max():.6f}]")
    print(f"   最大差异: {max_diff:.10f}")
    print(f"   平均差异: {mean_diff:.10f}")
    
    if max_diff < tolerance:
        print(f"   ✅ 差异在容忍范围内 (< {tolerance})")
        return True
    else:
        print(f"   ❌ 差异超出容忍范围 (>= {tolerance})")
        return False

def test_layer_by_layer_comparison():
    """测试逐层对比功能"""
    print("="*70)
    print("🧪 测试MLX和PyTorch CFM逐层对比")
    print("="*70)
    
    # 初始化IndexTTS2
    print("\\n📦 初始化IndexTTS2...")
    tts = IndexTTS2(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        use_mlx=True,
        diffusion_steps=25
    )
    
    # 检查是否有缓存数据
    cached_data = load_cached_inputs()
    if not cached_data:
        print("❌ 没有找到缓存数据，请先运行推理生成缓存")
        return
    
    # 进行对比
    print("\\n🔍 开始逐层对比...")
    try:
        mlx_output, pytorch_output = tts.compare_mlx_pytorch_cfm(cached_data)
        
        if mlx_output is not None and pytorch_output is not None:
            print("\\n✅ 对比完成")
            print(f"MLX输出形状: {mlx_output.shape}")
            print(f"PyTorch输出形状: {pytorch_output.shape}")
            
            # 最终输出对比
            print("\\n📊 最终输出对比:")
            compare_tensor_values(pytorch_output, mlx_output, "CFM最终输出", tolerance=1e-5)
        else:
            print("❌ 对比失败")
            
    except Exception as e:
        print(f"❌ 对比过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_layer_by_layer_comparison()
'''
    
    with open('test_production_layer_comparison.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print("✓ 已创建测试脚本: test_production_layer_comparison.py")

def main():
    """主函数"""
    print("="*70)
    print("🔧 在生产代码上添加输入输出逐层对比功能")
    print("="*70)
    
    # 检查缓存数据
    cached_data = load_cached_inputs()
    if not cached_data:
        print("❌ 没有找到缓存数据，请先运行推理生成缓存")
        return
    
    # 添加对比功能到CFM
    print("\\n📝 添加逐层对比功能到MLX CFM...")
    if add_layer_comparison_to_mlx_cfm():
        print("✅ MLX CFM对比功能添加成功")
    else:
        print("❌ MLX CFM对比功能添加失败")
    
    # 添加对比功能到推理代码
    print("\\n📝 添加对比功能到推理代码...")
    if add_comparison_to_infer_v2():
        print("✅ 推理代码对比功能添加成功")
    else:
        print("❌ 推理代码对比功能添加失败")
    
    # 创建测试脚本
    print("\\n📝 创建测试脚本...")
    create_test_script()
    
    print("\\n" + "="*70)
    print("✅ 所有功能添加完成")
    print("="*70)
    print("\\n📋 使用方法:")
    print("1. 运行测试脚本: python test_production_layer_comparison.py")
    print("2. 查看逐层对比结果")
    print("3. 使用前级缓存数据直接从CFM开始测试")

if __name__ == "__main__":
    main()


