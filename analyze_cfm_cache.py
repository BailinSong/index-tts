#!/usr/bin/env python3
"""
CFM 缓存数据分析工具
清晰展示 PyTorch 和 MLX 版本的 CFM 输入输出数据，方便后续测试使用
"""

import os
import pickle
import numpy as np
import torch
import mlx.core as mx
from typing import Dict, Any, List, Tuple
import time
from cfm_data_cache import CFMDataCache

class CFMCacheAnalyzer:
    """CFM 缓存数据分析器"""
    
    def __init__(self, cache_dir: str = "cfm_baseline_cache"):
        self.cache = CFMDataCache(cache_dir)
        self.cache_dir = cache_dir
    
    def list_cache_files(self) -> Dict[str, List[str]]:
        """列出所有缓存文件并分类"""
        files = os.listdir(self.cache_dir)
        
        pytorch_inputs = [f for f in files if f.startswith('cfm_pytorch_inputs_')]
        mlx_inputs = [f for f in files if f.startswith('cfm_mlx_inputs_')]
        pytorch_outputs = [f for f in files if f.startswith('cfm_pytorch_output_')]
        mlx_outputs = [f for f in files if f.startswith('cfm_mlx_output_')]
        
        # 按时间戳排序
        pytorch_inputs.sort()
        mlx_inputs.sort()
        pytorch_outputs.sort()
        mlx_outputs.sort()
        
        return {
            'pytorch_inputs': pytorch_inputs,
            'mlx_inputs': mlx_inputs,
            'pytorch_outputs': pytorch_outputs,
            'mlx_outputs': mlx_outputs
        }
    
    def analyze_input_data(self, pytorch_file: str, mlx_file: str) -> Dict[str, Any]:
        """分析 PyTorch 和 MLX 输入数据"""
        print(f"\n🔍 分析输入数据:")
        print(f"   PyTorch: {pytorch_file}")
        print(f"   MLX:     {mlx_file}")
        
        # 加载数据
        pytorch_data = self.cache.load_torch_inputs(pytorch_file)
        mlx_data = self.cache.load_mlx_inputs(mlx_file)
        
        analysis = {
            'pytorch': {},
            'mlx': {},
            'comparison': {}
        }
        
        # 分析 PyTorch 数据
        print(f"\n📊 PyTorch 输入数据:")
        for key, value in pytorch_data.items():
            if isinstance(value, torch.Tensor):
                # 检查是否为数值类型（非整数）
                if value.dtype in [torch.float32, torch.float64, torch.float16]:
                    analysis['pytorch'][key] = {
                        'shape': list(value.shape),
                        'dtype': str(value.dtype),
                        'device': str(value.device),
                        'min': float(value.min().item()),
                        'max': float(value.max().item()),
                        'mean': float(value.mean().item()),
                        'std': float(value.std().item())
                    }
                    print(f"   {key}: {value.shape} ({value.dtype})")
                    print(f"      range: [{analysis['pytorch'][key]['min']:.6f}, {analysis['pytorch'][key]['max']:.6f}]")
                    print(f"      mean: {analysis['pytorch'][key]['mean']:.6f}, std: {analysis['pytorch'][key]['std']:.6f}")
                else:
                    # 整数类型，只显示基本信息
                    analysis['pytorch'][key] = {
                        'shape': list(value.shape),
                        'dtype': str(value.dtype),
                        'device': str(value.device),
                        'min': int(value.min().item()),
                        'max': int(value.max().item()),
                        'values': value.tolist() if value.numel() <= 10 else f"<{value.numel()} values>"
                    }
                    print(f"   {key}: {value.shape} ({value.dtype})")
                    print(f"      range: [{analysis['pytorch'][key]['min']}, {analysis['pytorch'][key]['max']}]")
                    if value.numel() <= 10:
                        print(f"      values: {analysis['pytorch'][key]['values']}")
            else:
                analysis['pytorch'][key] = value
                print(f"   {key}: {value}")
        
        # 分析 MLX 数据
        print(f"\n📊 MLX 输入数据:")
        for key, value in mlx_data.items():
            if isinstance(value, mx.array):
                # 检查是否为数值类型（非整数）
                if value.dtype in [mx.float32, mx.float64, mx.float16]:
                    analysis['mlx'][key] = {
                        'shape': list(value.shape),
                        'dtype': str(value.dtype),
                        'min': float(mx.min(value)),
                        'max': float(mx.max(value)),
                        'mean': float(mx.mean(value)),
                        'std': float(mx.std(value))
                    }
                    print(f"   {key}: {value.shape} ({value.dtype})")
                    print(f"      range: [{analysis['mlx'][key]['min']:.6f}, {analysis['mlx'][key]['max']:.6f}]")
                    print(f"      mean: {analysis['mlx'][key]['mean']:.6f}, std: {analysis['mlx'][key]['std']:.6f}")
                else:
                    # 整数类型，只显示基本信息
                    analysis['mlx'][key] = {
                        'shape': list(value.shape),
                        'dtype': str(value.dtype),
                        'min': int(mx.min(value)),
                        'max': int(mx.max(value)),
                        'values': value.tolist() if value.size <= 10 else f"<{value.size} values>"
                    }
                    print(f"   {key}: {value.shape} ({value.dtype})")
                    print(f"      range: [{analysis['mlx'][key]['min']}, {analysis['mlx'][key]['max']}]")
                    if value.size <= 10:
                        print(f"      values: {analysis['mlx'][key]['values']}")
            else:
                analysis['mlx'][key] = value
                print(f"   {key}: {value}")
        
        # 比较数据
        print(f"\n🔍 输入数据比较:")
        for key in pytorch_data.keys():
            if key in mlx_data:
                if isinstance(pytorch_data[key], torch.Tensor) and isinstance(mlx_data[key], mx.array):
                    pytorch_np = pytorch_data[key].detach().cpu().numpy()
                    mlx_np = np.array(mlx_data[key])
                    
                    if pytorch_np.shape == mlx_np.shape:
                        diff = np.abs(pytorch_np - mlx_np)
                        max_diff = np.max(diff)
                        mean_diff = np.mean(diff)
                        
                        analysis['comparison'][key] = {
                            'shape_match': True,
                            'max_diff': float(max_diff),
                            'mean_diff': float(mean_diff),
                            'is_identical': max_diff < 1e-8
                        }
                        
                        print(f"   {key}: ✅ 形状匹配")
                        print(f"      最大差异: {max_diff:.8f}")
                        print(f"      平均差异: {mean_diff:.8f}")
                        if max_diff < 1e-8:
                            print(f"      ✅ 数据完全一致")
                        else:
                            print(f"      ⚠️  数据存在差异")
                    else:
                        analysis['comparison'][key] = {
                            'shape_match': False,
                            'pytorch_shape': list(pytorch_np.shape),
                            'mlx_shape': list(mlx_np.shape)
                        }
                        print(f"   {key}: ❌ 形状不匹配")
                        print(f"      PyTorch: {pytorch_np.shape}")
                        print(f"      MLX:     {mlx_np.shape}")
        
        return analysis
    
    def analyze_output_data(self, pytorch_file: str, mlx_file: str) -> Dict[str, Any]:
        """分析 PyTorch 和 MLX 输出数据"""
        print(f"\n🔍 分析输出数据:")
        print(f"   PyTorch: {pytorch_file}")
        print(f"   MLX:     {mlx_file}")
        
        # 加载数据
        pytorch_data = self.cache.load_torch_inputs(pytorch_file)
        mlx_data = self.cache.load_mlx_inputs(mlx_file)
        
        analysis = {
            'pytorch': {},
            'mlx': {},
            'comparison': {}
        }
        
        # 分析 PyTorch 输出
        if 'output' in pytorch_data:
            output = pytorch_data['output']
            analysis['pytorch']['output'] = {
                'shape': list(output.shape),
                'dtype': str(output.dtype),
                'min': float(output.min().item()),
                'max': float(output.max().item()),
                'mean': float(output.mean().item()),
                'std': float(output.std().item())
            }
            print(f"\n📊 PyTorch 输出数据:")
            print(f"   output: {output.shape} ({output.dtype})")
            print(f"      range: [{analysis['pytorch']['output']['min']:.6f}, {analysis['pytorch']['output']['max']:.6f}]")
            print(f"      mean: {analysis['pytorch']['output']['mean']:.6f}, std: {analysis['pytorch']['output']['std']:.6f}")
        
        # 分析 MLX 输出
        if 'output' in mlx_data:
            output = mlx_data['output']
            analysis['mlx']['output'] = {
                'shape': list(output.shape),
                'dtype': str(output.dtype),
                'min': float(mx.min(output)),
                'max': float(mx.max(output)),
                'mean': float(mx.mean(output)),
                'std': float(mx.std(output))
            }
            print(f"\n📊 MLX 输出数据:")
            print(f"   output: {output.shape} ({output.dtype})")
            print(f"      range: [{analysis['mlx']['output']['min']:.6f}, {analysis['mlx']['output']['max']:.6f}]")
            print(f"      mean: {analysis['mlx']['output']['mean']:.6f}, std: {analysis['mlx']['output']['std']:.6f}")
        
        # 比较输出
        if 'output' in pytorch_data and 'output' in mlx_data:
            pytorch_output = pytorch_data['output']
            mlx_output = mlx_data['output']
            
            pytorch_np = pytorch_output.detach().cpu().numpy()
            mlx_np = np.array(mlx_output)
            
            print(f"\n🔍 输出数据比较:")
            print(f"   PyTorch shape: {pytorch_np.shape}")
            print(f"   MLX shape:     {mlx_np.shape}")
            
            if pytorch_np.shape == mlx_np.shape:
                diff = np.abs(pytorch_np - mlx_np)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                
                analysis['comparison']['output'] = {
                    'shape_match': True,
                    'max_diff': float(max_diff),
                    'mean_diff': float(mean_diff),
                    'is_identical': max_diff < 1e-6
                }
                
                print(f"   ✅ 形状匹配")
                print(f"   最大差异: {max_diff:.8f}")
                print(f"   平均差异: {mean_diff:.8f}")
                
                if max_diff < 1e-6:
                    print(f"   ✅ 输出数据完全一致")
                else:
                    print(f"   ⚠️  输出数据存在差异")
                    print(f"   🔧 建议检查:")
                    print(f"      - 随机种子设置")
                    print(f"      - 权重加载")
                    print(f"      - 数值精度")
            else:
                analysis['comparison']['output'] = {
                    'shape_match': False,
                    'pytorch_shape': list(pytorch_np.shape),
                    'mlx_shape': list(mlx_np.shape)
                }
                print(f"   ❌ 形状不匹配")
                print(f"   🔧 这可能是由于不同的序列长度导致的")
        
        return analysis
    
    def create_test_data_package(self, output_dir: str = "cfm_test_data") -> str:
        """创建测试数据包，包含清晰的输入输出数据"""
        print(f"\n📦 创建测试数据包: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        files = self.list_cache_files()
        
        # 创建数据包
        package = {
            'metadata': {
                'created_at': time.time(),
                'pytorch_inputs_count': len(files['pytorch_inputs']),
                'mlx_inputs_count': len(files['mlx_inputs']),
                'pytorch_outputs_count': len(files['pytorch_outputs']),
                'mlx_outputs_count': len(files['mlx_outputs'])
            },
            'files': files,
            'data': {}
        }
        
        # 处理每一对输入数据
        for i, (pt_file, mlx_file) in enumerate(zip(files['pytorch_inputs'], files['mlx_inputs'])):
            print(f"\n处理输入数据对 {i+1}:")
            print(f"   PyTorch: {pt_file}")
            print(f"   MLX:     {mlx_file}")
            
            # 分析输入数据
            input_analysis = self.analyze_input_data(pt_file, mlx_file)
            package['data'][f'input_pair_{i+1}'] = {
                'pytorch_file': pt_file,
                'mlx_file': mlx_file,
                'analysis': input_analysis
            }
        
        # 处理每一对输出数据
        for i, (pt_file, mlx_file) in enumerate(zip(files['pytorch_outputs'], files['mlx_outputs'])):
            print(f"\n处理输出数据对 {i+1}:")
            print(f"   PyTorch: {pt_file}")
            print(f"   MLX:     {mlx_file}")
            
            # 分析输出数据
            output_analysis = self.analyze_output_data(pt_file, mlx_file)
            package['data'][f'output_pair_{i+1}'] = {
                'pytorch_file': pt_file,
                'mlx_file': mlx_file,
                'analysis': output_analysis
            }
        
        # 保存数据包
        package_file = os.path.join(output_dir, "cfm_test_package.pkl")
        with open(package_file, 'wb') as f:
            pickle.dump(package, f)
        
        print(f"\n✅ 测试数据包已保存: {package_file}")
        print(f"📊 包含 {len(package['data'])} 个数据对")
        
        # 创建简化的测试数据
        self.create_simplified_test_data(package, output_dir)
        
        return package_file
    
    def create_simplified_test_data(self, package: Dict, output_dir: str):
        """创建简化的测试数据，方便直接使用"""
        print(f"\n📝 创建简化测试数据...")
        
        # 创建 PyTorch 测试数据
        pytorch_test_data = {}
        for key, data in package['data'].items():
            if 'input' in key:
                pytorch_file = data['pytorch_file']
                pytorch_data = self.cache.load_torch_inputs(pytorch_file)
                pytorch_test_data[key] = pytorch_data
        
        pytorch_test_file = os.path.join(output_dir, "pytorch_test_data.pkl")
        with open(pytorch_test_file, 'wb') as f:
            pickle.dump(pytorch_test_data, f)
        print(f"   ✅ PyTorch 测试数据: {pytorch_test_file}")
        
        # 创建 MLX 测试数据
        mlx_test_data = {}
        for key, data in package['data'].items():
            if 'input' in key:
                mlx_file = data['mlx_file']
                mlx_data = self.cache.load_mlx_inputs(mlx_file)
                mlx_test_data[key] = mlx_data
        
        mlx_test_file = os.path.join(output_dir, "mlx_test_data.pkl")
        with open(mlx_test_file, 'wb') as f:
            pickle.dump(mlx_test_data, f)
        print(f"   ✅ MLX 测试数据: {mlx_test_file}")
        
        # 创建数据说明文件
        readme_file = os.path.join(output_dir, "README.md")
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write("# CFM 测试数据包\n\n")
            f.write("## 文件说明\n\n")
            f.write("- `cfm_test_package.pkl`: 完整的数据包，包含所有分析和元数据\n")
            f.write("- `pytorch_test_data.pkl`: PyTorch 格式的测试数据\n")
            f.write("- `mlx_test_data.pkl`: MLX 格式的测试数据\n\n")
            f.write("## 使用方法\n\n")
            f.write("```python\n")
            f.write("# 加载 PyTorch 测试数据\n")
            f.write("import pickle\n")
            f.write("with open('pytorch_test_data.pkl', 'rb') as f:\n")
            f.write("    pytorch_data = pickle.load(f)\n\n")
            f.write("# 加载 MLX 测试数据\n")
            f.write("with open('mlx_test_data.pkl', 'rb') as f:\n")
            f.write("    mlx_data = pickle.load(f)\n")
            f.write("```\n\n")
            f.write("## 数据对数量\n\n")
            f.write(f"- 输入数据对: {package['metadata']['pytorch_inputs_count']}\n")
            f.write(f"- 输出数据对: {package['metadata']['pytorch_outputs_count']}\n")
        
        print(f"   ✅ 说明文件: {readme_file}")
    
    def run_full_analysis(self):
        """运行完整分析"""
        print("🔧 CFM 缓存数据分析工具")
        print("=" * 50)
        
        # 列出缓存文件
        files = self.list_cache_files()
        
        print(f"\n📁 缓存文件统计:")
        print(f"   PyTorch 输入: {len(files['pytorch_inputs'])} 个")
        print(f"   MLX 输入:     {len(files['mlx_inputs'])} 个")
        print(f"   PyTorch 输出: {len(files['pytorch_outputs'])} 个")
        print(f"   MLX 输出:     {len(files['mlx_outputs'])} 个")
        
        if not files['pytorch_inputs'] or not files['mlx_inputs']:
            print("❌ 没有找到足够的缓存文件进行分析")
            return
        
        # 创建测试数据包
        package_file = self.create_test_data_package()
        
        print(f"\n🎉 分析完成!")
        print(f"📦 测试数据包: {package_file}")
        print(f"📁 输出目录: cfm_test_data/")


def main():
    """主函数"""
    analyzer = CFMCacheAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
