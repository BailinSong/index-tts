#!/usr/bin/env python3
"""
CFM 一致性分析工具
使用同一个 PyTorch CFM 前级缓存，转换为 MLX 格式进行一致性测试
"""

import os
import pickle
import numpy as np
import torch
import mlx.core as mx
from typing import Dict, Any, List, Tuple
import time
from cfm_data_cache import CFMDataCache

class CFMConsistencyAnalyzer:
    """CFM 一致性分析器"""
    
    def __init__(self, cache_dir: str = "cfm_baseline_cache"):
        self.cache = CFMDataCache(cache_dir)
        self.cache_dir = cache_dir
    
    def load_pytorch_inputs(self) -> Dict[str, Any]:
        """加载最新的 PyTorch CFM 输入缓存"""
        files = [f for f in os.listdir(self.cache_dir) if f.startswith('cfm_pytorch_inputs_')]
        if not files:
            raise FileNotFoundError("没有找到 PyTorch CFM 输入缓存文件")
        
        # 选择最新的文件
        files.sort()
        latest_file = files[-1]
        
        print(f"📁 加载 PyTorch CFM 输入缓存: {latest_file}")
        pytorch_inputs = self.cache.load_torch_inputs(latest_file)
        
        # 尝试加载 GPT 输出
        gpt_files = [f for f in os.listdir('.') if f.startswith('gpt_outputs_')]
        if gpt_files:
            gpt_files.sort()
            latest_gpt_file = gpt_files[-1]
            print(f"📁 加载 GPT 输出: {latest_gpt_file}")
            with open(latest_gpt_file, 'rb') as f:
                gpt_outputs = pickle.load(f)
            pytorch_inputs.update(gpt_outputs)
        
        return pytorch_inputs
    
    def convert_to_mlx_inputs(self, pytorch_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """将 PyTorch 输入转换为 MLX 格式"""
        print("🔄 将 PyTorch 输入转换为 MLX 格式...")
        mlx_inputs = self.cache.convert_to_mlx(pytorch_inputs)
        return mlx_inputs
    
    def test_cfm_consistency(self, pytorch_inputs: Dict[str, Any], mlx_inputs: Dict[str, Any]):
        """测试 CFM 一致性"""
        print("\n🔍 CFM 一致性测试")
        print("=" * 60)
        
        # 检查输入数据一致性
        print("\n📊 输入数据一致性检查:")
        self._check_input_consistency(pytorch_inputs, mlx_inputs)
        
        # 这里可以添加实际的 CFM 推理测试
        # 由于需要加载模型，这里先跳过
        print("\n⚠️  注意: 需要加载 CFM 模型进行实际推理测试")
        print("   建议使用 benchmark_v1_baseline.py 生成一致的输入缓存")
    
    def _check_input_consistency(self, pytorch_inputs: Dict[str, Any], mlx_inputs: Dict[str, Any]):
        """检查输入数据一致性"""
        print("   PyTorch 输入数据:")
        for key, value in pytorch_inputs.items():
            if isinstance(value, torch.Tensor):
                print(f"     {key}: {value.shape} ({value.dtype})")
                if value.dtype in [torch.float32, torch.float64, torch.float16]:
                    print(f"       range: [{value.min().item():.6f}, {value.max().item():.6f}]")
                    print(f"       mean: {value.mean().item():.6f}, std: {value.std().item():.6f}")
                else:
                    print(f"       range: [{value.min().item()}, {value.max().item()}]")
                    print(f"       values: {value.tolist()}")
            else:
                print(f"     {key}: {value}")
        
        print("\n   MLX 输入数据:")
        for key, value in mlx_inputs.items():
            if isinstance(value, mx.array):
                print(f"     {key}: {value.shape} ({value.dtype})")
                print(f"       range: [{mx.min(value):.6f}, {mx.max(value):.6f}]")
                print(f"       mean: {mx.mean(value):.6f}, std: {mx.std(value):.6f}")
            else:
                print(f"     {key}: {value}")
        
        print("\n   🔍 数据一致性比较:")
        for key in pytorch_inputs.keys():
            if key in mlx_inputs:
                if isinstance(pytorch_inputs[key], torch.Tensor) and isinstance(mlx_inputs[key], mx.array):
                    pytorch_np = pytorch_inputs[key].detach().cpu().numpy()
                    mlx_np = np.array(mlx_inputs[key])
                    
                    if pytorch_np.shape == mlx_np.shape:
                        diff = np.abs(pytorch_np - mlx_np)
                        max_diff = np.max(diff)
                        mean_diff = np.mean(diff)
                        
                        print(f"     {key}: ✅ 形状匹配")
                        print(f"       最大差异: {max_diff:.8f}")
                        print(f"       平均差异: {mean_diff:.8f}")
                        if max_diff < 1e-8:
                            print(f"       ✅ 数据完全一致")
                        else:
                            print(f"       ⚠️  数据存在差异")
                    else:
                        print(f"     {key}: ❌ 形状不匹配")
                        print(f"       PyTorch: {pytorch_np.shape}")
                        print(f"       MLX:     {mlx_np.shape}")
                else:
                    if pytorch_inputs[key] == mlx_inputs[key]:
                        print(f"     {key}: ✅ 数据一致")
                    else:
                        print(f"     {key}: ❌ 数据不一致")
                        print(f"       PyTorch: {pytorch_inputs[key]}")
                        print(f"       MLX:     {mlx_inputs[key]}")
    
    def create_consistent_test_data(self, pytorch_inputs: Dict[str, Any], mlx_inputs: Dict[str, Any]):
        """创建一致的测试数据包"""
        print("\n📦 创建一致的测试数据包...")
        
        # 创建测试数据包
        test_data = {
            'pytorch_inputs': pytorch_inputs,
            'mlx_inputs': mlx_inputs,
            'metadata': {
                'created_at': time.time(),
                'description': '使用同一个 PyTorch CFM 前级缓存生成的一致测试数据',
                'pytorch_shapes': {k: list(v.shape) if isinstance(v, torch.Tensor) else str(v) 
                                 for k, v in pytorch_inputs.items()},
                'mlx_shapes': {k: list(v.shape) if isinstance(v, mx.array) else str(v) 
                              for k, v in mlx_inputs.items()}
            }
        }
        
        # 保存测试数据
        os.makedirs("cfm_consistent_test", exist_ok=True)
        test_file = "cfm_consistent_test/consistent_test_data.pkl"
        
        with open(test_file, 'wb') as f:
            pickle.dump(test_data, f)
        
        print(f"   ✅ 一致测试数据已保存: {test_file}")
        
        # 创建说明文件
        readme_file = "cfm_consistent_test/README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write("# CFM 一致测试数据\n\n")
            f.write("## 说明\n\n")
            f.write("这个测试数据包使用同一个 PyTorch CFM 前级缓存生成，确保 PyTorch 和 MLX 使用完全相同的输入数据。\n\n")
            f.write("## 文件说明\n\n")
            f.write("- `consistent_test_data.pkl`: 包含 PyTorch 和 MLX 格式的相同输入数据\n\n")
            f.write("## 使用方法\n\n")
            f.write("```python\n")
            f.write("import pickle\n")
            f.write("with open('consistent_test_data.pkl', 'rb') as f:\n")
            f.write("    data = pickle.load(f)\n")
            f.write("    pytorch_inputs = data['pytorch_inputs']\n")
            f.write("    mlx_inputs = data['mlx_inputs']\n")
            f.write("```\n\n")
            f.write("## 数据形状\n\n")
            f.write("### PyTorch 输入:\n")
            for key, shape in test_data['metadata']['pytorch_shapes'].items():
                f.write(f"- {key}: {shape}\n")
            f.write("\n### MLX 输入:\n")
            for key, shape in test_data['metadata']['mlx_shapes'].items():
                f.write(f"- {key}: {shape}\n")
        
        print(f"   ✅ 说明文件已保存: {readme_file}")
        
        return test_file
    
    def run_consistency_analysis(self):
        """运行一致性分析"""
        print("🔧 CFM 一致性分析工具")
        print("=" * 50)
        
        try:
            # 加载 PyTorch CFM 输入缓存
            pytorch_inputs = self.load_pytorch_inputs()
            
            # 转换为 MLX 格式
            mlx_inputs = self.convert_to_mlx_inputs(pytorch_inputs)
            
            # 测试一致性
            self.test_cfm_consistency(pytorch_inputs, mlx_inputs)
            
            # 创建一致的测试数据包
            test_file = self.create_consistent_test_data(pytorch_inputs, mlx_inputs)
            
            print(f"\n🎉 一致性分析完成!")
            print(f"📦 测试数据包: {test_file}")
            print(f"📁 输出目录: cfm_consistent_test/")
            
        except Exception as e:
            print(f"❌ 分析失败: {e}")
            print("💡 建议先运行 benchmark_v1_baseline.py 生成 PyTorch CFM 输入缓存")


def main():
    """主函数"""
    analyzer = CFMConsistencyAnalyzer()
    analyzer.run_consistency_analysis()


if __name__ == "__main__":
    main()
