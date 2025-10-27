#!/usr/bin/env python3
"""
特定权重组件对比测试

专门测试 s2mel 中关键组件的权重差异
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append('/Users/bailin/index-tts')

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from indextts.utils.mlx_cache import MLXModelCache


class ComponentWeightTester:
    """组件权重测试器"""
    
    def __init__(self):
        self.pytorch_weights = {}
        self.mlx_weights = {}
    
    def load_weights(self, pytorch_checkpoint: str, mlx_model_name: str = "s2mel_cfm"):
        """加载权重"""
        print(f"📦 Loading weights...")
        
        # 加载 PyTorch 权重
        checkpoint = torch.load(pytorch_checkpoint, map_location='cpu')
        if 'net' in checkpoint:
            for key in checkpoint['net']:
                for param_name, param_value in checkpoint['net'][key].items():
                    self.pytorch_weights[f"{key}.{param_name}"] = param_value.cpu().numpy()
        else:
            self.pytorch_weights = {k: v.cpu().numpy() for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
        
        print(f"   PyTorch weights: {len(self.pytorch_weights)}")
        
        # 加载 MLX 权重
        if MLX_AVAILABLE:
            mlx_cache = MLXModelCache(cache_dir="checkpoints/mlx")
            mlx_state = mlx_cache.load_from_cache(mlx_model_name)
            if mlx_state:
                self.mlx_weights = {k: np.array(v) for k, v in mlx_state.items() if isinstance(v, mx.array)}
                print(f"   MLX weights: {len(self.mlx_weights)}")
            else:
                print("   ❌ No MLX cache found")
        else:
            print("   ⚠️ MLX not available")
    
    def test_cfm_weights(self):
        """测试 CFM 权重"""
        print(f"\n🔍 Testing CFM weights...")
        
        cfm_keys = [k for k in self.pytorch_weights.keys() if 'cfm' in k.lower()]
        mlx_cfm_keys = [k for k in self.mlx_weights.keys() if 'cfm' in k.lower()]
        
        print(f"   PyTorch CFM keys: {len(cfm_keys)}")
        print(f"   MLX CFM keys: {len(mlx_cfm_keys)}")
        
        # 检查关键 CFM 组件
        key_components = [
            'cfm.estimator.x_embedder',
            'cfm.estimator.t_embedder',
            'cfm.estimator.cond_projection',
            'cfm.estimator.transformer',
            'cfm.estimator.final_layer'
        ]
        
        for component in key_components:
            pt_keys = [k for k in cfm_keys if component in k]
            mlx_keys = [k for k in mlx_cfm_keys if component in k]
            
            print(f"   {component}:")
            print(f"      PyTorch: {len(pt_keys)} keys")
            print(f"      MLX: {len(mlx_keys)} keys")
            
            # 检查共同键的差异
            common_keys = set(pt_keys) & set(mlx_keys)
            if common_keys:
                max_diff = 0.0
                for key in common_keys:
                    pt_weight = self.pytorch_weights[key]
                    mlx_weight = self.mlx_weights[key]
                    if pt_weight.shape == mlx_weight.shape:
                        diff = np.max(np.abs(pt_weight - mlx_weight))
                        max_diff = max(max_diff, diff)
                
                print(f"      Max difference: {max_diff:.8f}")
                if max_diff < 1e-6:
                    print(f"      ✅ All weights match")
                else:
                    print(f"      ⚠️ Significant differences found")
    
    def test_transformer_weights(self):
        """测试 Transformer 权重"""
        print(f"\n🔍 Testing Transformer weights...")
        
        # 检查 Transformer 层权重
        transformer_keys = [k for k in self.pytorch_weights.keys() if 'transformer.layers' in k]
        mlx_transformer_keys = [k for k in self.mlx_weights.keys() if 'transformer.layers' in k]
        
        print(f"   PyTorch transformer keys: {len(transformer_keys)}")
        print(f"   MLX transformer keys: {len(mlx_transformer_keys)}")
        
        # 检查每一层的权重
        layer_indices = set()
        for key in transformer_keys:
            if 'transformer.layers.' in key:
                layer_idx = key.split('transformer.layers.')[1].split('.')[0]
                layer_indices.add(layer_idx)
        
        print(f"   Found {len(layer_indices)} transformer layers")
        
        for layer_idx in sorted(layer_indices):
            layer_prefix = f'transformer.layers.{layer_idx}'
            pt_layer_keys = [k for k in transformer_keys if layer_prefix in k]
            mlx_layer_keys = [k for k in mlx_transformer_keys if layer_prefix in k]
            
            print(f"   Layer {layer_idx}:")
            print(f"      PyTorch: {len(pt_layer_keys)} keys")
            print(f"      MLX: {len(mlx_layer_keys)} keys")
            
            # 检查关键权重
            key_weights = ['attention.wqkv.weight', 'attention.wo.weight', 'feed_forward.w1.weight', 'feed_forward.w2.weight']
            for weight_name in key_weights:
                pt_key = f'{layer_prefix}.{weight_name}'
                mlx_key = f'{layer_prefix}.{weight_name}'
                
                if pt_key in self.pytorch_weights and mlx_key in self.mlx_weights:
                    pt_weight = self.pytorch_weights[pt_key]
                    mlx_weight = self.mlx_weights[mlx_key]
                    
                    if pt_weight.shape == mlx_weight.shape:
                        diff = np.max(np.abs(pt_weight - mlx_weight))
                        status = "✅" if diff < 1e-6 else "⚠️"
                        print(f"      {status} {weight_name}: diff={diff:.8f}")
                    else:
                        print(f"      ❌ {weight_name}: shape mismatch {pt_weight.shape} vs {mlx_weight.shape}")
    
    def test_embedder_weights(self):
        """测试嵌入层权重"""
        print(f"\n🔍 Testing embedder weights...")
        
        embedder_types = ['x_embedder', 't_embedder', 'cond_projection', 'cond_embedder']
        
        for embedder_type in embedder_types:
            pt_keys = [k for k in self.pytorch_weights.keys() if embedder_type in k]
            mlx_keys = [k for k in self.mlx_weights.keys() if embedder_type in k]
            
            print(f"   {embedder_type}:")
            print(f"      PyTorch: {len(pt_keys)} keys")
            print(f"      MLX: {len(mlx_keys)} keys")
            
            # 检查权重差异
            common_keys = set(pt_keys) & set(mlx_keys)
            if common_keys:
                max_diff = 0.0
                for key in common_keys:
                    pt_weight = self.pytorch_weights[key]
                    mlx_weight = self.mlx_weights[key]
                    if pt_weight.shape == mlx_weight.shape:
                        diff = np.max(np.abs(pt_weight - mlx_weight))
                        max_diff = max(max_diff, diff)
                
                print(f"      Max difference: {max_diff:.8f}")
                if max_diff < 1e-6:
                    print(f"      ✅ All weights match")
                else:
                    print(f"      ⚠️ Significant differences found")
    
    def test_final_layer_weights(self):
        """测试最终层权重"""
        print(f"\n🔍 Testing final layer weights...")
        
        final_layer_keys = [k for k in self.pytorch_weights.keys() if 'final_layer' in k or 'final_mlp' in k]
        mlx_final_layer_keys = [k for k in self.mlx_weights.keys() if 'final_layer' in k or 'final_mlp' in k]
        
        print(f"   PyTorch final layer keys: {len(final_layer_keys)}")
        print(f"   MLX final layer keys: {len(mlx_final_layer_keys)}")
        
        # 检查最终层组件
        final_components = ['final_mlp', 'final_layer', 'wavenet', 'conv1', 'conv2']
        
        for component in final_components:
            pt_keys = [k for k in final_layer_keys if component in k]
            mlx_keys = [k for k in mlx_final_layer_keys if component in k]
            
            if pt_keys or mlx_keys:
                print(f"   {component}:")
                print(f"      PyTorch: {len(pt_keys)} keys")
                print(f"      MLX: {len(mlx_keys)} keys")
                
                # 检查权重差异
                common_keys = set(pt_keys) & set(mlx_keys)
                if common_keys:
                    max_diff = 0.0
                    for key in common_keys:
                        pt_weight = self.pytorch_weights[key]
                        mlx_weight = self.mlx_weights[key]
                        if pt_weight.shape == mlx_weight.shape:
                            diff = np.max(np.abs(pt_weight - mlx_weight))
                            max_diff = max(max_diff, diff)
                    
                    print(f"      Max difference: {max_diff:.8f}")
                    if max_diff < 1e-6:
                        print(f"      ✅ All weights match")
                    else:
                        print(f"      ⚠️ Significant differences found")
    
    def run_all_tests(self, pytorch_checkpoint: str, mlx_model_name: str = "s2mel_cfm"):
        """运行所有测试"""
        print(f"🧪 Component Weight Testing Suite")
        print(f"=" * 50)
        
        # 加载权重
        self.load_weights(pytorch_checkpoint, mlx_model_name)
        
        if not self.pytorch_weights:
            print("❌ No PyTorch weights loaded")
            return
        
        if not self.mlx_weights:
            print("❌ No MLX weights loaded")
            return
        
        # 运行各种测试
        self.test_cfm_weights()
        self.test_transformer_weights()
        self.test_embedder_weights()
        self.test_final_layer_weights()
        
        print(f"\n🎉 Component testing completed!")


def main():
    """主函数"""
    pytorch_checkpoint = "checkpoints/s2mel.pth"
    
    if not os.path.exists(pytorch_checkpoint):
        print(f"❌ PyTorch checkpoint not found: {pytorch_checkpoint}")
        return
    
    tester = ComponentWeightTester()
    tester.run_all_tests(pytorch_checkpoint)


if __name__ == "__main__":
    main()
