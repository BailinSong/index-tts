#!/usr/bin/env python3
"""
生产环境部署脚本
确保 MLX 模型在生产环境中正确部署和运行
"""

import os
import sys
import yaml
import time
import json
from pathlib import Path

def check_production_requirements():
    """检查生产环境要求"""
    print("🔍 检查生产环境要求...")
    
    requirements = {
        "mlx": False,
        "torch": False,
        "numpy": False,
        "yaml": False,
        "config_file": False,
        "model_files": False
    }
    
    # 检查 Python 包
    try:
        import mlx.core as mx
        requirements["mlx"] = True
        print("✅ MLX 可用")
    except ImportError:
        print("❌ MLX 不可用")
    
    try:
        import torch
        requirements["torch"] = True
        print("✅ PyTorch 可用")
    except ImportError:
        print("❌ PyTorch 不可用")
    
    try:
        import numpy as np
        requirements["numpy"] = True
        print("✅ NumPy 可用")
    except ImportError:
        print("❌ NumPy 不可用")
    
    try:
        import yaml
        requirements["yaml"] = True
        print("✅ PyYAML 可用")
    except ImportError:
        print("❌ PyYAML 不可用")
    
    # 检查配置文件
    if os.path.exists("checkpoints/config.yaml"):
        requirements["config_file"] = True
        print("✅ 配置文件存在")
    else:
        print("❌ 配置文件不存在")
    
    # 检查模型文件
    model_files = [
        "checkpoints/s2mel.pth",
        "checkpoints/mlx/s2mel.npz"
    ]
    
    missing_files = []
    for file_path in model_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} 存在")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path} 不存在")
    
    if not missing_files:
        requirements["model_files"] = True
    
    # 总结
    total_requirements = len(requirements)
    met_requirements = sum(requirements.values())
    
    print(f"\n📊 环境检查结果:")
    print(f"  总要求: {total_requirements}")
    print(f"  满足: {met_requirements}")
    print(f"  缺失: {total_requirements - met_requirements}")
    
    if met_requirements == total_requirements:
        print("✅ 生产环境要求全部满足")
        return True
    else:
        print("❌ 生产环境要求未完全满足")
        return False

def deploy_mlx_model():
    """部署 MLX 模型"""
    print("\n🚀 部署 MLX 模型...")
    
    try:
        from indextts.s2mel.modules.mlx_cfm import MLXCFM
        from indextts.utils.mlx_production_utils import fix_mlx_model_weights, test_mlx_model_production
        
        # 加载配置
        with open('checkpoints/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # 使用正确的配置路径
        config['DiT'] = config['s2mel']['DiT']
        config['style_encoder'] = config['s2mel']['style_encoder']
        config['wavenet'] = config['s2mel']['wavenet']
        
        # 创建 MLX 模型
        print("🏗️ 创建 MLX 模型...")
        mlx_model = MLXCFM(config)
        
        # 修复权重
        print("🔧 修复权重...")
        pytorch_weights_path = 'checkpoints/s2mel.pth'
        fix_success = fix_mlx_model_weights(mlx_model, pytorch_weights_path)
        
        if not fix_success:
            print("❌ 权重修复失败")
            return False
        
        # 测试性能
        print("⚡ 测试性能...")
        test_result = test_mlx_model_production(mlx_model)
        
        if not test_result['success']:
            print("❌ 性能测试失败")
            return False
        
        print("✅ MLX 模型部署成功")
        return True
        
    except Exception as e:
        print(f"❌ MLX 模型部署失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_production_monitor():
    """创建生产环境监控"""
    print("\n📊 创建生产环境监控...")
    
    monitor_code = '''#!/usr/bin/env python3
"""
生产环境监控脚本
监控 MLX 模型的性能和状态
"""

import time
import json
import psutil
import mlx.core as mx
from datetime import datetime

class MLXProductionMonitor:
    def __init__(self, model, config_path="mlx_production_config.yaml"):
        self.model = model
        self.config_path = config_path
        self.monitoring_data = []
        
    def get_system_info(self):
        """获取系统信息"""
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "mlx_memory_usage": mx.metal.memory_usage() if hasattr(mx.metal, 'memory_usage') else None
        }
    
    def test_model_performance(self):
        """测试模型性能"""
        import time
        
        # 创建测试输入
        batch_size = 2
        seq_len = 100
        input_dim = 512
        
        test_input = mx.random.normal((batch_size, seq_len, input_dim))
        
        # 预热
        for _ in range(5):
            _ = self.model.estimator.cond_projection(test_input)
        
        # 性能测试
        start_time = time.time()
        for _ in range(100):
            output = self.model.estimator.cond_projection(test_input)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        throughput = batch_size * seq_len / avg_time
        
        return {
            "avg_time_ms": avg_time * 1000,
            "throughput_tokens_per_sec": throughput,
            "output_shape": output.shape,
            "output_range": [float(output.min()), float(output.max())]
        }
    
    def monitor_cycle(self):
        """执行一次监控循环"""
        try:
            system_info = self.get_system_info()
            performance_info = self.test_model_performance()
            
            monitor_data = {
                **system_info,
                **performance_info
            }
            
            self.monitoring_data.append(monitor_data)
            
            # 只保留最近100条记录
            if len(self.monitoring_data) > 100:
                self.monitoring_data = self.monitoring_data[-100:]
            
            print(f"📊 监控数据: CPU={system_info['cpu_percent']:.1f}%, "
                  f"内存={system_info['memory_percent']:.1f}%, "
                  f"吞吐量={performance_info['throughput_tokens_per_sec']:.0f} tokens/s")
            
            return monitor_data
            
        except Exception as e:
            print(f"❌ 监控失败: {e}")
            return None
    
    def save_monitoring_data(self, filepath="mlx_production_monitor.json"):
        """保存监控数据"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.monitoring_data, f, indent=2)
            print(f"✅ 监控数据已保存到: {filepath}")
        except Exception as e:
            print(f"❌ 保存监控数据失败: {e}")
    
    def start_monitoring(self, duration_minutes=60, interval_seconds=30):
        """开始监控"""
        print(f"🔍 开始监控 {duration_minutes} 分钟，间隔 {interval_seconds} 秒...")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        while time.time() < end_time:
            self.monitor_cycle()
            time.sleep(interval_seconds)
        
        print("✅ 监控完成")
        self.save_monitoring_data()

if __name__ == "__main__":
    print("📊 MLX 生产环境监控")
    print("=" * 50)
    print("使用方法:")
    print("  python mlx_production_monitor.py")
    print("  # 或者导入使用:")
    print("  from mlx_production_monitor import MLXProductionMonitor")
'''
    
    with open('mlx_production_monitor.py', 'w', encoding='utf-8') as f:
        f.write(monitor_code)
    
    print("✅ 生产环境监控已创建: mlx_production_monitor.py")

def create_deployment_script():
    """创建部署脚本"""
    print("\n📦 创建部署脚本...")
    
    deployment_script = '''#!/bin/bash
# MLX 生产环境部署脚本

echo "🚀 MLX 生产环境部署"
echo "=================="

# 检查 Python 环境
echo "🔍 检查 Python 环境..."
python --version

# 检查依赖
echo "🔍 检查依赖..."
python -c "import mlx.core; print('✅ MLX 可用')" || echo "❌ MLX 不可用"
python -c "import torch; print('✅ PyTorch 可用')" || echo "❌ PyTorch 不可用"

# 运行环境检查
echo "🔍 运行环境检查..."
python -c "
from apply_production_fixes import check_production_requirements
check_production_requirements()
"

# 部署模型
echo "🚀 部署模型..."
python -c "
from apply_production_fixes import deploy_mlx_model
deploy_mlx_model()
"

# 运行测试
echo "🧪 运行测试..."
python automated_tests.py

echo "✅ 部署完成!"
'''
    
    with open('deploy_mlx_production.sh', 'w', encoding='utf-8') as f:
        f.write(deployment_script)
    
    # 设置执行权限
    os.chmod('deploy_mlx_production.sh', 0o755)
    
    print("✅ 部署脚本已创建: deploy_mlx_production.sh")

def main():
    """主函数"""
    print("🚀 MLX 生产环境部署")
    print("=" * 50)
    
    # 1. 检查生产环境要求
    if not check_production_requirements():
        print("❌ 生产环境要求未满足，请先解决依赖问题")
        return
    
    # 2. 部署 MLX 模型
    if not deploy_mlx_model():
        print("❌ MLX 模型部署失败")
        return
    
    # 3. 创建生产环境监控
    create_production_monitor()
    
    # 4. 创建部署脚本
    create_deployment_script()
    
    print("\n🎉 MLX 生产环境部署完成!")
    print("\n📋 部署文件:")
    print("  - indextts/utils/mlx_production_utils.py (生产工具)")
    print("  - mlx_production_config.yaml (生产配置)")
    print("  - mlx_production_monitor.py (生产监控)")
    print("  - deploy_mlx_production.sh (部署脚本)")
    
    print("\n🚀 使用方法:")
    print("  1. 运行部署脚本: ./deploy_mlx_production.sh")
    print("  2. 启动监控: python mlx_production_monitor.py")
    print("  3. 查看配置: cat mlx_production_config.yaml")

if __name__ == "__main__":
    main()
