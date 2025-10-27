#!/usr/bin/env python3
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
