"""
设备检测和优化工具模块
支持 CUDA、MPS、MLX、XPU 等多种设备
"""

import os
import torch
from typing import Optional, Dict, Any


def detect_best_device() -> str:
    """
    自动检测最佳可用设备
    
    Returns:
        str: 设备名称 (cuda, mps, mlx, xpu, cpu)
    """
    # 检查 CUDA
    if torch.cuda.is_available():
        return "cuda"
    
    # 检查 MLX (Apple Silicon 优化)
    try:
        import mlx.core as mx
        if mx.metal.is_available():
            return "mlx"
    except ImportError:
        pass
    
    # 检查 MPS (Apple Silicon)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        return "mps"
    
    # 检查 XPU (Intel)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    
    # 默认使用 CPU
    return "cpu"


def get_device_config(device: str) -> Dict[str, Any]:
    """
    获取设备特定的配置
    
    Args:
        device (str): 设备名称
        
    Returns:
        Dict[str, Any]: 设备配置字典
    """
    configs = {
        "cuda": {
            "supports_fp16": True,
            "supports_cuda_kernel": True,
            "supports_deepspeed": True,
            "memory_cleanup_func": torch.cuda.empty_cache,
            "synchronize_func": torch.cuda.synchronize,
            "device_type": "gpu"
        },
        "mlx": {
            "supports_fp16": True,
            "supports_cuda_kernel": False,
            "supports_deepspeed": False,
            "memory_cleanup_func": None,  # MLX 自动管理内存
            "synchronize_func": None,
            "device_type": "gpu"
        },
        "mps": {
            "supports_fp16": False,  # MPS 上 FP16 可能性能更差
            "supports_cuda_kernel": False,
            "supports_deepspeed": False,
            "memory_cleanup_func": torch.mps.empty_cache,
            "synchronize_func": None,
            "device_type": "gpu"
        },
        "xpu": {
            "supports_fp16": True,
            "supports_cuda_kernel": False,
            "supports_deepspeed": False,
            "memory_cleanup_func": torch.xpu.empty_cache,
            "synchronize_func": torch.xpu.synchronize,
            "device_type": "gpu"
        },
        "cpu": {
            "supports_fp16": False,
            "supports_cuda_kernel": False,
            "supports_deepspeed": False,
            "memory_cleanup_func": None,
            "synchronize_func": None,
            "device_type": "cpu"
        }
    }
    
    return configs.get(device, configs["cpu"])


def optimize_device_settings(device: str, use_fp16: Optional[bool] = None, 
                           use_cuda_kernel: Optional[bool] = None,
                           use_deepspeed: Optional[bool] = None) -> Dict[str, Any]:
    """
    根据设备类型优化设置
    
    Args:
        device (str): 设备名称
        use_fp16 (Optional[bool]): 是否使用 FP16
        use_cuda_kernel (Optional[bool]): 是否使用 CUDA 内核
        use_deepspeed (Optional[bool]): 是否使用 DeepSpeed
        
    Returns:
        Dict[str, Any]: 优化后的设置
    """
    config = get_device_config(device)
    
    # 自动优化设置
    if use_fp16 is None:
        use_fp16 = config["supports_fp16"]
    
    if use_cuda_kernel is None:
        use_cuda_kernel = config["supports_cuda_kernel"]
        
    if use_deepspeed is None:
        use_deepspeed = config["supports_deepspeed"]
    
    return {
        "device": device,
        "use_fp16": use_fp16,
        "use_cuda_kernel": use_cuda_kernel,
        "use_deepspeed": use_deepspeed,
        "config": config
    }


def cleanup_device_memory(device: str):
    """
    清理设备内存
    
    Args:
        device (str): 设备名称
    """
    config = get_device_config(device)
    cleanup_func = config.get("memory_cleanup_func")
    
    if cleanup_func:
        try:
            cleanup_func()
        except Exception as e:
            print(f"警告: 清理 {device} 内存时出错: {e}")


def synchronize_device(device: str):
    """
    同步设备操作
    
    Args:
        device (str): 设备名称
    """
    config = get_device_config(device)
    sync_func = config.get("synchronize_func")
    
    if sync_func:
        try:
            sync_func()
        except Exception as e:
            print(f"警告: 同步 {device} 时出错: {e}")


def get_device_info() -> Dict[str, Any]:
    """
    获取详细的设备信息
    
    Returns:
        Dict[str, Any]: 设备信息字典
    """
    info = {
        "available_devices": [],
        "best_device": detect_best_device(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch, "mps") and torch.backends.mps.is_available(),
        "xpu_available": hasattr(torch, "xpu") and torch.xpu.is_available(),
    }
    
    # 检查 CUDA
    if torch.cuda.is_available():
        info["available_devices"].append("cuda")
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_current_device"] = torch.cuda.current_device()
        info["cuda_device_name"] = torch.cuda.get_device_name()
    
    # 检查 MPS
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        info["available_devices"].append("mps")
    
    # 检查 XPU
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        info["available_devices"].append("xpu")
    
    # 检查 MLX
    try:
        import mlx.core as mx
        if mx.metal.is_available():
            info["available_devices"].append("mlx")
            info["mlx_available"] = True
        else:
            info["mlx_available"] = False
    except ImportError:
        info["mlx_available"] = False
    
    # 总是包含 CPU
    info["available_devices"].append("cpu")
    
    return info


def print_device_info():
    """打印设备信息"""
    info = get_device_info()
    
    print("=== IndexTTS 设备信息 ===")
    print(f"推荐设备: {info['best_device']}")
    print(f"PyTorch 版本: {info['torch_version']}")
    print(f"可用设备: {', '.join(info['available_devices'])}")
    
    if info["cuda_available"]:
        print(f"CUDA 设备数量: {info['cuda_device_count']}")
        print(f"当前 CUDA 设备: {info['cuda_current_device']}")
        print(f"CUDA 设备名称: {info['cuda_device_name']}")
    
    if info["mps_available"]:
        print("MPS (Apple Silicon) 可用")
    
    if info["xpu_available"]:
        print("XPU (Intel) 可用")
    
    if info["mlx_available"]:
        print("MLX (Apple Silicon 优化) 可用")
    
    print("========================")


if __name__ == "__main__":
    print_device_info()
