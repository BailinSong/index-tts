#!/usr/bin/env python3
"""
测试CFM输入格式要求
"""

import sys
import os
import torch
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from indextts.infer_v2 import IndexTTS2

def test_cfm_input_format():
    """测试CFM输入格式要求"""
    print("=== 测试CFM输入格式要求 ===\n")
    
    # 初始化TTS系统
    print("1. 初始化TTS系统...")
    tts = IndexTTS2()
    
    # 获取CFM estimator
    pytorch_cfm = tts.s2mel.models.cfm
    pytorch_estimator = pytorch_cfm.estimator
    
    print(f"   CFM estimator类型: {type(pytorch_estimator)}")
    
    # 检查forward方法的签名
    import inspect
    sig = inspect.signature(pytorch_estimator.forward)
    print(f"   Forward方法签名: {sig}")
    
    # 创建测试输入
    print("\n2. 创建测试输入...")
    batch_size = 1
    seq_len = 464
    feature_dim = 512
    mel_dim = 80
    
    # 创建测试张量
    x = torch.randn(batch_size, seq_len, feature_dim)
    prompt_x = torch.randn(batch_size, seq_len, feature_dim)
    x_lens = torch.tensor([seq_len])
    t = torch.zeros(batch_size)
    style = torch.randn(batch_size, 192)
    cond = torch.randn(batch_size, seq_len, feature_dim)
    
    print(f"   输入形状:")
    print(f"   x: {x.shape}")
    print(f"   prompt_x: {prompt_x.shape}")
    print(f"   x_lens: {x_lens.shape}")
    print(f"   t: {t.shape}")
    print(f"   style: {style.shape}")
    print(f"   cond: {cond.shape}")
    
    # 测试CFM调用
    print("\n3. 测试CFM调用...")
    try:
        with torch.no_grad():
            output = pytorch_estimator(
                x=x,
                prompt_x=prompt_x,
                x_lens=x_lens,
                t=t,
                style=style,
                cond=cond,
                mask_content=False
            )
        
        print(f"   ✅ CFM调用成功")
        print(f"   输出形状: {output.shape}")
        print(f"   输出范围: [{output.min():.6f}, {output.max():.6f}]")
        
    except Exception as e:
        print(f"   ❌ CFM调用失败: {e}")
        
        # 尝试不同的输入格式
        print("\n4. 尝试不同的输入格式...")
        
        # 尝试1: 调整x和prompt_x的维度
        try:
            x_alt = torch.randn(batch_size, mel_dim, seq_len)
            prompt_x_alt = torch.randn(batch_size, mel_dim, seq_len)
            
            print(f"   尝试格式1: x={x_alt.shape}, prompt_x={prompt_x_alt.shape}")
            
            with torch.no_grad():
                output = pytorch_estimator(
                    x=x_alt,
                    prompt_x=prompt_x_alt,
                    x_lens=x_lens,
                    t=t,
                    style=style,
                    cond=cond,
                    mask_content=False
                )
            
            print(f"   ✅ 格式1成功")
            print(f"   输出形状: {output.shape}")
            
        except Exception as e2:
            print(f"   ❌ 格式1失败: {e2}")
            
            # 尝试2: 调整cond的维度
            try:
                cond_alt = torch.randn(batch_size, feature_dim)
                
                print(f"   尝试格式2: cond={cond_alt.shape}")
                
                with torch.no_grad():
                    output = pytorch_estimator(
                        x=x,
                        prompt_x=prompt_x,
                        x_lens=x_lens,
                        t=t,
                        style=style,
                        cond=cond_alt,
                        mask_content=False
                    )
                
                print(f"   ✅ 格式2成功")
                print(f"   输出形状: {output.shape}")
                
            except Exception as e3:
                print(f"   ❌ 格式2失败: {e3}")
                
                # 尝试3: 检查CFM内部结构
                print("\n5. 检查CFM内部结构...")
                print(f"   CFM类型: {type(pytorch_cfm)}")
                print(f"   CFM属性: {dir(pytorch_cfm)}")
                
                if hasattr(pytorch_cfm, 'estimator'):
                    estimator = pytorch_cfm.estimator
                    print(f"   Estimator类型: {type(estimator)}")
                    print(f"   Estimator属性: {[attr for attr in dir(estimator) if not attr.startswith('_')]}")
                    
                    # 检查estimator的forward方法
                    if hasattr(estimator, 'forward'):
                        forward_sig = inspect.signature(estimator.forward)
                        print(f"   Estimator forward签名: {forward_sig}")
                        
                        # 检查参数类型
                        for param_name, param in forward_sig.parameters.items():
                            print(f"     {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}")
    
    return True

if __name__ == "__main__":
    try:
        success = test_cfm_input_format()
        if success:
            print("\n✅ CFM输入格式测试完成")
        else:
            print("\n❌ CFM输入格式测试失败")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
