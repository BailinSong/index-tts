#!/usr/bin/env python3
"""
调试CFM内部处理过程
"""

import sys
import os
import torch
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from indextts.infer_v2 import IndexTTS2

def debug_cfm_internal():
    """调试CFM内部处理过程"""
    print("=== 调试CFM内部处理过程 ===\n")
    
    # 初始化TTS系统
    print("1. 初始化TTS系统...")
    tts = IndexTTS2()
    
    # 获取CFM estimator
    pytorch_cfm = tts.s2mel.models.cfm
    pytorch_estimator = pytorch_cfm.estimator
    
    print(f"   CFM estimator类型: {type(pytorch_estimator)}")
    
    # 检查CFM的内部结构
    print("\n2. 检查CFM内部结构...")
    print(f"   CFM属性: {[attr for attr in dir(pytorch_cfm) if not attr.startswith('_')]}")
    print(f"   Estimator属性: {[attr for attr in dir(pytorch_estimator) if not attr.startswith('_')]}")
    
    # 检查estimator的关键组件
    print("\n3. 检查estimator关键组件...")
    if hasattr(pytorch_estimator, 'x_embedder'):
        print(f"   x_embedder: {type(pytorch_estimator.x_embedder)}")
        if hasattr(pytorch_estimator.x_embedder, 'weight'):
            print(f"   x_embedder权重形状: {pytorch_estimator.x_embedder.weight.shape}")
    
    if hasattr(pytorch_estimator, 'cond_x_merge_linear'):
        print(f"   cond_x_merge_linear: {type(pytorch_estimator.cond_x_merge_linear)}")
        if hasattr(pytorch_estimator.cond_x_merge_linear, 'weight'):
            print(f"   cond_x_merge_linear权重形状: {pytorch_estimator.cond_x_merge_linear.weight.shape}")
    
    if hasattr(pytorch_estimator, 'transformer'):
        print(f"   transformer: {type(pytorch_estimator.transformer)}")
    
    if hasattr(pytorch_estimator, 'wavenet'):
        print(f"   wavenet: {type(pytorch_estimator.wavenet)}")
    
    if hasattr(pytorch_estimator, 'final_layer'):
        print(f"   final_layer: {type(pytorch_estimator.final_layer)}")
    
    # 创建测试输入
    print("\n4. 创建测试输入...")
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
    
    # 尝试逐步调试CFM内部处理
    print("\n5. 逐步调试CFM内部处理...")
    try:
        # 检查x_embedder
        if hasattr(pytorch_estimator, 'x_embedder'):
            print("   检查x_embedder...")
            x_embedded = pytorch_estimator.x_embedder(x)
            print(f"   x_embedded形状: {x_embedded.shape}")
        
        # 检查cond_x_merge_linear
        if hasattr(pytorch_estimator, 'cond_x_merge_linear'):
            print("   检查cond_x_merge_linear...")
            print(f"   cond_x_merge_linear输入维度: {pytorch_estimator.cond_x_merge_linear.weight.shape[1]}")
            
            # 尝试合并x和cond
            try:
                # 检查x和cond的维度
                print(f"   x形状: {x.shape}")
                print(f"   cond形状: {cond.shape}")
                
                # 尝试不同的合并方式
                if x.shape == cond.shape:
                    print("   x和cond形状相同，尝试直接合并...")
                    # 这里可能需要检查CFM内部的具体实现
                    
            except Exception as e:
                print(f"   合并失败: {e}")
        
        # 尝试完整的CFM调用
        print("\n6. 尝试完整的CFM调用...")
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
        
        # 尝试理解错误
        print("\n7. 分析错误原因...")
        error_msg = str(e)
        print(f"   错误信息: {error_msg}")
        
        if "Sizes of tensors must match" in error_msg:
            print("   这是张量维度不匹配的错误")
            print("   可能的原因:")
            print("   1. x和cond的维度不匹配")
            print("   2. prompt_x的维度不正确")
            print("   3. CFM内部处理时的维度转换问题")
            
            # 尝试不同的输入格式
            print("\n8. 尝试不同的输入格式...")
            
            # 尝试1: 调整x的维度
            try:
                x_alt = torch.randn(batch_size, mel_dim, seq_len)
                print(f"   尝试格式1: x={x_alt.shape}")
                
                with torch.no_grad():
                    output = pytorch_estimator(
                        x=x_alt,
                        prompt_x=prompt_x,
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
                    
                    # 尝试3: 检查CFM源码
                    print("\n9. 检查CFM源码...")
                    try:
                        import inspect
                        source = inspect.getsource(pytorch_estimator.forward)
                        print("   CFM forward方法源码:")
                        print(source[:500] + "..." if len(source) > 500 else source)
                    except Exception as e4:
                        print(f"   无法获取源码: {e4}")
    
    return True

if __name__ == "__main__":
    try:
        success = debug_cfm_internal()
        if success:
            print("\n✅ CFM内部调试完成")
        else:
            print("\n❌ CFM内部调试失败")
    except Exception as e:
        print(f"\n❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()
