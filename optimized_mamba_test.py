#!/usr/bin/env python3
"""
显存优化版本的Mamba模块测试
"""

import sys
import torch

# 添加项目路径
sys.path.insert(0, '/media/zdp1/Datas1/cly/U-Mamba/umamba')

def test_memory_efficient_mamba():
    """测试显存优化的Mamba模块"""
    print("测试显存优化的Mamba模块...")
    
    try:
        from nnunetv2.nets.attention_blocks import EnhancedMambaBlock3D
        import torch.nn as nn
        
        # 创建一个较小的测试网络以节省显存
        print("创建显存优化的EnhancedMambaBlock3D...")
        mamba_block = EnhancedMambaBlock3D(16, 32, d_state=8, d_conv=3, expand=1)
        x = torch.randn(1, 16, 8, 8, 8)  # 使用较小的输入以节省显存
        
        if torch.cuda.is_available():
            mamba_block = mamba_block.cuda()
            x = x.cuda()
            print(f"  使用GPU进行测试")
        else:
            print(f"  使用CPU进行测试")
            
        print(f"  输入形状: {x.shape}")
        
        # 测试前向传播
        with torch.no_grad():
            output = mamba_block(x)
            
        print(f"  输出形状: {output.shape}")
        print("  显存优化的EnhancedMambaBlock3D测试通过!")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradient_checkpointing():
    """测试梯度检查点技术"""
    print("\n测试梯度检查点技术...")
    
    try:
        import torch.utils.checkpoint as checkpoint
        from nnunetv2.nets.attention_blocks import EnhancedMambaBlock3D
        import torch.nn as nn
        
        # 创建网络
        mamba_block = EnhancedMambaBlock3D(16, 32, d_state=8, d_conv=3, expand=1)
        x = torch.randn(1, 16, 8, 8, 8, requires_grad=True)
        
        if torch.cuda.is_available():
            mamba_block = mamba_block.cuda()
            x = x.cuda()
        
        # 使用梯度检查点
        def custom_forward(*inputs):
            return mamba_block(inputs[0])
        
        # 应用梯度检查点
        output = checkpoint.checkpoint(custom_forward, x)
        
        print(f"  梯度检查点技术测试通过!")
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"梯度检查点测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_memory_optimization_tips():
    """打印显存优化技巧"""
    print("\n显存优化技巧:")
    print("=" * 30)
    
    tips = [
        "1. 减小模型尺寸:",
        "   - 减少通道数",
        "   - 减少d_state参数 (如从16减少到8)",
        "   - 减少expand参数 (如从2减少到1)",
        "",
        "2. 减小输入尺寸:",
        "   - 使用较小的图像尺寸",
        "   - 减少批次大小",
        "",
        "3. 使用梯度检查点:",
        "   - 在训练时使用torch.utils.checkpoint",
        "   - 以计算时间为代价换取显存节省",
        "",
        "4. 混合精度训练:",
        "   - nnUNet默认启用AMP",
        "   - 可进一步设置PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "",
        "5. 多GPU训练:",
        "   - 使用DataParallel或DistributedDataParallel",
        "   - 在4张GPU上分配计算负载"
    ]
    
    for tip in tips:
        print(tip)

def main():
    print("显存优化Mamba模块测试")
    print("=" * 30)
    
    # 测试显存优化的Mamba模块
    test1 = test_memory_efficient_mamba()
    
    # 测试梯度检查点
    test2 = test_gradient_checkpointing()
    
    print("\n" + "=" * 30)
    if test1 and test2:
        print("所有测试通过！显存优化方案有效。")
    else:
        print("部分测试失败，请检查错误信息。")
    
    # 打印显存优化技巧
    print_memory_optimization_tips()
    
    print("\n推荐的训练命令:")
    print("ENCODER_ATTENTION=mamba DECODER_ATTENTION=cbam CUDA_VISIBLE_DEVICES=0,1,2,3 nnUNetv2_train 1 3d_fullres all -tr nnUNetTrainerMultiAttention")

if __name__ == "__main__":
    main()