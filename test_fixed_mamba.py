#!/usr/bin/env python3
"""
测试修复后的Mamba模块
"""

import sys
import torch

# 添加项目路径
sys.path.insert(0, '/media/zdp1/Datas1/cly/U-Mamba/umamba')

def test_fixed_mamba_blocks():
    """测试修复后的Mamba块"""
    print("测试修复后的Mamba块...")
    
    try:
        from nnunetv2.nets.attention_blocks import MambaBlock3D, EnhancedMambaBlock3D
        import torch.nn as nn
        
        # 测试MambaBlock3D
        print("1. 测试MambaBlock3D...")
        mamba_block = MambaBlock3D(32, 64)
        x = torch.randn(1, 32, 16, 16, 16)  # (B, C, H, W, D)
        
        if torch.cuda.is_available():
            mamba_block = mamba_block.cuda()
            x = x.cuda()
            
        with torch.no_grad():
            output = mamba_block(x)
            
        print(f"   输入形状: {x.shape}")
        print(f"   输出形状: {output.shape}")
        print("   MambaBlock3D测试通过!")
        
        # 测试EnhancedMambaBlock3D
        print("2. 测试EnhancedMambaBlock3D...")
        enhanced_mamba_block = EnhancedMambaBlock3D(32, 64)
        
        if torch.cuda.is_available():
            enhanced_mamba_block = enhanced_mamba_block.cuda()
            
        with torch.no_grad():
            enhanced_output = enhanced_mamba_block(x)
            
        print(f"   输入形状: {x.shape}")
        print(f"   输出形状: {enhanced_output.shape}")
        print("   EnhancedMambaBlock3D测试通过!")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_network_with_fixed_mamba():
    """测试使用修复后Mamba块的网络"""
    print("\n测试使用修复后Mamba块的网络...")
    
    try:
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiAttention import PlainConvUNetWithMultiAttention
        import torch.nn as nn
        
        # 创建一个小型网络进行测试
        network = PlainConvUNetWithMultiAttention(
            input_channels=1,
            n_stages=3,
            features_per_stage=(32, 64, 128),
            conv_op=nn.Conv3d,
            kernel_sizes=[(3, 3, 3)] * 3,
            strides=[(1, 1, 1), (2, 2, 2), (2, 2, 2)],
            n_conv_per_stage=[1] * 3,
            num_classes=2,
            n_conv_per_stage_decoder=[1] * 2,
            conv_bias=False,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            deep_supervision=False,
            encoder_attention="mamba",  # 使用Mamba注意力
            decoder_attention="cbam"
        )
        
        # 创建测试输入
        x = torch.randn(1, 1, 32, 32, 32)
        
        if torch.cuda.is_available():
            network = network.cuda()
            x = x.cuda()
        
        # 测试前向传播
        with torch.no_grad():
            output = network(x)
            
        print(f"   网络输入形状: {x.shape}")
        print(f"   网络输出形状: {output.shape}")
        print("   网络测试通过!")
        
        return True
        
    except Exception as e:
        print(f"网络测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("修复后的Mamba模块测试")
    print("=" * 30)
    
    # 测试Mamba块
    test1 = test_fixed_mamba_blocks()
    
    # 测试网络
    test2 = test_network_with_fixed_mamba()
    
    print("\n" + "=" * 30)
    if test1 and test2:
        print("所有测试通过！Mamba模块已修复成功。")
        print("\n现在可以重新运行训练命令:")
        print("ENCODER_ATTENTION=mamba DECODER_ATTENTION=cbam CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 3d_fullres all -tr nnUNetTrainerMultiAttention")
    else:
        print("测试失败，请检查错误信息。")

if __name__ == "__main__":
    main()