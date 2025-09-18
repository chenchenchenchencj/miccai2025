#!/usr/bin/env python3

import torch
import sys
import os

# 添加项目路径
sys.path.append('/media/zdp1/Datas1/cly/U-Mamba/umamba')

def test_mamba_unet():
    """测试集成Mamba的UNet"""
    try:
        # 导入必要的模块
        from nnunetv2.nets.attention_blocks import MAMBA_AVAILABLE
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiAttention import PlainConvUNetWithMultiAttention
        import torch.nn as nn
        
        print("Mamba可用性:", MAMBA_AVAILABLE)
        
        # 创建测试网络
        network = PlainConvUNetWithMultiAttention(
            input_channels=1,
            n_stages=5,
            features_per_stage=(32, 64, 128, 256, 320),
            conv_op=nn.Conv3d,
            kernel_sizes=[(3, 3, 3)] * 5,
            strides=[(1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
            n_conv_per_stage=[2] * 5,
            num_classes=2,
            n_conv_per_stage_decoder=[2] * 4,
            conv_bias=False,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            deep_supervision=True,
            encoder_attention="mamba",  # 使用Mamba注意力机制
            decoder_attention="cbam"
        )
        
        # 创建测试输入
        # 模拟一个小型3D医学图像输入 (batch_size=1, channels=1, depth=32, height=32, width=32)
        x = torch.randn(1, 1, 32, 32, 32)
        print(f"输入张量形状: {x.shape}")
        
        # 前向传播测试
        with torch.no_grad():
            output = network(x)
            
        if isinstance(output, list):
            print(f"输出是一个列表，包含{len(output)}个元素")
            for i, out in enumerate(output):
                print(f"  输出 {i} 形状: {out.shape}")
        else:
            print(f"输出张量形状: {output.shape}")
            
        print("Mamba UNet测试通过!")
        return True
        
    except Exception as e:
        print(f"Mamba UNet测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_mamba_unet():
    """测试集成增强版Mamba的UNet"""
    try:
        # 导入必要的模块
        from nnunetv2.nets.attention_blocks import MAMBA_AVAILABLE
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiAttention import PlainConvUNetWithMultiAttention
        import torch.nn as nn
        
        print("Mamba可用性:", MAMBA_AVAILABLE)
        
        # 创建测试网络
        network = PlainConvUNetWithMultiAttention(
            input_channels=1,
            n_stages=4,
            features_per_stage=(32, 64, 128, 256),
            conv_op=nn.Conv3d,
            kernel_sizes=[(3, 3, 3)] * 4,
            strides=[(1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
            n_conv_per_stage=[2] * 4,
            num_classes=2,
            n_conv_per_stage_decoder=[2] * 3,
            conv_bias=False,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            deep_supervision=True,
            encoder_attention="enhanced_mamba",  # 使用增强版Mamba注意力机制
            decoder_attention="coord_attention"
        )
        
        # 创建测试输入
        # 模拟一个小型3D医学图像输入 (batch_size=1, channels=1, depth=16, height=16, width=16)
        x = torch.randn(1, 1, 16, 16, 16)
        print(f"输入张量形状: {x.shape}")
        
        # 前向传播测试
        with torch.no_grad():
            output = network(x)
            
        if isinstance(output, list):
            print(f"输出是一个列表，包含{len(output)}个元素")
            for i, out in enumerate(output):
                print(f"  输出 {i} 形状: {out.shape}")
        else:
            print(f"输出张量形状: {output.shape}")
            
        print("增强版Mamba UNet测试通过!")
        return True
        
    except Exception as e:
        print(f"增强版Mamba UNet测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试集成Mamba的UNet...")
    
    # 测试基础Mamba UNet
    print("\n=== 测试基础Mamba UNet ===")
    success1 = test_mamba_unet()
    
    # 测试增强版Mamba UNet
    print("\n=== 测试增强版Mamba UNet ===")
    success2 = test_enhanced_mamba_unet()
    
    if success1 and success2:
        print("\n所有测试通过! Mamba模块已成功集成到UNet中。")
    else:
        print("\n部分测试失败，请检查实现。")