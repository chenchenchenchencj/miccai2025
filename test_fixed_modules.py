#!/usr/bin/env python3
"""
测试修复后的模块导入问题
"""

import sys
import torch

# 添加项目路径
sys.path.insert(0, '/media/zdp1/Datas1/cly/U-Mamba/umamba')

def test_module_imports():
    """测试模块导入"""
    print("测试模块导入...")
    
    try:
        from nnunetv2.nets.attention_blocks import (
            CoordAttentionBlock3D, 
            MambaBlock3D, 
            EnhancedMambaBlock3D
        )
        print("✓ 所有模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ 模块导入失败: {e}")
        return False

def test_module_functionality():
    """测试模块功能"""
    print("\n测试模块功能...")
    
    try:
        from nnunetv2.nets.attention_blocks import EnhancedMambaBlock3D
        import torch.nn as nn
        
        # 创建EnhancedMambaBlock3D实例
        block = EnhancedMambaBlock3D(32, 64, d_state=8, expand=1)
        x = torch.randn(1, 32, 16, 16, 16)  # (B, C, D, H, W)
        
        if torch.cuda.is_available():
            block = block.cuda()
            x = x.cuda()
            
        with torch.no_grad():
            output = block(x)
            
        print(f"✓ EnhancedMambaBlock3D功能测试通过")
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {output.shape}")
        return True
        
    except Exception as e:
        print(f"✗ 模块功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_network_creation():
    """测试网络创建"""
    print("\n测试网络创建...")
    
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
            encoder_attention="enhanced_mamba",  # 使用增强Mamba注意力
            decoder_attention="coord_attention"
        )
        
        # 创建测试输入
        x = torch.randn(1, 1, 32, 32, 32)
        
        if torch.cuda.is_available():
            network = network.cuda()
            x = x.cuda()
        
        # 测试前向传播
        with torch.no_grad():
            output = network(x)
            
        print(f"✓ 网络创建和前向传播测试通过")
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {output.shape}")
        return True
        
    except Exception as e:
        print(f"✗ 网络创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("修复后的模块测试")
    print("=" * 30)
    
    # 测试模块导入
    test1 = test_module_imports()
    
    # 测试模块功能
    test2 = test_module_functionality()
    
    # 测试网络创建
    test3 = test_network_creation()
    
    print("\n" + "=" * 30)
    if test1 and test2 and test3:
        print("所有测试通过！模块已正确修复。")
        print("\n现在可以重新运行训练命令:")
        print("python /media/zdp1/Datas1/cly/U-Mamba/run_memory_efficient_enhanced_mamba.py")
    else:
        print("测试失败，请检查错误信息。")

if __name__ == "__main__":
    main()