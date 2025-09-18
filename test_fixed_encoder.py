#!/usr/bin/env python3
"""
测试修复后的PlainConvEncoderWithAttention类
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, '/media/zdp1/Datas1/cly/U-Mamba/umamba')

def test_encoder_import():
    """测试编码器导入"""
    print("测试PlainConvEncoderWithAttention导入...")
    try:
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiAttention import PlainConvEncoderWithAttention
        print("✓ PlainConvEncoderWithAttention导入成功")
        return True
    except Exception as e:
        print(f"✗ PlainConvEncoderWithAttention导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_encoder_creation():
    """测试编码器创建"""
    print("\n测试PlainConvEncoderWithAttention创建...")
    try:
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiAttention import PlainConvEncoderWithAttention
        import torch.nn as nn
        
        # 创建编码器实例
        encoder = PlainConvEncoderWithAttention(
            input_channels=1,
            n_stages=3,
            features_per_stage=(32, 64, 128),
            conv_op=nn.Conv3d,
            kernel_sizes=[(3, 3, 3)] * 3,
            strides=[(1, 1, 1), (2, 2, 2), (2, 2, 2)],
            n_conv_per_stage=[1] * 3,
            attention_type="enhanced_mamba"
        )
        
        print("✓ PlainConvEncoderWithAttention创建成功")
        return True
    except Exception as e:
        print(f"✗ PlainConvEncoderWithAttention创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_network_creation():
    """测试完整网络创建"""
    print("\n测试完整网络创建...")
    try:
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiAttention import PlainConvUNetWithMultiAttention
        import torch.nn as nn
        
        # 创建完整网络
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
            encoder_attention="enhanced_mamba",
            decoder_attention="coord_attention"
        )
        
        print("✓ 完整网络创建成功")
        return True
    except Exception as e:
        print(f"✗ 完整网络创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("修复后的编码器测试")
    print("=" * 30)
    
    tests = [
        test_encoder_import,
        test_encoder_creation,
        test_network_creation
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 30)
    if all(results):
        print("所有测试通过！")
        print("\n现在可以尝试运行训练:")
        print("python /media/zdp1/Datas1/cly/U-Mamba/run_enhanced_mamba_ddp.py")
    else:
        print("部分测试失败，请检查错误信息。")

if __name__ == "__main__":
    main()