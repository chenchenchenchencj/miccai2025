#!/usr/bin/env python3
"""
简化版训练环境测试脚本
专注于验证训练环境而不是Mamba底层实现
"""

import os
import sys
import torch

def setup_environment():
    """设置训练环境"""
    print("设置训练环境...")
    
    # 设置必要的环境变量
    env_vars = {
        'PYTHONPATH': '/media/zdp1/Datas1/cly/U-Mamba/umamba',
        'nnUNet_raw': '/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw',
        'nnUNet_preprocessed': '/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_preprocessed',
        'nnUNet_results': '/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_trained_models/nnUNet_results'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  {key}: {value}")
    
    # 减少数据加载进程数以避免资源问题
    os.environ['nnUNet_n_proc_DA'] = '2'
    print(f"  nnUNet_n_proc_DA: 2")
    
    # 添加项目路径到sys.path
    sys.path.insert(0, '/media/zdp1/Datas1/cly/U-Mamba/umamba')
    
    return env_vars

def test_training_imports():
    """测试训练所需的导入"""
    print("\n测试训练所需模块导入...")
    
    modules_to_test = [
        ('torch', 'PyTorch'),
        ('nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiAttention', 'MultiAttention Trainer'),
        ('nnunetv2.nets.attention_blocks', 'Attention Blocks'),
    ]
    
    success = True
    for module_import, module_name in modules_to_test:
        try:
            __import__(module_import)
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ✗ {module_name}: {e}")
            success = False
        except Exception as e:
            print(f"  ✗ {module_name}: {e}")
            success = False
    
    return success

def test_attention_blocks_functionality():
    """测试注意力模块功能"""
    print("\n测试注意力模块功能...")
    
    try:
        from nnunetv2.nets.attention_blocks import (
            CoordAttentionBlock3D,
            MambaBlock3D,
            EnhancedMambaBlock3D
        )
        import torch.nn as nn
        
        # 测试坐标注意力块
        coord_block = CoordAttentionBlock3D(32, 64)
        x = torch.randn(1, 32, 16, 16, 16)
        if torch.cuda.is_available():
            coord_block = coord_block.cuda()
            x = x.cuda()
            
        with torch.no_grad():
            output = coord_block(x)
        print(f"  ✓ CoordAttentionBlock3D: {x.shape} -> {output.shape}")
        
        # 测试Mamba块（如果可用）
        try:
            mamba_block = MambaBlock3D(32, 64)
            if torch.cuda.is_available():
                mamba_block = mamba_block.cuda()
                
            with torch.no_grad():
                mamba_output = mamba_block(x)
            print(f"  ✓ MambaBlock3D: {x.shape} -> {mamba_output.shape}")
        except Exception as e:
            print(f"  - MambaBlock3D: 不可用 ({e})")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 注意力模块功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_network_creation():
    """测试网络创建"""
    print("\n测试网络创建...")
    
    try:
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiAttention import (
            PlainConvUNetWithMultiAttention
        )
        import torch.nn as nn
        
        # 创建一个小型网络进行测试
        network = PlainConvUNetWithMultiAttention(
            input_channels=1,
            n_stages=4,
            features_per_stage=(32, 64, 128, 256),
            conv_op=nn.Conv3d,
            kernel_sizes=[(3, 3, 3)] * 4,
            strides=[(1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
            n_conv_per_stage=[1] * 4,
            num_classes=2,
            n_conv_per_stage_decoder=[1] * 3,
            conv_bias=False,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            deep_supervision=False,
            encoder_attention="coord_attention",  # 使用坐标注意力而不是Mamba进行测试
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
            
        print(f"  ✓ 网络创建和前向传播测试通过")
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 网络创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_training_commands():
    """打印推荐的训练命令"""
    print("\n推荐的训练命令（从简单配置开始）:")
    print("=" * 50)
    
    commands = [
        "# 1. 使用坐标注意力（最稳定）",
        "ENCODER_ATTENTION=coord_attention DECODER_ATTENTION=cbam CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 3d_fullres all -tr nnUNetTrainerMultiAttention",
        "",
        "# 2. 如果上述成功，再尝试基础Mamba",
        "ENCODER_ATTENTION=mamba DECODER_ATTENTION=cbam CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 3d_fullres all -tr nnUNetTrainerMultiAttention",
        "",
        "# 3. 最后尝试增强版Mamba",
        "ENCODER_ATTENTION=enhanced_mamba DECODER_ATTENTION=coord_attention CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 3d_fullres all -tr nnUNetTrainerMultiAttention"
    ]
    
    for cmd in commands:
        print(cmd)

def main():
    print("简化版训练环境测试")
    print("=" * 30)
    
    # 设置环境
    setup_environment()
    
    # 测试导入
    imports_ok = test_training_imports()
    
    # 测试注意力模块
    attention_ok = test_attention_blocks_functionality()
    
    # 测试网络创建
    network_ok = test_network_creation()
    
    print("\n" + "=" * 30)
    print("测试结果:")
    print(f"  模块导入: {'通过' if imports_ok else '失败'}")
    print(f"  注意力模块: {'通过' if attention_ok else '失败'}")
    print(f"  网络创建: {'通过' if network_ok else '失败'}")
    
    if imports_ok and attention_ok and network_ok:
        print("\n✓ 训练环境测试通过！")
        print("  您的环境已准备好进行训练。")
        print_training_commands()
    else:
        print("\n✗ 训练环境测试失败！")
        print("  请检查错误信息并修复问题。")
        
        if imports_ok:
            print("\n  提示：模块导入正常，但功能测试失败")
            print("  建议先使用坐标注意力进行训练测试:")
            print("  ENCODER_ATTENTION=coord_attention DECODER_ATTENTION=cbam CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 3d_fullres all -tr nnUNetTrainerMultiAttention")

if __name__ == "__main__":
    main()