#!/usr/bin/env python3
"""
在修复causal-conv1d后测试训练环境
"""

import os
import sys

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

def test_imports():
    """测试必要的导入"""
    print("\n测试必要模块导入...")
    
    modules_to_test = [
        'torch',
        'mamba_ssm',
        'nnunetv2.nets.attention_blocks',
        'nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiAttention'
    ]
    
    success = True
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module}: {e}")
            success = False
        except Exception as e:
            print(f"  ✗ {module}: {e}")
            success = False
    
    return success

def test_attention_blocks():
    """测试注意力模块"""
    print("\n测试注意力模块...")
    
    try:
        from nnunetv2.nets.attention_blocks import MambaBlock3D, EnhancedMambaBlock3D, CoordAttentionBlock3D
        import torch
        
        # 测试MambaBlock3D
        block = MambaBlock3D(32, 64)
        x = torch.randn(1, 32, 16, 16, 16)
        if torch.cuda.is_available():
            block = block.cuda()
            x = x.cuda()
            
        with torch.no_grad():
            output = block(x)
        print(f"  ✓ MambaBlock3D: {x.shape} -> {output.shape}")
        
        # 测试EnhancedMambaBlock3D
        enhanced_block = EnhancedMambaBlock3D(32, 64)
        if torch.cuda.is_available():
            enhanced_block = enhanced_block.cuda()
            
        with torch.no_grad():
            enhanced_output = enhanced_block(x)
        print(f"  ✓ EnhancedMambaBlock3D: {x.shape} -> {enhanced_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 注意力模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_training_commands():
    """打印推荐的训练命令"""
    print("\n推荐的训练命令:")
    print("=" * 50)
    
    commands = [
        "# 使用基础Mamba编码器和CBAM解码器（推荐用于初始测试）",
        "ENCODER_ATTENTION=mamba DECODER_ATTENTION=cbam CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 3d_fullres all -tr nnUNetTrainerMultiAttention",
        "",
        "# 使用坐标注意力编码器和CBAM解码器",
        "ENCODER_ATTENTION=coord_attention DECODER_ATTENTION=cbam CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 3d_fullres all -tr nnUNetTrainerMultiAttention",
        "",
        "# 使用增强版Mamba编码器和坐标注意力解码器",
        "ENCODER_ATTENTION=enhanced_mamba DECODER_ATTENTION=coord_attention CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 3d_fullres all -tr nnUNetTrainerMultiAttention"
    ]
    
    for cmd in commands:
        print(cmd)

def main():
    print("训练环境测试")
    print("=" * 30)
    
    # 设置环境
    setup_environment()
    
    # 测试导入
    imports_ok = test_imports()
    
    # 测试注意力模块
    attention_ok = test_attention_blocks()
    
    print("\n" + "=" * 30)
    if imports_ok and attention_ok:
        print("✓ 环境测试通过！")
        print("  您的环境已准备好进行训练。")
        print_training_commands()
    else:
        print("✗ 环境测试失败！")
        print("  请检查错误信息并修复问题。")

if __name__ == "__main__":
    main()