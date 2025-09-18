#!/usr/bin/env python3
"""
多GPU训练脚本
帮助您在多张GPU上运行训练以解决显存不足问题
"""

import os
import sys

def setup_multi_gpu_environment():
    """设置多GPU环境"""
    print("设置多GPU训练环境...")
    
    # 设置环境变量
    env_vars = {
        'PYTHONPATH': '/media/zdp1/Datas1/cly/U-Mamba/umamba',
        'nnUNet_raw': '/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw',
        'nnUNet_preprocessed': '/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_preprocessed',
        'nnUNet_results': '/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_trained_models/nnUNet_results',
        'nnUNet_n_proc_DA': '2',  # 减少数据增强进程数
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'  # 避免内存碎片
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  {key}: {value}")
    
    # 添加项目路径
    sys.path.insert(0, '/media/zdp1/Datas1/cly/U-Mamba/umamba')
    
    return env_vars

def print_multi_gpu_commands():
    """打印多GPU训练命令"""
    print("\n多GPU训练命令:")
    print("=" * 50)
    
    commands = [
        "# 1. 使用DataParallel (推荐用于4张GPU)",
        "ENCODER_ATTENTION=coord_attention DECODER_ATTENTION=cbam CUDA_VISIBLE_DEVICES=0,1,2,3 nnUNetv2_train 1 3d_fullres all -tr nnUNetTrainerMultiAttention",
        "",
        "# 2. 如果上述成功，再尝试基础Mamba",
        "ENCODER_ATTENTION=mamba DECODER_ATTENTION=cbam CUDA_VISIBLE_DEVICES=0,1,2,3 nnUNetv2_train 1 3d_fullres all -tr nnUNetTrainerMultiAttention",
        "",
        "# 3. 如果显存仍然不足，可以减小网络规模",
        "UNET_3D_STAGES=4 ENCODER_ATTENTION=coord_attention DECODER_ATTENTION=cbam CUDA_VISIBLE_DEVICES=0,1,2,3 nnUNetv2_train 1 3d_fullres all -tr nnUNetTrainerMultiAttention"
    ]
    
    for cmd in commands:
        print(cmd)

def print_memory_saving_tips():
    """打印节省显存的技巧"""
    print("\n节省显存的技巧:")
    print("=" * 30)
    
    tips = [
        "1. 减少网络深度:",
        "   设置 UNET_3D_STAGES=4 或更少",
        "",
        "2. 减少数据增强进程数:",
        "   设置 nnUNet_n_proc_DA=1",
        "",
        "3. 使用更小的输入尺寸:",
        "   在数据预处理阶段使用更小的图像尺寸",
        "",
        "4. 启用显存优化:",
        "   设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "",
        "5. 使用混合精度训练:",
        "   nnUNet默认使用AMP，通常不需要额外设置",
        "",
        "6. 分布式训练:",
        "   如果有多台机器，可以考虑使用torchrun进行分布式训练"
    ]
    
    for tip in tips:
        print(tip)

def main():
    print("多GPU训练配置工具")
    print("=" * 30)
    
    # 设置环境
    setup_multi_gpu_environment()
    
    # 打印训练命令
    print_multi_gpu_commands()
    
    # 打印节省显存的技巧
    print_memory_saving_tips()
    
    print("\n建议首先尝试坐标注意力配置，因为它对显存要求较低。")
    print("如果成功运行，再尝试Mamba配置。")

if __name__ == "__main__":
    main()