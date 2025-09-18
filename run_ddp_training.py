#!/usr/bin/env python3
"""
用于运行多GPU分布式训练的脚本
"""

import os
import sys
import subprocess

def run_ddp_training():
    """运行DDP多GPU训练"""
    print("设置多GPU分布式训练...")
    
    # 设置环境变量
    env_vars = {
        'PYTHONPATH': '/media/zdp1/Datas1/cly/U-Mamba/umamba',
        'nnUNet_raw': '/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw',
        'nnUNet_preprocessed': '/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_preprocessed',
        'nnUNet_results': '/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_trained_models/nnUNet_results',
        'nnUNet_n_proc_DA': '2',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'
    }
    
    # 导出环境变量
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  {key}: {value}")
    
    # 添加项目路径
    sys.path.insert(0, '/media/zdp1/Datas1/cly/U-Mamba/umamba')
    
    # 构建训练命令
    # 使用torchrun进行分布式训练
    ngpus = 4  # 使用4个GPU
    command = [
        'torchrun',
        f'--nproc_per_node={ngpus}',
        '/media/zdp1/Datas1/cly/U-Mamba/umamba/nnunetv2/run/run_training.py',
        '1', '3d_fullres', 'all', 
        '-tr', 'nnUNetTrainerMultiAttention'
    ]
    
    # 添加注意力机制环境变量到命令中
    env_prefix = [
        'ENCODER_ATTENTION=coord_attention',
        'DECODER_ATTENTION=cbam'
    ]
    
    print(f"\n运行命令:")
    print(f"{' '.join(env_prefix)} {' '.join(command)}")
    
    try:
        # 执行训练命令
        result = subprocess.run(env_prefix + command, check=True)
        print("训练完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"训练失败: {e}")
        return False

def print_alternative_commands():
    """打印替代的多GPU训练命令"""
    print("\n替代的多GPU训练命令:")
    print("=" * 50)
    
    commands = [
        "# 1. 使用torchrun进行分布式训练 (推荐)",
        "export PYTHONPATH=/media/zdp1/Datas1/cly/U-Mamba/umamba",
        "export nnUNet_raw=/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw",
        "export nnUNet_preprocessed=/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_preprocessed", 
        "export nnUNet_results=/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_trained_models/nnUNet_results",
        "export nnUNet_n_proc_DA=2",
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "",
        "torchrun --nproc_per_node=4 /media/zdp1/Datas1/cly/U-Mamba/umamba/nnunetv2/run/run_training.py 1 3d_fullres all -tr nnUNetTrainerMultiAttention",
        "",
        "# 2. 如果torchrun不可用，可以尝试使用CUDA_VISIBLE_DEVICES配合较小的网络",
        "export UNET_3D_STAGES=4  # 减少网络深度",
        "CUDA_VISIBLE_DEVICES=0,1,2,3 nnUNetv2_train 1 3d_fullres all -tr nnUNetTrainerMultiAttention"
    ]
    
    for cmd in commands:
        print(cmd)

def main():
    print("多GPU分布式训练设置")
    print("=" * 30)
    
    # 首先尝试运行DDP训练
    success = run_ddp_training()
    
    if not success:
        print("\nDDP训练启动失败，以下是替代方案:")
        print_alternative_commands()

if __name__ == "__main__":
    main()