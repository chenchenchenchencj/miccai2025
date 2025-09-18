#!/usr/bin/env python3
"""
显存优化版Enhanced Mamba 3D UNet DDP训练脚本
"""

import os
import sys
import subprocess

def setup_memory_efficient_environment():
    """设置显存优化的环境变量"""
    # 设置环境变量以优化显存使用
    env_vars = {
        'PYTHONPATH': '/media/zdp1/Datas1/cly/U-Mamba/umamba',
        'nnUNet_raw': '/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw',
        'nnUNet_preprocessed': '/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_preprocessed',
        'nnUNet_results': '/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_trained_models/nnUNet_results',
        'nnUNet_n_proc_DA': '1',  # 减少数据增强进程数
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',  # 避免内存碎片
        'ENCODER_ATTENTION': 'enhanced_mamba',
        'DECODER_ATTENTION': 'coord_attention',
        'UNET_3D_STAGES': '5'  # 限制网络深度以节省显存
    }
    
    # 设置环境变量
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"设置环境变量 {key}: {value}")
    
    return env_vars

def run_memory_efficient_ddp_training(num_gpus=2):
    """运行显存优化的DDP训练"""
    print(f"\n开始使用 {num_gpus} 个GPU进行显存优化的DDP分布式训练...")
    
    # 构建训练命令
    command = [
        'torchrun',
        f'--nproc_per_node={num_gpus}',
        '/media/zdp1/Datas1/cly/U-Mamba/umamba/nnunetv2/run/run_training.py',
        '1', '3d_fullres', 'all',
        '-tr', 'nnUNetTrainerMultiAttention'
    ]
    
    print(f"执行命令: {' '.join(command)}")
    
    try:
        # 执行训练
        result = subprocess.run(command, check=True)
        print("训练完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"训练失败: {e}")
        return False

def main():
    print("显存优化版Enhanced Mamba 3D UNet DDP训练")
    print("=" * 50)
    
    # 设置显存优化环境
    setup_memory_efficient_environment()
    
    # 获取GPU数量参数（默认使用2个GPU以节省显存）
    num_gpus = 2
    if len(sys.argv) > 1:
        try:
            num_gpus = int(sys.argv[1])
        except ValueError:
            print(f"无效的GPU数量参数，使用默认值 {num_gpus}")
    
    print(f"使用 {num_gpus} 个GPU进行训练")
    print("网络阶段数限制为5以节省显存")
    print("Mamba参数已优化以减少显存使用")
    
    # 运行训练
    success = run_memory_efficient_ddp_training(num_gpus)
    
    if success:
        print("\n训练成功完成!")
    else:
        print("\n训练失败，请检查错误信息!")

if __name__ == "__main__":
    main()