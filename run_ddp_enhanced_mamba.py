#!/usr/bin/env python3
"""
运行坐标注意力增强Mamba块的DDP分布式训练
"""

import os
import sys
import subprocess

def setup_environment():
    """设置训练环境"""
    # 设置环境变量
    env_vars = {
        'PYTHONPATH': '/media/zdp1/Datas1/cly/U-Mamba/umamba',
        'nnUNet_raw': '/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw',
        'nnUNet_preprocessed': '/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_preprocessed',
        'nnUNet_results': '/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_trained_models/nnUNet_results',
        'nnUNet_n_proc_DA': '1',  # 减少数据加载进程以节省内存
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
        'ENCODER_ATTENTION': 'enhanced_mamba',
        'DECODER_ATTENTION': 'coord_attention'
    }
    
    # 设置环境变量
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"设置环境变量 {key}: {value}")
    
    # 添加项目路径到sys.path
    sys.path.insert(0, '/media/zdp1/Datas1/cly/U-Mamba/umamba')
    
    return env_vars

def run_ddp_training(num_gpus=4):
    """运行DDP训练"""
    print(f"\n开始使用 {num_gpus} 个GPU进行DDP分布式训练...")
    
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
    print("坐标注意力增强Mamba块DDP分布式训练")
    print("=" * 50)
    
    # 设置环境
    setup_environment()
    
    # 获取GPU数量参数
    num_gpus = 4
    if len(sys.argv) > 1:
        try:
            num_gpus = int(sys.argv[1])
        except ValueError:
            print(f"无效的GPU数量参数，使用默认值 {num_gpus}")
    
    print(f"使用 {num_gpus} 个GPU进行训练")
    
    # 运行训练
    success = run_ddp_training(num_gpus)
    
    if success:
        print("\n训练成功完成!")
    else:
        print("\n训练失败，请检查错误信息!")

if __name__ == "__main__":
    main()