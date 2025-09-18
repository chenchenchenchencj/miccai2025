#!/usr/bin/env python3
"""
正确实现DDP训练的脚本
"""

import os
import sys
import subprocess
import torch

def check_gpu_availability():
    """检查GPU可用性"""
    if not torch.cuda.is_available():
        print("错误: CUDA不可用")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个GPU:")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    return True

def setup_environment():
    """设置训练环境"""
    # 设置环境变量
    env_vars = {
        'PYTHONPATH': '/media/zdp1/Datas1/cly/U-Mamba/umamba',
        'nnUNet_raw': '/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw',
        'nnUNet_preprocessed': '/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_preprocessed',
        'nnUNet_results': '/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_trained_models/nnUNet_results',
        'nnUNet_n_proc_DA': '1',  # 减少数据增强进程数以节省内存
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',  # 避免内存碎片
        'ENCODER_ATTENTION': 'enhanced_mamba',
        'DECODER_ATTENTION': 'coord_attention',
        'UNET_3D_STAGES': '5'  # 限制网络深度以节省显存
    }
    
    # 添加batch size调整因子（如果指定）
    if len(sys.argv) > 2:
        try:
            batch_factor = float(sys.argv[2])
            if 0 < batch_factor <= 1.0:
                env_vars['BATCH_SIZE_FACTOR'] = str(batch_factor)
                print(f"设置batch size因子为: {batch_factor}")
            else:
                print("警告: batch size因子应在(0, 1.0]范围内")
        except ValueError:
            print("警告: 无效的batch size因子参数")
    
    # 设置环境变量
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"设置环境变量 {key}={value}")
    
    # 添加项目路径到sys.path
    sys.path.insert(0, '/media/zdp1/Datas1/cly/U-Mamba/umamba')
    
    return env_vars

def run_ddp_training(num_gpus):
    """运行DDP训练"""
    print(f"\n开始使用 {num_gpus} 个GPU进行DDP分布式训练...")
    
    # 构建训练命令，使用nnUNet自带的DDP机制
    command = [
        'python',
        '/media/zdp1/Datas1/cly/U-Mamba/umamba/nnunetv2/run/run_training.py',
        '1', '3d_fullres', 'all',
        '-tr', 'nnUNetTrainerMultiAttention',
        '-num_gpus', str(num_gpus)
    ]
    
    print(f"执行命令: {' '.join(command)}")
    
    # 设置环境变量
    env = os.environ.copy()
    
    try:
        # 执行训练
        result = subprocess.run(command, env=env, check=True)
        print("训练完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"训练失败: {e}")
        return False

def show_batch_size_info():
    """显示batch size相关信息"""
    print("\nBatch Size调整说明:")
    print("=" * 30)
    print("通过设置BATCH_SIZE_FACTOR环境变量来调整batch size:")
    print("  1.0 (默认) = 原始batch size")
    print("  0.5        = batch size减半")
    print("  0.25       = batch size减少到1/4")
    print("\n示例命令:")
    print("  python /media/zdp1/Datas1/cly/U-Mamba/run_correct_ddp_training.py 4 0.5")

def main():
    print("正确实现的Enhanced Mamba 3D UNet DDP训练")
    print("=" * 50)
    
    # 显示batch size信息
    show_batch_size_info()
    
    # 检查GPU可用性
    if not check_gpu_availability():
        return
    
    # 获取GPU数量参数
    num_gpus = torch.cuda.device_count()
    if len(sys.argv) > 1:
        try:
            num_gpus = min(int(sys.argv[1]), num_gpus)
        except ValueError:
            print(f"无效的GPU数量参数，使用检测到的GPU数量 {num_gpus}")
    
    if num_gpus < 1:
        print("至少需要1个GPU进行训练")
        return
    
    print(f"\n使用 {num_gpus} 个GPU进行训练")
    
    # 设置环境
    setup_environment()
    
    # 运行训练
    success = run_ddp_training(num_gpus)
    
    if success:
        print("\n训练成功完成!")
    else:
        print("\n训练失败，请检查错误信息!")

if __name__ == "__main__":
    main()