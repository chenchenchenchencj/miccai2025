#export nnUNet_results="/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_trained_models"#换成自己的路径
#export nnUNet_preprocessed="/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_preprocessed"
#export nnUNet_raw="/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw"
#export CUDA_VISIBLE_DEVICES=0
#nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainerUMambaBot #206是数据集号码，0是不用交叉验证，直接划分好训练集和验证集
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys
import torch
import platform
import shutil
import psutil


def run_command(cmd):
    """运行系统命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"错误: {str(e)}"


def get_system_info():
    """获取系统信息"""
    print("=" * 50)
    print("           系统配置检查")
    print("=" * 50)
    print()

    # 1. 系统信息
    print("1. 系统信息:")
    print("-" * 40)
    print(f"系统: {platform.system()} {platform.release()}")
    print(f"平台: {platform.platform()}")
    print(f"处理器架构: {platform.machine()}")
    print()


def get_cpu_info():
    """获取CPU信息"""
    print("2. CPU信息:")
    print("-" * 40)
    if platform.system() == "Linux":
        cpu_info = run_command("lscpu | grep 'Model name' | cut -d':' -f2 | xargs")
        print(f"型号: {cpu_info}")

        cpu_cores = run_command("nproc")
        print(f"核心数: {cpu_cores}")
    else:
        print(f"处理器: {platform.processor()}")
    print()


def get_memory_info():
    """获取内存信息"""
    print("3. 内存信息:")
    print("-" * 40)
    mem = psutil.virtual_memory()
    print(f"总内存: {mem.total / (1024 ** 3):.2f} GB")
    print(f"可用内存: {mem.available / (1024 ** 3):.2f} GB")
    print()


def get_gpu_info():
    """获取GPU信息"""
    print("4. GPU信息:")
    print("-" * 40)
    if shutil.which("nvidia-smi"):
        gpu_info = run_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
        print("GPU详情:")
        for i, line in enumerate(gpu_info.split('\n')):
            print(f"  GPU {i}: {line}")

        driver_version = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1")
        print(f"驱动版本: {driver_version}")
    else:
        print("未找到NVIDIA驱动或nvidia-smi命令")
    print()


def get_cuda_info():
    """获取CUDA信息"""
    print("5. CUDA信息:")
    print("-" * 40)
    if shutil.which("nvcc"):
        cuda_version = run_command("nvcc --version | grep 'release' | awk '{print $6}' | cut -c2-")
        print(f"CUDA版本: {cuda_version}")
    else:
        print("未找到nvcc，CUDA可能未安装或未正确配置路径")
    print()


def get_python_info():
    """获取Python和PyTorch信息"""
    print("6. Python和PyTorch信息:")
    print("-" * 40)
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU索引: {torch.cuda.current_device()}")
    print()


def get_disk_info():
    """获取磁盘信息"""
    print("7. 磁盘空间信息:")
    print("-" * 40)
    if platform.system() == "Linux":
        disk_info = run_command("df -h / | tail -1")
        print(f"根分区: {disk_info}")
    else:
        disk = psutil.disk_usage('/')
        print(f"总空间: {disk.total / (1024 ** 3):.2f} GB")
        print(f"可用空间: {disk.free / (1024 ** 3):.2f} GB")
    print()


def main():
    """主函数"""
    get_system_info()
    get_cpu_info()
    get_memory_info()
    get_gpu_info()
    get_cuda_info()
    get_python_info()
    get_disk_info()

    print("=" * 50)
    print("           检查完成")
    print("=" * 50)


if __name__ == "__main__":
    main()