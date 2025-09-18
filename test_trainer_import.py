#!/usr/bin/env python3

import sys
import os

# 添加umamba到路径
sys.path.insert(0, '/media/zdp1/Datas1/cly/U-Mamba/umamba')

def test_imports():
    """测试我们创建的训练器是否可以正确导入"""
    try:
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUNetDepth import nnUNetTrainerUNetDepth
        print("✓ nnUNetTrainerUNetDepth 导入成功")
    except Exception as e:
        print(f"✗ nnUNetTrainerUNetDepth 导入失败: {e}")
    
    try:
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainerFlexibleUNetDepth import nnUNetTrainerFlexibleUNetDepth
        print("✓ nnUNetTrainerFlexibleUNetDepth 导入成功")
    except Exception as e:
        print(f"✗ nnUNetTrainerFlexibleUNetDepth 导入失败: {e}")
        
    try:
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainerOptimized3DUNet import nnUNetTrainerOptimized3DUNet
        print("✓ nnUNetTrainerOptimized3DUNet 导入成功")
    except Exception as e:
        print(f"✗ nnUNetTrainerOptimized3DUNet 导入失败: {e}")
        
    # 测试基本训练器
    try:
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        print("✓ nnUNetTrainer 基类导入成功")
    except Exception as e:
        print(f"✗ nnUNetTrainer 基类导入失败: {e}")

if __name__ == "__main__":
    print("测试训练器导入...")
    test_imports()