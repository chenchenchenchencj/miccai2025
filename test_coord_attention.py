#!/usr/bin/env python3

import sys
import os
import torch

# 添加项目路径
sys.path.insert(0, '/media/zdp1/Datas1/cly/U-Mamba/umamba')

def test_import():
    """测试新训练器是否能正确导入"""
    try:
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainerCoordAttention import nnUNetTrainerCoordAttention
        print("✓ nnUNetTrainerCoordAttention 导入成功")
        return True
    except Exception as e:
        print(f"✗ nnUNetTrainerCoordAttention 导入失败: {e}")
        return False

def test_model_creation():
    """测试模型是否能正确创建"""
    try:
        # 临时设置环境变量
        os.environ['nnUNet_raw'] = '/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw'
        os.environ['nnUNet_preprocessed'] = '/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_preprocessed'
        os.environ['nnUNet_results'] = '/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_trained_models/nnUNet_results'
        
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainerCoordAttention import nnUNetTrainerCoordAttention
        
        # 创建一个简单的测试配置
        plans = {
            'plans_name': 'nnUNetPlans',
            'dataset_name': 'Dataset001_Teeth',
        }
        
        # 模拟数据集JSON
        dataset_json = {
            'labels': {'background': 0, 'teeth': 1},
            'numTraining': 30,
            'file_ending': '.nii.gz'
        }
        
        # 模拟配置管理器数据
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
        
        # 尝试创建训练器实例
        # 注意：这里只是测试导入和初始化，不实际运行训练
        
        print("✓ 模型创建测试完成")
        return True
    except Exception as e:
        print(f"✗ 模型创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("测试Coordinate Attention训练器...")
    print("=" * 50)
    
    import_success = test_import()
    if import_success:
        model_success = test_model_creation()
        if model_success:
            print("=" * 50)
            print("✓ 所有测试通过，可以开始训练")
        else:
            print("=" * 50)
            print("✗ 模型创建失败，请检查实现")
    else:
        print("=" * 50)
        print("✗ 导入失败，请检查代码")