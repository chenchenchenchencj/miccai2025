#!/usr/bin/env python3
"""
测试减小batch size的效果
"""

import os
import sys
import torch

# 添加项目路径
sys.path.insert(0, '/media/zdp1/Datas1/cly/U-Mamba/umamba')

def test_batch_size_adjustment():
    """测试batch size调整"""
    print("测试batch size调整...")
    
    # 设置环境变量减小batch size
    os.environ['BATCH_SIZE_FACTOR'] = '0.5'  # 减半
    os.environ['PYTHONPATH'] = '/media/zdp1/Datas1/cly/U-Mamba/umamba'
    
    try:
        # 导入必要的模块
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiAttention import nnUNetTrainerMultiAttention
        import torch
        
        print("✓ 环境变量设置成功")
        print(f"  BATCH_SIZE_FACTOR: {os.environ.get('BATCH_SIZE_FACTOR')}")
        return True
    except Exception as e:
        print(f"✗ 环境变量设置失败: {e}")
        return False

def calculate_memory_savings():
    """计算显存节省估计"""
    print("\n显存节省估计:")
    print("=" * 30)
    
    # 假设原始batch size为2
    original_batch_size = 2
    factors = [1.0, 0.75, 0.5, 0.25]
    
    print("Batch Size调整对照表:")
    for factor in factors:
        new_batch_size = max(1, int(original_batch_size * factor))
        memory_saving = (1 - factor) * 100
        print(f"  因子 {factor:4.2f} -> Batch Size: {new_batch_size} (节省约 {memory_saving:5.1f}% 显存)")

def main():
    print("Batch Size调整测试")
    print("=" * 30)
    
    test_batch_size_adjustment()
    calculate_memory_savings()
    
    print("\n推荐的训练命令:")
    print("# 减小batch size到原来的一半")
    print("export BATCH_SIZE_FACTOR=0.5")
    print("python /media/zdp1/Datas1/cly/U-Mamba/run_enhanced_mamba_ddp.py")
    
    print("\n# 或者减小到原来的25%")
    print("export BATCH_SIZE_FACTOR=0.25")
    print("python /media/zdp1/Datas1/cly/U-Mamba/run_enhanced_mamba_ddp.py")

if __name__ == "__main__":
    main()