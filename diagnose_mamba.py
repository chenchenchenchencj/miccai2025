#!/usr/bin/env python3
import sys
import os

# 添加项目路径
sys.path.append('/media/zdp1/Datas1/cly/U-Mamba/umamba')

def check_environment():
    """检查环境变量设置"""
    print("=== 环境变量检查 ===")
    required_vars = ['PYTHONPATH', 'nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']
    for var in required_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"{var}: {value}")
    print()

def check_mamba_installation():
    """检查Mamba安装状态"""
    print("=== Mamba模块检查 ===")
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA设备数量: {torch.cuda.device_count()}")
            print(f"当前CUDA设备: {torch.cuda.current_device()}")
            print(f"当前设备名称: {torch.cuda.get_device_name()}")
    except Exception as e:
        print(f"PyTorch检查失败: {e}")
    
    try:
        import mamba_ssm
        print(f"Mamba SSM已安装, 版本: {getattr(mamba_ssm, '__version__', 'Unknown')}")
        
        from mamba_ssm import Mamba
        print("Mamba类导入成功")
    except ImportError as e:
        print(f"Mamba SSM导入失败: {e}")
        return False
    except Exception as e:
        print(f"Mamba SSM其他错误: {e}")
        return False
    
    try:
        from nnunetv2.nets.attention_blocks import MAMBA_AVAILABLE
        print(f"MAMBA_AVAILABLE标志: {MAMBA_AVAILABLE}")
    except Exception as e:
        print(f"检查MAMBA_AVAILABLE失败: {e}")
    
    print()

def test_mamba_blocks():
    """测试Mamba块功能"""
    print("=== Mamba块功能测试 ===")
    try:
        import torch
        from nnunetv2.nets.attention_blocks import MambaBlock3D, EnhancedMambaBlock3D
        
        # 测试基础Mamba块
        print("测试基础Mamba块...")
        block = MambaBlock3D(32, 64)
        if torch.cuda.is_available():
            block = block.cuda()
            
        x = torch.randn(1, 32, 16, 16, 16)
        if torch.cuda.is_available():
            x = x.cuda()
            
        with torch.no_grad():
            output = block(x)
        print(f"基础Mamba块测试通过 - 输入: {x.shape}, 输出: {output.shape}")
        
        # 测试增强Mamba块
        print("测试增强Mamba块...")
        enhanced_block = EnhancedMambaBlock3D(32, 64)
        if torch.cuda.is_available():
            enhanced_block = enhanced_block.cuda()
            
        with torch.no_grad():
            enhanced_output = enhanced_block(x)
        print(f"增强Mamba块测试通过 - 输入: {x.shape}, 输出: {enhanced_output.shape}")
        
    except Exception as e:
        print(f"Mamba块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()

def check_data_preprocessing():
    """检查数据预处理状态"""
    print("=== 数据预处理检查 ===")
    try:
        import os
        raw_path = os.environ.get('nnUNet_raw', '/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw')
        preprocessed_path = os.environ.get('nnUNet_preprocessed', '/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_preprocessed')
        
        print(f"原始数据路径: {raw_path}")
        if os.path.exists(raw_path):
            datasets = os.listdir(raw_path)
            print(f"数据集列表: {datasets}")
        else:
            print("原始数据路径不存在")
            
        print(f"预处理数据路径: {preprocessed_path}")
        if os.path.exists(preprocessed_path):
            datasets = os.listdir(preprocessed_path)
            print(f"预处理数据集列表: {datasets}")
        else:
            print("预处理数据路径不存在")
    except Exception as e:
        print(f"数据预处理检查失败: {e}")
    
    print()

def main():
    """主诊断函数"""
    print("U-Mamba系统诊断工具")
    print("=" * 50)
    
    check_environment()
    check_mamba_installation()
    test_mamba_blocks()
    check_data_preprocessing()
    
    print("诊断完成。如果所有检查都通过，但训练仍然失败，请尝试:")
    print("1. 减少数据增强进程数: export nnUNet_n_proc_DA=2")
    print("2. 使用基础Mamba而不是增强版Mamba")
    print("3. 减小网络规模或批次大小")
    print("4. 检查是否有足够的GPU内存")

if __name__ == "__main__":
    main()