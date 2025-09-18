#!/usr/bin/env python3
"""
修复和测试Mamba模块的脚本
解决causal-conv1d和Mamba模块的兼容性问题
"""

import sys
import os
import torch

def test_causal_conv1d_correctly():
    """正确测试causal-conv1d"""
    print("正确测试causal-conv1d...")
    
    try:
        from causal_conv1d import causal_conv1d_fn
        print("✓ 成功导入causal_conv1d_fn")
        
        # 使用正确的参数形状
        batch_size, seq_len, dim = 2, 10, 64
        # 注意：causal_conv1d期望输入形状为 (batch, dim, seq_len)
        x = torch.randn(batch_size, dim, seq_len)
        # 权重形状应为 (dim, width) 其中width是卷积核宽度
        weight = torch.randn(dim, 4)  # width=4
        
        if torch.cuda.is_available():
            x = x.cuda()
            weight = weight.cuda()
            
        # 测试causal_conv1d函数
        with torch.no_grad():
            result = causal_conv1d_fn(x, weight)
            
        print(f"✓ causal_conv1d_fn测试通过")
        print(f"  输入形状: {x.shape}")
        print(f"  权重形状: {weight.shape}")
        print(f"  输出形状: {result.shape}")
        
        return True
        
    except ImportError as e:
        print(f"✗ causal_conv1d导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ causal_conv1d测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mamba_ssm():
    """测试Mamba SSM"""
    print("\n测试Mamba SSM...")
    
    try:
        import mamba_ssm
        from mamba_ssm import Mamba
        print(f"✓ Mamba SSM成功导入，版本: {getattr(mamba_ssm, '__version__', '未知')}")
        
        # 创建Mamba模型
        model = Mamba(
            d_model=64,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        
        # 创建测试数据
        batch_size, seq_len, dim = 1, 8, 64
        x = torch.randn(batch_size, seq_len, dim)
        
        if torch.cuda.is_available():
            model = model.cuda()
            x = x.cuda()
            
        # 测试前向传播
        with torch.no_grad():
            y = model(x)
            
        print("✓ Mamba前向传播测试通过")
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {y.shape}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Mamba SSM导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ Mamba SSM测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_custom_mamba_implementation():
    """测试自定义Mamba实现"""
    print("\n测试自定义Mamba实现...")
    
    try:
        # 添加正确的路径
        sys.path.insert(0, '/media/zdp1/Datas1/cly/U-Mamba/umamba')
        sys.path.insert(0, '/media/zdp1/Datas1/cly/U-Mamba')
        
        # 检查文件是否存在
        mamba_file = '/media/zdp1/Datas1/cly/U-Mamba/umamba/nnunetv2/mamba.py'
        if not os.path.exists(mamba_file):
            print(f"✗ 自定义Mamba文件不存在: {mamba_file}")
            return False
            
        # 直接从文件导入
        from nnunetv2.mamba import SS2D
        print("✓ 自定义SS2D成功导入")
        
        # 创建模型和测试数据
        model = SS2D(d_model=32)
        x = torch.randn(1, 32, 16, 16)  # (batch, channels, height, width)
        
        if torch.cuda.is_available():
            model = model.cuda()
            x = x.cuda()
            
        # 测试前向传播
        with torch.no_grad():
            y = model(x)
            
        print("✓ 自定义SS2D前向传播测试通过")
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {y.shape}")
        
        return True
        
    except ImportError as e:
        print(f"✗ 自定义SS2D导入失败: {e}")
        # 列出可用的模块
        try:
            import nnunetv2
            print(f"  nnunetv2模块路径: {nnunetv2.__path__}")
        except:
            pass
        return False
    except Exception as e:
        print(f"✗ 自定义SS2D测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_installation_status():
    """检查安装状态"""
    print("检查安装状态...")
    
    # 检查causal-conv1d安装路径
    try:
        import causal_conv1d
        print(f"  causal_conv1d路径: {causal_conv1d.__file__}")
        print(f"  causal_conv1d版本: {getattr(causal_conv1d, '__version__', '未知')}")
    except:
        print("  causal_conv1d未正确安装")
    
    # 检查mamba_ssm安装路径
    try:
        import mamba_ssm
        print(f"  mamba_ssm路径: {mamba_ssm.__file__}")
        print(f"  mamba_ssm版本: {getattr(mamba_ssm, '__version__', '未知')}")
    except:
        print("  mamba_ssm未正确安装")
    
    # 检查自定义实现
    try:
        sys.path.insert(0, '/media/zdp1/Datas1/cly/U-Mamba/umamba')
        import nnunetv2.mamba
        print(f"  自定义mamba路径: {nnunetv2.mamba.__file__}")
    except:
        print("  自定义mamba未找到")

def provide_solutions():
    """提供解决方案"""
    print("\n推荐的解决方案:")
    print("=" * 40)
    
    solutions = [
        "1. 重新安装causal-conv1d和mamba-ssm:",
        "   pip uninstall -y causal-conv1d mamba-ssm",
        "   pip install causal-conv1d>=1.2.0",
        "   pip install mamba-ssm --no-cache-dir",
        "",
        "2. 如果仍有问题，尝试使用conda安装:",
        "   conda install -c conda-forge mamba-ssm",
        "",
        "3. 检查CUDA版本兼容性:",
        "   确保PyTorch版本与CUDA版本兼容",
        "   Mamba要求CUDA 11.6+但不支持CUDA 12.x",
        "",
        "4. 降级PyTorch到CUDA 11.8版本:",
        "   pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118",
        "",
        "5. 使用项目中的自定义实现作为备选方案"
    ]
    
    for solution in solutions:
        print(solution)

def main():
    print("Mamba模块修复和测试工具")
    print("=" * 30)
    
    # 检查安装状态
    check_installation_status()
    
    # 测试各个组件
    test1 = test_causal_conv1d_correctly()
    test2 = test_mamba_ssm()
    test3 = test_custom_mamba_implementation()
    
    print("\n" + "=" * 30)
    print("测试结果汇总:")
    print(f"  causal_conv1d测试: {'通过' if test1 else '失败'}")
    print(f"  Mamba SSM测试: {'通过' if test2 else '失败'}")
    print(f"  自定义Mamba测试: {'通过' if test3 else '失败'}")
    
    if test2 or test3:
        print("\n✓ Mamba功能基本可用，可以尝试进行训练!")
        print("  建议从简单配置开始:")
        print("  ENCODER_ATTENTION=mamba DECODER_ATTENTION=cbam CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 3d_fullres all -tr nnUNetTrainerMultiAttention")
    else:
        print("\n✗ Mamba功能存在问题，请尝试推荐的解决方案。")
        provide_solutions()

if __name__ == "__main__":
    main()