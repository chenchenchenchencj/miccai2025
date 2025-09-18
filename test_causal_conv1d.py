#!/usr/bin/env python3
"""
测试重新安装的causal-conv1d是否解决了兼容性问题
"""

import sys
import torch

def test_causal_conv1d():
    """测试causal-conv1d安装"""
    print("测试causal-conv1d安装...")
    
    try:
        # 尝试导入causal_conv1d
        import causal_conv1d
        print(f"✓ causal_conv1d成功导入")
        print(f"  版本: {getattr(causal_conv1d, '__version__', '未知')}")
        
        # 测试基本功能
        from causal_conv1d import causal_conv1d_fn
        
        # 创建测试数据
        batch_size, seq_len, dim = 2, 10, 64
        x = torch.randn(batch_size, dim, seq_len)
        weight = torch.randn(dim, 1, 4)  # (dim, 1, kernel_size)
        
        if torch.cuda.is_available():
            x = x.cuda()
            weight = weight.cuda()
            
        # 测试causal_conv1d函数
        with torch.no_grad():
            result = causal_conv1d_fn(x, weight)
            
        print(f"✓ causal_conv1d_fn测试通过")
        print(f"  输入形状: {x.shape}")
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

def test_mamba_with_causal_conv1d():
    """测试Mamba与causal-conv1d的集成"""
    print("\n测试Mamba与causal-conv1d的集成...")
    
    try:
        # 尝试导入Mamba
        from mamba_ssm import Mamba
        print("✓ Mamba成功导入")
        
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
        print(f"✗ Mamba导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ Mamba测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_custom_mamba_implementation():
    """测试自定义Mamba实现"""
    print("\n测试自定义Mamba实现...")
    
    try:
        # 添加项目路径
        sys.path.append('/media/zdp1/Datas1/cly/U-Mamba/umamba')
        
        # 导入自定义实现
        from mamba import SS2D
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
        
    except Exception as e:
        print(f"✗ 自定义SS2D测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("causal-conv1d重新安装验证测试")
    print("=" * 40)
    
    # 测试causal_conv1d
    test1_success = test_causal_conv1d()
    
    # 测试Mamba集成
    test2_success = test_mamba_with_causal_conv1d()
    
    # 测试自定义实现
    test3_success = test_custom_mamba_implementation()
    
    print("\n" + "=" * 40)
    if test1_success and (test2_success or test3_success):
        print("✓ 所有测试通过！causal-conv1d重新安装成功。")
        print("  您现在可以尝试运行训练了。")
    else:
        print("✗ 部分测试失败，请检查安装或环境配置。")
        print("  建议查看详细的错误信息以进一步诊断问题。")

if __name__ == "__main__":
    main()