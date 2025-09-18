#!/usr/bin/env python3
import torch
import sys

print("开始测试Mamba模块...")

# 测试1: 检查PyTorch
print("1. 检查PyTorch...")
try:
    import torch
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA版本: {torch.version.cuda}")
except Exception as e:
    print(f"   PyTorch检查失败: {e}")
    sys.exit(1)

# 测试2: 检查Mamba SSM
print("2. 检查Mamba SSM...")
try:
    import mamba_ssm
    print(f"   Mamba SSM已安装")
    
    from mamba_ssm import Mamba
    print(f"   Mamba类导入成功")
except ImportError as e:
    print(f"   Mamba SSM导入失败: {e}")
    print("   请确保已正确安装mamba_ssm")
    sys.exit(1)
except Exception as e:
    print(f"   Mamba SSM其他错误: {e}")
    sys.exit(1)

# 测试3: 测试Mamba模块功能
print("3. 测试Mamba模块功能...")
try:
    # 创建一个简单的Mamba模块
    model = Mamba(
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
    )
    
    # 创建测试数据
    x = torch.randn(2, 10, 64)  # (batch, seq_len, dim)
    
    # 如果CUDA可用，将模型和数据移到GPU上
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
    
    # 前向传播
    with torch.no_grad():
        y = model(x)
    
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {y.shape}")
    print("   Mamba模块功能测试通过!")
    
except Exception as e:
    print(f"   Mamba模块功能测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("所有测试通过! Mamba模块工作正常。")