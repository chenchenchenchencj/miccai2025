#!/usr/bin/env python3
"""
测试DDP是否正确初始化的脚本
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup_ddp(rank, world_size):
    """设置DDP环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # 为每个进程设置不同的GPU
    torch.cuda.set_device(rank)
    print(f"Rank {rank}/{world_size-1}: 初始化完成，使用GPU {rank}")

def cleanup_ddp():
    """清理DDP环境"""
    dist.destroy_process_group()

def test_ddp_functionality(rank, world_size):
    """测试DDP功能"""
    setup_ddp(rank, world_size)
    
    # 检查DDP状态
    print(f"Rank {rank}: DDP已初始化")
    print(f"  World size: {dist.get_world_size()}")
    print(f"  Rank: {dist.get_rank()}")
    print(f"  Backend: {dist.get_backend()}")
    
    # 创建一个简单的模型
    model = torch.nn.Linear(10, 10).cuda(rank)
    print(f"Rank {rank}: 模型创建完成")
    
    # 创建DDP包装的模型
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    print(f"Rank {rank}: DDP模型包装完成")
    
    # 创建一些测试数据
    data = torch.randn(20, 10).cuda(rank)
    targets = torch.randn(20, 10).cuda(rank)
    
    # 前向传播
    outputs = ddp_model(data)
    loss = torch.nn.functional.mse_loss(outputs, targets)
    print(f"Rank {rank}: 前向传播完成，损失值 = {loss.item()}")
    
    # 反向传播
    loss.backward()
    print(f"Rank {rank}: 反向传播完成")
    
    cleanup_ddp()
    print(f"Rank {rank}: 清理完成")

def main():
    print("DDP初始化测试")
    print("=" * 30)
    
    if not torch.cuda.is_available():
        print("CUDA不可用，无法测试DDP")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个GPU")
    
    # 获取测试GPU数量
    test_gpus = gpu_count
    if len(sys.argv) > 1:
        try:
            test_gpus = min(int(sys.argv[1]), gpu_count)
        except ValueError:
            print(f"无效的GPU数量参数，使用默认值 {test_gpus}")
    
    print(f"使用 {test_gpus} 个GPU进行测试")
    
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    # 启动多进程测试
    print("启动DDP测试...")
    mp.spawn(test_ddp_functionality,
             args=(test_gpus,),
             nprocs=test_gpus,
             join=True)
    
    print("DDP测试完成!")

if __name__ == "__main__":
    main()