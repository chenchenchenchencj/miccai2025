#!/usr/bin/env python3
"""
测试DDP修复的脚本
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

def setup_distributed(rank, world_size):
    """设置分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化分布式进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # 为每个进程设置不同的GPU
    torch.cuda.set_device(rank)
    print(f"Rank {rank}: Using GPU {rank}")

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def test_device_assignment(rank, world_size):
    """测试设备分配"""
    print(f"Rank {rank}: 初始化分布式环境")
    setup_distributed(rank, world_size)
    
    # 检查当前设备
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Rank {rank}: Current device index = {current_device}, Device name = {device_name}")
    
    # 创建一个简单的模型并移到对应设备
    model = torch.nn.Linear(10, 10).to(current_device)
    ddp_model = DDP(model, device_ids=[current_device])
    
    # 创建一些随机数据并移到对应设备
    data = torch.randn(20, 10).to(current_device)
    targets = torch.randn(20, 10).to(current_device)
    
    # 简单的前向传播
    outputs = ddp_model(data)
    loss = torch.nn.functional.mse_loss(outputs, targets)
    
    print(f"Rank {rank}: Forward pass completed, loss = {loss.item()}")
    
    # 简单的反向传播
    loss.backward()
    print(f"Rank {rank}: Backward pass completed")
    
    cleanup_distributed()
    print(f"Rank {rank}: Cleanup completed")

def main():
    print("DDP设备分配测试")
    print("=" * 30)
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("CUDA不可用，退出")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个GPU:")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 获取GPU数量
    world_size = gpu_count
    if len(sys.argv) > 1:
        try:
            world_size = min(int(sys.argv[1]), gpu_count)
        except ValueError:
            print(f"无效的GPU数量参数，使用默认值 {world_size}")
    
    print(f"使用 {world_size} 个GPU进行测试")
    
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    # 启动多进程
    mp.spawn(test_device_assignment,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    
    print("DDP设备分配测试完成!")

if __name__ == "__main__":
    main()