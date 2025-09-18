#!/usr/bin/env python3
"""
调试DDP训练的脚本
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
    
    # 设置当前GPU设备
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def simple_model_test(rank, world_size):
    """简单的模型测试"""
    print(f"Rank {rank}: 初始化分布式环境")
    setup_distributed(rank, world_size)
    
    # 创建一个简单的模型
    model = torch.nn.Linear(10, 10).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # 创建一些随机数据
    data = torch.randn(20, 10).to(rank)
    targets = torch.randn(20, 10).to(rank)
    
    # 简单的前向传播
    outputs = ddp_model(data)
    loss = torch.nn.functional.mse_loss(outputs, targets)
    
    print(f"Rank {rank}: 前向传播完成，损失值 = {loss.item()}")
    
    # 简单的反向传播
    loss.backward()
    print(f"Rank {rank}: 反向传播完成")
    
    cleanup_distributed()
    print(f"Rank {rank}: 清理完成")

def main():
    print("DDP训练调试脚本")
    print("=" * 30)
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("CUDA不可用，退出")
        return
    
    print(f"CUDA设备数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 获取GPU数量
    world_size = torch.cuda.device_count()
    if len(sys.argv) > 1:
        try:
            world_size = int(sys.argv[1])
        except ValueError:
            print(f"无效的GPU数量参数，使用默认值 {world_size}")
    
    print(f"使用 {world_size} 个GPU进行测试")
    
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    # 启动多进程
    mp.spawn(simple_model_test,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    
    print("DDP测试完成!")

if __name__ == "__main__":
    main()