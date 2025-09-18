#!/bin/bash

# 设置环境变量
export PYTHONPATH=/media/zdp1/Datas1/cly/U-Mamba/umamba:$PYTHONPATH
export nnUNet_raw="/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw"
export nnUNet_preprocessed="/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_preprocessed"
export nnUNet_results="/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_trained_models/nnUNet_results"
export nnUNet_n_proc_DA=1  # 减少数据加载进程以节省内存
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 打印环境变量确认
echo "环境变量设置:"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  nnUNet_raw: $nnUNet_raw"
echo "  nnUNet_preprocessed: $nnUNet_preprocessed"
echo "  nnUNet_results: $nnUNet_results"
echo "  nnUNet_n_proc_DA: $nnUNet_n_proc_DA"
echo ""

# 设置注意力机制
export ENCODER_ATTENTION=enhanced_mamba
export DECODER_ATTENTION=coord_attention

echo "使用注意力机制配置:"
echo "  编码器注意力: $ENCODER_ATTENTION"
echo "  解码器注意力: $DECODER_ATTENTION"
echo ""

# 设置GPU数量
NGPUS=4
if [ ! -z "$1" ]; then
    NGPUS=$1
fi

echo "使用GPU数量: $NGPUS"
echo ""

# 构建训练命令
TRAIN_SCRIPT="/media/zdp1/Datas1/cly/U-Mamba/umamba/nnunetv2/run/run_training.py"
echo "训练命令:"
echo "torchrun --nproc_per_node=$NGPUS $TRAIN_SCRIPT 1 3d_fullres all -tr nnUNetTrainerMultiAttention"
echo ""

# 执行训练
echo "开始DDP分布式训练..."
torchrun --nproc_per_node=$NGPUS $TRAIN_SCRIPT 1 3d_fullres all -tr nnUNetTrainerMultiAttention