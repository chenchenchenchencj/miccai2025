#!/bin/bash

# Enhanced Mamba 3D UNet with Coord Attention - DDP Training Script

# 设置环境变量
export PYTHONPATH=/media/zdp1/Datas1/cly/U-Mamba/umamba:$PYTHONPATH
export nnUNet_raw="/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw"
export nnUNet_preprocessed="/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_preprocessed"
export nnUNet_results="/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_trained_models/nnUNet_results"
export nnUNet_n_proc_DA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========================================"
echo "Enhanced Mamba 3D UNet DDP Training"
echo "========================================"
echo "Environment variables:"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  nnUNet_raw: $nnUNet_raw"
echo "  nnUNet_preprocessed: $nnUNet_preprocessed"
echo "  nnUNet_results: $nnUNet_results"
echo "  nnUNet_n_proc_DA: $nnUNet_n_proc_DA"
echo ""

# 设置注意力机制
export ENCODER_ATTENTION=enhanced_mamba
export DECODER_ATTENTION=coord_attention

echo "Attention mechanisms:"
echo "  Encoder: $ENCODER_ATTENTION"
echo "  Decoder: $DECODER_ATTENTION"
echo ""

# 设置GPU数量 (默认使用4个GPU)
NGPUS=4
if [ ! -z "$1" ]; then
    NGPUS=$1
fi

echo "Using $NGPUS GPUs"
echo ""

# 运行DDP训练
echo "Starting DDP training with torchrun..."
echo "Command: torchrun --nproc_per_node=$NGPUS /media/zdp1/Datas1/cly/U-Mamba/umamba/nnunetv2/run/run_training.py 1 3d_fullres all -tr nnUNetTrainerMultiAttention"
echo ""

torchrun --nproc_per_node=$NGPUS /media/zdp1/Datas1/cly/U-Mamba/umamba/nnunetv2/run/run_training.py 1 3d_fullres all -tr nnUNetTrainerMultiAttention