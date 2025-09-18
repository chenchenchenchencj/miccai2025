#!/bin/bash

# 设置环境变量
export PYTHONPATH=/media/zdp1/Datas1/cly/U-Mamba/umamba:$PYTHONPATH
export nnUNet_raw="/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw"
export nnUNet_preprocessed="/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_preprocessed"
export nnUNet_results="/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_trained_models/nnUNet_results"
export nnUNet_n_proc_DA=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 打印环境变量确认
echo "环境变量设置:"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  nnUNet_raw: $nnUNet_raw"
echo "  nnUNet_preprocessed: $nnUNet_preprocessed"
echo "  nnUNet_results: $nnUNet_results"
echo "  nnUNet_n_proc_DA: $nnUNet_n_proc_DA"
echo ""

# 根据参数选择不同的注意力机制组合
if [ "$1" == "enhanced" ]; then
    echo "使用增强版Mamba编码器 + 坐标注意力解码器"
    ENCODER_ATTENTION=enhanced_mamba
    DECODER_ATTENTION=coord_attention
elif [ "$1" == "basic" ]; then
    echo "使用基础Mamba编码器 + CBAM解码器"
    ENCODER_ATTENTION=mamba
    DECODER_ATTENTION=cbam
elif [ "$1" == "coord" ]; then
    echo "使用坐标注意力编码器 + CBAM解码器"
    ENCODER_ATTENTION=coord_attention
    DECODER_ATTENTION=cbam
elif [ "$1" == "ddp" ]; then
    echo "使用DDP分布式训练"
    echo "启动4 GPU分布式训练..."
    torchrun --nproc_per_node=4 /media/zdp1/Datas1/cly/U-Mamba/umamba/nnunetv2/run/run_training.py 1 3d_fullres all -tr nnUNetTrainerMultiAttention
    exit 0
else
    echo "使用默认配置: 坐标注意力编码器 + CBAM解码器"
    ENCODER_ATTENTION=coord_attention
    DECODER_ATTENTION=cbam
fi

# 设置GPU设备
if [ -z "$2" ]; then
    GPU_DEVICES="0,1,2,3"
else
    GPU_DEVICES=$2
fi

echo "使用GPU设备: $GPU_DEVICES"
echo ""

# 构建训练命令
TRAIN_CMD="nnUNetv2_train 1 3d_fullres all -tr nnUNetTrainerMultiAttention"
echo "训练命令: CUDA_VISIBLE_DEVICES=$GPU_DEVICES ENCODER_ATTENTION=$ENCODER_ATTENTION DECODER_ATTENTION=$DECODER_ATTENTION $TRAIN_CMD"
echo ""

# 设置注意力机制环境变量并执行训练
CUDA_VISIBLE_DEVICES=$GPU_DEVICES ENCODER_ATTENTION=$ENCODER_ATTENTION DECODER_ATTENTION=$DECODER_ATTENTION $TRAIN_CMD