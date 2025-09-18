#!/bin/bash

# U-Mamba训练配置脚本
echo "设置U-Mamba训练环境..."

# 设置必要的环境变量
export PYTHONPATH=/media/zdp1/Datas1/cly/U-Mamba/umamba:$PYTHONPATH
export nnUNet_raw="/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw"
export nnUNet_preprocessed="/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_preprocessed"
export nnUNet_results="/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_trained_models/nnUNet_results"

# 可选：减少数据增强进程数以降低内存使用
export nnUNet_n_proc_DA=2

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
else
    echo "使用默认配置: 增强版Mamba编码器 + 坐标注意力解码器"
    ENCODER_ATTENTION=enhanced_mamba
    DECODER_ATTENTION=coord_attention
fi

# 设置GPU设备
GPU_DEVICE=${2:-0}
echo "使用GPU设备: $GPU_DEVICE"
echo ""

# 构建训练命令
TRAIN_CMD="CUDA_VISIBLE_DEVICES=$GPU_DEVICE nnUNetv2_train 1 3d_fullres all -tr nnUNetTrainerMultiAttention"
echo "训练命令: $TRAIN_CMD"
echo ""

# 设置注意力机制环境变量并执行训练
ENCODER_ATTENTION=$ENCODER_ATTENTION DECODER_ATTENTION=$DECODER_ATTENTION $TRAIN_CMD