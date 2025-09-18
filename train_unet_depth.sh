#!/bin/bash

# 设置必要的环境变量
export nnUNet_raw="/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw"
export nnUNet_preprocessed="/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_preprocessed"
export nnUNet_results="/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_results"

# 创建目录（如果不存在）
mkdir -p $nnUNet_raw
mkdir -p $nnUNet_preprocessed
mkdir -p $nnUNet_results

# 添加umamba到PYTHONPATH
export PYTHONPATH="/media/zdp1/Datas1/cly/U-Mamba/umamba":$PYTHONPATH

echo "环境变量设置完成:"
echo "nnUNet_raw: $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "nnUNet_results: $nnUNet_results"
echo "PYTHONPATH: $PYTHONPATH"

# 检查是否提供了数据集ID
if [ -z "$1" ]; then
    echo "请提供数据集ID，例如:"
    echo "  ./train_unet_depth.sh 1"
    echo "  ./train_unet_depth.sh 1 0  # 指定fold"
    exit 1
fi

DATASET_ID=$1
FOLD=${2:-all}  # 如果没有提供fold，默认使用'all'
TRAINER=${3:-nnUNetTrainerUNetDepth}  # 如果没有提供训练器，默认使用nnUNetTrainerUNetDepth

echo "开始训练:"
echo "  数据集ID: $DATASET_ID"
echo "  Fold: $FOLD"
echo "  训练器: $TRAINER"

# 运行训练
nnUNetv2_train $DATASET_ID 3d_fullres $FOLD -tr $TRAINER
