#!/bin/bash

# 设置必要的环境变量
export nnUNet_raw="/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw"
export nnUNet_preprocessed="/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_preprocessed"
export nnUNet_results="/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_trained_models/nnUNet_results"

# 添加umamba到PYTHONPATH
export PYTHONPATH="/media/zdp1/Datas1/cly/U-Mamba/umamba":$PYTHONPATH

echo "================== 环境变量设置 =================="
echo "nnUNet_raw: $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "nnUNet_results: $nnUNet_results"
echo "PYTHONPATH: $PYTHONPATH"
echo "================================================="

# 检查数据集ID
if [ -z "$1" ]; then
    echo "使用默认数据集ID: 1"
    DATASET_ID=1
else
    DATASET_ID=$1
fi

# 检查fold
if [ -z "$2" ]; then
    echo "使用默认fold: all"
    FOLD="all"
else
    FOLD=$2
fi

# 检查训练器
if [ -z "$3" ]; then
    echo "使用默认训练器: nnUNetTrainerUNetDepth"
    TRAINER="nnUNetTrainerUNetDepth"
else
    TRAINER=$3
fi

echo ""
echo "================== 训练配置 =================="
echo "数据集ID: $DATASET_ID"
echo "Fold: $FOLD"
echo "训练器: $TRAINER"
echo "=============================================="

# 运行训练命令
echo ""
echo "开始训练..."
echo "命令: nnUNetv2_train $DATASET_ID 3d_fullres $FOLD -tr $TRAINER"

nnUNetv2_train $DATASET_ID 3d_fullres $FOLD -tr $TRAINER