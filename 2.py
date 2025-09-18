import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import join

# 初始化预测器
predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# 从训练好的模型文件夹初始化
model_path = "/media/zdp1/Datas1/cly/U-Mamba/umamba/nnunetv2/training/nnUNetTrainer"  # 替换为你的模型路径
predictor.initialize_from_trained_model_folder(
    model_training_output_dir=model_path,
    use_folds=None,  # 自动检测可用fold
    checkpoint_name='/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_trained_models/nnUNet_results/Dataset001_Teeth/SegResNet__nnUNetPlans__3d_fullres/fold_all/checkpoint_best.pth'
)

# 执行推理
input_file = "/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw/Dataset001_Teeth/detr/NPZ_0001_0000.nii.gz"  # 替换为输入文件路径
output_file = "/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw/Dataset001_Teeth/NPZ_0001_0000.nii.gz"  # 替换为输出文件路径

predictor.predict_from_files(
    list_of_lists_or_source_folder=input_file,
    output_folder_or_list_of_truncated_output_files=output_file,
    save_probabilities=False,
    overwrite=True
)