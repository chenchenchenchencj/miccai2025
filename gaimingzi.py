import os
import re


def rename_files_to_npz(folder_path):
    """
    将文件夹内的文件按 NPZ_XXX.nii.gz 格式批量重命名
    命名规则：NPZ_001.nii.gz, NPZ_002.nii.gz, ..., NPZ_099.nii.gz 等

    参数:
        folder_path: 包含需要重命名文件的文件夹路径
    """
    # 获取文件夹中所有文件（不包括子文件夹）
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # 仅处理.nii.gz文件（可选）
    # all_files = [f for f in all_files if f.lower().endswith('.nii.gz')]

    # 按文件名自然排序（保持原顺序）
    all_files.sort(key=lambda f: [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', f)])

    # 开始重命名
    for i, filename in enumerate(all_files, start=1):
        # 构造新文件名：NPZ_XXX.nii.gz（XXX为3位数字序号）
        new_name = f"NPZ_{i:04d}_0000{os.path.splitext(filename)[1]}"

        # 如果是双层扩展名（如.nii.gz）
        if filename.lower().endswith('.nii.gz'):
            new_name = f"NPZ_{i:04d}_0000.nii.gz"

        # 完整文件路径
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)

        # 重命名文件
        os.rename(old_path, new_path)
        print(f"重命名: {filename} -> {new_name}")


# 使用示例
if __name__ == "__main__":
    # 替换为您的实际文件夹路径
    target_folder = "/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw/Dataset001_Teeth/imagesTr"

    # 执行重命名操作（谨慎使用，建议先备份数据）
    rename_files_to_npz(target_folder)
    # nnUNet_convert_decathlon_task -i /home/zdp1/U-Mamba/data/nnUNet_raw/Dataset001_CBCT
    #nnUNet_plan_and_preprocess -t 1
    #export PYTHONPATH=/media/zdp1/Datas1/cly/U-Mamba/umamba:$PYTHONPATH
    #CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 1 3d_lowres all -tr nnUNetTrainerUMambaEnc
    #CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 1 3d_lowres all -tr nnUNetTrainerUMambaBot

    # CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict     -i /media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw/Dataset001_Teeth/detr     -o /media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw/Dataset001_Teeth/result1     -d 1     -c 3d_fullres    -tr nnUNetTrainer -f all
    # CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 1 3d_lowres 2
    # CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 1 3d_fullres all
    #CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 1 3d_fullres all -tr nnUNetTrainerUNetDepth
    # CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict     -i /media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw/Dataset001_Teeth/detr     -o /media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw/Dataset001_Teeth/result1     -d 1     -c 3d_lowres     -tr nnUNetTrainer     -f all
    # CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict     -i /media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw/Dataset001_Teeth/detr     -o /media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw/Dataset001_Teeth/result2     -d 1     -c 3d_fullres     -tr SegResNet -f all
    #ENCODER_ATTENTION = enhanced_mamba DECODER_ATTENTION = coord_attention CUDA_VISIBLE_DEVICES = 0 nnUNetv2_train 1 3d_fullres all - tr nnUNetTrainerMultiAttention