import nibabel as nib
import numpy as np

label_path = "/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw/Dataset001_Teeth/labelsTr/NPZ_0001.nii.gz"
img = nib.load(label_path)
data = img.get_fdata()
print("Unique labels:", np.unique(data))
#nnUNet_n_proc_DA=0 CUDA_VISIBLE_DEVICES=0 nnUNetv2_train Dataset001_Teeth 3d all -tr nnUNetTrainerUMambaEnc -device cuda
# export PYTHONPATH=/media/zdp1/Datas1/cly/U-Mamba/umamba: