import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.file_path_utilities import get_output_folder
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, isdir

def run_inference(
    input_folder,
    output_folder,
    dataset_id,
    configuration,
    trainer_name="SegResNet",
    folds=(0, 1, 2, 3, 4),
    checkpoint_name="checkpoint_final.pth",
    device="cuda",
    step_size=0.5,
    disable_tta=True,
    save_probabilities=False,
    overwrite=True,
    num_processes_preprocessing=2,
    num_processes_segmentation_export=2,
    folder_with_segs_from_prev_stage=None,
    verbose=False
):
    # 获取模型输出目录
    model_folder = get_output_folder(dataset_id, trainer_name, "nnUNetPlans", configuration)
    if not isdir(output_folder):
        maybe_mkdir_p(output_folder)
    # 设置设备
    device = torch.device(device)
    # 创建推理器
    predictor = nnUNetPredictor(
        tile_step_size=step_size,
        use_gaussian=True,
        use_mirroring=not disable_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=verbose,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        model_folder,
        folds,
        checkpoint_name=checkpoint_name
    )
    predictor.predict_from_files(
        input_folder,
        output_folder,
        save_probabilities=save_probabilities,
        overwrite=overwrite,
        num_processes_preprocessing=num_processes_preprocessing,
        num_processes_segmentation_export=num_processes_segmentation_export,
        folder_with_segs_from_prev_stage=folder_with_segs_from_prev_stage,
        num_parts=1,
        part_id=0
    )

if __name__ == "__main__":
    # 这里填写你的实际参数
    run_inference(
        input_folder="/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw/Dataset001_Teeth/detr",
        output_folder="/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_raw/Dataset001_Teeth/result",
        dataset_id="1",
        configuration="3d_lowres",
        trainer_name="SegResNet",
        folds=(0, 1, 2, 3, 4),
        checkpoint_name="/media/zdp1/Datas1/cly/U-Mamba/data/nnUNet_trained_models/nnUNet_results/Dataset001_Teeth/SegResNet__nnUNetPlans__3d_lowres/fold_all/checkpoint_final.pth",
        device="cuda",
        step_size=0.5,
        disable_tta=True,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        verbose=True
    )