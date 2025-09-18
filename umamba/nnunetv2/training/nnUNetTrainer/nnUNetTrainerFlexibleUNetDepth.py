import os
import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


class nnUNetTrainerFlexibleUNetDepth(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # 从环境变量获取网络深度设置，如果未设置则使用默认值
        self.target_num_stages = int(os.environ.get('UNET_TARGET_STAGES', -1))
        
        # 可以根据网络深度调整其他超参数
        if self.target_num_stages > 6:
            # 更深的网络可能需要更小的学习率
            self.initial_lr = 1e-3
        elif self.target_num_stages > 7:
            # 非常深的网络需要更小的学习率和更大的权重衰减
            self.initial_lr = 5e-4
            self.weight_decay = 5e-5

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        构建具有灵活深度的3D UNet网络
        可以通过设置环境变量 UNET_TARGET_STAGES 来控制网络深度:
        - 设置为-1（默认）：自动根据patch size调整深度
        - 设置为具体数值（如5, 6, 7等）：使用指定的阶段数
        """
        # 获取原始配置参数
        num_stages = len(configuration_manager.conv_kernel_sizes)
        dim = len(configuration_manager.conv_kernel_sizes[0])
        conv_op = convert_dim_to_conv_op(dim)

        label_manager = plans_manager.get_label_manager(dataset_json)

        # 获取目标阶段数（网络深度）
        target_num_stages = int(os.environ.get('UNET_TARGET_STAGES', -1))
        
        if target_num_stages == -1:
            # 自动调整网络深度
            patch_size = configuration_manager.patch_size
            smallest_patch_dim = min(patch_size)
            
            # 对于3D分割，根据patch size确定合适的网络深度
            if smallest_patch_dim < 48:
                target_num_stages = 4
            elif smallest_patch_dim < 96:
                target_num_stages = 5
            elif smallest_patch_dim < 128:
                target_num_stages = 6
            else:
                target_num_stages = min(7, num_stages)  # 最多7个阶段
        else:
            # 使用指定的阶段数，但确保不超过合理范围
            target_num_stages = max(3, min(target_num_stages, 8))
        
        # 确保不超过原始配置的阶段数
        target_num_stages = min(target_num_stages, num_stages)
        
        print(f"构建网络: 原始阶段数={num_stages}, 目标阶段数={target_num_stages}")
        
        # 根据目标阶段数调整配置
        features_per_stage = [min(configuration_manager.UNet_base_num_features * 2 ** i,
                                 configuration_manager.unet_max_num_features) for i in range(target_num_stages)]
        
        # 调整卷积核大小和步长配置
        conv_kernel_sizes = configuration_manager.conv_kernel_sizes[:target_num_stages]
        pool_op_kernel_sizes = configuration_manager.pool_op_kernel_sizes[:target_num_stages]
        
        # 调整每个阶段的卷积层数
        if hasattr(configuration_manager, 'n_conv_per_stage_encoder'):
            n_conv_per_stage_encoder = configuration_manager.n_conv_per_stage_encoder[:target_num_stages]
        else:
            # 默认每个阶段2个卷积层
            n_conv_per_stage_encoder = [2] * target_num_stages
            
        if hasattr(configuration_manager, 'n_conv_per_stage_decoder'):
            n_conv_per_stage_decoder = configuration_manager.n_conv_per_stage_decoder[:target_num_stages-1]
        else:
            # 默认每个阶段2个卷积层
            n_conv_per_stage_decoder = [2] * (target_num_stages - 1)

        # 构建网络
        model = PlainConvUNet(
            input_channels=num_input_channels,
            n_stages=target_num_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=conv_kernel_sizes,
            strides=pool_op_kernel_sizes,
            num_classes=label_manager.num_segmentation_heads,
            n_conv_per_stage=n_conv_per_stage_encoder,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=True,
            norm_op=get_matching_instancenorm(conv_op),
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            deep_supervision=enable_deep_supervision
        )
        
        # 初始化权重
        model.apply(InitWeights_He(1e-2))
        
        return model