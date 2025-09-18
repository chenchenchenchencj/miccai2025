import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


class nnUNetTrainerUNetDepth(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # 可以在这里调整训练参数
        # 例如，更深的网络可能需要更小的学习率
        # self.initial_lr = 1e-3

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        构建具有可调整深度的3D UNet网络
        """
        # 获取原始配置参数
        num_stages = len(configuration_manager.conv_kernel_sizes)
        dim = len(configuration_manager.conv_kernel_sizes[0])
        conv_op = convert_dim_to_conv_op(dim)

        label_manager = plans_manager.get_label_manager(dataset_json)

        # 调整网络深度
        # 原始阶段数
        original_num_stages = num_stages
        
        # 根据图像尺寸和GPU内存情况调整网络深度
        patch_size = configuration_manager.patch_size
        smallest_patch_dim = min(patch_size)
        
        # 对于3D分割，根据patch size确定合适的网络深度
        # 一般来说，较小的patch size不需要太深的网络
        if smallest_patch_dim < 64:
            target_num_stages = min(5, original_num_stages)  # 减少阶段数
        elif smallest_patch_dim < 128:
            target_num_stages = min(6, original_num_stages)  # 适中阶段数
        else:
            target_num_stages = min(7, original_num_stages)  # 增加阶段数，但不超过7
        
        # 确保我们不会增加阶段数超过原始配置
        target_num_stages = min(target_num_stages, original_num_stages)
        
        print(f"调整网络深度: 原始阶段数={original_num_stages}, 调整后阶段数={target_num_stages}")
        
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