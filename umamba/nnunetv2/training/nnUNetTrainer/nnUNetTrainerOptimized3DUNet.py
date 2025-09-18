import os
import torch
from torch import nn
import numpy as np
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


class nnUNetTrainerOptimized3DUNet(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # 从环境变量获取网络深度设置
        self.target_num_stages = int(os.environ.get('UNET_3D_STAGES', -1))
        
        # 根据网络深度调整超参数
        if self.target_num_stages > 6:
            # 更深的网络需要更小的学习率和更大的权重衰减
            self.initial_lr = 5e-3
            self.weight_decay = 5e-5

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        构建针对3D分割任务优化的UNet网络
        
        网络深度可以通过环境变量 UNET_3D_STAGES 控制:
        - -1 (默认): 自动根据patch size和GPU内存优化
        - 4-7: 指定具体的网络阶段数
        
        对于3D分割任务的特殊优化:
        1. 根据patch size自动调整网络深度
        2. 优化特征图数量以平衡精度和内存使用
        3. 调整卷积块数量以适应3D计算复杂度
        """
        # 获取原始配置参数
        original_num_stages = len(configuration_manager.conv_kernel_sizes)
        dim = len(configuration_manager.conv_kernel_sizes[0])
        conv_op = convert_dim_to_conv_op(dim)

        label_manager = plans_manager.get_label_manager(dataset_json)

        # 验证是否为3D任务
        if dim != 3:
            raise ValueError("此训练器专为3D分割任务设计，当前任务不是3D任务")
        
        # 获取目标阶段数（网络深度）
        target_num_stages_env = int(os.environ.get('UNET_3D_STAGES', -1))
        
        if target_num_stages_env == -1:
            # 自动调整网络深度，针对3D任务优化
            patch_size = configuration_manager.patch_size
            volume = np.prod(patch_size)  # 3D patch的体素数量
            
            # 根据3D patch体积确定合适的网络深度
            if volume < 64*64*64:  # 相对较小的patch
                target_num_stages = 5
            elif volume < 128*128*128:  # 中等大小patch
                target_num_stages = 6
            else:  # 大型patch
                target_num_stages = min(7, original_num_stages)
        else:
            # 使用指定的阶段数
            target_num_stages = max(4, min(target_num_stages_env, 8))
        
        # 确保不超过原始配置的阶段数
        target_num_stages = min(target_num_stages, original_num_stages)
        
        print(f"3D UNet优化配置: 原始阶段数={original_num_stages}, 目标阶段数={target_num_stages}")
        
        # 针对3D任务优化特征图数量，避免显存不足
        base_num_features = configuration_manager.UNet_base_num_features
        max_num_features = configuration_manager.unet_max_num_features
        
        # 对于3D任务，适当减少特征图数量以节省显存
        if target_num_stages >= 6:
            base_num_features = min(base_num_features, 24)  # 减少基础特征数
            max_num_features = min(max_num_features, 256)   # 减少最大特征数
        
        features_per_stage = [min(base_num_features * 2 ** i, max_num_features) for i in range(target_num_stages)]
        print(f"特征图数量: {features_per_stage}")
        
        # 调整卷积核大小和步长配置
        conv_kernel_sizes = configuration_manager.conv_kernel_sizes[:target_num_stages]
        pool_op_kernel_sizes = configuration_manager.pool_op_kernel_sizes[:target_num_stages]
        
        # 针对3D任务优化卷积块数量
        # 更深的网络减少每个阶段的卷积层数以控制计算复杂度
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
            
        # 对于更深的网络，减少部分阶段的卷积层数
        if target_num_stages >= 6:
            # 减少最后几个阶段的卷积层数以控制计算量
            for i in range(max(0, target_num_stages-2), target_num_stages):
                n_conv_per_stage_encoder[i] = min(n_conv_per_stage_encoder[i], 2)

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
