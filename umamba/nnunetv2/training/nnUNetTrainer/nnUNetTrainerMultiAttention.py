import os
import torch
from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.nets.attention_blocks import CoordAttentionBlock3D, MambaBlock3D, EnhancedMambaBlock3D
from nnunetv2.nets.unet_decoder_with_attention import UNetDecoderWithAttention


class PlainConvUNetWithMultiAttention(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: tuple,
                 conv_op: type,
                 kernel_sizes: tuple,
                 strides: tuple,
                 n_conv_per_stage: tuple,
                 num_classes: int,
                 n_conv_per_stage_decoder: tuple,
                 conv_bias: bool = False,
                 norm_op: type = nn.InstanceNorm3d,
                 norm_op_kwargs: dict = None,
                 dropout_op: type = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: type = nn.LeakyReLU,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 encoder_attention: str = "coord_attention",
                 decoder_attention: str = "cbam"):
        """
        带有多注意力机制的3D UNet实现
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
            
        # 使用带有注意力机制的编码器
        self.encoder = PlainConvEncoderWithAttention(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
            n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
            dropout_op_kwargs, nonlin, nonlin_kwargs, encoder_attention)
        
        # 使用带注意力机制的解码器
        self.decoder = UNetDecoderWithAttention(
            self.encoder, 
            num_classes, 
            n_conv_per_stage_decoder, 
            deep_supervision,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            attention_type=decoder_attention
        )

    def forward(self, x):
        skips = self.encoder(x)
        result = self.decoder(skips)
        return result

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == self.encoder.conv_op.dim, "Input size format incorrect. Example: (64, 128, 128, 128) for 3D"
        output = self.encoder.compute_conv_feature_map_size(input_size)
        output += self.decoder.compute_conv_feature_map_size(input_size)
        return output


class PlainConvEncoderWithAttention(PlainConvEncoder):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: tuple,
                 conv_op: type,
                 kernel_sizes: tuple,
                 strides: tuple,
                 n_conv_per_stage: tuple,
                 conv_bias: bool = False,
                 norm_op: type = nn.InstanceNorm3d,
                 norm_op_kwargs: dict = None,
                 dropout_op: type = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: type = nn.LeakyReLU,
                 nonlin_kwargs: dict = None,
                 attention_type: str = "coord_attention"):
        """
        带有注意力机制的编码器
        """
        super(PlainConvEncoder, self).__init__()
        self.input_channels = input_channels
        self.n_stages = n_stages
        self.features_per_stage = features_per_stage
        self.conv_op = conv_op
        self.output_channels = features_per_stage
        self.conv_bias = conv_bias
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.strides = strides
        self.kernel_sizes = kernel_sizes
        self.return_skips = True
        
        # 定义池化操作
        self.stages = nn.ModuleList()
        for s in range(n_stages):
            stage = nn.Sequential()
            
            # 添加下采样层（除了第一阶段）
            if s > 0:
                stage.add_module('downsample', 
                                 conv_op(features_per_stage[s-1], features_per_stage[s], 
                                        kernel_size=strides[s], stride=strides[s], bias=False))
            
            # 添加卷积块（带注意力机制）
            for i in range(n_conv_per_stage[s]):
                # 正确计算输入和输出通道数
                in_channels = input_channels if s == 0 and i == 0 else features_per_stage[s]
                
                # 根据注意力类型添加相应的注意力块
                if attention_type == "coord_attention":
                    stage.add_module(f'conv_block_{i}', 
                                     CoordAttentionBlock3D(in_channels, features_per_stage[s]))
                elif attention_type == "mamba":
                    stage.add_module(f'conv_block_{i}', 
                                     MambaBlock3D(in_channels, features_per_stage[s]))
                elif attention_type == "enhanced_mamba":
                    stage.add_module(f'conv_block_{i}', 
                                     EnhancedMambaBlock3D(in_channels, features_per_stage[s]))
                else:  # 默认不使用注意力机制
                    from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
                    stage.add_module(f'conv_block_{i}',
                                     StackedConvBlocks(
                                         n_conv_per_stage=1,
                                         conv_op=conv_op,
                                         input_channels=in_channels,
                                         output_channels=features_per_stage[s],
                                         kernel_size=kernel_sizes[s],
                                         stride=1,
                                         conv_bias=conv_bias,
                                         norm_op=norm_op,
                                         norm_op_kwargs=norm_op_kwargs,
                                         dropout_op=dropout_op,
                                         dropout_op_kwargs=dropout_op_kwargs,
                                         nonlin=nonlin,
                                         nonlin_kwargs=nonlin_kwargs
                                     ))
            
            self.stages.append(stage)
            
        # 初始化权重
        self.apply(InitWeights_He(1e-2))
    def _create_conv_block(self, in_channels, out_channels):
        """
        创建普通的卷积块
        """
        conv_op = self.conv_op
        norm_op = self.norm_op
        nonlin = self.nonlin
        
        return nn.Sequential(
            conv_op(in_channels, out_channels, kernel_size=3, padding=1, bias=self.conv_bias),
            norm_op(out_channels, **self.norm_op_kwargs) if self.norm_op_kwargs else norm_op(out_channels),
            nonlin(**self.nonlin_kwargs) if self.nonlin_kwargs else nonlin()
        )
        
    def forward(self, x):
        """
        重写forward方法以确保正确返回skip连接
        """
        skips = []
        for s in self.stages:
            x = s(x)
            if self.return_skips:
                skips.append(x)
        if self.return_skips:
            return skips
        else:
            return x


class nnUNetTrainerMultiAttention(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        # ---- 先读取环境变量并设置默认属性，避免父类__init__里调用 _set_batch_size_and_oversample 时找不到属性 ----
        self.encoder_attention = os.environ.get('ENCODER_ATTENTION', 'coord_attention')
        self.decoder_attention = os.environ.get('DECODER_ATTENTION', 'cbam')
        self.target_num_stages = int(os.environ.get('UNET_3D_STAGES', -1))
        self.batch_size_factor = float(os.environ.get('BATCH_SIZE_FACTOR', 1.0))
        # 父类初始化前先假设非DDP，父类里可能会调用 _set_batch_size_and_oversample
        self.is_ddp = False
        # 先调用父类完成基础初始化（其中会调用 _set_batch_size_and_oversample）
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        # ---- 之后再确定DDP真实状态并打印 ----
        self.is_ddp = dist.is_available() and dist.is_initialized()
        if self.is_ddp:
            self.local_rank = dist.get_rank()
            print(f"Local rank {self.local_rank}: DDP环境已初始化")
            print(f"  World size: {dist.get_world_size()}")
            print(f"  Backend: {dist.get_backend()}")
        else:
            self.local_rank = 0
            print(f"单GPU/非DDP模式，使用设备: {device}")

        # 根据网络深度调整优化器超参数（若需要）
        if self.target_num_stages > 6:
            self.initial_lr = 1e-3
            self.weight_decay = 3e-5

        # 若 batch_size_factor != 1，在父类初始化已经设置过 batch_size 的情况下我们再显式调整一次并提示
        if self.batch_size_factor != 1.0:
            original_batch_size = getattr(self, 'batch_size', None)
            if original_batch_size is not None:
                new_batch_size = max(1, int(original_batch_size * self.batch_size_factor))
                if new_batch_size != self.batch_size:
                    print(f"根据因子再次调整batch size: {self.batch_size} -> {new_batch_size}")
                    self.batch_size = new_batch_size

    def _set_batch_size_and_oversample(self):
        """覆写: 在父类__init__期间调用。需确保所用属性已存在。"""
        batch_size_factor = getattr(self, 'batch_size_factor', 1.0)
        is_ddp = getattr(self, 'is_ddp', False)

        if not is_ddp:
            original_batch_size = self.configuration_manager.batch_size
            self.batch_size = max(1, int(original_batch_size * batch_size_factor))
            print(f"单GPU模式，batch size设置为: {self.batch_size} (factor={batch_size_factor})")
        else:
            original_batch_size = self.configuration_manager.batch_size
            adjusted_batch_size = max(1, int(original_batch_size * batch_size_factor))
            world_size = dist.get_world_size()
            my_rank = dist.get_rank()

            # 处理batch size 小于 GPU 数量：自动扩展或给出提示
            if adjusted_batch_size < world_size:
                if my_rank == 0:
                    print(f"警告: 原始batch size={adjusted_batch_size} < GPU数={world_size}。自动设置为{world_size} (每GPU 1)。\n"
                          f"若需自定义请设置环境变量：export BATCH_SIZE_FACTOR=比例 或 修改plans中的batch size。")
                adjusted_batch_size = world_size

            global_batch_size = adjusted_batch_size
            batch_size_per_GPU = int(np.ceil(global_batch_size / world_size))

            sample_id_low = 0 if my_rank == 0 else np.sum([batch_size_per_GPU] * my_rank)
            sample_id_high = np.sum([batch_size_per_GPU] * (my_rank + 1))
            # 截断到 global_batch_size 避免越界
            sample_id_high = min(sample_id_high, global_batch_size)

            oversample = [True if not i < round(global_batch_size * (1 - self.oversample_foreground_percent)) else False
                          for i in range(global_batch_size)]

            if sample_id_low >= global_batch_size:  # 多余进程（极端情况下）
                print(f"Rank {my_rank}: 没有可分配样本，进程将空转。建议减少GPU或增大batch size。")
                self.batch_size = 0
                self.oversample_foreground_percent = 0.0
                return

            effective_span = sample_id_high - sample_id_low
            if effective_span < batch_size_per_GPU:
                batch_size_per_GPU = effective_span

            if sample_id_high / global_batch_size < (1 - self.oversample_foreground_percent):
                oversample_percent = 0.0
            elif sample_id_low / global_batch_size > (1 - self.oversample_foreground_percent):
                oversample_percent = 1.0
            else:
                oversample_percent = sum(oversample[sample_id_low:sample_id_high]) / max(1, batch_size_per_GPU)

            print(f"DDP模式 - worker {my_rank}: global_batch {global_batch_size}, perGPU {batch_size_per_GPU}, oversample {oversample_percent:.3f} (factor={batch_size_factor})")
            self.batch_size = batch_size_per_GPU
            self.oversample_foreground_percent = oversample_percent

    def run_training(self):
        """覆写，增加tqdm进度条 (仅rank0)。通过环境变量 PROGRESS_BAR=1 开启。"""
        use_bar = os.environ.get('PROGRESS_BAR', '0').lower() in ('1', 'true', 'yes', 'y')
        if use_bar:
            try:
                from tqdm import tqdm
            except ImportError:
                if self.local_rank == 0:
                    print('未安装tqdm，关闭进度条。 pip install tqdm 可开启。')
                use_bar = False
        self.on_train_start()
        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            self.on_train_epoch_start()
            train_outputs = []
            if use_bar and self.local_rank == 0:
                iter_obj = tqdm(range(self.num_iterations_per_epoch), desc=f'Epoch {epoch} [train]', leave=False)
            else:
                iter_obj = range(self.num_iterations_per_epoch)
            for batch_id in iter_obj:
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                if use_bar and self.local_rank == 0:
                    val_iter = tqdm(range(self.num_val_iterations_per_epoch), desc=f'Epoch {epoch} [val]', leave=False)
                else:
                    val_iter = range(self.num_val_iterations_per_epoch)
                for batch_id in val_iter:
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)
            self.on_epoch_end()
        self.on_train_end()

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        构建带有多种注意力机制的3D UNet网络
        
        网络深度可以通过环境变量 UNET_3D_STAGES 控制:
        - -1 (默认): 自动根据patch size和GPU内存优化
        - 4-7: 指定具体的网络阶段数
        
        注意力机制可以通过环境变量控制:
        - ENCODER_ATTENTION: 编码器注意力机制 ("coord_attention", "mamba", "enhanced_mamba", "none")
        - DECODER_ATTENTION: 解码器注意力机制 ("cbam", "coord_attention", "se_block", "eca", "sa", "self_attention")
        
        特点:
        1. 集成多种注意力机制增强特征表示
        2. 自动调整网络深度以适应3D任务
        3. 优化特征图数量平衡精度和内存使用
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
        
        # 获取注意力类型
        encoder_attention = os.environ.get('ENCODER_ATTENTION', 'coord_attention')
        decoder_attention = os.environ.get('DECODER_ATTENTION', 'cbam')
        
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
        
        print(f"3D UNet with Multi Attention配置:")
        print(f"  原始阶段数={original_num_stages}, 目标阶段数={target_num_stages}")
        print(f"  编码器注意力机制={encoder_attention}, 解码器注意力机制={decoder_attention}")
        
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
        model = PlainConvUNetWithMultiAttention(
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
            deep_supervision=enable_deep_supervision,
            encoder_attention=encoder_attention,
            decoder_attention=decoder_attention
        )
        
        return model
