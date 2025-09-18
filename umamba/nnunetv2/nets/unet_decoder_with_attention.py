import torch
from torch import nn
import torch.nn.functional as F
from typing import Union, Tuple, List
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim


class UNetDecoderWithAttention(nn.Module):
    def __init__(self,
                 encoder,
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision: bool = False,
                 nonlin=nn.LeakyReLU,
                 nonlin_kwargs=None,
                 dropout_op=None,
                 dropout_op_kwargs=None,
                 norm_op=nn.InstanceNorm3d,
                 norm_op_kwargs=None,
                 attention_type: str = "coord_attention"):
        """
        带有注意力机制的UNet解码器
        
        Args:
            encoder: 编码器网络
            num_classes: 分类数量
            n_conv_per_stage: 每个阶段的卷积层数量
            deep_supervision: 是否使用深度监督
            nonlin: 非线性激活函数
            nonlin_kwargs: 非线性激活函数参数
            dropout_op: dropout操作
            dropout_op_kwargs: dropout操作参数
            norm_op: 归一化操作
            norm_op_kwargs: 归一化操作参数
            attention_type: 注意力类型 ("coord_attention", "se_block", "cbam", "eca", "sa", "self_attention")
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage必须有n_stages_encoder - 1个条目"

        stages = []
        upsample_layers = []
        seg_layers = []
        
        # 初始化注意力模块类型
        self.attention_type = attention_type
        
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]
            
            # 上采样层
            upsample_layers.append(UpsampleLayer(
                conv_op=encoder.conv_op,
                input_channels=input_features_below,
                output_channels=input_features_skip,
                pool_op_kernel_size=stride_for_upsampling,
                mode='nearest'
            ))

            # 解码器阶段 - 添加注意力机制
            stage_modules = []
            
            # 第一个卷积块 (融合跳跃连接和上采样特征)
            stage_modules.append(ConvDropoutNormReLU(
                conv_op=encoder.conv_op,
                input_channels=2 * input_features_skip,
                output_channels=input_features_skip,
                kernel_size=encoder.kernel_sizes[-(s + 1)],
                conv_bias=encoder.conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=dropout_op,
                dropout_op_kwargs=dropout_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            ))
            
            # 添加注意力模块
            if attention_type == "coord_attention":
                stage_modules.append(CoordAttention3D(
                    inp=input_features_skip,
                    oup=input_features_skip
                ))
            elif attention_type == "se_block":
                stage_modules.append(SEBlock3D(
                    channel=input_features_skip,
                    reduction=16
                ))
            elif attention_type == "cbam":
                stage_modules.append(CBAM3D(
                    in_planes=input_features_skip,
                    ratio=16,
                    kernel_size=7
                ))
            elif attention_type == "eca":
                stage_modules.append(ECABlock3D(
                    channel=input_features_skip
                ))
            elif attention_type == "sa":
                stage_modules.append(SpatialAttention3DImproved(
                    kernel_size=7
                ))
            elif attention_type == "self_attention":
                stage_modules.append(SelfAttention3D(
                    in_channels=input_features_skip
                ))
            
            # 其余的卷积块
            for i in range(n_conv_per_stage[s-1] - 1):
                stage_modules.append(ConvDropoutNormReLU(
                    conv_op=encoder.conv_op,
                    input_channels=input_features_skip,
                    output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 1)],
                    conv_bias=encoder.conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                ))
                
                # 在每个卷积块后也可以添加注意力模块（可选）
                if attention_type == "coord_attention":
                    stage_modules.append(CoordAttention3D(
                        inp=input_features_skip,
                        oup=input_features_skip
                    ))
                elif attention_type == "se_block":
                    stage_modules.append(SEBlock3D(
                        channel=input_features_skip,
                        reduction=16
                    ))
                elif attention_type == "cbam":
                    stage_modules.append(CBAM3D(
                        in_planes=input_features_skip,
                        ratio=16,
                        kernel_size=7
                    ))
                elif attention_type == "eca":
                    stage_modules.append(ECABlock3D(
                        channel=input_features_skip
                    ))
                elif attention_type == "sa":
                    stage_modules.append(SpatialAttention3DImproved(
                        kernel_size=7
                    ))
                elif attention_type == "self_attention":
                    stage_modules.append(SelfAttention3D(
                        in_channels=input_features_skip
                    ))
            
            stages.append(nn.Sequential(*stage_modules))
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        前向传播
        """
        lres_input = skips[-1]
        skip_connections = skips[:-1][::-1]  # 反转顺序
        
        # 深度监督输出
        seg_outputs = []
        
        for s in range(len(self.stages)):
            # 上采样
            x = self.upsample_layers[s](lres_input)
            
            # 融合跳跃连接
            x = torch.cat((x, skip_connections[s]), 1)
            
            # 通过解码器阶段
            x = self.stages[s](x)
            
            # 保存用于深度监督的输出
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == len(self.stages) - 1:  # 最后一层输出
                seg_outputs.append(self.seg_layers[s](x))
            
            # 更新lres_input用于下一次迭代
            lres_input = x

        if not self.deep_supervision:
            return seg_outputs[-1]
        else:
            return seg_outputs[::-1]  # 反转顺序以匹配期望的分辨率顺序

    def compute_conv_feature_map_size(self, input_size):
        """
        计算特征图大小
        """
        skip_sizes = []
        for s in range(len(self.encoder.output_channels)):
            skip_sizes.append([*input_size, self.encoder.output_channels[s]])
            input_size = [i // j for i, j in zip(input_size, self.encoder.strides[s + 1])] if s < len(self.encoder.strides) - 1 else input_size

        # 计算解码器部分的特征图大小
        feature_sizes = []
        for s in range(len(self.stages)):
            # 上采样后的特征图大小
            upsampled_size = [i * j for i, j in zip(skip_sizes[-(s+1)][:-1], self.encoder.strides[-(s+1)])]
            feature_sizes.append([*upsampled_size, skip_sizes[-(s+2)][-1] * 2])  # 融合后的通道数
            
            # 每个卷积块的输出大小
            for i in range(len(self.stages[s])):
                if isinstance(self.stages[s][i], ConvDropoutNormReLU):
                    feature_sizes.append([*upsampled_size, self.stages[s][i].conv.out_channels])

        return sum([i[-1] * prod(i[:-1]) for i in feature_sizes])


def prod(iterable):
    """
    计算可迭代对象中所有元素的乘积
    """
    result = 1
    for x in iterable:
        result *= x
    return result


class UpsampleLayer(nn.Module):
    def __init__(self,
                 conv_op,
                 input_channels: int,
                 output_channels: int,
                 pool_op_kernel_size: Union[List[int], Tuple[int, ...]],
                 mode: str = 'nearest'):
        """
        上采样层
        """
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        """
        前向传播
        """
        x = torch.nn.functional.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x


class ConvDropoutNormReLU(nn.Module):
    def __init__(self,
                 conv_op,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool,
                 norm_op,
                 norm_op_kwargs: dict,
                 dropout_op,
                 dropout_op_kwargs: dict,
                 nonlin,
                 nonlin_kwargs: dict):
        """
        卷积- Dropout - 归一化 - ReLU 模块
        """
        super().__init__()
        # 卷积层
        self.conv = conv_op(input_channels, output_channels, kernel_size=kernel_size,
                            padding=[(i - 1) // 2 for i in kernel_size] if isinstance(kernel_size, (list, tuple)) else (kernel_size - 1) // 2,
                            bias=conv_bias)
                            
        # Dropout层
        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)
        else:
            self.dropout = None
            
        # 归一化层
        if norm_op is not None:
            self.norm = norm_op(output_channels, **norm_op_kwargs) if norm_op_kwargs is not None else norm_op(output_channels)
        else:
            self.norm = None
            
        # 激活函数
        self.nonlin = nonlin(**nonlin_kwargs) if nonlin_kwargs is not None else nonlin()
        
        # 初始化权重
        if conv_op == nn.Conv2d:
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        elif conv_op == nn.Conv3d:
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        """
        前向传播
        """
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.nonlin(x)
        return x


class CoordAttention3D(nn.Module):
    """
    3D Coordinate Attention模块
    """
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttention3D, self).__init__()
        # 三个方向的自适应平均池化
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))   # Depth方向池化 -> (D, 1, 1)
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))   # Height方向池化 -> (1, H, 1)
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))   # Width方向池化 -> (1, 1, W)
        
        mip = max(8, inp // reduction)
        
        # 用于处理连接后的特征
        self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(mip)
        self.act = nn.ReLU(inplace=True)
        
        # 三个方向的注意力权重生成
        self.conv_d = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_h = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        
        n, c, d, h, w = x.size()
        
        # 分别在三个轴向上进行池化
        x_d = self.pool_d(x)   # (n, c, d, 1, 1)
        x_h = self.pool_h(x)   # (n, c, 1, h, 1)
        x_w = self.pool_w(x)   # (n, c, 1, 1, w)
        
        # 处理Depth方向特征
        y_d = self.conv1(x_d)
        y_d = self.bn1(y_d)
        y_d = self.act(y_d)
        a_d = self.conv_d(y_d).sigmoid()  # (n, oup, d, 1, 1)
        
        # 处理Height方向特征
        y_h = self.conv1(x_h)
        y_h = self.bn1(y_h)
        y_h = self.act(y_h)
        a_h = self.conv_h(y_h).sigmoid()  # (n, oup, 1, h, 1)
        
        # 处理Width方向特征
        y_w = self.conv1(x_w)
        y_w = self.bn1(y_w)
        y_w = self.act(y_w)
        a_w = self.conv_w(y_w).sigmoid()  # (n, oup, 1, 1, w)
        
        # 应用注意力权重
        out = identity * a_d * a_h * a_w
        
        return out


class SEBlock3D(nn.Module):
    """
    3D Squeeze-and-Exitation Block
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention3D(nn.Module):
    """
    3D通道注意力模块
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention3D(nn.Module):
    """
    3D空间注意力模块
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM3D(nn.Module):
    """
    3D Convolutional Block Attention Module
    """
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM3D, self).__init__()
        self.ca = ChannelAttention3D(in_planes, ratio)
        self.sa = SpatialAttention3D(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class ECABlock3D(nn.Module):
    """
    Efficient Channel Attention (ECA) Block 3D版本
    """
    def __init__(self, channel, k_size=3):
        super(ECABlock3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        
        # Multi-scale information fusion
        y = self.sigmoid(y)
        
        return x * y.expand_as(x)


class SpatialAttention3DImproved(nn.Module):
    """
    改进的空间注意力模块
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention3DImproved, self).__init__()
        
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算平均通道和最大通道
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_compress = torch.cat([avg_out, max_out], dim=1)
        
        # 应用卷积和批归一化
        x_out = self.conv1(x_compress)
        x_out = self.bn(x_out)
        
        return self.sigmoid(x_out) * x


class SelfAttention3D(nn.Module):
    """
    3D自注意力机制
    """
    def __init__(self, in_channels):
        super(SelfAttention3D, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, depth, height, width = x.size()
        
        # 计算查询、键和值
        proj_query = self.query_conv(x).view(batch_size, -1, depth * height * width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, depth * height * width)
        proj_value = self.value_conv(x).view(batch_size, -1, depth * height * width)
        
        # 计算注意力权重
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        # 应用注意力权重
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, depth, height, width)
        
        out = self.gamma * out + x
        return out