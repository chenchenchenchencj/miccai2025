import torch
import torch.nn as nn
import torch.nn.functional as F

# 检查Mamba模块是否可用
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available, some Mamba features will be disabled")


class MambaBlock3D(nn.Module):
    """
    基于Mamba的3D块
    """
    def __init__(self, in_channels, out_channels, d_state=16, d_conv=3, expand=2):
        super(MambaBlock3D, self).__init__()
        # 卷积层
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Mamba层
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=out_channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.use_mamba = True
        else:
            print("Warning: Mamba module not available, using standard conv block")
            self.use_mamba = False
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        if self.use_mamba:
            # 为Mamba处理准备数据 (B, C, H, W, D) -> 处理每个轴向切片
            x = self.process_with_mamba(x)
        
        return x
    
    def process_with_mamba(self, x):
        """
        使用Mamba处理3D特征图
        正确处理维度以适应Mamba的期望输入格式
        """
        B, C, H, W, D = x.shape
        # 对每个D切片应用Mamba
        output_slices = []
        for d in range(D):
            slice_2d = x[:, :, :, :, d]  # (B, C, H, W)
            # 重塑为(B, H*W, C)以适应Mamba的输入要求
            slice_reshaped = slice_2d.permute(0, 2, 3, 1).reshape(B, H*W, C)
            # 应用Mamba
            slice_mamba = self.mamba(slice_reshaped)
            # 重塑回(B, C, H, W)
            slice_out = slice_mamba.reshape(B, H, W, C).permute(0, 3, 1, 2)
            output_slices.append(slice_out)
        
        # 重新组合
        output = torch.stack(output_slices, dim=-1)  # (B, C, H, W, D)
        return output


class EnhancedMambaBlock3D(nn.Module):
    """
    显存优化版增强Mamba 3D块
    只在一个轴向上应用Mamba以节省显存
    """
    def __init__(self, in_channels, out_channels, d_state=8, d_conv=3, expand=1):
        super(EnhancedMambaBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if MAMBA_AVAILABLE:
            # 使用更小的参数以节省显存
            self.mamba_d = Mamba(
                d_model=out_channels,
                d_state=d_state,      # 减少d_state从16到8
                d_conv=d_conv,
                expand=expand,        # 减少expand从2到1
            )
            # 为了节省显存，我们只在最重要的轴向上应用Mamba
            self.use_mamba = True
        else:
            self.use_mamba = False
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        if self.use_mamba:
            x = self.process_with_mamba_3d(x)
        
        return x
    
    def process_with_mamba_3d(self, x):
        """
        在一个轴向上应用Mamba以节省显存
        """
        B, C, D, H, W = x.shape
        
        # 只在D轴向应用Mamba以节省显存
        x_d = x.permute(0, 3, 4, 2, 1).contiguous()  # (B, H, W, D, C)
        x_d = x_d.view(B, H*W, D, C).permute(0, 2, 1, 3).contiguous()  # (B, D, H*W, C)
        x_d = x_d.view(B*D, H*W, C)
        
        # 分批处理以节省显存
        batch_size = min(16, B*D)  # 进一步减少批次大小以节省显存
        processed = []
        for i in range(0, B*D, batch_size):
            end_idx = min(i + batch_size, B*D)
            batch = x_d[i:end_idx]
            processed_batch = self.mamba_d(batch)
            processed.append(processed_batch)
        
        x_d = torch.cat(processed, dim=0)
        x_d = x_d.view(B, D, H*W, C).permute(0, 3, 1, 2).contiguous()  # (B, C, D, H*W)
        x_d = x_d.view(B, C, D, H, W)
        
        # 融合结果
        x = x + x_d  # 只融合D轴向的结果以节省显存
        return x


class CoordAttention3D(nn.Module):
    """
    3D Coordinate Attention模块
    通过分别关注空间坐标信息来增强特征表示
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
        #print(f"CoordAttention3D input size: {x.size()}")
        
        # 分别在三个轴向上进行池化
        x_d = self.pool_d(x)   # (n, c, d, 1, 1)
        x_h = self.pool_h(x)   # (n, c, 1, h, 1)
        x_w = self.pool_w(x)   # (n, c, 1, 1, w)
        
        #print(f"After pooling - x_d: {x_d.size()}, x_h: {x_h.size()}, x_w: {x_w.size()}")
        
        # 为了连接，我们需要将它们扩展到相同的空间维度
        # 首先，我们对每个特征图进行处理，然后生成注意力权重
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
        
        #print(f"Attention weights - a_d: {a_d.size()}, a_h: {a_h.size()}, a_w: {a_w.size()}")
        
        # 应用注意力权重
        out = identity * a_d * a_h * a_w
        
        #print(f"CoordAttention3D output size: {out.size()}")
        
        return out


class CoordAttentionBlock3D(nn.Module):
    """
    带有3D Coordinate Attention的卷积块
    """
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAttentionBlock3D, self).__init__()
        # 使用合适的通道数进行卷积
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.ca = CoordAttention3D(out_channels, out_channels, reduction)
        
    def forward(self, x):
        residual = x
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.ca(x)
        
        # 添加残差连接
        x = x + residual
        
        return x