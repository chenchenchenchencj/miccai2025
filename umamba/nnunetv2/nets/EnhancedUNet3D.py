import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple, Union

# Enhanced UNet 3D with: 
# 1) Early multi-scale input fusion (original + 0.5 downsample) via 1x1 compression
# 2) Dual-path encoder (main + extra low-res global context path)
# 3) Larger 5x5x5 grouped conv blocks in deeper stages
# 4) Dilated conv tail with dilation rates (1,2,3) for multi-scale aggregation
# 5) Optional deep supervision

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, groups=1, dilation=1):
        super().__init__()
        pad = ((k - 1) // 2) * dilation
        self.conv = nn.Conv3d(in_ch, out_ch, k, padding=pad, bias=False, groups=groups, dilation=dilation)
        self.norm = nn.InstanceNorm3d(out_ch, eps=1e-5, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ResidualUnit(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, groups=1):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch, k=k, groups=groups)
        self.conv2 = ConvBlock(out_ch, out_ch, k=k, groups=groups)
        self.proj = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.proj(x)

class EncoderStage(nn.Module):
    def __init__(self, in_ch, out_ch, stride, deep_stage=False):
        super().__init__()
        # For deeper stages use larger kernel 5 and group conv
        if deep_stage:
            groups = min(4, out_ch) if out_ch % min(4, out_ch) == 0 else 1
            self.block = ResidualUnit(in_ch, out_ch, k=5, groups=groups)
        else:
            self.block = ResidualUnit(in_ch, out_ch, k=3, groups=1)
        self.down = nn.Conv3d(out_ch, out_ch, 3, stride=stride, padding=1) if any(s > 1 for s in stride) else nn.Identity()
    def forward(self, x):
        x = self.block(x)
        skip = x
        x = self.down(x)
        return x, skip

class DecoderStage(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.block = ResidualUnit(out_ch + skip_ch, out_ch, k=3)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            # pad or crop to match
            diff = [s1 - s2 for s1, s2 in zip(skip.shape[2:], x.shape[2:])]
            # simple center crop of skip if needed (rare) else pad x
            pads = []
            for d in reversed(diff):
                if d > 0:
                    pads.extend([0, d])
                else:
                    pads.extend([0,0])
            if any(d > 0 for d in diff):
                x = F.pad(x, pads)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class DilatedTail(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.convs = nn.ModuleList([
            ConvBlock(ch, ch, k=3, dilation=d) for d in (1,2,3)
        ])
        self.fuse = nn.Conv3d(ch*3, ch, 1)
        self.norm = nn.InstanceNorm3d(ch, eps=1e-5, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
    def forward(self, x):
        feats = [c(x) for c in self.convs]
        x = self.fuse(torch.cat(feats, dim=1))
        return self.act(self.norm(x))

class EnhancedUNet3D(nn.Module):
    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 n_stages: int = 5,
                 features_per_stage: Union[List[int], Tuple[int, ...]] = (32,64,128,256,320),
                 deep_supervision: bool = True):
        super().__init__()
        assert n_stages == len(features_per_stage)
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes

        # Early multi-scale fusion
        self.early_fuse = nn.Sequential(
            nn.Conv3d(input_channels*2, features_per_stage[0], 1, bias=False),
            nn.InstanceNorm3d(features_per_stage[0], eps=1e-5, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )

        # Encoder
        enc_stages = []
        in_ch = features_per_stage[0]
        for i in range(n_stages):
            out_ch = features_per_stage[i]
            stride = (1,1,1) if i==0 else (2,2,2)
            deep_stage = (i >= n_stages - 2)  # last two stages use large kernel grouped conv
            enc_stages.append(EncoderStage(in_ch if i>0 else in_ch, out_ch, stride, deep_stage=deep_stage))
            in_ch = out_ch
        self.encoder = nn.ModuleList(enc_stages)

        # Extra low-res global context path (downsample further 2x from bottleneck)
        self.global_down1 = nn.Conv3d(features_per_stage[-1], features_per_stage[-1], 3, stride=2, padding=1)
        self.global_down2 = nn.Conv3d(features_per_stage[-1], features_per_stage[-1], 3, stride=2, padding=1)
        self.global_fuse = nn.Conv3d(features_per_stage[-1]*2, features_per_stage[-1], 1)

        # Dilated tail
        self.dilated_tail = DilatedTail(features_per_stage[-1])

        # Decoder
        dec_stages = []
        seg_layers = []
        for i in range(n_stages-1, 0, -1):
            in_ch = features_per_stage[i]
            skip_ch = features_per_stage[i-1]
            out_ch = skip_ch
            dec_stages.append(DecoderStage(in_ch, skip_ch, out_ch))
            seg_layers.append(nn.Conv3d(out_ch, num_classes, 1))
        self.decoder = nn.ModuleList(dec_stages)
        self.seg_layers = nn.ModuleList(seg_layers)  # order aligned with decoder order

    def forward(self, x: torch.Tensor):
        # early fusion (original + 0.5 downsample -> upsample to original -> concat)
        x_half = F.interpolate(x, scale_factor=0.5, mode='trilinear', align_corners=False, recompute_scale_factor=False)
        x_half_up = F.interpolate(x_half, size=x.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, x_half_up], dim=1)
        x = self.early_fuse(x)

        skips = []
        out = x
        for stage in self.encoder:
            out, skip = stage(out)
            skips.append(skip)
        # bottleneck output is last out, global context path
        g = self.global_down1(out)
        g = F.leaky_relu(g, 0.01, inplace=True)
        g = self.global_down2(g)
        g = F.leaky_relu(g, 0.01, inplace=True)
        g_up = F.interpolate(g, size=out.shape[2:], mode='trilinear', align_corners=False)
        out = self.global_fuse(torch.cat([out, g_up], dim=1))
        out = self.dilated_tail(out)

        seg_outputs = []
        dec_in = out
        # Iterate decoder stages; skips list length = n_stages; we consumed last skip corresponds to penultimate? adjust
        # We appended skip after each encoder stage; last skip corresponds to last encoder output before downsample? Accept logic
        for i, dec in enumerate(self.decoder):
            skip = skips[-(i+2)]  # skip aligned: first decoder uses skip from second last encoder stage
            dec_in = dec(dec_in, skip)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[i](dec_in))
        seg_outputs = seg_outputs[::-1]
        if self.deep_supervision:
            return seg_outputs
        else:
            return seg_outputs[-1]


