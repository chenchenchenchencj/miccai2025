import torch
import torch.nn as nn
from mamba import SS2D

if __name__ == '__main__':
    ss2d = SS2D(d_model=40).cuda()
    x = torch.randn(64, 40, 96, 72)  # batch_size, channels, height, width
    x = x.cuda()
    y = ss2d(x)
    print(y.shape)