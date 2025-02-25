from pytorch_revgrad import RevGrad
from torch import nn
import torch

class PrintSize(nn.Module):
    def __init__(self):
        super(PrintSize, self).__init__()
    def forward(self, x):
        print(x.shape)
        return x
    
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def channel_disc(channel_size):
    return nn.Sequential(
        RevGrad(),
        nn.AdaptiveAvgPool2d(1),  # to [B, channel_size, 1, 1]
        nn.Conv2d(channel_size, 1, 1, 1, autopad(1, None), groups=1),  # to [B, 1 , 1, 1]
        nn.Flatten(),
    ).cuda()
    
def no_channel_disc():
    return nn.Sequential(
        RevGrad(),
        nn.AdaptiveAvgPool2d(1),  # to [B, 1, 1]
        nn.Flatten(),
    ).cuda()