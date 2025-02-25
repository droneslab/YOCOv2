import torch
from torch import nn

''' ======================================
        Top-k Ranking
    ====================================== '''
# From the paper: Contrastive Attention Maps for Self-supervised Co-localization
# https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
class ChannelAttention(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return y

class TopKRank(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.k = int(channel_size*0.5)
        self.ca = ChannelAttention(channel_size)
        
    def forward(self, x):
        
        channel_ranks = self.ca(x).squeeze()
        _,sorted_idxs = torch.sort(channel_ranks, dim=1, descending=True) # Sort channels by priority
        sorted_k_idxs = sorted_idxs[:,:self.k]
        out = torch.zeros(x.shape[0], self.k, x.shape[2], x.shape[3])
        for i in range(x.shape[0]):
            x_ = x[i,:]
            i_ = sorted_k_idxs[i,:]
            out[i,...] = x_[i_,:]
        return out.cuda()