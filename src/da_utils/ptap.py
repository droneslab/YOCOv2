import torch
from torch import nn
from .discs import no_channel_disc

''' ======================================
        Pixel-wise Top-k Attention Pooling
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
        return x * y.expand_as(x)

class PTAP(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.k = int(channel_size*0.5)
        self.ca = ChannelAttention(channel_size)
        
    def forward(self, x):
        Fw = self.ca(x)
        Fw,_ = torch.sort(Fw, dim=1, descending=True) # Sort activations by priority
        return (1/self.k)*torch.sum(Fw[:, :self.k, :, :], dim=1) # CAML paper method (PTAP)
    
class PTAPLoss(nn.Module):
    def __init__(self, loss_type, lfc, mfc, sfc):
        super().__init__()
        self.loss_type = loss_type
        
        self.lf_ptap = PTAP(lfc).cuda()
        self.mf_ptap = PTAP(mfc).cuda()
        self.sf_ptap = PTAP(sfc).cuda()
        
        if self.loss_type == 'disc':
            self.lf_disc = no_channel_disc()
            self.mf_disc = no_channel_disc()
            self.sf_disc = no_channel_disc()
            
        self.Dcrit = nn.BCEWithLogitsLoss()
    
    def forward(self, lf, mf, sf, da_preds, da_images, da_labels, device):
        da_labels = da_labels.float().cuda()
                
        Dlf_obj_center = self.lf_ptap(lf)
        Dlf_preds = self.lf_disc(Dlf_obj_center.cuda()).squeeze()
        Dlf = self.Dcrit(Dlf_preds, da_labels)
        
        Dmf_obj_center = self.mf_ptap(mf)
        Dmf_preds = self.mf_disc(Dmf_obj_center.cuda()).squeeze()
        Dmf = self.Dcrit(Dmf_preds, da_labels)
        
        Dsf_obj_center = self.sf_ptap(sf)
        Dsf_preds = self.sf_disc(Dsf_obj_center.cuda()).squeeze()
        Dsf = self.Dcrit(Dsf_preds, da_labels)
        
        return Dlf, Dmf, Dsf
    