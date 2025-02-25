import torch
from torch import nn
from sklearn.cluster import AgglomerativeClustering
from .discs import channel_disc
import numpy as np

class YOCOv0Loss(nn.Module):
    def __init__(self, loss_type='disc', nc=0):
        super().__init__()
        self.loss_type = loss_type
        self.nc = nc
                
        if self.loss_type == 'disc':
            self.clust = AgglomerativeClustering(metric='cosine', linkage='average', n_clusters=nc+1)
            self.lf_disc = channel_disc(nc+1)
            self.mf_disc = channel_disc(nc+1)
            self.sf_disc = channel_disc(nc+1)
            
        self.Dcrit = nn.BCEWithLogitsLoss()
    
    def cluster_inst(self, feats):
        # For each fmap in batch
        b,c,h,w = feats.shape
        batch_groups = torch.zeros((b, self.nc+1, h, w))
        for i in range(b):
            feat = feats[i,:]
            feat = feat.reshape(feat.shape[0], feat.shape[1]*feat.shape[2]).detach().cpu().numpy()
            try:
                assignments = self.clust.fit_predict(feat)
            except:
                continue
            uniques = np.unique(assignments)
            for lbl in range(self.nc+1):
                if lbl not in uniques:
                    continue
                batch_groups[i,lbl,:] = torch.mean(feats[i, assignments == lbl, :], dim=0)
        return batch_groups
    
    def forward(self, lf, mf, sf, da_preds, da_images, da_labels, device):
        da_labels = da_labels.float().cuda()
                
        Dlf_groups = self.cluster_inst(lf)
        Dlf_preds = self.lf_disc(Dlf_groups.cuda()).squeeze()
        Dlf = self.Dcrit(Dlf_preds, da_labels)
        
        Dmf_groups = self.cluster_inst(mf)
        Dmf_preds = self.mf_disc(Dmf_groups.cuda()).squeeze()
        Dmf = self.Dcrit(Dmf_preds, da_labels)
        
        Dsf_groups = self.cluster_inst(sf)
        Dsf_preds = self.sf_disc(Dsf_groups.cuda()).squeeze()
        Dsf = self.Dcrit(Dsf_preds, da_labels)
        
        return Dlf, Dmf, Dsf