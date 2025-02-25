import torch
from torch import nn
from kmeans_pytorch import kmeans as kmeans_clustering
from .discs import no_channel_disc

class KMeansLoss(nn.Module):
    def __init__(self, loss_type='disc'):
        super().__init__()
        self.loss_type = loss_type
                
        if self.loss_type == 'disc':
            self.lf_disc = no_channel_disc()
            self.mf_disc = no_channel_disc()
            self.sf_disc = no_channel_disc()
            
        self.Dcrit = nn.BCEWithLogitsLoss()
        
    def cluster_inst_agg(self, lf, mf, sf):
        _,_,rh,rw = lf.shape
        mf = resize(mf, (rh,rw), antialias=True)
        sf = resize(sf, (rh,rw), antialias=True)
        feats = torch.cat((lf,mf,sf), dim=1)
        
        b,c,h,w = feats.shape
        feats = feats.reshape(b,c, h*w)
        output_clusters = torch.zeros((b,2,h,w))
        for i in range(b):
            feat = feats[i,:]
            cluster_ids, cluster_centers = kmeans_clustering(
                X=feat, num_clusters=2, device=torch.device('cuda:0'), tqdm_flag=False
            )
            output_clusters[i,:] = cluster_centers.reshape((2,h,w))
        return output_clusters
    
    def cluster_inst(self, feats):        
        b,c,h,w = feats.shape
        feats = feats.reshape(b,c, h*w)
        output_clusters = torch.zeros((b,1,h,w))
        for i in range(b):
            feat = feats[i,:]
            cluster_ids, cluster_centers = kmeans_clustering(
                X=feat, num_clusters=2, device=torch.device('cuda:0'), tqdm_flag=False
            )
            obj_idx = torch.argmax(torch.bincount(cluster_ids)).item()
            obj_center = cluster_centers[obj_idx,:].reshape(h,w)[None,:]
            output_clusters[i,:] = obj_center
        return output_clusters
    
    def forward(self, lf, mf, sf, da_preds, da_images, da_labels, device):
        da_labels = da_labels.float().cuda()
                
        Dlf_obj_center = self.cluster_inst(lf)
        Dlf_preds = self.lf_disc(Dlf_obj_center.cuda()).squeeze()
        Dlf = self.Dcrit(Dlf_preds, da_labels)
        
        Dmf_obj_center = self.cluster_inst(mf)
        Dmf_preds = self.mf_disc(Dmf_obj_center.cuda()).squeeze()
        Dmf = self.Dcrit(Dmf_preds, da_labels)
        
        Dsf_obj_center = self.cluster_inst(sf)
        Dsf_preds = self.sf_disc(Dsf_obj_center.cuda()).squeeze()
        Dsf = self.Dcrit(Dsf_preds, da_labels)
        
        return Dlf, Dmf, Dsf