import torch
from torch import nn
import numpy as np
from ultralytics.utils import ops
from torchvision.transforms.functional import resize
from copy import deepcopy
import torchvision.transforms.functional as F
from sklearn.cluster import AgglomerativeClustering
from .discs import channel_disc
from .sff import TopKRank

# Class that handles any ROI (i.e., bounding box) DA
class ROILoss(nn.Module):
    def __init__(self, loss_type='disc', tk=False, lfc=64, mfc=128, sfc=256):
        super().__init__()
        self.loss_type = loss_type
        self.tk = tk
                    
        if self.tk:
            self.lf_tk = TopKRank(lfc).cuda()
            self.mf_tk = TopKRank(mfc).cuda()
            self.sf_tk = TopKRank(sfc).cuda()
            
        if self.loss_type == 'disc':
            # Clustering interface (cosine, complete (max) linkage, tow distance)
            self.clust = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='complete', distance_threshold=0.1)
            self.lf_disc = channel_disc(lfc if not self.tk else lfc//2)
            self.mf_disc = channel_disc(mfc if not self.tk else mfc//2)
            self.sf_disc = channel_disc(sfc if not self.tk else sfc//2)
            
            
        self.Dcrit = nn.BCEWithLogitsLoss()
        
    def get_boxes(self, preds):
        preds = ops.non_max_suppression(
            preds,
            iou_thres=0.0,
        )
        # preds, list with len batch_size
        return [b[:, :4] for b in preds]
    
    def calc_average_box_size_batch(self, batch_boxes: list):
        tops, lefts, bots, rights = ([],[],[],[])
        for box in [b for b in batch_boxes if b.shape[0] != 0]: # only consider images that have boxes
            tops.append(torch.mean(box[:,0]).item())
            lefts.append(torch.mean(box[:,1]).item())
            bots.append(torch.mean(box[:,2]).item())
            rights.append(torch.mean(box[:,3]).item())
        # If this batch had no detections, return None
        if not tops:
            return None
        else:
            top_mean = np.mean(tops)
            left_mean = np.mean(lefts)
            bot_mean = np.mean(bots)
            right_mean = np.mean(rights)
            h = int(bot_mean - top_mean)
            w = int(right_mean - left_mean)
            return max(h,w)
    
    def crop_fmap(self, fmap, boxes, newsize):
        crops = torch.zeros((boxes.shape[0], fmap.shape[0], newsize, newsize))
        for c in range(boxes.shape[0]):
            box = boxes[c,:]
            t,l,b,r = [int(x.item()) for x in box]
            h = b-t
            w = r-l
            # Fractional (float) box coords might trunc down to 0, fix that here
            h = 1 if h == 0 else h
            w = 1 if w == 0 else w
            crops[c,:] = F.resized_crop(fmap, t,l,h,w, (newsize, newsize), antialias=True)
        return crops

    def hierarchical_clustering_numpy(self, crops):
        # Reshape & numpy
        features = crops.reshape(crops.shape[0], crops.shape[1]*crops.shape[2]*crops.shape[3]).numpy()
        assignments = self.clust.fit_predict(features)
        uniques = np.unique(assignments)
        # Group tensor [uniques, mean crops through the first (num detections) dimension]
        mean_groups = torch.zeros([uniques.shape[0]] + list(crops.shape[1:]))
        for idx in uniques:
            mean_groups[idx,:] = torch.mean(crops[assignments == idx, :], dim=0)
        return mean_groups
    
    # Takes source and target fmap crops, clusters, discriminates, and returns a combined loss
    def hierarchical_cluster(self, source_crops, target_crops, disc, device):
        loss = torch.zeros((1)).to(device)
        for lbl,crops in enumerate([source_crops, target_crops]): # Enumerate gets us easy label, source=0 target=1
            if crops.shape[0] == 0 or torch.sum(torch.isnan(crops)):
                continue
            elif crops.shape[0] == 1:
                mean_groups = crops.to(device)
            else:
                try:
                    mean_groups = self.hierarchical_clustering_numpy(crops).to(device)
                except:
                    continue
                
            # Discriminate each group
            Dpreds = disc(mean_groups)
            Dtarget = torch.zeros((mean_groups.shape[0],1)) if lbl == 0 else torch.ones((mean_groups.shape[0],1))
            Dtarget = Dtarget.float().to(device)
            Dloss = self.Dcrit(Dpreds, Dtarget)
            loss += Dloss
        return loss
    
    # CL implementation from ViSGA
    def contrastive_cluster(self, source_crops, target_crops, device, margin=1):
        loss = torch.zeros((1)).to(device)
        if source_crops.shape[0] == 0 or target_crops.shape[0] == 0:
            return loss
         
        for crop in source_crops:
            source_emb = torch.flatten(crop)
            target_embs = torch.flatten(target_crops, start_dim=1)
            norms = torch.norm(source_emb[None,:] - target_embs, p=2, dim=1)
            nni = torch.argmin(norms)
            lh = norms[nni]
            rh_norms = norms[norms != lh]
            rh = torch.sum(torch.maximum(torch.zeros_like(rh_norms), margin-rh_norms))
            loss += (lh + rh)
        return loss
    
    def forward(self, lf, mf, sf, da_preds, da_images, da_labels, device):
        # Original image height/width (what boxes will be scaled to), get boxes
        img_size = (da_images.shape[-2], da_images.shape[-1])
        boxes = self.get_boxes(da_preds) # xyxy (tlbr)
        
        source_idxs = (da_labels == 0).nonzero(as_tuple=True)[0]
        target_idxs = (da_labels == 1).nonzero(as_tuple=True)[0]
        
        # Either do instance discrimination (if detections) or skip
        Dlf = torch.zeros((1)).to(device)
        Dmf = torch.zeros((1)).to(device)
        Dsf = torch.zeros((1)).to(device)
        is_detections = np.sum([b.shape[0] for b in boxes]) > 0
        
        if is_detections:
            # Scale boxes to feature map sizes
            lf_size = (lf.shape[-2], lf.shape[-1])
            mf_size = (mf.shape[-2], mf.shape[-1])
            sf_size = (sf.shape[-2], sf.shape[-1])
            lboxes = [ops.scale_boxes(img_size, b.detach().clone(), lf_size) for b in boxes]
            mboxes = [ops.scale_boxes(img_size, b.detach().clone(), mf_size) for b in boxes]
            sboxes = [ops.scale_boxes(img_size, b.detach().clone(), sf_size) for b in boxes]
        
            # Get common resize shape based on each box scale statistics
            lbox_mean_size = self.calc_average_box_size_batch(lboxes) # max of average height, width of boxes across batch
            mbox_mean_size = self.calc_average_box_size_batch(mboxes)
            sbox_mean_size = self.calc_average_box_size_batch(sboxes)
            
            # Filter features by Top-K best ranked
            if self.tk:
                lf = self.lf_tk(lf)
                mf = self.mf_tk(mf)
                sf = self.sf_tk(sf)
                    
            # Crop feature maps and resize for all boxes at each scale
            lcrops = [self.crop_fmap(lf[i,:], lboxes[i], lbox_mean_size) for i in range(len(lboxes))] if lbox_mean_size else []
            mcrops = [self.crop_fmap(mf[i,:], mboxes[i], mbox_mean_size) for i in range(len(mboxes))] if mbox_mean_size else []
            scrops = [self.crop_fmap(sf[i,:], sboxes[i], sbox_mean_size) for i in range(len(sboxes))] if sbox_mean_size else []
            
            # Loss per scale
            if lcrops:
                # Collect all source, target crops and collect into one source, target tensor [X, 64, 6, 6]
                source_lcrops, target_lcrops = (torch.cat([lcrops[i] for i in source_idxs], dim=0), torch.cat([lcrops[j] for j in target_idxs], dim=0))
                if self.loss_type == 'disc':
                    Dlf = self.hierarchical_cluster(source_lcrops, target_lcrops, self.lf_disc, device).to(device)
                else:
                    Dlf = self.contrastive_cluster(source_lcrops, target_lcrops, device).to(device)
                
            if mcrops:
                source_mcrops, target_mcrops = (torch.cat([mcrops[i] for i in source_idxs], dim=0), torch.cat([mcrops[j] for j in target_idxs], dim=0))
                if self.loss_type == 'disc':
                    Dmf = self.hierarchical_cluster(source_mcrops, target_mcrops, self.mf_disc, device).to(device)
                else:
                    Dmf = self.contrastive_cluster(source_mcrops, target_mcrops, device).to(device)
            
            if scrops:
                source_scrops, target_scrops = (torch.cat([scrops[i] for i in source_idxs], dim=0), torch.cat([scrops[j] for j in target_idxs], dim=0))
                if self.loss_type == 'disc':
                    Dsf = self.hierarchical_cluster(source_scrops, target_scrops, self.sf_disc, device).to(device)
                else:
                    Dsf = self.contrastive_cluster(source_scrops, target_scrops, device).to(device)
                    
        return Dlf, Dmf, Dsf
