import torch
from torch import nn
from ultralytics.utils.loss import v8DetectionLoss
from copy import deepcopy
import torchvision.transforms.functional as F
import numpy as np
from da_utils.roi import ROILoss
from da_utils.kmeans import KMeansLoss
from da_utils.discs import channel_disc
from da_utils.ptap import PTAPLoss
from da_utils.yocov0 import YOCOv0Loss

# Custom YOCO loss
class YOCOLoss:
    def __init__(self, model, yoco_bool, args, cmd_args):
        self.da = yoco_bool
        self.args = args
        self.cmd_args = cmd_args
        self.model = model
        self.da_type = cmd_args.da_type
        self.da_loss = cmd_args.da_loss
        self.fm = cmd_args.fm
        
        self.detection_loss = v8DetectionLoss(model)
        if self.fm:
            self.l1 = torch.nn.L1Loss()
                
        if self.da:
            from domain_data import DomainDataset
            from torch.utils.data import DataLoader
                        
            self.domain_ds = DomainDataset(args.batch, self.cmd_args.train_ds, self.cmd_args.test_ds, args)
            self.domain_dl = DataLoader(self.domain_ds, batch_sampler=self.domain_ds.sampler)

            # Model that comes in is a custom YOLO Model instance, regular torch sequential is under `model.model`
            self.torch_model = model.model
            
            # Arch type
            # mtype = str(model.args.model)
            mtype = args.save_dir.split('/')[-1]
            if 'yolov' in mtype:
                self.arch = mtype.split('yolov')[-1].split('.')[0]
                if '_' in self.arch:
                    self.arch = self.arch.split('_')[0]
            else:
                self.arch = mtype.split('yocov')[-1].split('_')[0] # Fine tuning
                
            self.yolo_ver = self.arch[0]
            self.yolo_size = self.arch[1]
            
            # Dict to store feature maps each forward call
            self.features = {}
            
            # Where target feature maps are for DA for each YOLO arch
            # TODO: Deduce these hardcoded dict values from yaml file or model config some how?
            self.feature_layers = {
                '5': {'9':'backbone', '17':'det_large', '20':'det_medium', '23':'det_small'},
                '6': {'9':'backbone', '19':'det_large', '23':'det_medium', '27':'det_small'},
                '8': {'9':'backbone', '15':'det_large', '18':'det_medium', '21':'det_small'},
            }
            
            # Depth of feature channels for discriminators
            self.feat_depths = {
                '5': {
                    'n': {'Bf': 256, 'Lf': 64,  'Mf': 128, 'Sf': 256},
                    's': {'Bf': 512, 'Lf': 128, 'Mf': 256, 'Sf': 512},
                    'm': {'Bf': 768, 'Lf': 192, 'Mf': 384, 'Sf': 768}
                },
                '6': {
                    'n': {'Bf': 256, 'Lf': 32, 'Mf': 64,  'Sf': 128},
                    's': {'Bf': 512, 'Lf': 64, 'Mf': 128, 'Sf': 256},
                },
                '8': {
                    'n': {'Bf': 256, 'Lf': 64,  'Mf': 128, 'Sf': 256},
                    's': {'Bf': 512, 'Lf': 128, 'Mf': 256, 'Sf': 512},
                    'm': {'Bf': 576, 'Lf': 192, 'Mf': 384, 'Sf': 576}
                },
            }
            
            # Register hooks to save intermediate feature maps
            for name, layer in self.torch_model.named_children():
                if name in list(self.feature_layers[self.yolo_ver].keys()):
                    layer.register_forward_hook(self.save_features(self.feature_layers[self.yolo_ver][name], self.features))
        
            # Image Discriminator (always present)
            self.Dbf = channel_disc(self.feat_depths[self.yolo_ver][self.yolo_size]['Bf'])

            # Instance Adaptation
            if self.da_type == 'roi' or self.da_type=='tk':
                self.inst_loss = ROILoss(loss_type=self.da_loss, 
                                        tk=True if self.da_type=='tk' else False,
                                        lfc=self.feat_depths[self.yolo_ver][self.yolo_size]['Lf'],
                                        mfc=self.feat_depths[self.yolo_ver][self.yolo_size]['Mf'],
                                        sfc=self.feat_depths[self.yolo_ver][self.yolo_size]['Sf'])
            elif self.da_type == 'kmeans':
                self.inst_loss = KMeansLoss(loss_type=self.da_loss)
            elif self.da_type == 'ptap':
                self.inst_loss = PTAPLoss(loss_type=self.da_loss, 
                                        lfc=self.feat_depths[self.yolo_ver][self.yolo_size]['Lf'],
                                        mfc=self.feat_depths[self.yolo_ver][self.yolo_size]['Mf'],
                                        sfc=self.feat_depths[self.yolo_ver][self.yolo_size]['Sf'])
            elif self.da_type == 'yocov0':
                self.inst_loss = YOCOv0Loss(loss_type=self.da_loss, nc=len(np.unique(self.domain_ds.labels)))
                
            self.Dcrit = nn.BCEWithLogitsLoss()                
        
    def save_features(self, name, features):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
        
    def fm_loss(self, lf,mf,sf, da_labels):
        fm_loss = 0
        fs = [lf,mf,sf]
        ws = [1.0 / (2**((3-1)-i)) for i in range(3)]
        for i in range(3):
            f = fs[i]
            f_source = f[da_labels == 0,:]
            f_target = f[da_labels == 1,:]
            w = ws[i]
            fm_loss += w*(self.l1(f_source, f_target))
        return fm_loss        
    
    def __call__(self, preds, batch):
        
        # Supervised YOLO loss (source domain data)
        loss, loss_items = self.detection_loss(preds, batch)
                        
        if self.da:
            # Get batch of mixed domain data
            da_images, da_labels = next(iter(self.domain_dl))
            
            # Device
            device = next(self.model.parameters()).device
            dtype = next(self.model.parameters()).dtype
            da_images = da_images.to(device, dtype=dtype)
            
            # Forward pass
            if self.model.training:
                self.model.eval()
                da_preds, _ = self.model(da_images)
                self.model.train()
            else:
                da_preds, _ = self.model(da_images)
                
            # Image discrimination
            bf = self.features['backbone'].float()
            Dbf = self.Dcrit(self.Dbf(bf).squeeze(), da_labels.float().cuda())
            
            # Get detection feature maps
            lf = self.features['det_large'].float()
            mf = self.features['det_medium'].float()
            sf = self.features['det_small'].float()
            
            Dlf, Dmf, Dsf = self.inst_loss(lf, mf, sf, da_preds, da_images, da_labels, device)
            
            fm = torch.zeros((1)).to(loss.device)
            if self.fm:
                fm = self.fm_loss(lf, mf, sf, da_labels)
                loss = loss + Dbf + Dlf + Dmf + Dsf + fm
                loss_items = torch.cat((loss_items, Dbf.reshape(1), Dlf.reshape(1), Dmf.reshape(1), Dsf.reshape(1), fm.reshape(1)))   
            else:    
                loss = loss + Dbf + Dlf + Dmf + Dsf
                loss_items = torch.cat((loss_items, Dbf.reshape(1), Dlf.reshape(1), Dmf.reshape(1), Dsf.reshape(1)))   
            
        
        return loss, loss_items
