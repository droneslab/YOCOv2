from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import glob
import torch
from kmeans_pytorch import kmeans
from numpy.random import shuffle
import torchvision as tv
import matplotlib
import matplotlib.cm as cm
import numpy as np
from kornia.enhance import add_weighted  
from torchvision.transforms.functional import resize
from torch import nn

def save_features(name, features):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

# --- Setup model
# model = YOLO(f'../../logs/yolo_v8n_mars2/weights/last.pt')
# model = YOLO(f'../../logs/yolo_v8n_asteroid3/last.pt')
# model = YOLO(f'../../logs/moon64_last.pt')
model = YOLO(f'../../logs/mars2_last.pt')
features = {}
for name, layer in model.model.model.named_children():
    if name in ['15', '18', '21']:
        layer.register_forward_hook(save_features(name, features))

paths = glob.glob('/home/tj/data/yoco_journal/mars/mars2/images/test/*.png')
# paths = glob.glob('/home/tj/data/yoco_journal/asteroid/asteroid3/test/*.png')
# paths = sorted(glob.glob('/home/tj/data/yoco_journal/moon/moon64/images/test/*.png'))
# paths = glob.glob('/home/tj/data/yoco_journal/asteroid/asteroid1/images/test/*.png')

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

f = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Conv2d(4, 1, 1, 1, autopad(1, None), groups=1),
    nn.Flatten(),
)

shuffle(paths)
for path in paths:
    img = tv.io.read_image(path)
    
    results = model.predict(path)
    lf = features['15'][0,:]
    mf = features['18'][0,:]
    sf = features['21'][0,:]
        
    c,h,w = lf.shape
    mff = resize(mf, (h,w))
    sff = resize(sf, (h,w))
    af = torch.cat((lf,mff,sff))
    
    nc = 4
    thresh = 0.10
    
    viss = []
    img = (img-torch.min(img))/(torch.max(img)-torch.min(img))
    for fs in [lf,mf,sf,af]:
        viss.append(img)
        
        c,h,w = fs.shape
        fs = fs.reshape(fs.shape[0], fs.shape[1]*fs.shape[2])
        fs_ids_x, fs_centers = kmeans(
            X=fs, num_clusters=nc, device=torch.device('cuda:0')
        )
        
        print(torch.bincount(fs_ids_x))
        
        # obj_idx = torch.argmax(torch.bincount(fs_ids_x)).item()
        # obj_center = fs_centers[obj_idx,:].reshape(h,w)[None,:]
        
        # obj_maps = fs[fs_ids_x == obj_idx,:]
        # obj_max,_ = torch.max(obj_maps, dim=0)
        # obj_max = obj_max.reshape(h,w)[None,:]
        
        # selects = torch.mean(fs_centers, dim=-1) > 0.1
        # fs_centers = fs_centers[selects]
        
        # fs_centers = torch.functional.normalize(fs_centers, dim=-1)
        
        fs_centers = (fs_centers-torch.min(fs_centers))/(torch.max(fs_centers)-torch.min(fs_centers))
        
        # obj_centers = fs_centers.reshape(torch.sum(selects).item(),h,w)
        # obj_centers = obj_centers[torch.mean(obj_centers)]
        obj_centers = fs_centers.reshape(nc,h,w)
        
        print(f(obj_centers))
        
        obj_mean = torch.mean(obj_centers, dim=0)[None,:]
        
        obj_centers = torch.cat((obj_centers, obj_mean), dim=0)
                
        # for fsx in [fs0, fs1]:
        for i in range(obj_centers.shape[0]):
        
        # for fsx in [obj_max, obj_center]:
            fsx = obj_centers[i,:]
            # f_vis = torch.mean(fsx, dim=0).reshape(h,w)[None,:]
            
            # f_vis = fsx.reshape(h,w)[None,:]
            
            print(torch.mean(fsx))
            
            f_vis = fsx[None,:]
            
            f_vis = f_vis.permute(1,2,0).detach().cpu().numpy()
            f_norm = matplotlib.colors.Normalize(vmin=f_vis.min().item(), vmax=f_vis.max().item())
            f_heat = torch.Tensor(cm.jet(f_norm(np.squeeze(f_vis)))[:,:,:3]).permute(2,0,1)
            f_heat = tv.transforms.Resize((512,512))(f_heat)
            f_vis = add_weighted(img, 0.5, f_heat, 0.5, 0.)
            viss.append(f_vis)
    
    grid = tv.utils.make_grid(viss, nrow=6)
    plt.imshow(grid.permute(1,2,0).cpu().numpy())
    plt.show()
    