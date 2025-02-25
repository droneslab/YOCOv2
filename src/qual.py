from ultralytics import YOLO
import sys
from ultralytics.utils import yaml_load
from glob import glob
from PIL import Image, ImageDraw, ImageFont
from alive_progress import alive_bar
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
font_size = 16

models = {
      # 5M, ROI Discs (stock, roi disc, roi fm disc, roi tk fm disc)
      'mars': {
            'dpath': '/home/drones/data/hirise/map-proj-v3_2/',
            'ypath': '../datasets/vm/mars_hirise.yaml',
            'stock': '/mnt/nas/data/yoco_journal/logs/stocks_midtarget/yolov5m_mars1->mars2/weights/last.pt',
            'roi_disc': '/mnt/nas/YOCOv2_backup/YOCOv2/logs/yocov5m_mars1->mars_hirise_ROI_Disc_50/weights/last.pt',
            'roi_fm_disc': '/mnt/nas/YOCOv2_backup/YOCOv2/logs/yocov5m_mars1->mars_hirise_ROI_FM_Disc_50//weights/last.pt',
            'roi_tk_fm_disc': '/mnt/nas/YOCOv2_backup/YOCOv2/logs/yocov5m_mars1->mars_hirise_ROI_TK_FM_Disc_50/weights/last.pt',
      },
      # 6S, Feature Clustering (stock, YOCOv0, KMeans, PTAP)
      'asteroid': {
            'dpath': '/mnt/nas/data/yoco_journal/asteroid/orex/images_512/test/',
            'ypath': '../datasets/vm/orex.yaml',
            'stock': '/mnt/nas/YOCOv2_backup/YOCOv2/logs/yolov6s_asteroid1->orex_Stock_200/weights/last.pt',
            'yocov0': '/mnt/nas/YOCOv2_backup/YOCOv2/logs/yocov6s_asteroid1->orex_YOCOv0_200/weights/last.pt',
            'kmeans': '/mnt/nas/YOCOv2_backup/YOCOv2/logs/yocov6s_asteroid1->orex_KMeans_200/weights/last.pt',
            'ptap': '/mnt/nas/YOCOv2_backup/YOCOv2/logs/yocov6s_asteroid1->orex_PTAP_200/weights/last.pt',
      },
      # 8M, Contrastive (stock, roi contrastive, roi fm contrastive, roi tk fm contrastive)
      'moon': {
            'dpath': '/mnt/nas/data/yoco_journal/moon/moon100/images/val/',
            'ypath': '../datasets/vm/moon100.yaml',
            'stock': '/mnt/nas/data/yoco_journal/logs/stocks_midtarget/yolov8m_moon64->moon100/weights/last.pt',
            'roi_contrastive': '/mnt/nas/data/yoco_journal/logs/082024/yocov8m_moon64->moon100_ROI_Contrastive/weights/last.pt',
            'roi_fm_contrastive': '/mnt/nas/data/yoco_journal/logs/082024/yocov8m_moon64->moon100_ROI_FM_Contrastive/weights/last.pt',
            'roi_tk_fm_contrastive': '/mnt/nas/data/yoco_journal/logs/082024/yocov8m_moon64->moon100_ROI_TK_FM_Contrastive/weights/last.pt'
      }
}

# default conf 0.25 iou 0.7
conf=0.25
iou =0.7
scene = 'moon'
dpath = models[scene]['dpath']
ypath = models[scene]['ypath']
img_ext = '.' + glob(f'{dpath}/*')[0].split('.')[-1]

save_root = f'../results/qual/{scene}/conf{conf}_iou{iou}/'
Path(save_root).mkdir(parents=True, exist_ok=True)

for model_name in [m for m in list(models[scene].keys()) if 'path' not in m]:
      yaml = yaml_load(ypath)

      model = YOLO(models[scene][model_name])
      results = model(source=dpath, save_txt=True, save_conf=True, conf=conf, iou=iou)
      save_dir = results[-1].save_dir
      
      msave_path = f'{save_root}/{model_name}/'
      Path(msave_path + '/labels/').mkdir(parents=True, exist_ok=True)
      Path(msave_path + '/images/').mkdir(parents=True, exist_ok=True)

      # box_font = ImageFont.truetype(font_path, size=font_size)

      classes = yaml['names']
      colors = {
            0: 'crimson',
            1: 'limegreen',
            2: 'blue'
      }

      lbl_files = glob(f'{save_dir}/labels/*.txt')
      with alive_bar(len(lbl_files)) as bar:
            for lbl in lbl_files:
                  lbl_name = lbl.split('/')[-1]
                  img_path = dpath + lbl_name.replace('.txt', img_ext)
                  img = Image.open(img_path).convert('RGB')
                  iw,ih = img.size
            
                  draw = ImageDraw.Draw(img)
                  
                  with open(lbl, 'r') as f:
                        box_lines = f.readlines()
                  
                  for line in box_lines:
                        vals = [float(x) for x in line.split(' ')]
                        c,xn,yn,wn,hn,conf = vals
                        x = xn*iw
                        y = yn*ih
                        w = wn*iw
                        h = hn*ih
                        
                        wr = w/2
                        hr = h/2
                        
                        l = int(x-wr)
                        r = int(x+wr)
                        t = int(y-hr)
                        b = int(y+hr)
                        
                        col = colors[c]
                        
                        # Box
                        draw.rectangle(((l,t),(r,b)), outline=col, width=4)
                        
                        # Label
                        # label = '{} {:.2f}'.format(classes[c], conf)
                        # width = draw.textlength(label, font=box_font)
                        # label_pos = (l,t-font_size)
                        # draw.rectangle(((l,t-font_size), (l+width, t)), fill=col)
                        # draw.text(label_pos, label, font=box_font)
                  
                  save_name = lbl_name.replace('.txt', '.png')
                  img.save(f'{msave_path}/images/{save_name}')
            
                  # Save label
                  Path(lbl).rename(f'{msave_path}/labels/{lbl_name}')
            
                  bar()
                  

# COMPS
dirs = glob(f'{save_root}/*')

flist = glob(f'{dpath}/*')
flist = [s.split('/')[-1] for s in flist]

img_dirs = [s + '/images/' for s in dirs]
img_dirs = [s for s in img_dirs if 'stock' in s] + [s for s in img_dirs if 'stock' not in s] # put stock first

comps_save = f'{save_root}/comps/'
Path(comps_save).mkdir(parents=True, exist_ok=True)

for img_name in flist:
    imgs = []
    lbls = []
    img_name = img_name.replace('.jpg', '.png')
    for img_dir in img_dirs:
        try:
            img = Image.open(f'{img_dir}/{img_name}')
            imgs.append(img)
            lbls.append(img_dir)
        except:
            continue
    if not imgs:
        continue

    fig, axs = plt.subplots(1,len(imgs))
    fig.set_size_inches(15, 4)
    for i in range(len(imgs)):
        lbl = ' '.join(lbls[i].split('/images')[0].split('/')[-1].split('_'))
        if len(imgs) > 1:
            axs[i].imshow(imgs[i])
            axs[i].set_title(lbl)
        else:
            axs.imshow(imgs[i])
            axs.set_title(lbl)
    plt.savefig(f'{comps_save}/{img_name}')
    plt.close()