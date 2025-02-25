import cv2, sys, glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from alive_progress import alive_bar

def read_lbl(lbl_path):
    with open(lbl_path, 'r') as f:
        lines = f.readlines()
        
    boxes = []
    for line in lines:
        line = line.strip()
        vals = line.split(' ')
        vals = [float(x) for x in vals]
        c,x,y,w,h = vals
        boxes.append((c,x,y,w,h))
    return boxes

colors = {
    0: 'crimson',
    1: 'limegreen',
    2: 'blue'
}

ds_path = sys.argv[1]
split = sys.argv[2]
img_paths = glob.glob(f'{ds_path}/images/{split}/*.png')
sp = f'{ds_path}/ground_truth/{split}/'
Path(sp).mkdir(parents=True, exist_ok=True)

with alive_bar(len(img_paths)) as bar:
    for img_path in img_paths:
        img = Image.open(img_path).convert('RGB')
        iw,ih = img.size
        draw = ImageDraw.Draw(img)
        num = img_path.split('/')[-1].split('.')[0]
        boxes = read_lbl(img_path.replace('images', 'labels').replace('.png', '.txt'))
        
        for box in boxes:
            c,xn,yn,wn,hn = box
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
            draw.rectangle(((l,t),(r,b)), outline=col, width=4)
            
        im_name = img_path.split('/')[-1]
        img.save(f'{sp}/{im_name}')
        
        bar()
        
        # plt.imshow(img)
        # # plt.show()
        # plt.savefig('test.png')
                