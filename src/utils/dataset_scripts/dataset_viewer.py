import cv2, sys, glob
import matplotlib.pyplot as plt
import numpy as np

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

ds_path = sys.argv[1]
split = sys.argv[2]
# img_paths = glob.glob(f'{ds_path}/images/{split}/*.png')
# np.random.shuffle(img_paths)

img_paths = ['/mnt/nas/data/yoco_journal/moon/moon100/images/val/2922.png']

for img_path in img_paths:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ih,iw,_ = img.shape
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
        
        cv2.rectangle(img, (l,t), (r,b), (255,255,255), 1)
    
    plt.imshow(img)
    # plt.show()
    plt.savefig('test.png')
                