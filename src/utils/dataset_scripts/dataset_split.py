import sys
dpath = sys.argv[1]
from glob import glob
from pathlib import Path
from numpy.random import shuffle
image_fs = glob(f'{dpath}/images/*.png')
num_images = len(image_fs)
num_train = int(num_images*0.8)
num_val = int(num_images*0.1)
num_test = int(num_images*0.1)
print(f'Num images: {num_images}')
print(f'Splits:\n\tTrain: {num_train}\n\tVal:   {num_val}\n\tTest:  {num_test}')
print('Shuffling and moving...')

shuffle(image_fs)
train_fs = image_fs[:num_train]
val_fs = image_fs[num_train:num_train+num_val]
test_fs = image_fs[num_train+num_val:]

Path(f'{dpath}/images/train/').mkdir(parents=True, exist_ok=True)
Path(f'{dpath}/images/val/').mkdir(parents=True, exist_ok=True)
Path(f'{dpath}/images/test/').mkdir(parents=True, exist_ok=True)
Path(f'{dpath}/labels/train/').mkdir(parents=True, exist_ok=True)
Path(f'{dpath}/labels/val/').mkdir(parents=True, exist_ok=True)
Path(f'{dpath}/labels/test/').mkdir(parents=True, exist_ok=True)

for img_path in train_fs:
    img_dest = f'{dpath}/images/train/' + img_path.split('/')[-1]
    lbl_path = img_path.replace('images', 'labels').replace('.png', '.txt')
    lbl_dest = img_dest.replace('images', 'labels').replace('.png', '.txt')
    Path(img_path).rename(img_dest)
    Path(lbl_path).rename(lbl_dest)
    
for img_path in val_fs:
    img_dest = f'{dpath}/images/val/' + img_path.split('/')[-1]
    lbl_path = img_path.replace('images', 'labels').replace('.png', '.txt')
    lbl_dest = img_dest.replace('images', 'labels').replace('.png', '.txt')
    Path(img_path).rename(img_dest)
    Path(lbl_path).rename(lbl_dest)
    
for img_path in test_fs:
    img_dest = f'{dpath}/images/test/' + img_path.split('/')[-1]
    lbl_path = img_path.replace('images', 'labels').replace('.png', '.txt')
    lbl_dest = img_dest.replace('images', 'labels').replace('.png', '.txt')
    Path(img_path).rename(img_dest)
    Path(lbl_path).rename(lbl_dest)
    