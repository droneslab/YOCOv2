import sys
dpath = sys.argv[1]
from glob import glob
from pathlib import Path
from numpy.random import shuffle
image_fs = glob(f'{dpath}/*.png')
num_images = len(image_fs)
num_train = int(num_images*0.9)
num_test = int(num_images*0.1)
print(f'Num images: {num_images}')
print(f'Splits:\n\tTrain: {num_train}\n\tTest:  {num_test}')
print('Shuffling and moving...')

shuffle(image_fs)
train_fs = image_fs[:num_train]
test_fs = image_fs[num_train:]

Path(f'{dpath}/images/train/').mkdir(parents=True, exist_ok=True)
Path(f'{dpath}/images/test/').mkdir(parents=True, exist_ok=True)

for img_path in train_fs:
    img_dest = f'{dpath}/images/train/' + img_path.split('/')[-1]
    Path(img_path).rename(img_dest)
    
for img_path in test_fs:
    img_dest = f'{dpath}/images/test/' + img_path.split('/')[-1]
    Path(img_path).rename(img_dest)
    