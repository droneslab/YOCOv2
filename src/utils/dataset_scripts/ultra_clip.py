import sys
import glob

dpath = sys.argv[1]

train_lbls = glob.glob(f'{dpath}/labels/train/*.txt')
val_lbls = glob.glob(f'{dpath}/labels/val/*.txt')
test_lbls = glob.glob(f'{dpath}/labels/test/*.txt')

all_lbls = train_lbls + val_lbls + test_lbls
for lbl in all_lbls:
    with open(lbl, 'r') as f:
        lbl_lines = f.readlines()
        lbl_lines = [l.strip() for l in lbl_lines]
    
    new_lines = ''
    for l in lbl_lines:
        vals = [float(x) for x in l.split(' ')]
        vals[0] = int(vals[0])
        vals = [0.0 if x < 0 else x for x in vals]
        vals = [1.0 if x > 1 else x for x in vals]
        new_lines += ' '.join([str(x) for x in vals])
        new_lines += '\n'
    
    with open(lbl, 'w') as f:
        f.write(new_lines)