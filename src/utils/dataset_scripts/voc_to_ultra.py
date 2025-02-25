import xml.etree.ElementTree as ET
from tqdm import tqdm
import yaml
import glob
from pathlib import Path


def convert_label(lbl_path, save_path, split=''):
    def convert_box(size, box):
        dw, dh = 1. / size[0], 1. / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh

    inst = lbl_path.split('/')[-1].split('.')[0]
    out_file = open(f'{save_path}/{split}/{inst}.txt', 'w')

    tree = ET.parse(lbl_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        xmlbox = obj.find('bndbox')
        bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
        cls_id = 0
        out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')

# Convert
path = f'/mnt/Space-Vision/data/yoco_journal/cityscapes/'
vocdir = f'{path}/voc/'
ultradir = f'{path}/ultra/'
save_dir = f'{ultradir}/labels/'
Path(save_dir).mkdir(parents=True, exist_ok=True)

voc_img_paths = glob.glob(f'{vocdir}/JPEGImages/*/*.jpg')
for voc_img_path in tqdm(voc_img_paths):
    split = voc_img_path.split('/')[-2]
    inst = voc_img_path.split('/')[-1].split('.')[0]
    lbl_path = f'{vocdir}/Annotations/{split}/{inst}.xml'
    convert_label(lbl_path, save_dir, split=split)
