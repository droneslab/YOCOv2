'''
Converts all Ultralytics YOLO-formatted txt files in dataset train/val/test folders to PASCAL VOC XML
https://medium.com/@Spritan/convert-yolo-annotations-to-voc-ee5745b05851
https://github.com/Spritan/Yolo_to_VOC/blob/main/app.ipynb
https://github.com/rmalav15/visualize-voc-format-data
'''

import xml.etree.ElementTree as ET
import glob
import sys
from pathlib import Path
import os
from tqdm import tqdm

dpath = sys.argv[1]

def yolo_to_voc(yolo_path, voc_path, image_width, image_height, img_dir):
    # Read YOLO annotations from file
    with open(yolo_path, 'r') as file:
        lines = file.readlines()

    # Create the root element for the XML tree
    root = ET.Element("annotation")

    # Add the 'folder' element to the root
    folder = ET.SubElement(root, "folder")
    folder.text = img_dir  # Replace with the appropriate folder name

    # Add the 'filename' element to the root
    filename = ET.SubElement(root, "filename")
    filename.text = os.path.basename(yolo_path).replace('.txt', '.jpg')

    # Add the 'size' element to the root
    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    height = ET.SubElement(size, "height")
    depth = ET.SubElement(size, "depth")
    width.text = str(image_width)
    height.text = str(image_height)
    depth.text = "3"

    # Process each line in the YOLO file (each line corresponds to a bounding box)
    for line in lines:
        parts = line.split()
        class_id, center_x, center_y, bbox_width, bbox_height = map(float, parts)

        # Add the 'object' element to the root for each bounding box
        obj = ET.SubElement(root, "object")

        # Add the 'name' element to the 'object' element
        name = ET.SubElement(obj, "name")
        name.text = str(int(class_id) + 1)  # Class ID (YOLO uses 0-based indexing)

        # Add the 'pose' element to the 'object' element
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"

        # Add the 'truncated' element to the 'object' element
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"

        # Add the 'difficult' element to the 'object' element
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"

        # Add the 'bndbox' element to the 'object' element
        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        ymin = ET.SubElement(bndbox, "ymin")
        xmax = ET.SubElement(bndbox, "xmax")
        ymax = ET.SubElement(bndbox, "ymax")

        # Calculate and set bounding box coordinates based on YOLO format
        xmin.text = str(int((center_x - bbox_width / 2) * image_width))
        ymin.text = str(int((center_y - bbox_height / 2) * image_height))
        xmax.text = str(int((center_x + bbox_width / 2) * image_width))
        ymax.text = str(int((center_y + bbox_height / 2) * image_height))

    # Create an XML tree and write it to the specified VOC XML file
    tree = ET.ElementTree(root)
    tree.write(voc_path)

for split in ['train', 'val', 'test']:
    yolo_dir_path = f'{dpath}/labels/{split}/'
    voc_dir_path = f'{dpath}/labels/{split}/voc/'
    Path(voc_dir_path).mkdir(parents=True, exist_ok=True)
    image_width = 512  # Replace with the actual width of the image
    image_height = 512  # Replace with the actual height of the image
    
    yolo_files = glob.glob(yolo_dir_path + '/*.txt')
    
    img_dir = os.path.abspath(f'{dpath}/images/{split}/')

    for file in tqdm(yolo_files, desc=f'{split}'):
        voc_file = os.path.basename(file).replace('.txt', '.xml')
        voc_path = os.path.join(voc_dir_path,voc_file)
        yolo_to_voc(file, voc_path, image_width, image_height, img_dir)    
