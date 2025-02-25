# From: https://github.com/nuggy875/mask2bbox_cityscape/blob/master/main.py

import os
import os.path as osp
import glob
import json
from xml.etree.ElementTree import Element, SubElement, ElementTree

load_json_path = '/mnt/Space-Vision/data/yoco_journal/cityscapes/gtFine/test'
save_xml_path = '/mnt/Space-Vision/data/yoco_journal/cityscapes/voc/Annotations/test/'

if __name__ == "__main__":
    anno_json_list = glob.glob(osp.join(load_json_path, '*/*.json'))

    for anno_mask in anno_json_list:
        fn_element = anno_mask.split('/')[-1].split('.')[0].split('_')[0:3]
        file_name = fn_element[0] + '_' + fn_element[1] + '_' + fn_element[2] + '_leftImg8bit'
        file_folder = fn_element[0]

        with open(anno_mask) as json_file:          # For Images
            json_data = json.load(json_file)

            img_height = json_data['imgHeight']     # Image Height
            img_width = json_data['imgWidth']       # Image Widht
            img_objects = json_data['objects']      # Object (polygons)
            objects_num = len(img_objects)          # Object Num

            # Make XML
            root = Element('annotation')
            SubElement(root, 'folder').text = 'cityscape_hazy'
            SubElement(root, 'filename').text = file_name + '.png'    ## 수정

            size = SubElement(root, 'size')
            SubElement(size, 'width').text = str(img_width)
            SubElement(size, 'height').text = str(img_height)
            SubElement(size, 'depth').text = str(3)

            SubElement(root, 'segmented').text = '0'

            for i, obj in enumerate(img_objects):   # For Objects
                x_min = img_width
                y_min = img_height
                x_max = 0
                y_max = 0

                for inf in obj['polygon']:          # For Polygons in Objects
                    x = inf[0]
                    y = inf[1]
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                # print('min({}, {}), max({}, {}), width:{}, height:{}'.format(x_min, y_min, x_max, y_max, x_max-x_min, y_max-y_min))

                # Make Bounding Box
                class_usage=True
                # if obj['label']=='person' or obj['label']=='rider':
                #     obj_name = 'person'
                # elif obj['label']=='bicycle':
                #     obj_name = 'bicycle'
                if obj['label']=='car' or obj['label']=='truck':
                    obj_name = 'car'
                # elif obj['label']=='bus':
                #     obj_name = 'bus'
                # elif obj['label']=='motorcycle':
                #     obj_name = 'motorbike'
                else:
                    class_usage=False
                    
                if class_usage:
                    obj_bndbox = [x_min, y_min, x_max, y_max]
                    # print('obj_name:{}, obj_bndbox:{}'.format(obj_name, obj_bndbox))
                    
                    obj = SubElement(root, 'object')
                    SubElement(obj, 'name').text = obj_name
                    SubElement(obj, 'pose').text = 'Unspecified'
                    SubElement(obj, 'truncated').text = '0'
                    SubElement(obj, 'difficult').text = '0'
                    bbox = SubElement(obj, 'bndbox')
                    SubElement(bbox, 'xmin').text = str(x_min)
                    SubElement(bbox, 'ymin').text = str(y_min)
                    SubElement(bbox, 'xmax').text = str(x_max)
                    SubElement(bbox, 'ymax').text = str(y_max)
            tree = ElementTree(root)
            save_path_dir = save_xml_path+'/'
            if not osp.exists(save_path_dir):
                os.makedirs(save_path_dir)

            tree.write(save_path_dir+'/{}.xml'.format(file_name), encoding='utf-8')