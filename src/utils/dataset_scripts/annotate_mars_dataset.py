import cv2
import numpy as np
from glob import glob
import sys
from pathlib import Path
from alive_progress import alive_bar

VISUALIZE = False

def get_aabbs(mask, min_size=3, use_blobs=False):
    # Function to get the bounding boxes (aabb = axis aligned bounding box):
    if use_blobs:
        _,img = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
        _,mask = cv2.connectedComponents(img.astype(np.uint8))
        
        aabbs = get_aabbs(mask,min_size=min_size)

    else:
        ids = np.unique(mask)[1:]

        aabbs = [] 
        for id in ids:
            individual = mask == id
            pixels = np.argwhere(individual)
            pixels[:,[0,1]] = pixels[:,[1,0]]
            min = pixels.min(axis=0).tolist()
            max = pixels.max(axis=0).tolist()
            
            if (max[0]-min[0] > min_size) and (max[1]-min[1] > min_size):
                aabbs.append(min+max)

    aabbs = np.array(aabbs)

    return aabbs

def draw_aabbs(img, aabbs, color, width=1):
    # Draw the bounding box:
    for idx in range(0,aabbs.shape[0]):
        bl_corner = aabbs[idx,0:2].astype(int)
        tr_corner = aabbs[idx,2:].astype(int)
        img = cv2.rectangle(img, tuple(bl_corner), tuple(tr_corner), color, width)
    return img

# Create dir for visualized labels
if VISUALIZE: Path(sys.argv[1] + '/visualized_labels/').mkdir(parents=True, exist_ok=True)

img_paths = glob(sys.argv[1] + '/images/*.png')
with alive_bar(len(img_paths)) as bar:
    for img_path in img_paths:
        # Get the corresponding image mask:
        num = img_path.split('/')[-1].split('.')[0]
        labels = cv2.imread(f'{sys.argv[1]}/masks/{num}.png')
        img = cv2.imread(img_path)
        img_h,img_w,img_c = img.shape

        craters = labels[:,:,0]

        # Masks for green and red channels:
        green_mask = labels[:,:,1] > 1
        red_mask = labels[:,:,2] > 1

        # Get the masks for rocks, dunes, and mountains:
        rock_mask = np.bitwise_and(green_mask, red_mask)
        dune_mask = np.bitwise_xor(green_mask, rock_mask)
        mountain_mask = np.bitwise_xor(green_mask,red_mask)

        dunes = np.zeros(craters.shape)
        mountains = np.zeros(craters.shape)
        rocks = np.zeros(craters.shape)

        dunes[dune_mask] = labels[dune_mask,1]
        mountains[mountain_mask] = labels[mountain_mask,2]
        rocks[rock_mask] = labels[rock_mask,1]

        # Get the bounding boxes:
        crater_aabbs = get_aabbs(craters)
        dune_aabbs = get_aabbs(dunes)
        mountain_aabbs = get_aabbs(mountains)
        rock_aabbs = get_aabbs(rocks,use_blobs=True)
        
        '''
        Write out annotations in ultralytics format (in same images dir)
            - The *.txt file should be formatted with one line per object in [class x_center y_center width height] format
            - Box coordinates must be in normalized xywh format (from 0 to 1)
            - If your boxes are in pixels, you should divide x_center and width by image width, and y_center and height by image height. 
            - Class numbers should be zero-indexed (start with 0)
                - [crater: 0, dune: 1, mountain: 2]
        '''
        anno_lines = ''
        
        # Craters
        for box in crater_aabbs:
            l,t,r,b = box
            w = r-l
            h = b-t
            x_center = l + (w/2)
            y_center = t + (h/2)
            anno_lines += f'0 {x_center/img_w} {y_center/img_h} {w/img_w} {h/img_h}\n'
        
        for box in dune_aabbs:
            l,t,r,b = box
            w = r-l
            h = b-t
            x_center = l + (w/2)
            y_center = t + (h/2)
            anno_lines += f'1 {x_center/img_w} {y_center/img_h} {w/img_w} {h/img_h}\n'
            
        for box in mountain_aabbs:
            l,t,r,b = box
            w = r-l
            h = b-t
            x_center = l + (w/2)
            y_center = t + (h/2)
            anno_lines += f'2 {x_center/img_w} {y_center/img_h} {w/img_w} {h/img_h}\n'
            
        with open(f'{sys.argv[1]}/images/{num}.txt', 'w') as f:
            f.write(anno_lines)

        ###############################
        ### Draw the bounding boxes ###
        ###############################
        if VISUALIZE:
            # Loop through all craters:
            outlined = img
            outlined = draw_aabbs(outlined, crater_aabbs, (255,0,0))
            outlined = draw_aabbs(outlined, dune_aabbs, (0,255,0))
            outlined = draw_aabbs(outlined, mountain_aabbs, (0,0,255))
            outlined = draw_aabbs(outlined, rock_aabbs, (0,255,255))

            # Draw the boxes:
            cv2.imwrite(f'{sys.argv[1]}/visualized_labels/{num}.png',outlined)
            
        bar()
