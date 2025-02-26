import os
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from numpy.random import shuffle, choice
import glob
from ultralytics.data.augment import classify_augmentations
from ultralytics.data.utils import check_det_dataset
from PIL import Image
import math


class DomainSampler(Sampler):
    def __init__(self, batch_size, num_source_samples=12000, num_target_samples=12000):
        # Assume a concat [source_train_samples + target_train_samples] indices list, 
        #       where idxs [0 to num_source_samples] domain 0, idxs [num_source_samples + 1 to num_source_samples + num_target_samples] domain 1
        self.source_idxs = list(range(num_source_samples))            
        self.target_idxs = list(range(num_source_samples, num_source_samples+num_target_samples))
        self.batch_size = batch_size
        self.idx_size = min(num_source_samples, num_target_samples)
        
    def create_batches(self):
        alternating_idxs = [None]*(self.idx_size*2)
        alternating_idxs[::2] = choice(self.source_idxs, self.idx_size)
        alternating_idxs[1::2] = choice(self.target_idxs, self.idx_size)
        batches = [alternating_idxs[i:i + self.batch_size] for i in range(0, len(alternating_idxs), self.batch_size)]
        return batches

    def __iter__(self):
        self.batches = self.create_batches()
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

class DomainDataset(Dataset):
    def __init__(self, batch_size, source_yaml, target_yaml, args):
        
        source_data = check_det_dataset(source_yaml)
        source_img_path = source_data['train']
        target_data = check_det_dataset(target_yaml)
        target_img_path = target_data['train']
        
        if isinstance(source_img_path, list):
            source_img_path = source_img_path[0]
            
        if isinstance(target_img_path, list):
            target_img_path = target_img_path[0]
                        
        self.transform = classify_augmentations(
                            size=args.imgsz,
                            hflip=args.fliplr,
                            vflip=args.flipud,
                            erasing=args.erasing,
                            auto_augment=args.auto_augment,
                            hsv_h=args.hsv_h,
                            hsv_s=args.hsv_s,
                            hsv_v=args.hsv_v,
                        )
        
        self.source_img_files = glob.glob(f'{source_img_path}/*')
        self.source_labels = [0]*len(self.source_img_files)
        self.target_img_files = glob.glob(f'{target_img_path}/*')
        self.target_labels = [1]*len(self.target_img_files)
                
        if len(self.target_img_files) < len(self.source_img_files):
            copies = int(math.ceil(len(self.source_img_files)/len(self.target_img_files)))
            new_target_img_files = self.target_img_files*copies
            new_target_labels = self.target_labels*copies
            self.target_img_files = new_target_img_files[:len(self.source_img_files)]
            self.target_labels = new_target_labels[:len(self.source_img_files)]
        
        self.image_paths = self.source_img_files + self.target_img_files
        self.labels = self.source_labels + self.target_labels
        
        self.sampler = DomainSampler(batch_size, len(self.source_img_files), len(self.target_img_files))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = image.convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
