import numpy as np 
import os
import cv2
from os.path import join, split, isdir, isfile, abspath
import torch
from PIL import Image
import random
import collections
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from .transforms import get_transformations


class SemanLineDataset(Dataset):
    def __init__(self, root_dir, df_names, split='train', transform=None):
        lines = df_names['file_name'].tolist()
        lines_with_flip = lines.copy()
        for l in lines:
            lines_with_flip.append(str(l)+'_flip')
        lines = lines_with_flip
        self.image_path = [join(root_dir, str(i)+".jpg") for i in lines]
        self.data_path = [join(root_dir, str(i)+".npy") for i in lines]
        self.split = split
        self.transform = transform
    
    def __getitem__(self, item):
        assert isfile(self.image_path[item]), self.image_path[item]
        image = cv2.imread(self.image_path[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        data = np.load(self.data_path[item], allow_pickle=True).item()
        hough_space_label8 = data["hough_space_label8"].astype(np.float32)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        hough_space_label8 = torch.from_numpy(hough_space_label8).unsqueeze(0)
        if self.split == 'val':
            gt_coords = data["coords"]
            return image, hough_space_label8, gt_coords, self.image_path[item].split('/')[-1]
        elif self.split == 'train':
            return image, hough_space_label8, self.image_path[item].split('/')[-1]

    def __len__(self):
        return len(self.image_path)


class SemanLineDatasetTest(Dataset):
    def __init__(self, root_dir, transform=None):
        root_path_dir = Path(root_dir)
        self.image_path = [str(l) for l in list(root_path_dir.glob('*.jpeg')) +\
                                           list(root_path_dir.glob('*.jpg'))  +\
                                           list(root_path_dir.glob('*.png'))]
        self.transform = transform
        
    def __getitem__(self, item):
        assert isfile(self.image_path[item]), self.image_path[item]
        image = cv2.imread(self.image_path[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, self.image_path[item]

    def __len__(self):
        return len(self.image_path)


def get_loader(root_dir, df_names, batch_size, img_size=(400,400), 
                                               num_thread=4, 
                                               pin=True, 
                                               test=False, 
                                               split='train',
                                               transform_name='no_aug'):

    if test is False:
        transform_train = get_transformations(transform_name)
        dataset = SemanLineDataset(root_dir=root_dir, 
                                   df_names=df_names, 
                                   transform=transform_train, 
                                   split=split)
    else:
        transform_test = get_transformations('test_aug', image_size=img_size)
        dataset = SemanLineDatasetTest(root_dir=root_dir, 
                                       transform=transform_test)

    data_loader = DataLoader(dataset=dataset, 
                             batch_size=batch_size, 
                             shuffle=True, 
                             num_workers=num_thread,
                             pin_memory=pin)
    return data_loader

        
