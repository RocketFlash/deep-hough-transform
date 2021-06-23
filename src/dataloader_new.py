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
from .utils import Line, LineAnnotation, line2hough

def nearest8(x):
    return int(np.round(x/8)*8)


def get_lines_from_txt(label_file):
    lines = []
    with open(label_file) as f:
        data = f.readlines()
        for line in data:
            data1 = line.strip().split(',')
            if len(data1) <= 3:
                continue
            data1 = [int(float(x)) for x in data1 if x!=''] # x1, y1, x2, y2
            if data1[1]==data1[3] and data1[0]==data1[2]:
                continue
            lines.append([data1[1], data1[0], data1[3], data1[2]]) # y1, x1, y2, x2
    return lines


def get_annotation(image_size, lines, num_directions):
    H, W, _ = image_size
    lines = [Line(l) for l in lines]

    return LineAnnotation(size=[H, W], divisions=num_directions, lines=lines)


def lines_to_keypoints(lines):
    keypoints = []
    line_indexes = []

    for l_idx, l in enumerate(lines):
        keypoints.append((l[0], l[1]))
        keypoints.append((l[2], l[3]))
        line_indexes.append(l_idx)
        line_indexes.append(l_idx)
    
    return keypoints, line_indexes


def keypoints_to_lines(keypoints, line_indexes):
    lines = []
    line_indexes = np.array(line_indexes)
    line_indexes_unique = np.unique(line_indexes)
    keypoints = np.array(keypoints)

    for li in line_indexes_unique:
        line_keypoints = keypoints[line_indexes==li, :]

        if line_keypoints[0][1] >= line_keypoints[1][1]:
            lines.append(list(line_keypoints[1])+list(line_keypoints[0]))
        else:
            lines.append(list(line_keypoints[0])+list(line_keypoints[1]))
    return lines


def prepare_data(label_file, transform=None, 
                             num_directions=12,
                             numangle=100,
                             numrho=100,
                             ext='.jpeg'):

    image_path = label_file.with_suffix(ext)

    im = cv2.imread(str(image_path))
    image_size = im.shape
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    lines = get_lines_from_txt(label_file)

    if transform is not None:
        keypoints, line_idxs = lines_to_keypoints(lines)
        transformed = transform(image=im, keypoints=keypoints, line_idxs=line_idxs)
        im = transformed['image']
        transformed_keypoints = transformed['keypoints']
        transformed_line_idxs = transformed['line_idxs']
        lines = keypoints_to_lines(transformed_keypoints, transformed_line_idxs)

    annotation = get_annotation(image_size, lines, num_directions)
    
    # newH = nearest8(image_size[0])
    # newW = nearest8(image_size[1])
    
    # print(im.shape)
    # im = cv2.resize(im, (newW, newH))
    # annotation.resize(size=[newH, newW])
    
    _, newH, newW = im.shape
    hough_space_label = np.zeros((numangle, numrho))

    for l in annotation.lines:
        theta, r = line2hough(l, numAngle=numangle, numRho=numrho, size=(newH, newW))
        if r<numrho:
            hough_space_label[theta, r] += 1

    hough_space_label = cv2.GaussianBlur(hough_space_label, (5,5), 0)

    if hough_space_label.max() > 0:
        hough_space_label = hough_space_label / hough_space_label.max()

    gt_coords = [l.coord for l in annotation.lines]

    data = dict({
        "hough_space_label8": hough_space_label,
        "coords": np.array(gt_coords)
    })

    return im, data


class SemanLineDataset(Dataset):
    def __init__(self, root_dir, df_names,
                                 num_directions=12,
                                 numangle=100,
                                 numrho=100,
                                 imgs_ext='.jpeg',
                                 split='train', 
                                 transform=None):
        lines = df_names['file_name'].tolist()
        self.label_paths = [join(root_dir, str(i)+".txt") for i in lines]
        
        self.num_directions = num_directions
        self.numangle = numangle
        self.numrho = numrho
        self.imgs_ext = imgs_ext
        self.split = split
        self.transform = transform
    
    def __getitem__(self, item):
        assert isfile(self.label_paths[item]), self.label_paths[item]

        image, data = prepare_data(Path(self.label_paths[item]), transform=self.transform,
                                                        num_directions=self.num_directions,
                                                        numangle=self.numangle,
                                                        numrho=self.numrho,
                                                        ext=self.imgs_ext)

        hough_space_label8 = data["hough_space_label8"].astype(np.float32)
        hough_space_label8 = torch.from_numpy(hough_space_label8).unsqueeze(0)

        if self.split == 'val':
            gt_coords = data["coords"]
            return image, hough_space_label8, gt_coords, self.label_paths[item].split('/')[-1]
        elif self.split == 'train':
            return image, hough_space_label8, self.label_paths[item].split('/')[-1]

    def __len__(self):
        return len(self.label_paths)


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

        
