import numpy as np
import cv2
from PIL import Image
import argparse
import os, sys

from os.path import join, split, splitext, abspath, isfile
sys.path.insert(0, abspath(".."))
sys.path.insert(0, abspath("."))

from pathlib import Path
from sklearn.model_selection import KFold
from skimage.measure import label, regionprops
from src.utils import Line, LineAnnotation, line2hough
from easydict import EasyDict

import pandas as pd
import matplotlib.pyplot as plt


def nearest8(x):
    return int(np.round(x/8)*8)


def vis_anno(image, annotation):
    mask = annotation.oriental_mask()
    mask_sum = mask.sum(axis=0).astype(bool)
    image_cp = image.copy()
    image_cp[mask_sum, ...] = [0, 255, 0]
    mask = np.zeros((image.shape[0], image.shape[1]))
    mask[mask_sum] = 1
    return image_cp, mask


def get_lines_from_txt(label_file, for_viz=False):
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


def get_annotation(image_size, label_file, num_directions, h_flip=False):
    H, W, _ = image_size
    lines_raw = get_lines_from_txt(label_file)
    lines = []
    
    if h_flip:
        lines = [Line([l[0], W-1-l[1], l[2], W-1-l[3]]) for l in lines_raw]
    else:
        lines = [Line(l) for l in lines_raw]
    
    annotation = LineAnnotation(size=[H, W], divisions=num_directions, lines=lines)
    return annotation


def show_image(image):
    fig, ax = plt.subplots(1, figsize=(50, 30))
    ax.imshow(image)
    plt.show()

    
def visualize_data(label_file):
    print(label_file)
    
    lines = get_lines_from_txt(label_file, for_viz=True)
    image_path = label_file.with_suffix('.jpeg')
    if isfile(image_path):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, _ = image.shape
    else:
        print(f"Warning: image {image_path} doesnt exist!")
        return
    
    for line in lines:
        y1, x1, y2, x2 = line
        cv2.line(image, (x1,y1), (x2,y2), (0, 255, 0), 2)
    

def prepare_data(args, labels_files, save_dir='./'):

    for idx, label_file in tqdm(enumerate(labels_files), total=len(labels_files)):
        image_path = label_file.with_suffix('.jpeg')
        file_name = image_path.stem

        if isfile(image_path):
            im = cv2.imread(str(image_path))
            image_size = im.shape
            im = cv2.resize(im, (args.fixsize, args.fixsize))
        else:
            print(f"Warning: image {image_path} doesnt exist!")
            continue
        for argument in range(2):
            if argument == 0:
                annotation = get_annotation(image_size, label_file, args.num_directions)
            else:
                im = cv2.flip(im, 1)
                file_name = file_name + '_flip'
                annotation = get_annotation(image_size, label_file, args.num_directions, h_flip=True)

            # resize image and annotations
            if args.fixsize is not None:
                newH = nearest8(args.fixsize)
                newW = nearest8(args.fixsize)
            else:
                newH = nearest8(image_size[0])
                newW = nearest8(image_size[1])
            
            im = cv2.resize(im, (newW, newH))
            annotation.resize(size=[newH, newW])
            
            vis, mask = vis_anno(im, annotation)
            hough_space_label = np.zeros((args.numangle, args.numrho))

            for l in annotation.lines:
                theta, r = line2hough(l, numAngle=args.numangle, numRho=args.numrho, size=(newH, newW))
                hough_space_label[theta, r] += 1

            hough_space_label = cv2.GaussianBlur(hough_space_label, (5,5), 0)

            if hough_space_label.max() > 0:
                hough_space_label = hough_space_label / hough_space_label.max()

            gt_coords = []
            for l in annotation.lines:
                gt_coords.append(l.coord)
            gt_coords = np.array(gt_coords)
            data = dict({
                "hough_space_label8": hough_space_label,
                "coords": gt_coords
            })

            save_name = save_dir / file_name

            np.save(save_name, data)
            cv2.imwrite(str(save_name) + '.jpg', im)
            cv2.imwrite(str(save_name) + '_p_label.jpg', hough_space_label*255)
            cv2.imwrite(str(save_name) + '_vis.jpg', vis)
            cv2.imwrite(str(save_name) + '_mask.jpg', mask*255)


def make_k_fold_split(labels_files, k=5, save_dir='./'):
    RANDOM_STATE = 28

    labels_files_names = [l.stem for l in labels_files]
    df_folds = pd.DataFrame(labels_files_names, columns =['file_name'])
    kfold = KFold(n_splits=k, random_state=RANDOM_STATE, shuffle=True)


    for fold, (train_index, valid_index) in enumerate(kfold.split(labels_files_names)):
        df_folds.loc[valid_index, 'fold'] = int(fold)

    df_folds['fold'] = df_folds['fold'].astype(int)
    df_folds.to_csv(save_dir / 'folds.csv', index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare semantic line data format.")
    parser.add_argument('--root', type=str, required=True, help='the data root dir.')
    parser.add_argument('--num_directions', type=int, default=12, help='the division of semicircular angle')
    parser.add_argument('--save-dir', type=str, required=True, help='save-dir')
    parser.add_argument('--fixsize', type=int, default=None, help='fix resize of images and annotations')
    parser.add_argument('--numangle', type=int, default=80, required=True)
    parser.add_argument('--numrho', type=int, default=80, required=True)
    parser.add_argument('--is_notebook', default=False, action='store_true')
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()

    data_dir = Path(abspath(args.root)).resolve()
    save_dir = Path(abspath(args.save_dir)).resolve()

    save_dir.mkdir(exist_ok=True)

    labels_files = list(data_dir.glob('*.txt'))

    if args.is_notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm.auto import tqdm

    prepare_data(args, labels_files, save_dir=save_dir)
    make_k_fold_split(labels_files, k=args.k, save_dir=save_dir)

