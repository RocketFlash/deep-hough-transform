import argparse
import os
import random
import time
from os.path import isfile, join, split

import torch
import numpy as np
import tqdm
import yaml
import cv2
from skimage.measure import label, regionprops

from src.logger import Logger
from src.dataloader import get_loader
from src.model.network import Net
from src.utils import reverse_mapping, visulize_mapping
from src.basic_ops import *

parser = argparse.ArgumentParser(description='PyTorch Semantic-Line Training')
# arguments from command line
parser.add_argument('--config', default="./configs/default_config.yml", help="path to config file")
parser.add_argument('--model', required=True, help='path to the pretrained model')
parser.add_argument('--tmp', default="", help='tmp')
args = parser.parse_args()

assert os.path.isfile(args.config)
CONFIGS = yaml.load(open(args.config))

# merge configs
if args.tmp != "" and args.tmp != CONFIGS["MISC"]["TMP"]:
    CONFIGS["MISC"]["TMP"] = args.tmp

os.makedirs(CONFIGS["MISC"]["TMP"], exist_ok=True)
logger = Logger(os.path.join(CONFIGS["MISC"]["TMP"], "log.txt"))

def main():

    logger.info(args)

    model = Net(numAngle=CONFIGS["MODEL"]["NUMANGLE"], numRho=CONFIGS["MODEL"]["NUMRHO"], backbone=CONFIGS["MODEL"]["BACKBONE"])
    model = model.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])

    if args.model:
        if isfile(args.model):
            logger.info("=> loading pretrained model '{}'".format(args.model))
            checkpoint = torch.load(args.model)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.model, checkpoint['epoch']))
        else:
            logger.info("=> no pretrained model found at '{}'".format(args.model))
    # dataloader
    test_loader = get_loader(CONFIGS["DATA"]["TEST_DIR"], CONFIGS["DATA"]["TEST_LABEL_FILE"], 
                                batch_size=1, num_thread=CONFIGS["DATA"]["WORKERS"], test=True)

    logger.info("Data loading done.")

    
    
    logger.info("Start testing.")
    total_time = test(test_loader, model, args)
    
    logger.info("Test done! Total %d imgs at %.4f secs without image io, fps: %.3f" % (len(test_loader), total_time, len(test_loader) / total_time))

        
def test(test_loader, model, args):
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        bar = tqdm.tqdm(test_loader)
        iter_num = len(test_loader.dataset)
        ftime = 0
        ntime = 0
        for i, data in enumerate(bar):
            t = time.time()
            
            images, names = data
            
            img_size = images.shape[2:]
            images = images.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])
            size = (img_size[0], img_size[1])        
            key_points = model(images)
            key_points = torch.sigmoid(key_points)
            ftime += (time.time() - t)
            t = time.time()
            visualize_save_path = os.path.join(CONFIGS["MISC"]["TMP"], 'visualize_test')
            os.makedirs(visualize_save_path, exist_ok=True)

            binary_kmap = key_points.squeeze().cpu().numpy() > CONFIGS['MODEL']['THRESHOLD']
            kmap_label = label(binary_kmap, connectivity=1)
            props = regionprops(kmap_label)
            plist = []
            for prop in props:
                plist.append(prop.centroid)

            #size = (400, 400) #change it when using other dataset.
            b_points = reverse_mapping(plist, numAngle=CONFIGS["MODEL"]["NUMANGLE"], numRho=CONFIGS["MODEL"]["NUMRHO"], size=(400, 400))
            scale_w = size[1] / 400
            scale_h = size[0] / 400
            for i in range(len(b_points)):
                y1 = int(np.round(b_points[i][0] * scale_h))
                x1 = int(np.round(b_points[i][1] * scale_w))
                y2 = int(np.round(b_points[i][2] * scale_h))
                x2 = int(np.round(b_points[i][3] * scale_w))
                if x1 == x2:
                    angle = -np.pi / 2
                else:
                    angle = np.arctan((y1-y2) / (x1-x2))
                (x1, y1), (x2, y2) = get_boundary_point(y1, x1, angle, size[0], size[1])
                b_points[i] = (y1, x1, y2, x2)

            
            vis = visulize_mapping(b_points, size, names[0])
            cv2.imwrite(join(visualize_save_path, names[0].split('/')[-1]), vis)
            np_data = np.array(b_points)
            np.save(join(visualize_save_path, names[0].split('/')[-1].split('.')[0]), np_data)

    logger.info('forward time for total images: %.6f' % ftime)
    logger.info('post-processing time for total images: %.6f' % ntime)
    return ftime + ntime

if __name__ == '__main__':
    main()
