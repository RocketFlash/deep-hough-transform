import torch
import numpy as np
import math
import cv2 
import os
import torchvision
from .metric import EA_metric
from .basic_ops import *
import shutil
from torchvision import transforms


def draw_line(y, x, angle, image, color=(0,0,255), num_directions=24):
    '''
    Draw a line with point y, x, angle in image with color.
    '''
    cv2.circle(image, (x, y), 2, color, 2)
    H, W = image.shape[:2]
    angle = int2arc(angle, num_directions)
    point1, point2 = get_boundary_point(y, x, angle, H, W)
    cv2.line(image, point1, point2, color, 2)
    return image

def convert_line_to_hough(line, size=(32, 32)):
    H, W = size
    theta = line.angle()
    alpha = theta + np.pi / 2
    if theta == -np.pi / 2:
        r = line.coord[1] - W/2
    else:
        k = np.tan(theta)
        y1 = line.coord[0] - H/2
        x1 = line.coord[1] - W/2
        r = (y1 - k*x1) / np.sqrt(1 + k**2)
    return alpha, r

def line2hough(line, numAngle, numRho, size=(32, 32)):
    H, W = size
    alpha, r = convert_line_to_hough(line, size)

    irho = int(np.sqrt(H*H + W*W) + 1) / ((numRho - 1))
    itheta = np.pi / numAngle

    r = int(np.round(r / irho)) + int((numRho) / 2)
    alpha = int(np.round(alpha / itheta))
    if alpha >= numAngle:
        alpha = numAngle - 1
    return alpha, r

def reverse_mapping(point_list, numAngle, numRho, size=(32, 32)):
    H, W = size
    irho = int(np.sqrt(H*H + W*W) + 1) / ((numRho - 1))
    itheta = np.pi / numAngle
    b_points = []

    for (thetai, ri) in point_list:
        theta = thetai * itheta
        r = ri - numRho // 2
        cosi = np.cos(theta) / irho
        sini = np.sin(theta) / irho
        if sini == 0:
            x = np.round(r / cosi + W / 2)
            b_points.append((0, int(x), H-1, int(x)))
        else:
            # print('k = %.4f', - cosi / sini)
            # print('b = %.2f', np.round(r / sini + W * cosi / sini / 2 + H / 2))
            angle = np.arctan(- cosi / sini)
            y = np.round(r / sini + W * cosi / sini / 2 + H / 2)
            p1, p2 = get_boundary_point(int(y), 0, angle, H, W)
            if p1 is not None and p2 is not None:
                b_points.append((p1[1], p1[0], p2[1], p2[0]))
    return b_points

def visulize_mapping(b_points, size, filename):
    img = cv2.imread(filename) #change the path when using other dataset.
    img = cv2.resize(img, size)
    for (y1, x1, y2, x2) in b_points:
        img = cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), thickness=5)
    return img


def visulize_mapping_orig(b_points, size, filename):
    img = cv2.imread(filename) #change the path when using other dataset.
    img_h, img_w, _ = img.shape
    sw = img_w / size[1]
    sh = img_h / size[0]
    for (y1, x1, y2, x2) in b_points:
        x1_s = int(sw * x1)
        x2_s = int(sw * x2)
        y1_s = int(sh * y1)
        y2_s = int(sh * y2)
        img = cv2.line(img, (x1_s, y1_s), (x2_s, y2_s), (255, 255, 0), thickness=9)
    return img


def caculate_precision(b_points, gt_coords, thresh=0.90):
    N = len(b_points)
    if N == 0:
        return 0, 0
    ea = np.zeros(N, dtype=np.float32)
    for i, coord_p in enumerate(b_points):
        if coord_p[0]==coord_p[2] and coord_p[1]==coord_p[3]:
            continue
        l_pred = Line(list(coord_p))
        for coord_g in gt_coords:
            l_gt = Line(list(coord_g))
            ea[i] = max(ea[i], EA_metric(l_pred, l_gt))
    return (ea >= thresh).sum(), N

def caculate_recall(b_points, gt_coords, thresh=0.90):
    N = len(gt_coords)
    if N == 0:
        return 1.0, 0
    ea = np.zeros(N, dtype=np.float32)
    for i, coord_g in enumerate(gt_coords):
        l_gt = Line(list(coord_g))
        for coord_p in b_points:
            if coord_p[0]==coord_p[2] and coord_p[1]==coord_p[3]:
                continue
            l_pred = Line(list(coord_p))
            ea[i] = max(ea[i], EA_metric(l_pred, l_gt))
    return (ea >= thresh).sum(), N


def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(path, filename))
    if is_best:
        shutil.copyfile(os.path.join(path, filename), os.path.join(path, 'model_best.pth'))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def batch_grid(images):
    image_grid = torchvision.utils.make_grid(images, nrow=4)
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.255]
    )
    return inv_normalize(image_grid)

class DayHourMinute(object):
    def __init__(self, seconds):
        self.days = int(seconds // 86400)
        self.hours = int((seconds- (self.days * 86400)) // 3600)
        self.minutes = int((seconds - self.days * 86400 - self.hours * 3600) // 60)

