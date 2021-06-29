import torch
import numpy as np 
import torch.nn as nn

from .backbone.fpn import FPN101, FPN50, FPN18, ResNext50_FPN
from .backbone.mobilenet import MobileNet_FPN
from .backbone.vgg_fpn import VGG_FPN
from .backbone.res2net import res2net50_FPN

from .dht import DHT_Layer
from .deep_hought_torch import DHT_Layer as DHT_Layer_torch

class Net(nn.Module):
    def __init__(self, numAngle, numRho, backbone, hough_cuda=True):
        super(Net, self).__init__()
        if backbone == 'resnet18':
            self.backbone = FPN18(pretrained=True, output_stride=32)
            output_stride = 32
        if backbone == 'resnet50':
            self.backbone = FPN50(pretrained=True, output_stride=16)
            output_stride = 16
        if backbone == 'resnet101':
            self.backbone = FPN101(output_stride=16)
            output_stride = 16
        if backbone == 'resnext50':
            self.backbone = ResNext50_FPN(output_stride=16)
            output_stride = 16
        if backbone == 'vgg16':
            self.backbone = VGG_FPN()
            output_stride = 16
        if backbone == 'mobilenetv2':
            self.backbone = MobileNet_FPN()
            output_stride = 32
        if backbone == 'res2net50':
            self.backbone = res2net50_FPN()
            output_stride = 32
        
        DHT_Layer_v = DHT_Layer if hough_cuda else DHT_Layer_torch
        if backbone == 'mobilenetv2':
            self.dht_detector1 = DHT_Layer_v(32, 32, numAngle=numAngle, numRho=numRho)
            self.dht_detector2 = DHT_Layer_v(32, 32, numAngle=numAngle, numRho=numRho // 2)
            self.dht_detector3 = DHT_Layer_v(32, 32, numAngle=numAngle, numRho=numRho // 4)
            self.dht_detector4 = DHT_Layer_v(32, 32, numAngle=numAngle, numRho=numRho // (output_stride // 4))
            
            self.last_conv = nn.Sequential(
                nn.Conv2d(128, 1, 1)
            )
        else:
            self.dht_detector1 = DHT_Layer_v(256, 128, numAngle=numAngle, numRho=numRho)
            self.dht_detector2 = DHT_Layer_v(256, 128, numAngle=numAngle, numRho=numRho // 2)
            self.dht_detector3 = DHT_Layer_v(256, 128, numAngle=numAngle, numRho=numRho // 4)
            self.dht_detector4 = DHT_Layer_v(256, 128, numAngle=numAngle, numRho=numRho // (output_stride // 4))
            
            self.last_conv = nn.Sequential(
                nn.Conv2d(512, 1, 1)
            )

        self.numAngle = numAngle
        self.numRho = numRho

    def upsample_cat(self, p1, p2, p3, p4):
        p1 = nn.functional.interpolate(p1, size=(self.numAngle, self.numRho), mode='bilinear')
        p2 = nn.functional.interpolate(p2, size=(self.numAngle, self.numRho), mode='bilinear')
        p3 = nn.functional.interpolate(p3, size=(self.numAngle, self.numRho), mode='bilinear')
        p4 = nn.functional.interpolate(p4, size=(self.numAngle, self.numRho), mode='bilinear')
        return torch.cat([p1, p2, p3, p4], dim=1)

    def forward(self, x):
        p1, p2, p3, p4 = self.backbone(x)
      
        p1 = self.dht_detector1(p1)
        p2 = self.dht_detector2(p2)
        p3 = self.dht_detector3(p3)
        p4 = self.dht_detector4(p4)

        cat = self.upsample_cat(p1, p2, p3, p4)
        logist = self.last_conv(cat)

        return logist
