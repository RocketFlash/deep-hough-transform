import torch
import torch.nn as nn
import numpy as np
# from skimage.transform import hough_line
from .cython_hough.hough_line_custom import hough_line
import matplotlib.pyplot as plt

PI = 3.14159265358979323846


class DHT_Layer(nn.Module):
    def __init__(self, input_dim, dim, numAngle, numRho):
        super(DHT_Layer, self).__init__()
        self.fist_conv = nn.Sequential(
            nn.Conv2d(input_dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.dht = DHT(numAngle=numAngle, numRho=numRho)
        self.convs = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.fist_conv(x)
        x = self.dht(x)
        x = self.convs(x)
        return x


class DHT(nn.Module):
    def __init__(self, numAngle, numRho):
        super(DHT, self).__init__()       
        self.line_agg = C_dht(numAngle, numRho)

    def forward(self, x):
        accum = self.line_agg(x)
        return accum




def line_accum_forward(inpt, numrho, numangle):
    N, C, H, W = inpt.shape
    accum = torch.zeros(N, C, numangle, numrho).type_as(inpt)

    for im_idx, img_i in enumerate(inpt):
        for ch_idx, channel_i in enumerate(img_i):
            in_t = channel_i.cpu().numpy()
            out = hough_line(in_t, numangle, numrho, H, W)
            plt.imshow(out)
            plt.show()
            break
            accum[im_idx, ch_idx] = torch.from_numpy(out)
    return accum

            

class C_dht_Function(torch.autograd.Function):
    @staticmethod 
    def forward(ctx, feat, numangle, numrho):
        N, C, H, W = feat.size()
        out = line_accum_forward(feat, numrho, numangle)


        ctx.save_for_backward(feat)
        ctx.numangle = numangle
        ctx.numrho = numrho
        return out
        
    @staticmethod
    def backward(ctx, grad_output):
        feat = ctx.saved_tensors[0]
        N, C, H, W = feat.size()
        numangle = ctx.numangle
        numrho = ctx.numrho

        tabSin, tabCos = init_table(numangle, numrho, H, W) 

        out = torch.zeros_like(feat).type_as(feat)
        # out = dh.backward(grad_output.contiguous(), out, feat, numangle, numrho)
        grad_in = out[0]
        return grad_in, None, None


class C_dht(torch.nn.Module):
    def __init__(self, numAngle, numRho):
        super(C_dht, self).__init__()
        self.numAngle = numAngle
        self.numRho = numRho
    
    def forward(self, feat):
        return C_dht_Function.apply(feat, self.numAngle, self.numRho)
