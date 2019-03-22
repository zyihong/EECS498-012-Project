import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import png
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from colormap.colors import Color, hex2rgb
from sklearn.metrics import average_precision_score as ap_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm




class twotimes_conv(nn.Module):
    def __init__(self,inch,outch):
        super(twotimes_conv,self).__init__()
        self.doubleconv=nn.Sequential(
            nn.Conv2d(in_channels=inch,out_channels=outch,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=outch, out_channels=outch, kernel_size=3,padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x=self.doubleconv(x)
        return x



class G1_Net(nn.Module):
    def __init__(self, n_class=3):
        super(G1_Net, self).__init__()
        self.n_class = n_class
        self.twoconv1=twotimes_conv(4,64)#256
        self.maxpool=nn.MaxPool2d(2,2)#128
        self.twoconv2=twotimes_conv(64,128)#128
        # nn.MaxPool2d(2,2),#64
        self.twoconv3=twotimes_conv(128,256)#64
        self.twoconv32=twotimes_conv(256,512)
        self.dconv32=nn.ConvTranspose2d(512,256,stride=2,kernel_size=2)
        self.twocon42=twotimes_conv(512,256)
        self.dconv3=nn.ConvTranspose2d(256,128,stride=2,kernel_size=2)#128
        self.twocon4=twotimes_conv(256,128)
        self.dconv4=nn.ConvTranspose2d(128,64,stride=2,kernel_size=2)#256
        self.twocon5=twotimes_conv(128,64)
        self.final=nn.Sequential(nn.Conv2d(64,self.n_class,1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x0 = self.twoconv1(x)
        x1=self.maxpool(x0)
        x1=self.twoconv2(x1)
        x2=self.maxpool(x1)
        x2=self.twoconv3(x2)
        x22=self.maxpool(x2)
        # x2=self.dconv3(x2)
        x22=self.twoconv32(x22)
        x22=self.dconv32(x22)
        x32=torch.cat((x2,x22),1)
        x32=self.twocon42(x32)
        x32=self.dconv3(x32)
        x3=torch.cat((x1,x32),1)
        x3=self.twocon4(x3)
        x3=self.dconv4(x3)
        x4=torch.cat((x0,x3),1)
        x4=self.twocon5(x4)
        x=self.final(x4)
        return x

