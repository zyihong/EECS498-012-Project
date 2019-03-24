import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
#import png
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
#from colormap.colors import Color, hex2rgb
from sklearn.metrics import average_precision_score as ap_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from torch.autograd import Variable
import torch.nn.functional as F




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

class G2_Net(nn.Module):

    def __init__(self, kernel_size=3):
        super(G2_Net, self).__init__()
        #input size (6,256,256)
        self.conv1 = nn.Sequential(
            nn.Conv2d(6,32,kernel_size,stride = 1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32,32,kernel_size,stride = 1, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(3,stride=2,padding=1))
        #(3,128,128)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size,stride = 1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64,64,kernel_size,stride = 1, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(3,stride=2,padding=1))
        #(64,64,64)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size,stride = 1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128,128,kernel_size,stride = 1, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(3,stride=2,padding=1))
        #(128,32,32)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size,stride = 1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256,256,kernel_size,stride = 1, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(3,stride=2,padding=1))
        #(256,16,16)
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(256,128,kernel_size,stride = 1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(128,128,kernel_size,stride = 1, padding=2),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        #(128,32,32)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256,64,kernel_size,stride = 1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(64,64,kernel_size,stride = 1, padding=2),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        #(64,64,64)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128,32,kernel_size,stride = 1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(32,32,kernel_size,stride = 1, padding=2),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        #(3,256,256)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64,3,kernel_size,stride = 1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(3,3,kernel_size,stride = 1, padding=2),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        
    def forward(self,inputs):
        e1 = self.conv1(inputs)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        d4 = self.deconv4(e4)
        d3_in = torch.cat((e3,d4),1)
        d3 = self.deconv3(d3_in)
        d2_in = torch.cat((e2,d3),1)
        d2 = self.deconv2(d2_in)
        d1_in = torch.cat((e1,d2),1)
        d1 = self.deconv1(d1_in)
        return d1


class Discriminator(nn.Module):

    def __init__(self, kernel_size=5, dim=64):
        super(Discriminator, self).__init__()
        self.kernel_size = kernel_size
        self.dim = dim
        self.conv1 = nn.Conv2d(3, self.dim, self.kernel_size, stride=2, padding=2)
        #(64,128,128)
        #LeakyRelu
        self.conv2 = nn.Conv2d(self.dim, 2*self.dim,kernel_size,stride=2,padding=2)
        self.bn1 = nn.BatchNorm2d(2*self.dim)
        #LeakyRelu
        #(128,64,64)
        self.conv3 = nn.Conv2d(2*self.dim,4*self.dim,kernel_size,stride=2,padding=2)
        self.bn2 = nn.BatchNorm2d(4*self.dim)
        #LeakyRelu
        #(256,32,32)
        self.conv4 = nn.Conv2d(4*self.dim,8*self.dim,kernel_size,stride=2,padding=2)
        self.bn3 = nn.BatchNorm2d(8*self.dim)
        #LeakyRelu
        #(512,16,16)
        self.conv5 = nn.Conv2d(8*self.dim,8*self.dim,kernel_size,stride=2,padding=2)
        self.bn4 = nn.BatchNorm2d(8*self.dim)
        #LeakyRelu
        #(512,8,8)
        self.fc = nn.Linear(8*10*8*self.dim,1)
        
    def forward(self, x):
        batch_size, channel, height, width = x.shape
        output = self.conv1(x)
        output = F.leaky_relu(output)
        output = self.conv2(output)
        output = self.bn1(output)
        output = F.leaky_relu(output)
        output = self.conv3(output)
        output = self.bn2(output)
        output = F.leaky_relu(output)
        output = self.conv4(output)
        output = self.bn3(output)
        output = F.leaky_relu(output)
        output = self.conv5(output)
        output = self.bn4(output)
        output = F.leaky_relu(output)
        
        output = output.view(-1, 8*10*8*self.dim)
        output = self.fc(output)
        
        return output