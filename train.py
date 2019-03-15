import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

import math
from collections import OrderedDict


class Colorization(nn.Module):
    # You will implement a simple version of vgg11 (https://arxiv.org/pdf/1409.1556.pdf)
    # Since the shape of image in CIFAR10 is 32x32x3, much smaller than 224x224x3,
    # the number of channels and hidden units are decreased compared to the architecture in paper
    def __init__(self):
        super(Colorization, self).__init__()

        self.conv1=nn.Conv2d(1, 64, kernel_size=(3,3),padding=(1,1),stride=2)
        self.relu=nn.ReLU()
        self.batch1=nn.BatchNorm2d(64)

        self.conv2=nn.Conv2d(64, 128, kernel_size=(3,3),padding=(1,1),stride=2)
        self.batch2=nn.BatchNorm2d(128)

        self.conv3=nn.Conv2d(128, 256, kernel_size=(3,3),padding=(1,1),stride=2)
        self.batch3=nn.BatchNorm2d(256)

        self.conv4=nn.Conv2d(256,512,3,padding=(1,1),stride=1)
        self.batch4=nn.BatchNorm2d(512)

        self.conv5=nn.Conv2d(512, 512, 3, padding=(2, 2), stride=1,dilation=(2,2))
        self.batch5=nn.BatchNorm2d(512)

        self.conv6=nn.Conv2d(512, 512, 3, padding=(1, 1), stride=1)
        self.batch6=nn.BatchNorm2d(512)

        self.dconv1=nn.ConvTranspose2d(512,256,stride=2,kernel_size=4,padding=1)
        self.dconv2=nn.ConvTranspose2d(256,2,stride=4,kernel_size=6,padding=1)




    def forward(self, x):
        y=x.clone()
        x=self.conv1(x)
        x=self.relu(x)
        x=self.batch1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.batch2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.batch3(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.batch4(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.batch5(x)

        x = self.conv6(x)
        x = self.relu(x)
        x = self.batch6(x)

        x=self.dconv1(x)
        x=self.dconv2(x)
        x = torch.cat((y,x),dim=1)
        return x


def train(trains,label, net, criterion, optimizer, device):
    for epoch in range(5):  # loop over the dataset multiple times
        optimizer.zero_grad()

        outputs = net(trains)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        # print statistics
        print(loss.item())

    picture1 = cv2.cvtColor(outputs[0].cpu().data.numpy().transpose(1, 2, 0).astype('uint8'), cv2.COLOR_Lab2RGB)
    plt.imshow(picture1)
    plt.show()
    picture2 = cv2.cvtColor(label[0].cpu().data.numpy().transpose(1, 2, 0).astype('uint8'), cv2.COLOR_Lab2RGB)
    plt.imshow(picture2)
    plt.show()


def test(testloader, net, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # images = F.pad(images, (64, 64, 64, 64), 'constant', 0)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

def mse_loss(input,target):
    return torch.sum((input[:,1:]-target[:,1:])**2)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    root='./data/images/Train/'
    trainimages=os.listdir(root)
    batch_size=10
    X = []
    Y = []
    net = Colorization().to(device)
    criterion = mse_loss
    optimizer = optim.Adam(net.parameters(), lr=0.05)
    for i,image in enumerate(trainimages):
        pic=cv2.imread(root+image)
        # graypic=cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
        pic=cv2.cvtColor(pic,cv2.COLOR_BGR2Lab)
        # plt.imshow(cv2.cvtColor(pic,cv2.COLOR_Lab2RGB))
        # plt.show()
        X.append(pic[:,:,0].reshape((1,256,256)))
        Y.append(pic)
        if (i+1)%batch_size==0:
            print(i+1)
            X=torch.tensor(X).to(device=device,dtype=torch.float)
            Y=torch.tensor(Y).to(device=device,dtype=torch.float)
            # plt.imshow(cv2.cvtColor(Y[0].cpu().data.numpy().astype('uint8'),cv2.COLOR_Lab2RGB))
            # plt.show()
            Y=Y.permute(0,3,1,2)
            train(X,Y,net,criterion,optimizer,device)
            X=[]
            Y=[]





if __name__ == "__main__":
    main()

