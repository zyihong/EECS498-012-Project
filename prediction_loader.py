import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.misc import imread, imresize
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import csv
import json
from PIL import Image
import tarfile
import glob
import h5py
import cv2
from readvideo import parse_seg,readvideo
import math
from torchvision import transforms



VIDEO_DATASET = "videos/reshaped.hdf5"
SEG_PATH = "segmentation.txt"
KEYPOINTS = "keypoints.h5"

'''
VIDEO_DATASET = "videos/reshaped.hdf5"
SEG_PATH = "natops/data/segmentation.txt"
KEYPOINTS = "keypoints.h5"
'''
FOLDER_DIR = '/home/emily/SURREAL/git_copy/surreal/video_prediction/'


class NATOPSData(Dataset):
    #__depth = []
    #__flow = []
    #__segm = []
    #__normal = []
    #__video = []
    #__keypoint = []
    #__label = []
    #__appearance = []

    def __init__(self, video_dataset,seg_path,keypoints,data_len=30,transform=None):
        self.transform = transform
        # Open and load text file including the whole training data
        self.seg_list = parse_seg(seg_path)
        self.keypoints = pd.HDFStore(keypoints)
        #self.motion_video_path = video_dataset
        #self.motion_file = h5py.File(video_dataset,'r+')    
        #file handle for video dataset
        self.data_len = data_len
        self.f = h5py.File(video_dataset,'r')
        self.to_tensor = transforms.ToTensor()

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        #print('load file')
        '''
        index: 0 - 9599
        subject_no: 0 - 19
        gesture_no: 0 - 23
        numOfRepeat: 0 - 19
        '''
        #subject_idx = int(index/(24*20))
        #gesture_idx = int(index/20%24)
        '''
        numOfRepeat = int(index/(24*20))
        subject_idx = int(index/24%20)
        gesture_idx = int(index%24)
        '''

        gesture_idx = int(index/(24*20))
        subject_idx = int(index/20%20)
        numOfRepeat = int(index%20)


        motion = np.zeros((self.data_len,64,64,3),dtype=np.uint8)
        keypoint = np.zeros((self.data_len,18))
        
        motion_temp,keypoint_temp = readvideo(self.f,subject_idx, gesture_idx, numOfRepeat,self.seg_list,self.keypoints)
        length = min(len(keypoint_temp),len(motion_temp))
        center_x = keypoint_temp[:,2]
        center_x = center_x.reshape((-1,1))
        center_y = keypoint_temp[:,3]
        center_y = center_y.reshape((-1,1))

        keypoint_temp[:,::2] = (keypoint_temp[:,::2] - center_x + 90) * 64/180
        keypoint_temp[:,1:18:2] = (keypoint_temp[:,1:18:2] - center_y + 90) * 64/180

        motion[:min(self.data_len,length)] = motion_temp[:min(self.data_len,length)]
        keypoint[:min(self.data_len,length)] = keypoint_temp[:min(self.data_len,length)]
    
        #print("problem with %d motion data"%index)

        # Convert image and label to torch tensors
        motion_tensor = torch.zeros((self.data_len,3,64,64))
        for t in range(self.data_len):
            motion_tensor[t,:,:,:] = self.to_tensor(motion[t,:,:,:])
        #appearance is first frame of motion
        appearance_tensor = None
        appearance = motion[0]
        appearance_tensor = self.to_tensor(appearance)

        keypoint = torch.from_numpy(np.asarray(keypoint))

        l = np.zeros(24)
        l[gesture_idx] = 1
        l = torch.from_numpy(l)

        return motion_tensor, keypoint,appearance_tensor,l

    # Override to give PyTorch size of dataset
    def __len__(self):
        return 7680

'''
def main():
    batch_size = 2
    dset_train = NATOPSData(FOLDER_DIR+VIDEO_DATASET,FOLDER_DIR+SEG_PATH,FOLDER_DIR+KEYPOINTS)
    train_loader = DataLoader(dset_train, batch_size, shuffle=True, num_workers=1)
    motion_tensor, keypoint,appearance_tensor,last_frame_tensor = next(iter(train_loader))
    

    #shuffle y_a to get y_a',y_m
    #frames = int(motion.numpy().shape[1]/10)  
    image_first = appearance_tensor.permute(0,2,3,1).numpy()[0,:,:,:]
    
    plt.imshow(image_first)
    joints_loc = keypoint.numpy()[0,0,:]
    plt.plot(joints_loc[::2], joints_loc[1:18:2], 'r+')
    plt.savefig('images/image_first.png')
    plt.close()
    image_last = last_frame_tensor.permute(0,2,3,1).numpy()[0,:,:,:]

    plt.imshow(image_last)
    joints_loc = keypoint.numpy()[0,-1,:]
    plt.plot(joints_loc[::2], joints_loc[1:18:2], 'r+')
    plt.savefig('images/image_last.png')
    plt.close()

    for i in range(30):
        

        # plt.show()
        #plt.savefig('render/image%d.png'%i)
        #print(image)
        #plt.close()
        joints_loc = keypoint.numpy()[0,i,:]
        print('joints shape',joints_loc.shape)
        plt.plot(joints_loc[::2], joints_loc[1:18:2], 'go', linewidth=2, markersize=12)
        plt.savefig('images/keypoints%d.png'%i)
        #plt.savefig('image%d.png'%f)
        plt.waitforbuttonpress()
        #plt.savefig('render/segm%d.png'%i)
        plt.close()
if __name__ == '__main__':
    main()
'''
