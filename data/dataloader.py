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


FOLDER_DATASET = "/data/out"


class FacadeDataset(Dataset):
    def __init__(self, dataset, flag, dataDir='', data_range=(0, 8), n_class=5, onehot=False):
        self.onehot = onehot
        # assert(flag in ['train', 'eval', 'test', 'test_dev', 'kaggle'])
        print("load " + flag + " dataset start")
        # print("    from: %s" % dataDir)
        print("    range: [%d, %d)" % (data_range[0], data_range[1]))
        imgs = dataset[0]
        segm = dataset[1]
        self.dataset = []
        base = imgs[0, 0]
        base = base.to(torch.float32)
        base_img = base.permute(2,0,1)
        for i in range(data_range[0], data_range[1]):
            target_seg = segm[0, i]
            target_seg = target_seg.to(torch.float32)
            
            #target_seg = target_seg.view(1,target_seg.size()[0],-1)
            
            #target_seg = target_seg.permute(2,0,1)
            zeros = torch.zeros(target_seg.shape)
            ones = torch.ones(target_seg.shape)
            mask = torch.where(target_seg > 0, ones, zeros)
            mask.unsqueeze_(2)
            target_seg.unsqueeze_(2)
            target_seg = target_seg.permute(2,0,1)
            #print('target_seg size',target_seg.size())
            #img = torch.cat((base, target_seg*100.0), 2)
            #img=img.permute(2,0,1)
            label = imgs[0, i]
            label=label.permute(2,0,1)
            mask = torch.cat((mask, mask, mask), 2)
            #print('mask size',mask.size())
            mask = mask.permute(2,0,1)

            base_seg = segm[0, 0]
            base_seg = base_seg.to(torch.float32)
            zeros = torch.zeros(base_seg.shape)
            ones = torch.ones(base_seg.shape)
            base_mask = torch.where(base_seg > 0, ones, zeros)
            base_mask.unsqueeze_(2)
            base_mask = torch.cat((base_mask, base_mask, base_mask), 2)
            base_mask = base_mask.permute(2, 0, 1)
            # img = Image.open(os.path.join(dataDir,flag,'eecs442_%04d.jpg' % i))

            # pngreader = png.Reader(filename=os.path.join(dataDir,flag,'eecs442_%04d.png' % i))
            # w,h,row,info = pngreader.read()
            # label = np.array(list(row)).astype('uint8')

            # Normalize input image
            # img = np.asarray(img).astype("f").transpose(2, 0, 1)/128.0-1.0
            # Convert to n_class-dimensional onehot matrix
            # label_ = np.asarray(label)
            # label = np.zeros((n_class, img.shape[1], img.shape[2])).astype("i")
            # for j in range(n_class):
            #     label[j, :] = label_ == j
            self.dataset.append((base_img,target_seg,label, mask, base_mask))
        print("load dataset done")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        base_img,img, label, mask, base_mask = self.dataset[index]
        # label = torch.FloatTensor(label)
        # if not self.onehot:
        #     label = torch.argmax(label, dim=0)
        # else:
        #     label = label.long()

        return base_img.to(dtype=torch.float32),img.to(dtype=torch.float32), label.to(dtype=torch.float32), mask.to(dtype=torch.float32), base_mask.to(dtype=torch.float32)


class MotionData(Dataset):
    #__depth = []
    #__flow = []
    #__segm = []
    #__normal = []
    __data = []
    __annotation = []
    __img = []
    __keypoint = []

    def __init__(self, folder_dataset, transform=None):
        self.transform = transform
        # Open and load text file including the whole training data
        with open('data/files.csv', mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                # depth path
                #self.__depth.append(row['depth'])        
                # flow path
                #self.__flow.append(row['flow'])
                # segm path
                #self.__segm.append(row['segm'])        
                # normal path
                #self.__normal.append(row['normal'])
                self.__data.append(row['data'])
                # annotation path
                self.__annotation.append(row['annotation'])
                #image path
                self.__img.append(row['img'])
                self.__keypoint.append(row['keypoint'])

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        print('load file')
        h5f = h5py.File(self.__data[index],'r')
        print('finish loading')
        data_len = 130
        depth= np.zeros((data_len,240,320))
        try:
            print('load depth')
            depth_temp = h5f['depth'][:] 
            print('finish loading')
            length = len(depth_temp)
            #if(length <= data_len):
            #    depth = np.pad(depth,((0,data_len-length),(0,0),(0,0)),'constant')
            depth[:min(data_len,length)] = depth_temp[:min(data_len,length)]
        except:
            print("problem with %d depth data"%index)
              
        flow = np.zeros((data_len,240,320,2))
        try:
            flow_temp = h5f['gtflow'][:] 
            length = len(flow_temp)
            #if(length <= data_len):
            #    flow = np.pad(flow,((0,data_len-length),(0,0),(0,0)),'constant')
            flow[:min(data_len,length)] = flow_temp[:min(data_len,length)]
        except:
            print("problem with %d flow data"%index)
        segm = np.zeros((data_len,240,320))
        try:
            segm_temp = h5f['segm'][:] 
            length = len(segm_temp)
            #if(length <= data_len):
            #    segm = np.pad(segm,((0,data_len-length),(0,0),(0,0)),'constant')
            segm[:min(data_len,length)] = segm_temp[:min(data_len,length)]
        except:
            print("problem with %d segm data"%index)
        normal = np.zeros((data_len,240,320,3))
        try:
            normal_temp = h5f['normal'][:] 
            length = len(normal_temp)
            #if(length <= data_len):
            #    normal = np.pad(normal,((0,data_len-length),(0,0),(0,0)),'constant')
            normal[:min(data_len,length)] = normal_temp[:min(data_len,length)]
        except:
            print("problem with %d normal data"%index)
        #annotation
        try:
            with open(self.__annotation[index]) as f:
                    annotation = json.load(f)
        except:
            print("problem with annotation")
        #images
        img = np.zeros((data_len,240,320,3),dtype=int)
        try:
            
            #dimension (H,W,C)
            
            tarfile.open(self.__img[index]).extractall(self.__img[index].split(".")[0])
            
            length = len(glob.glob("%s/*/*" %(self.__img[index].split(".")[0])))
            print('image',length)
            for i in range(min(data_len,length)):
                path = glob.glob("%s/*/Image%04d.png" %(self.__img[index].split(".")[0],i))
                # print('path',path)
                with Image.open(path[0]) as img_temp:
                    img_temp = img_temp.convert('RGB')
                    img_temp = np.asarray(img_temp)
                    img_temp = img_temp.astype(int)
                    img[i] = img_temp
                #img[i] = torch.from_numpy(img_temp,dtype=torch.int)
        except:
            print("problem with %d image"%index)        

        #keypoint
        h5f_keypoint = h5py.File(self.__keypoint[index],'r')

        keypoint = np.zeros((2,43,data_len))
        try:
            keypoint_temp = h5f_keypoint['joints2D'][:]
            print('keypoint shape',keypoint_temp.shape)
            length = keypoint_temp.shape[2]
            #if(length <= data_len):
            #    normal = np.pad(normal,((0,data_len-length),(0,0),(0,0)),'constant')
            keypoint[:,:,:min(data_len,length)] = keypoint_temp[:,:,:min(data_len,length)]
        except:
            print("problem with %d keypoint data"%index)
   
        # Convert image and label to torch tensors
        depth = torch.from_numpy(np.asarray(depth))
        
        
        flow = torch.from_numpy(np.asarray(flow))
        
        segm = torch.from_numpy(np.asarray(segm))
        
        normal = torch.from_numpy(np.asarray(normal))
        
        img = torch.from_numpy(np.asarray(img))

        keypoint = torch.from_numpy(np.asarray(keypoint))
        
        return depth,flow,segm,normal,annotation,img,keypoint

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__data)


def load_data():
    dset_train = MotionData(FOLDER_DATASET)
    train_loader = DataLoader(dset_train, batch_size=1, shuffle=True, num_workers=0)
    depth,flow,segm,normal,annotation,img,keypoint = next(iter(train_loader))
    print("keypoint shape",keypoint.shape)
    print('Batch shape:',depth.numpy().shape, flow.numpy().shape,img.numpy().shape)
    # frames = int(img.numpy().shape[1]/10)
    # for i in range(frames):
    #     idx = i * 10
    #     image = img.numpy()[0,idx,:,:,:]
    #
    #     ax0 = plt.subplot(231)
    #     ax0.imshow(image)
    #     # plt.show()
    #     #plt.savefig('render/image%d.png'%i)
    #     #print(image)
    #     #plt.close()
    #     joints_loc = keypoint.numpy()[0,:,:,idx]
    #     print('joints shape',joints_loc.shape)
    #     ax0.plot(joints_loc[0,:], 240-joints_loc[1,:], 'r+')
    #     #plt.savefig('image%d.png'%f)
    #
    #
    #
    #     ax1 = plt.subplot(232)
    #     ax1.imshow(depth.numpy()[0,idx,:,:])
    #     #plt.show()
    #     #plt.waitforbuttonpress()
    #     #plt.savefig('render/depth%d.png'%i)
    #     ax2 = plt.subplot(233)
    #     ax2.imshow(segm.numpy()[0,idx,:,:])
    #     #plt.show()
    #     ax3 = plt.subplot(234)
    #     ax3.imshow(flow.numpy()[0,idx,:,:,0])
    #     ax4 = plt.subplot(235)
    #     ax4.imshow(normal.numpy()[0,idx,:,:,:])
    #     # plt.waitforbuttonpress()
    #     #plt.savefig('render/segm%d.png'%i)
    #     plt.close()

    return depth,flow,segm,normal,annotation,img,keypoint

