import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from predict_gan import Generator2,Discriminator,Generator1
from torch.utils.data import DataLoader
from prediction_loader import NATOPSData
from tensorboardX import SummaryWriter
import time
import os
from torchvision.utils import save_image,make_grid

MODEL_DIR = 'models-2/'
TRAIN_DIR = '/home/emily/SURREAL/git_copy/surreal/video_prediction/'
SAVE_STEP = 500
GENERATOR_ONE_PATH = './models-2/generator_one-41-500.ckpt'
GENERATOR_TWO_PATH = './models-2/generator_two-41-500.ckpt'
DISCRIMINATOR_PATH = './models-2/discriminator-41-500.ckpt'
LOAD_FROM_CHECKPOINT = True
IMAGE_PATH = "save_images/"
TRAIN_OR_EVAL = 'E'

def train_step(step,motion_video, keypoints,base_img,last_frame, generator_one, generator_two, discriminator, L1_criterion,
    BCE_criterion, gen_train_op1, gen_train_op2, dis_train_op1, device, epoch,batch_size=10,q=18,p=128,T=10):
    '''
    Function for training.
    '''
    start = time.time()
    running_loss = 0.0
    
    base_img = base_img.to(device)
    keypoints = keypoints.type(torch.FloatTensor)
    keypoints = keypoints.to(device)
    motion_video = motion_video.to(device)
    last_frame = last_frame.to(device)

    gen_train_op1.zero_grad()
    gen_train_op2.zero_grad()
    dis_train_op1.zero_grad()

    prev_keypoints = keypoints[:,:T]
    #prev_keypoints = prev_keypoints.type(torch.FloatTensor)
    post_keypoints = keypoints[:,T*2:T*3]
    #post_keypoints = post_keypoints.type(torch.FloatTensor)


    z = torch.rand((batch_size,p)).to(device)

    G1 = generator_one(base_img,last_frame,prev_keypoints,post_keypoints,z)

    g1_loss = L1_criterion(G1, motion_video)
    g1_loss.backward(retain_graph=True)
    gen_train_op1.step()

    #Generator 2
    DiffMap = generator_two(G1,base_img)
    G2 = G1 + DiffMap
    
    g2_loss,d_loss = 0,0

    #Discriminator
    for t in range(T*3):
        triplet = torch.cat([motion_video[:,t,:,:,:], G2[:,t,:,:,:], base_img], dim=0)
        #print("triplet",triplet.size())
        D_z = discriminator(triplet)
        D_z = torch.clamp(D_z, 0.0, 1.0)
        # print('DZ',D_z.size())
        D_z_pos_x_target, D_z_neg_g2, D_z_neg_x = torch.split(D_z,batch_size) #batch size
        D_z_pos = D_z_pos_x_target
        D_z_neg = torch.cat([D_z_neg_g2, D_z_neg_x], 0)
        
        #Generator 2 loss
        g2_loss += BCE_criterion(D_z_neg, torch.ones((2*batch_size,1)).cuda())
        #g2_loss = BCE_criterion(D_z_neg, torch.ones((10))) #2*batch size
        PoseMaskLoss2 = L1_criterion(G2[:,t,:,:,:],motion_video[:,t,:,:,:])
        g2_loss += 50*PoseMaskLoss2

        #discriminator loss
        d_loss += BCE_criterion(D_z_pos, torch.ones((batch_size,1)).cuda())
        d_loss += BCE_criterion(D_z_neg, torch.zeros((batch_size*2,1)).cuda())
        #d_loss = BCE_criterion(D_z_pos, torch.ones((5)))
        #d_loss += BCE_criterion(D_z_neg, torch.zeros((10)))
        d_loss /= 2

    g2_loss.backward(retain_graph=True)
    gen_train_op2.step()

    d_loss.backward()
    dis_train_op1.step()

    g1_running_loss = g1_loss.item()
    g2_running_loss = g2_loss.item()
    d_running_loss = d_loss.item()

    if (step + 1) % SAVE_STEP == 0:
        torch.save(generator_one.state_dict(), os.path.join(
            MODEL_DIR, 'generator_one-{}-{}.ckpt'.format(epoch + 1, step + 1)))
        torch.save(generator_two.state_dict(), os.path.join(
            MODEL_DIR, 'generator_two-{}-{}.ckpt'.format(epoch + 1, step + 1)))
        torch.save(discriminator.state_dict(), os.path.join(
            MODEL_DIR, 'discriminator-{}-{}.ckpt'.format(epoch + 1, step + 1)))


    end = time.time()
    print('[epoch %d] g1_loss: %.3f g2_loss: %.3f d_loss: %.3f elapsed time %.3f' %
        (epoch, g1_running_loss,g2_running_loss,d_running_loss, end-start))
    return g1_running_loss,g2_running_loss,d_running_loss,G2,G1


def train(device):
    N =10
    T = 10
    q =18
    p=128

    dset_train = NATOPSData("videos/reshaped.hdf5","natops/data/segmentation.txt","keypoints.h5")
    train_loader = DataLoader(dset_train, batch_size=10, shuffle=True, num_workers=1)
    
    #models
    generator_one = Generator1(N,q,p).to(device)
    generator_two = Generator2().to(device)
    discriminator = Discriminator().to(device)

    #loss functions
    #criterion = nn.L1Loss()  # TODO decide loss
    L1_criterion = nn.L1Loss()
    BCE_criterion = nn.BCELoss()
    #optimizer = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=0)

    #optimizers
    gen_train_op1 = optim.Adam(generator_one.parameters(), lr=1e-3, betas=(0.5, 0.999))
    gen_train_op2 = optim.Adam(generator_two.parameters(), lr=1e-3, betas=(0.5, 0.999))
    dis_train_op1 = optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999))

    if LOAD_FROM_CHECKPOINT:
        generator_one.load_state_dict(torch.load(GENERATOR_ONE_PATH))
        generator_two.load_state_dict(torch.load(GENERATOR_TWO_PATH))
        discriminator.load_state_dict(torch.load(DISCRIMINATOR_PATH))
    
    writer = SummaryWriter('plots/exps-2')
    iteration = 0
    if TRAIN_OR_EVAL == 'T':
        print('\nStart training generator1')
        for epoch in range(50):  # TODO decide epochs
            print('-----------------Epoch = %d-----------------' % (epoch + 1))
            generator_one = generator_one.train()
            generator_two = generator_two.train()
            discriminator = discriminator.train()
            for step, (motion_video, keypoints,base_img,last_frame) in enumerate(tqdm(train_loader)):
                g1_running_loss,g2_running_loss,d_running_loss,G2_out,G1_out = train_step(step,motion_video, keypoints,base_img,
                    last_frame, generator_one,generator_two,discriminator, L1_criterion, BCE_criterion, 
                        gen_train_op1,gen_train_op2,dis_train_op1, device, epoch + 1)
                writer.add_scalar('G1_loss', g1_running_loss, iteration)
                writer.add_scalar('G2_loss', g2_running_loss, iteration)
                writer.add_scalar('D_loss', d_running_loss, iteration)
                writer.add_image('G2_5', G2_out[0,5,:,:,:], iteration)
                writer.add_image('G2_10', G2_out[0,10,:,:,:], iteration)
                writer.add_image('G2_15', G2_out[0,15,:,:,:], iteration)
                writer.add_image('label_15', motion_video[0,15,:,:,:], iteration)
                writer.add_image('G2_17', G2_out[0,17,:,:,:], iteration)
                writer.add_image('G2_25', G2_out[0,25,:,:,:], iteration)
                writer.add_image('G1_15', G1_out[0,15,:,:,:], iteration)
                iteration += 1
            if epoch%10 == 0:
                save_image(make_grid(G2_out[0,:,:,:,:],nrow=10),IMAGE_PATH + "G2.png",nrow=1)
                save_image(make_grid(motion_video[0,:,:,:,:],nrow=10),IMAGE_PATH + "labels.png",nrow=1)
    else:
        motion_video, keypoints,base_img,last_frame = next(iter(train_loader))
        evaluate(motion_video, keypoints,base_img,last_frame, generator_one,generator_two,device)

        

def evaluate(motion_video, keypoints,base_img,last_frame, generator_one,generator_two,device,batch_size=10,q=18,p=128,T=10):
    base_img = base_img.to(device)
    keypoints = keypoints.type(torch.FloatTensor)
    keypoints = keypoints.to(device)
    motion_video = motion_video.to(device)
    last_frame = last_frame.to(device)

    prev_keypoints = keypoints[:,:T]
    post_keypoints = keypoints[:,T*2:T*3]


    z = torch.rand((batch_size,p)).to(device)

    G1 = generator_one(base_img,last_frame,prev_keypoints,post_keypoints,z)

    DiffMap = generator_two(G1,base_img)
    G2 = G1 + DiffMap
    '''
    save_image(make_grid(G2[0,:,:,:,:],nrow=10),IMAGE_PATH + "G2_0.png",nrow=1)
    save_image(make_grid(motion_video[0,:,:,:,:],nrow=10),IMAGE_PATH + "labels_0.png",nrow=1)
    save_image(make_grid(G2[1,:,:,:,:],nrow=10),IMAGE_PATH + "G2_1.png",nrow=1)
    save_image(make_grid(motion_video[1,:,:,:,:],nrow=10),IMAGE_PATH + "labels_1.png",nrow=1)
    save_image(make_grid(G2[2,:,:,:,:],nrow=10),IMAGE_PATH + "G2_2.png",nrow=1)
    save_image(make_grid(motion_video[2,:,:,:,:],nrow=10),IMAGE_PATH + "labels_2.png",nrow=1)
    save_image(make_grid(G2[3,:,:,:,:],nrow=10),IMAGE_PATH + "G2_3.png",nrow=1)
    save_image(make_grid(motion_video[3,:,:,:,:],nrow=10),IMAGE_PATH + "labels_3.png",nrow=1)
    save_image(make_grid(G2[4,:,:,:,:],nrow=10),IMAGE_PATH + "G2_4.png",nrow=1)
    save_image(make_grid(motion_video[4,:,:,:,:],nrow=10),IMAGE_PATH + "labels_4.png",nrow=1)
    save_image(make_grid(G2[5,:,:,:,:],nrow=10),IMAGE_PATH + "G2_5.png",nrow=1)
    save_image(make_grid(motion_video[5,:,:,:,:],nrow=10),IMAGE_PATH + "labels_5.png",nrow=1)
    '''
    
    for b in range(10):
        batch_img = G2[b,:,:,:,:]
        batch_label = motion_video[b,:,:,:,:]
        for t in range(3*T):
            save_image(batch_img[t,:,:,:],IMAGE_PATH +"%d/"%b+"G2_%d.png"%t,nrow=1)
            save_image(batch_label[t,:,:,:],IMAGE_PATH +"%d_label/"%b+"label_%d.png"%t,nrow=1)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train(device)
    print('test')


if __name__ == "__main__":
    main()
