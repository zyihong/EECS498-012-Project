import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from predict_gan import Generator2,Appearance_D,Generator1,Motion_D
from torch.utils.data import DataLoader
from prediction_loader import NATOPSData
from tensorboardX import SummaryWriter
import time
import os
from torchvision.utils import save_image,make_grid

MODEL_DIR = 'models-8/'
TRAIN_DIR = '/home/emily/SURREAL/git_copy/surreal/video_prediction/'
SAVE_STEP = 500
GENERATOR_ONE_PATH = './models-5/generator_one-10-1000.ckpt'
GENERATOR_TWO_PATH = './models-5/generator_two-10-1000.ckpt'
DISCRIMINATOR_A_PATH = './models-5/discriminator-10-1000.ckpt'
DISCRIMINATOR_M_PATH = './models-5/discriminator-10-1000.ckpt'
LOAD_FROM_CHECKPOINT = False
IMAGE_PATH = "save_images/v8/"
TRAIN_OR_EVAL = 'T'

def train_step(step,motion_video, keypoints,base_img,y_l, G, D_a,D_m, L1_criterion,
    BCE_criterion, G_solver,D_a_solver,D_m_solver, device, epoch,batch_size=4,q=18,p=128,T=10):
    '''
    Function for training.
    '''
    start = time.time()
    running_loss = 0.0
    
    base_img = base_img.to(device)
    keypoints = keypoints.type(torch.FloatTensor)
    keypoints = keypoints.to(device)
    
    motion_video = motion_video.to(device)
    y_l = y_l.type(torch.FloatTensor)
    y_l =y_l.to(device)

    G_solver.zero_grad()
    D_a_solver.zero_grad()
    D_m_solver.zero_grad()

    #prev_keypoints = keypoints[:,:T]
    #prev_keypoints = prev_keypoints.type(torch.FloatTensor)
    #post_keypoints = keypoints[:,T*2:T*3]
    #post_keypoints = post_keypoints.type(torch.FloatTensor)


    z = torch.rand((batch_size,p)).to(device)

    G1 = G(base_img,keypoints,z)

    #g1_loss = L1_criterion(G1, motion_video)
    #g1_loss.backward(retain_graph=True)
    #gen_train_op1.step()

    #Generator 2
    #DiffMap = generator_two(G1,base_img)
    #G2 = G1 + DiffMap
    
    g1_loss,g2_loss,d_a_loss,d_m_loss = 0,0,0,0

    s_r_a,r_l1_a,r_l2_a = D_a(motion_video,base_img)
    s_f_a,f_l1_a,f_l2_a = D_a(G1,base_img)
    d_a_loss += BCE_criterion(s_r_a, torch.ones((batch_size,1)).cuda())
    d_a_loss += BCE_criterion(s_f_a, torch.zeros((batch_size,1)).cuda())

    s_r_m,l_r,r_l1_m,r_l2_m = D_m(motion_video,base_img,y_l)
    s_f_m,l_f,f_l1_m,f_l2_m = D_m(G1,base_img,y_l)

    d_m_loss += BCE_criterion(s_r_m, torch.ones((batch_size,1)).cuda())
    d_m_loss += BCE_criterion(s_f_m, torch.zeros((batch_size,1)).cuda())

    g_cross_entropy_loss = BCE_criterion(s_f_a, torch.ones((batch_size,1)).cuda())
    g_cross_entropy_loss += BCE_criterion(s_f_m, torch.ones((batch_size,1)).cuda())

    g1_loss += g_cross_entropy_loss
    PoseMaskLoss1 = L1_criterion(G1,motion_video)
    g1_loss += 10*PoseMaskLoss1




    g1_loss.backward(retain_graph=True)
    G_solver.step()

    d_a_loss.backward(retain_graph=True)
    D_a_solver.step()

    d_m_loss.backward()
    D_m_solver.step()


    g_running_loss = g1_loss.item()
    d_a_running_loss = d_a_loss.item()
    d_m_running_loss = d_m_loss.item()

    if (step + 1) % SAVE_STEP == 0:
        torch.save(G.state_dict(), os.path.join(
            MODEL_DIR, 'G-{}-{}.ckpt'.format(epoch + 1, step + 1)))
        torch.save(D_a.state_dict(), os.path.join(
            MODEL_DIR, 'D_a-{}-{}.ckpt'.format(epoch + 1, step + 1)))
        torch.save(D_m.state_dict(), os.path.join(
            MODEL_DIR, 'D_m-{}-{}.ckpt'.format(epoch + 1, step + 1)))


    end = time.time()
    print('[epoch %d] g1_loss: %.3f g2_loss: %.3f d_loss: %.3f elapsed time %.3f' %
        (epoch, g_running_loss,d_a_running_loss,d_m_running_loss, end-start))
    return g_running_loss,d_a_running_loss,d_m_running_loss,G1,g_cross_entropy_loss,PoseMaskLoss1


def train(device):
    N =4
    T = 10
    q =18
    p=128
    c = 24

    dset_train = NATOPSData("videos/reshaped.hdf5","natops/data/segmentation.txt","keypoints.h5")
    train_loader = DataLoader(dset_train, batch_size=4, shuffle=True, num_workers=1)
    
    #models
    G = Generator1(N,q,p).to(device)
    D_a = Appearance_D().to(device)
    D_m = Motion_D(q,c).to(device)

    #loss functions
    #criterion = nn.L1Loss()  # TODO decide loss
    L1_criterion = nn.L1Loss()
    BCE_criterion = nn.BCELoss()
    #optimizer = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=0)

    #optimizers
    G_solver = optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))
    D_m_solver = optim.Adam(D_m.parameters(), lr=1e-3,betas=(0.5, 0.999))
    D_a_solver = optim.Adam(D_a.parameters(), lr=1e-3,betas=(0.5, 0.999))

    if LOAD_FROM_CHECKPOINT:
        G.load_state_dict(torch.load(GENERATOR_ONE_PATH))
        D_a.load_state_dict(torch.load(DISCRIMINATOR_A_PATH))
        D_m.load_state_dict(torch.load(DISCRIMINATOR_M_PATH))
    
    writer = SummaryWriter('plots/exps-8')
    iteration = 0
    if TRAIN_OR_EVAL == 'T':
        print('\nStart training generator1')
        for epoch in range(50):  # TODO decide epochs
            print('-----------------Epoch = %d-----------------' % (epoch + 1))
            G = G.train()
            D_a = D_a.train()
            D_m = D_m.train()
            for step, (motion_video, keypoints,base_img,y_l) in enumerate(tqdm(train_loader)):
                g_running_loss,d_a_running_loss,d_m_running_loss,G1_out,g_cross_entropy_loss,PoseMaskLoss1 = train_step(step,motion_video, keypoints,base_img,
                    y_l, G, D_a,D_m, L1_criterion, BCE_criterion, G_solver,D_a_solver,D_m_solver, device, epoch,N)
                
                if iteration%100 == 0:
                    writer.add_scalar('G1_loss', g_running_loss, iteration)
                    writer.add_scalar('G1_loss_l1', PoseMaskLoss1*10, iteration)
                    writer.add_scalar('G1_loss_cross_entropy',g_cross_entropy_loss , iteration)
                    writer.add_scalar('D_a_loss', d_a_running_loss, iteration)
                    writer.add_scalar('D_m_loss', d_m_running_loss, iteration)
                    writer.add_image('G1 image', G1_out[0,15,:,:,:], iteration)
                    print('G1_size', G1_out.size())
                    writer.add_video("G1 single video", G1_out[0,:,:,:,:].view(1,30,3,64,64),iteration)
                    writer.add_video("G1_1 video", G1_out, iteration)

                    writer.add_video("label video", motion_video, iteration)
                iteration += 1
            if epoch%10 == 0:
                save_image(make_grid(G1_out[0,:,:,:,:],nrow=10),IMAGE_PATH + "G2.png",nrow=1)
                save_image(make_grid(motion_video[0,:,:,:,:],nrow=10),IMAGE_PATH + "labels.png",nrow=1)
    else:
        motion_video, keypoints,base_img,y_l = next(iter(train_loader))
        evaluate(motion_video, keypoints,base_img,y_l, generator_one,generator_two,device)

        

def evaluate(motion_video, keypoints,base_img,y_l, generator_one,generator_two,device,batch_size=4,q=18,p=128,T=10):
    base_img = base_img.to(device)
    keypoints = keypoints.type(torch.FloatTensor)
    keypoints = keypoints.to(device)
    motion_video = motion_video.to(device)
    y_l =y_l.to(device)


    z = torch.rand((batch_size,p)).to(device)

    G1 = generator_one(base_img,keypoints,z)

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
    
    for b in range(batch_size):
        batch_img = G1[b,:,:,:,:]
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
