from data.dataloader import load_data, FacadeDataset
from models import *
from matplotlib import pyplot as plt
import numpy as np


def train(trainloader, generator_one,generator_two,discriminator, L1_criterion, 
    BCE_criterion,gen_train_op1,gen_train_op2,dis_train_op1, device, epoch):
    '''
    Function for training.
    '''
    start = time.time()
    running_loss = 0.0
    generator_one = generator_one.train()
    generator_two = generator_two.train()
    discriminator = discriminator.train()
    for step,(base_img,target_seg, labels, mask, base_mask) in enumerate(trainloader):
        #base_img = conditional image(IA) labels = target_image(IB) target_seg = pose_target(IB')
        base_img = base_img.to(device)
        target_seg = target_seg.to(device)
        # plt.imshow(images[4,-1].cpu().numpy().astype('uint8'))
        # plt.show()
        labels = labels.to(device)
        mask = mask.to(device)
        base_mask=base_mask.to(device)

        gen_train_op1.zero_grad()
        gen_train_op2.zero_grad()
        dis_train_op1.zero_grad()


        #print sizes 
        print('base img', base_img.size())
        print('target seg', target_seg.size())
        print('labels',labels.size())
        #Generator 1
        images=torch.cat((base_img, target_seg*100.0), 1)
        G1 = generator_one(images)

        g1_loss = L1_criterion(G1*base_mask*20,labels*base_mask*20)+L1_criterion(G1*mask*9,labels*mask*9)+L1_criterion(G1*6,labels*6)
        g1_loss.backward(retain_graph=True)
        gen_train_op1.step()

        #Generator 2
        DiffMap = generator_two(torch.cat((G1, base_img), 1))
        G2 = G1 + DiffMap
        
        #Discriminator
        triplet = torch.cat([labels, G2, base_img], dim=0)
        D_z = discriminator(triplet)
        D_z = torch.clamp(D_z, 0.0, 1.0)
        print('DZ',D_z.size())
        D_z_pos_x_target, D_z_neg_g2, D_z_neg_x = torch.split(D_z,5) #batch size
        D_z_pos = D_z_pos_x_target
        D_z_neg = torch.cat([D_z_neg_g2, D_z_neg_x], 0)
        
        #Generator 2 loss
        #g2_loss = BCE_criterion(D_z_neg, torch.ones((2)).cuda())
        g2_loss = BCE_criterion(D_z_neg, torch.ones((10))) #2*batch size
        PoseMaskLoss2 = L1_criterion(G2 * mask, labels * mask)
        L1Loss2 = L1_criterion(G2, labels) + PoseMaskLoss2
        g2_loss += 50*L1Loss2


        gen_train_op2.zero_grad()
        g2_loss.backward(retain_graph=True)
        gen_train_op2.step()

        #discriminator loss
        #d_loss = BCE_criterion(D_z_pos, torch.ones((1)).cuda())
        #d_loss += BCE_criterion(D_z_neg, torch.zeros((2)).cuda())
        d_loss = BCE_criterion(D_z_pos, torch.ones((5)))
        d_loss += BCE_criterion(D_z_neg, torch.zeros((10)))
        d_loss /= 2

        dis_train_op1.zero_grad()
        d_loss.backward()
        dis_train_op1.step()
    
        g1_running_loss = g1_loss.item()
        g2_running_loss = g2_loss.item()
        d_running_loss = d_loss.item()
    end = time.time()
    print('[epoch %d] g1_loss: %.3f g2_loss: %.3f d_loss: %.3f elapsed time %.3f' %
          (epoch, g1_running_loss,g2_running_loss,d_running_loss, end-start))


def test_G1(testloader, net, criterion, device):
    '''
    Function for testing.
    '''
    losses = 0.
    cnt = 0
    with torch.no_grad():
        net = net.eval()
        for images, labels, mask, base_mask in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            output = net(images)

            loss = criterion(output * mask, labels * mask)+criterion(output,labels)
            losses += loss.item()
            cnt += 1
    print(losses / cnt)
    image=output[0].permute(1, 2, 0).cpu().numpy()
    if np.min(image)<0:
        image -= np.min(image)
    image/=np.max(image)
    image*=255
    image=image.astype('uint8')
    plt.imshow(image)
    plt.show()
    image = labels[0].permute(1, 2, 0).cpu().numpy()
    plt.imshow(image.astype('uint8'))
    plt.show()
    return (losses/cnt)


def generator(device):

    depth, flow, segm, normal, annotation, img, keypoint = load_data()
    train_data = FacadeDataset(dataset=(img, segm), flag='train', data_range=(75, 125), onehot=False)
    train_loader = DataLoader(train_data, batch_size=5,shuffle=True)
    test_data = FacadeDataset(dataset=(img, segm), flag='test', data_range=(50, 75), onehot=False)
    test_loader = DataLoader(test_data, batch_size=1,shuffle=True)
    
    #models
    generator_one = G1_Net().to(device)
    generator_two = G2_Net().to(device)
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

    print('\nStart training generator1')
    for epoch in range(50):  # TODO decide epochs
        print('-----------------Epoch = %d-----------------' % (epoch + 1))
        train(train_loader, generator_one,generator_two,discriminator,
            L1_criterion, BCE_criterion, gen_train_op1,gen_train_op2,dis_train_op1, device, epoch + 1)
        # TODO create your evaluation set, load the evaluation set and test on evaluation set
        #evaluation_loader = train_loader
        #test_G1(evaluation_loader, net, criterion, device)

    #test_G1(test_loader, net, criterion, device)

    # print('1')
    # plt.imshow(img[0, 0, :, :, :])
    # plt.show()
    #
    # plt.imshow(segm[0, 0, :, :])
    # plt.show()
    #
    # print('2')
    # plt.imshow(img[0, 126, :, :])
    # plt.show()
    #
    # plt.imshow(segm[0, 126, :, :])
    # plt.show()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator(device)
    print('test')


if __name__ == "__main__":
    main()

