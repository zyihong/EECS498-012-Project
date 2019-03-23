from data.dataloader import load_data, FacadeDataset
from models import *
from matplotlib import pyplot as plt
import numpy as np


def train_G1(trainloader, net, criterion, optimizer, device, epoch):
    '''
    Function for training.
    '''
    start = time.time()
    running_loss = 0.0
    net = net.train()
    for images, labels, mask in tqdm(trainloader):
        images = images.to(device)
        # plt.imshow(images[4,-1].cpu().numpy().astype('uint8'))
        # plt.show()
        labels = labels.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output*mask, labels*mask)
        loss.backward()
        optimizer.step()
        running_loss = loss.item()
    end = time.time()
    print('[epoch %d] loss: %.3f elapsed time %.3f' %
          (epoch, running_loss, end-start))


def test_G1(testloader, net, criterion, device):
    '''
    Function for testing.
    '''
    losses = 0.
    cnt = 0
    with torch.no_grad():
        net = net.eval()
        for images, labels, mask in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            output = net(images)

            loss = criterion(output * mask, labels * mask)
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


def generator1(device):

    depth, flow, segm, normal, annotation, img, keypoint = load_data()
    train_data = FacadeDataset(dataset=(img, segm), flag='train', data_range=(50, 110), onehot=False)
    train_loader = DataLoader(train_data, batch_size=5,shuffle=True)
    test_data = FacadeDataset(dataset=(img, segm), flag='test', data_range=(110, 125), onehot=False)
    test_loader = DataLoader(test_data, batch_size=1,shuffle=True)
    net = G1_Net().to(device)

    criterion = nn.L1Loss()  # TODO decide loss
    optimizer = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=0)

    print('\nStart training generator1')
    for epoch in range(20):  # TODO decide epochs
        print('-----------------Epoch = %d-----------------' % (epoch + 1))
        train_G1(train_loader, net, criterion, optimizer, device, epoch + 1)
        # TODO create your evaluation set, load the evaluation set and test on evaluation set
        evaluation_loader = train_loader
        test_G1(evaluation_loader, net, criterion, device)

    test_G1(test_loader, net, criterion, device)

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
    generator1(device)
    print('test')


if __name__ == "__main__":
    main()

