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
    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)
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
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            loss = criterion(output, labels)
            losses += loss.item()
            cnt += 1
    print(losses / cnt)
    return (losses/cnt)


def generator1(device):

    depth, flow, segm, normal, annotation, img, keypoint = load_data()
    train_data = FacadeDataset(dataset=(img, segm), flag='train', data_range=(0, 80), onehot=False)
    train_loader = DataLoader(train_data, batch_size=5)

    net = G1_Net().to(device)

    criterion = nn.CrossEntropyLoss()  # TODO decide loss
    optimizer = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=0)

    print('\nStart training generator1')
    for epoch in range(5):  # TODO decide epochs
        print('-----------------Epoch = %d-----------------' % (epoch + 1))
        train_G1(train_loader, net, criterion, optimizer, device, epoch + 1)
        # TODO create your evaluation set, load the evaluation set and test on evaluation set
        evaluation_loader = train_loader
        test_G1(evaluation_loader, net, criterion, device)
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

