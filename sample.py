from data.dataloader import load_data, FacadeDataset
from models import *
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

MODEL_DIR = 'models/'
SAVE_STEP = 10
GENERATOR_ONE_PATH = './models/generator_one-51-20.ckpt'
GENERATOR_TWO_PATH = './models/generator_two-51-20.ckpt'
DISCRIMINATOR_PATH = './models/discriminator-51-20.ckpt'
LOAD_FROM_CHECKPOINT = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

generator_one = G1_Net().to(device)
generator_two = G2_Net().to(device)
discriminator = Discriminator().to(device)
if LOAD_FROM_CHECKPOINT:
    generator_one.load_state_dict(torch.load(GENERATOR_ONE_PATH))
    generator_two.load_state_dict(torch.load(GENERATOR_TWO_PATH))
    # discriminator.load_state_dict(torch.load(DISCRIMINATOR_PATH))


def test_G2(testloader, generator_one, generator_two, discriminator, L1_criterion, BCE_criterion,  device):
    losses = 0.
    cnt = 0
    with torch.no_grad():
        generator_one = generator_one.eval()
        generator_two = generator_two.eval()
        # discriminator = discriminator.eval()
        it = 0
        for step, (base_img, target_seg, labels, mask, base_mask) in enumerate(testloader):
            base_img = base_img.to(device)
            target_seg = target_seg.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            base_mask = base_mask.to(device)

            images = torch.cat((base_img, target_seg * 100.0), 1)
            G1 = generator_one(images)

            DiffMap = generator_two(torch.cat((G1, base_img), 1))
            G2 = G1 + DiffMap

            # triplet = torch.cat([labels, G2, base_img], dim=0)
            # D_z = discriminator(triplet)
            # D_z = torch.clamp(D_z, 0.0, 1.0)
            # # print('DZ',D_z.size())
            # D_z_pos_x_target, D_z_neg_g2, D_z_neg_x = torch.split(D_z, 2)  # batch size
            # D_z_neg = torch.cat([D_z_neg_g2, D_z_neg_x], 0)
            #
            # g2_loss = BCE_criterion(D_z_neg, torch.ones((4, 1)).to(device=device))  # 2*batch size
            # PoseMaskLoss2 = L1_criterion(G2 * mask, labels * mask)
            # L1Loss2 = L1_criterion(G2, labels) + PoseMaskLoss2
            # g2_loss += 50 * L1Loss2
            #
            # losses += g2_loss.item()
            # cnt += 1

            for i in range(G2.shape[0]):
                image = G2[i].permute(1, 2, 0).cpu().numpy()
                if np.min(image) < 0:
                    image -= np.min(image)
                image /= np.max(image)
                image *= 255
                image = image.astype('uint8')
                plt.imshow(image)
                plt.show()
                plt.imsave('./images/img{}'.format(it),image)
                it+=1
                # image = labels[i].permute(1, 2, 0).cpu().numpy()
                # plt.imshow(image.astype('uint8'))
                # plt.show()
    # return (losses / cnt)



depth, flow, segm, normal, annotation, img, keypoint = load_data()
# train_data = FacadeDataset(dataset=(img, segm), flag='train', data_range=(75, 125), onehot=False)
# train_loader = DataLoader(train_data, batch_size=2,shuffle=True)
test_data = FacadeDataset(dataset=(img, segm), flag='test', data_range=(0,130),onehot=False)
test_loader = DataLoader(test_data, batch_size=2,shuffle=False)
L1_criterion = nn.L1Loss()
BCE_criterion = nn.BCELoss()

test_G2(test_loader, generator_one,generator_two,discriminator, L1_criterion,
            BCE_criterion, device)