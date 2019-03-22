from data.dataloader import load_data
from models import *
from matplotlib import pyplot as plt
import numpy as np

def main():
    depth, flow, segm, normal, annotation, img, keypoint = load_data()

    # print('1')
    # plt.imshow(img[0, 0, :, :, :])
    # plt.show()
    #
    # plt.imshow(segm[0, 0, :, :])
    # plt.show()
    #
    # print('2')
    # plt.imshow(img[0, 60, :, :, :])
    # plt.show()
    #
    # plt.imshow(segm[0, 60, :, :])
    # plt.show()





    print('test')


if __name__ == "__main__":
    main()

