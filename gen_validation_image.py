import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

import utils
from utils import to_var, save_img
from mymodels import VAE

from torch.autograd import Variable

device = 'cpu' if not torch.cuda.is_available() else 'cuda'
print('device: ', device)

def main(args):

    if not args.data == 'mnist':
        args.data == 'face'
        # facescrub-5
        args.epochs = 10000
        args.img_channel = 3
        args.num_labels = 5
        args.learning_rate = 0.005
        args.save_test_sample = 1000
        args.save_recon_img = 1000
        args.model_path = 'save_models/face/vae-91-1001.ckpt'
    else:
        args.model_path = 'save_models/mnist/vae-4-3001.ckpt'

    # load weights
    #vae = VAE(args.latent_size).to(device)
    vae = VAE(args.latent_size, args.num_labels, args.img_channel).to(device)
    vae.load_state_dict(torch.load(args.model_path))


    for _ in range(5):
        # fix a style for a row
        style = torch.randn(args.latent_size).cuda()
        style = style.view(1,-1)
        img_row = None
        for number in range(args.num_labels):
            # condition (label)
            c = torch.LongTensor([number]).cuda()
            c = c.view(1,-1)

            # inference
            x = vae.inference(style, c)

            # form a numpy image
            if args.data == 'mnist':
                x = x.view((28,28,1)).data.cpu().detach().numpy()
                x = x.repeat(3,2)
            else:
                x = x.view((3,28,28)).data.cpu().detach().numpy()
                x = x.transpose(1,2,0)

            if img_row is None:
                img_row = x
            else:
                img_row = np.hstack((img_row,x))
            # img_row = np.hstack((img_row,x))
        if 'img' not in locals():
            img = img_row
        else:
            img = np.vstack((img, img_row))
    plt.imshow(img)
    plt.show()
    exit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--figroot", default='fig')
    parser.add_argument("--display_row", default=5)
    parser.add_argument("--save_test_sample", type=int, default=100)
    parser.add_argument("--save_recon_img", type=int, default=100)


    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--latent_size", type=int, default=20)
    parser.add_argument("--img_size", default=28)
    parser.add_argument("--img_channel", type=int, default=1)
    parser.add_argument("--num_labels", type=int, default=10)
    parser.add_argument("--conditional", action='store_true')

    parser.add_argument("--save_model", type=int, default=1000)
    parser.add_argument("--save_root", default='save_models')
    parser.add_argument("--data", default='mnist')

    args = parser.parse_args()

    main(args)
