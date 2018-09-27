import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST, ImageFolder
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

from utils import to_var, save_img
from mymodels import VAE

from torch.autograd import Variable

device = 'cpu' if not torch.cuda.is_available() else 'cuda'
print('device: ', device)

def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return BCE + KLD

def main(args):

    # datasets = OrderedDict()
    # datasets['train'] = MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
    
    if args.data == 'mnist':
        ### CVAE on MNIST ###
        n_transform = transforms.Compose([transforms.ToTensor()])
        dataset = MNIST('data', transform=n_transform, download=True)
        args.img_size = 28
        args.epochs = 10000
        args.img_channel = 1
        args.num_labels = 10
        args.learning_rate = 0.001
        args.save_test_sample = 1000
        args.save_recon_img = 1000
    else:
        args.data == 'face'
        ### CVAE on facescrub5 ###
        n_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
        dataset = ImageFolder('facescrub-5', transform=n_transform)
        args.img_size = 32
        args.epochs = 10000
        args.img_channel = 3
        args.num_labels = 5
        args.learning_rate = 0.001
        args.save_test_sample = 1000
        args.save_recon_img = 1000


    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    vae = VAE(args.latent_size, args.num_labels, args.img_channel, args.img_size).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    # fixed noise
    fix_noise = Variable(torch.randn((args.batch_size, args.latent_size)).cuda())

    # tracker_global = defaultdict(torch.FloatTensor)

    num_iter = 0

    for epoch in range(args.epochs):
        for _, batch in enumerate(data_loader, 0):
            img, label  = Variable(batch[0].cuda()), Variable(batch[1].cuda())
            
            # recon_img, mean, log_var, z = vae(img)

            # CVAE
            # if img.size()[0]!= 64:
            #     print('batch size:',img.size()[0])
            recon_img, mean, log_var, z = vae(img, label)

            loss = loss_fn(recon_img, img, mean, log_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_iter += 1

            if num_iter % args.print_every == 0:
                print("Batch {:04d}/{}, Loss {:9.4f}".format(num_iter, len(data_loader)-1, loss.data.item()))

            if num_iter % args.save_test_sample == 0 and len(label) == 64:
                c = label*0+2 # TODO: given condition
                # c = torch.LongTensor([number]).cuda()
                x = vae.inference(fix_noise, c)
                save_img(args, x.detach(), num_iter)
            
            if num_iter % args.save_recon_img == 0:
                save_img(args, recon_img.detach(), num_iter, recon=True)

            # save the model checkpoints
            if num_iter % args.save_model == 0:
                if not(os.path.exists(os.path.join(args.save_root))):
                    os.mkdir(os.path.join(args.save_root))
                if not(os.path.exists(os.path.join(args.save_root, args.data))):
                    os.mkdir(os.path.join(args.save_root, args.data))
                torch.save(vae.state_dict(), os.path.join(args.save_root, args.data,'vae-{}-{}.ckpt'.format(epoch+1, num_iter+1)))

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
