import torch
import torch.nn as nn
# from utils import to_var, idx2onehot
from torch.autograd import Variable
import numpy as np
import math
import utils

# device = 'cpu' if torch.cuda.is_available() else 'cuda'

class VAE(nn.Module):
    def __init__(self, latent_variable_size, num_class, num_channel, img_size):
        super().__init__()
        self.num_channel = num_channel
        self.num_class = num_class
        self.img_size = img_size
        self.img_size_s = int(img_size/4)
        ndf,ngf = 64, 64

        # VAE Encoder
        if self.img_size == 28:
            self.Encoder = nn.Sequential(
                nn.Conv2d((self.num_channel+self.num_class), ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2),

                nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf*2),
                nn.LeakyReLU(0.2),

                nn.Conv2d(ndf*2, ndf*4, 7, 1, 0, bias=False),
                nn.BatchNorm2d(ndf*4),
            )
        else:
            self.Encoder = nn.Sequential(
                nn.Conv2d((self.num_channel+self.num_class), ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2),

                nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf*2),
                nn.LeakyReLU(0.2),

                nn.Conv2d(ndf*2, ndf*4, 7, 4, 1, bias=False),
                nn.BatchNorm2d(ndf*4),
            )

        self.mu = nn.Linear(ndf*4*1*1, latent_variable_size)
        self.logvar = nn.Linear(ndf*4*1*1, latent_variable_size)

        # VAE Decoder
        if self.img_size == 28:
            self.Decoder_input = nn.Sequential(
            nn.Linear(latent_variable_size+num_class, ndf*4*7*7, bias=False),
            nn.LeakyReLU(0.2),
            )
        else:
            self.Decoder_input = nn.Sequential(
            nn.Linear(latent_variable_size+num_class, ndf*4*8*8, bias=False),
            nn.LeakyReLU(0.2),
            )
        # self.Decoder_input = nn.Sequential(
        #     nn.Linear(latent_variable_size+num_class, ndf*4*7*7, bias=False),
        #     nn.LeakyReLU(0.2),
        # )

        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2),
    
            nn.ConvTranspose2d(ngf*2, self.num_channel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_channel),
    
            nn.Sigmoid()
        )
        # if self.img_size == 28:
        #     self.Decoder = nn.Sequential(
        #         nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
        #         nn.BatchNorm2d(ngf*2),
        #         nn.LeakyReLU(0.2),
    
        #         nn.ConvTranspose2d(ngf*2, self.num_channel, 4, 2, 1, bias=False),
        #         nn.BatchNorm2d(self.num_channel),
    
        #         nn.Sigmoid()
        #     )
        # else:
        #     self.Decoder = nn.Sequential(
        #     nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf*2),
        #     nn.LeakyReLU(0.2),

        #     nn.ConvTranspose2d(ngf*2, self.num_channel, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(self.num_channel),

        #     nn.Sigmoid()
        #     )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) # uniform dist. within [0, 1]
        return eps.mul(std).add_(mu)
    
    def encode(self, x, c):
        if c is not None:
            # restructure input
            qq = utils.idx2onehot(c.view(-1,1), self.num_class)
            qq = qq.view(-1,self.num_class,1,1).expand(-1,-1,self.img_size,self.img_size)
            x = torch.cat([x, qq],1)
        
        x = self.Encoder(x)
        x = x.view(x.size(0), -1)
        
        # exit()
        mu, logvar = self.mu(x), self.logvar(x)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z
    
    def decode(self, z, c):
        if c is not None:
            # restructure input
            qq = utils.idx2onehot(c.view(-1,1), self.num_class)
            z = torch.cat([z, qq],1)

        recon_x = self.Decoder_input(z)
        recon_x = recon_x.view(recon_x.size(0), -1, self.img_size_s, self.img_size_s)
        recon_x = self.Decoder(recon_x)
        return recon_x

    # modify CVAE
    def inference(self, x, c=None):
        return self.decode(x, c)

    # modify CVAE
    def forward(self, x, c=None):
        mu, logvar, z = self.encode(x, c)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar, z

    '''
    def forward(self, x, c=None):

        batch_size = x.size(0)

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5 * log_var)
        # eps = to_var(torch.randn([batch_size, self.latent_size]))
        eps = torch.randn([batch_size, self.latent_size]).to(device)
        z = eps * std + means

        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def inference(self, n=1, c=None):

        batch_size = n
        z = to_var(torch.randn([batch_size, self.latent_size]))

        recon_x = self.decoder(z, c)

        return recon_x
    '''

