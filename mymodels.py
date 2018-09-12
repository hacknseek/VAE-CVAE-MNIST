import torch
import torch.nn as nn
# from utils import to_var, idx2onehot
from torch.autograd import Variable
import numpy as np
import utils

# device = 'cpu' if torch.cuda.is_available() else 'cuda'

class VAE(nn.Module):
    def __init__(self, latent_variable_size):
        super().__init__()
        nc = 1
        ndf,ngf = 64, 64
        num_class = 10 # CVAE

        # VAE Encoder
        self.Encoder = nn.Sequential(
            nn.Conv2d((nc+num_class), ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*2, ndf*4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(ndf*4),
        )

        self.mu = nn.Linear(ndf*4*1*1, latent_variable_size)
        self.logvar = nn.Linear(ndf*4*1*1, latent_variable_size)

        # VAE Decoder
        self.Decoder_input = nn.Sequential(
            nn.Linear(latent_variable_size+num_class, ndf*4*7*7, bias=False),
            nn.LeakyReLU(0.2),
        )

        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nc),

            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) # uniform dist. within [0, 1]
        return eps.mul(std).add_(mu)
    
    def encode(self, x, c):
        if c is not None:
            # restructure input
            qq = utils.idx2onehot(c.view(-1,1), 10)
            qq = qq.view(-1,10,1,1).expand(-1,-1,28,28)
            x = torch.cat([x, qq],1)
            # print('x:',x.size())
        
        x = self.Encoder(x)
        x = x.view(x.size(0), -1)
        mu, logvar = self.mu(x), self.logvar(x)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z
    
    def decode(self, z, c):
        if c is not None:
            # restructure input
            qq = utils.idx2onehot(c.view(-1,1), 10)
            z = torch.cat([z, qq],1)

        # print('z:', z.size())
        recon_x = self.Decoder_input(z)
        recon_x = recon_x.view(recon_x.size(0), -1, 7, 7)
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

