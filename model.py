import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.image_size = 64
        self.num_channels = 1
        self.noise_dim = 100
        self.embed_dim = 30
        self.latent_dim = self.noise_dim + self.embed_dim
        self.ngf = 64

        # based on: https://github.com/pytorch/examples/blob/master/dcgan/main.py
        self.netG = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (num_channels) x 32 x 32
        )

    def forward(self, embed_vector, z):
        embed_vector = embed_vector.unsqueeze(2).unsqueeze(3)
        latent_vector = torch.cat([embed_vector, z], 1)
        output = self.netG(latent_vector)
        return output


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.image_size = 64
        self.num_channels = 1
        self.embed_dim = 30
        # self.projected_embed_dim = 128
        self.ndf = 64
        self.B_dim = 128
        self.C_dim = 16
        self.netD_1 = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(self.num_channels, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # output size (ndf*4) x 4 x 4
        )

        # self.projector = Concat_embed(self.embed_dim, self.projected_embed_dim)

        self.netD_2 = nn.Sequential(
            # state size. (ndf*2) x 4 x 4
            # nn.Conv2d(self.ndf * 8 + self.projected_embed_dim, 1, 4, 1, 0, bias=False),
            nn.Conv2d(self.ndf * 4 + self.embed_dim, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inp, embed):
        # print(embed.size())
        x_intermediate = self.netD_1(inp)
        #print("inter",x_intermediate.size())
        replicated_embed = embed.repeat(4, 4, 1, 1).permute(2, 3, 0, 1)
        # print(replicated_embed.size())
        # print(x_intermediate.size())
        x = torch.cat([x_intermediate, replicated_embed], 1)
        x = self.netD_2(x)
        #print("x size",x.size())
        #print("x reshape",x.view(-1,1).squeeze(1).size())
        return x.view(-1, 1).squeeze(1), x_intermediate
