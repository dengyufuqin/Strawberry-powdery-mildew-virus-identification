import torch.nn as nn
from spectral import SpectralNorm
import numpy as np
import torch

class Self_Attn(nn.Module):
    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.channel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax =nn.Softmax(dim=-1)
    def forward(self, x):
        m_batchsize, C, width,height =x.size()
        proj_query =self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention =self.softmax(energy)
        proj_value =self.value_conv(x).view(m_batchsize, -1, width*height)

        out =torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma*out + x
        return out, attention

class NetG(nn.Module):
    def __init__(self, ngf, nz, num_classes=5):
        super(NetG, self).__init__()
        self.classes = num_classes

        self.layer1 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(nz+self.classes, ngf * 16, kernel_size=4, stride=1, padding=0, bias=False)),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(ngf * 4, ngf*2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True)
        )

        self.layer5 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(128, 'relu')
        # self.attn3 = Self_Attn(64, 'relu')

    def forward(self, x, label):
        label = label.view(label.size(0), label.size(1), 1, 1)
        label = label.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, label], dim=1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out, p1 = self.attn1(out)
        out = self.layer4(out)
        out, p2 = self.attn2(out)
        out = self.layer5(out)
        # out, p3 = self.attn3(out)
        out = self.layer6(out)
        return out, p1, p2



class NetD(nn.Module):
    def __init__(self, ndf, num_classes=5):
        super(NetD, self).__init__()
        self.classes = num_classes

        self.layer1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(3+num_classes, ndf, kernel_size=4, stride=2, padding=1, bias=False)),
            # nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            # nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            # nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            # nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer5 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ndf * 8, ndf*16, 4, 1, 0, bias=False)),
            # nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2, True)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(ndf*16, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )
        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')
        # self.attn3 = Self_Attn(1024, 'relu')

    # 定义NetD的前向传播
    def forward(self, x, label):
        label = label.view(label.size(0), label.size(1), 1, 1)
        label = label.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, label], dim=1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out, p1 = self.attn1(out)
        out = self.layer4(out)
        out, p2 = self.attn2(out)
        out = self.layer5(out)
        # out, p3 = self.attn3(out)
        out = self.layer6(out)
        return out.squeeze(), p1, p2
