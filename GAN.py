import argparse
import torch
import torchvision
import torchvision.utils as vutils
import torch.nn as nn
from random import randint
from GAN_model import NetD, NetG
from torch import autograd
import numpy as np
from torch.backends import cudnn


cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--imageSize', type=int, default=128)
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epoch', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--data_path', default='./GAN/', help='folder to train data')
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument('--outf', default='./generate/', help='folder to output images and model checkpoints')
parser.add_argument('--lambd', type=float, default=10, help='Gradient penalty lambda hyperparameter')
opt = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def label2onehot(labels, dim):
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out


transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(opt.imageSize),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
dataset = torchvision.datasets.ImageFolder(opt.data_path, transform=transforms)

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    drop_last=True,
)

netG = NetG(opt.ngf, opt.nz, 2).to(device)
netD = NetD(opt.ndf, 2).to(device)

netG.load_state_dict(torch.load('./generate/netG_310.pth'))
netD.load_state_dict(torch.load('./generate/netD_310.pth'))


optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0, 0.9))
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0004, betas=(0, 0.9))

for epoch in range(10, opt.epoch + 1):
    for i, (imgs, labels) in enumerate(dataloader):
        label = label2onehot(labels, 2).to(device)
        real_imgs = imgs.to(device)
        noise = torch.randn(opt.batchSize, opt.nz, 1, 1)
        noise = noise.to(device)
        d_out_real, dr1, dr2 = netD(real_imgs, label)
        d_loss_real = -torch.mean(d_out_real)
        fake_imgs, gr1, gr2 = netG(noise, label)
        d_out_fake, df1, df2 = netD(fake_imgs, label)
        d_loss_fake = d_out_fake.mean()
        d_loss = d_loss_real + d_loss_fake

        alpha =torch.randn(opt.batchSize, 1, 1, 1).cuda().expand_as(real_imgs)
        interpolated = alpha*real_imgs.data + (1-alpha)*fake_imgs.data
        interpolated.requires_grad_(True)
        out, _, _ = netD(interpolated, label)
        grad = torch.autograd.grad(
            outputs=out,
            inputs=interpolated,
            grad_outputs=torch.ones(out.size()).cuda(),
            retain_graph=True,
            create_graph=True,
            only_inputs=True
        )[0]
        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

        d_loss = d_loss + opt.lambd * d_loss_gp
        optimizerD.zero_grad()
        optimizerG.zero_grad()
        d_loss.backward()
        optimizerD.step()


        fake_imgs, _, _ = netG(noise, label)
        g_out_fake, _, _ = netD(fake_imgs, label)
        g_loss_fake = -g_out_fake.mean()
        optimizerD.zero_grad()
        optimizerG.zero_grad()
        g_loss_fake.backward()
        optimizerG.step()


        print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
              % (epoch, opt.epoch, i, len(dataloader), d_loss.item(), g_loss_fake.item()))

    if epoch % 10 == 0:
        for i in range(2):
            labels = np.zeros(64)
            for j in range(64):
                labels[j] = i
            labels = torch.from_numpy(labels)
            label = label2onehot(labels, 2).to(device)
            fake_imgs, _, _ = netG(noise, label)
            vutils.save_image(fake_imgs.data,
                                  '%s/fake_samples_epoch_%d_class_%d.png' % (opt.outf, epoch, i),
                                  normalize=True)
    if epoch % 10 == 0:
        torch.save(netG.state_dict(), '%s/netG_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_%d.pth' % (opt.outf, epoch))