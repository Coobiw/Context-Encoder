import numpy as np
import torch
from torch import nn
# from torch.utils.tensorboard import SummaryWriter
import time


class Net_G(nn.Module):
    def __init__(self):
        super(Net_G, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2)
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(512, 4096, 4, 1, 0),
            nn.BatchNorm2d(4096),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1, 1),
            nn.BatchNorm2d(4096),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block8 = nn.Sequential(
            nn.ConvTranspose2d(4096, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.block9 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.block10 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.block11 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.block12 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.block13 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, 2, 1)
        )

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        out = self.tanh(x)
        return out


if __name__ == '__main__':
    t1 = time.time()
    x = torch.rand(1, 1, 128, 128).cuda()
    print("input_size:", x.size())
    net = Net_G().cuda()
    net.eval()
    ouput = net(x)
    print("output_size:", ouput.size())
    t2 = time.time()
    print(t2 - t1)
    # with SummaryWriter(log_dir='netG_structure',comment='Net_FCN') as w:
    #     w.add_graph(net,(x,))