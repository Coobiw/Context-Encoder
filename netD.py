import numpy as np
import torch
from torch import nn
# from torch.utils.tensorboard import SummaryWriter
import cfg
class Net_D(nn.Module):
    def __init__(self):
        super(Net_D, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True),
            nn.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.MaxPool2d(2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.MaxPool2d(2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.MaxPool2d(2)
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, 0)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = x.view(-1,1)
        out = self.sigmoid(x)

        return out

if __name__ == '__main__':
    x = torch.rand(4, 1, 128, 128)
    print("input_size:", x.size())
    net = Net_D()
    y = net(x)
    print(y)
    print("output_size:", y.size())
    # with SummaryWriter(log_dir='netD_structure',comment='Net_FCN') as w:
    #     w.add_graph(net,(x,))