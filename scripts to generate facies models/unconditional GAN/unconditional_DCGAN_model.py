from torch import nn
import torch


class Generator80x100(nn.Module):
    def __init__(self):
        super(Generator80x100, self).__init__()

        self.Gnet = nn.Sequential(
            nn.ConvTranspose2d(100, 512, kernel_size=4,stride=2, padding=0, bias=False),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(512, 256, kernel_size=5,stride=(2, 3), padding=1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(256, 512, kernel_size=5,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(512, 128, kernel_size=4,stride=2, padding=(0, 1), bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(128, 1, kernel_size=4,stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.reshape(-1, 100, 1, 1)
        return self.Gnet(z)


class Discriminator80x100(nn.Module):
    def __init__(self):
        super(Discriminator80x100, self).__init__()

        self.Dnet = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.Dnet(x).reshape(-1,1)
