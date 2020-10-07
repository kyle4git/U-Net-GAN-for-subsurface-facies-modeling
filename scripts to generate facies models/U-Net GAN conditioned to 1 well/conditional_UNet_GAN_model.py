import torch
from torch import nn


class cGenerator(nn.Module):
    def __init__(self):
        super(cGenerator, self).__init__()

        def block(in_channel, out_channel, ks,
                  up=False, down=False, normalize=True, relu=True, tanh=False):

            layers = []
            if down:
                layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=ks, stride=2, bias=False))
            if up:
                layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size=ks, stride=2, bias=False))
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channel))
            if relu:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            if tanh:
                layers.append(nn.Tanh())

            return layers

        self.net1 = nn.Sequential(*block(1, 32, 4, down=True, normalize=False))  # [39, 49]
        self.net2 = nn.Sequential(*block(32, 64, 3, down=True))  # [19,24]
        self.net3 = nn.Sequential(*block(64, 128, (3, 4), down=True))  # [9,11]
        self.net4 = nn.Sequential(*block(128, 256, (3, 5), down=True))  # [4, 4]

        self.net5 = nn.Sequential(*block(256, 512, 4, down=True))  # [1,1]

        self.net6 = nn.Sequential(*block(512 + 100, 256, 4, up=True))  # [4,4]
        self.net7 = nn.Sequential(*block(1024, 128, (3, 5), up=True))  # [9,11]
        self.net8 = nn.Sequential(*block(512, 64, (3, 4), up=True))  # [19,24]
        self.net9 = nn.Sequential(*block(256, 32, 3, up=True))  # [39,49]

        self.net10 = nn.Sequential(*block(128, 1, 4, up=True, normalize=False,
                                          relu=False, tanh=True))  # [80,100]
        
        self.znet1 = nn.Sequential(*block(100, 512, 4, up=True)) # [1,1]==>[4,4]
        self.znet2 = nn.Sequential(*block(512, 256, (3,5), up=True)) # [4,4]==>[9,11]
        self.znet3 = nn.Sequential(*block(256, 128, (3,4), up=True)) # [9,11]==>[14,24]
        self.znet4 = nn.Sequential(*block(128, 64, 3, up=True)) # [19,24]==>[39,49]

    def forward(self, x0, z):
        z = z.view(-1, 100, 1, 1)
        z1 = self.znet1(z)
        z2 = self.znet2(z1)
        z3 = self.znet3(z2)
        z4 = self.znet4(z3)
        
        x1 = self.net1(x0)
        x2 = self.net2(x1)
        x3 = self.net3(x2)
        x4 = self.net4(x3)
        x5 = self.net5(x4)

        x5 = torch.cat([x5, z], dim=1)

        x6 = self.net6(x5)
        x6 = torch.cat([x4, x6, z1], dim=1)

        x7 = self.net7(x6)
        x7 = torch.cat([x3, x7, z2], dim=1)

        x8 = self.net8(x7)
        x8 = torch.cat([x2, x8, z3], dim=1)

        x9 = self.net9(x8)
        x9 = torch.cat([x1, x9, z4], dim=1)

        x10 = self.net10(x9)
        del x0, x1, x2, x3, x4, x5, x6, x7, x8, x9
        del z1, z2, z3, z4

        return x10.view(-1, 1, 80, 100)


class cDiscriminator(nn.Module):
    def __init__(self):
        super(cDiscriminator, self).__init__()

        def block(in_channel, out_channel, ks, normalize=True,relu=True, sig=False):
            layers = []

            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=ks, stride=2, bias=False))
            if normalize:
                layers.append(nn.BatchNorm2d(out_channel))
            if relu:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            if sig:
                layers.append(nn.Sigmoid())

            return layers

        self.Dnet = nn.Sequential(
            *block(2, 64, 4, normalize=False),
            *block(64, 128, 3),
            *block(128, 256, (3, 4)),
            *block(256, 1, (3, 5), normalize=False, relu=False, sig=True),
        )

    def forward(self, image, mask):
        x = torch.cat([image, mask], 1)
        score = self.Dnet(x)
        score = score.reshape(-1, 16).mean(dim=1).reshape(-1, 1)
        del x
        return score