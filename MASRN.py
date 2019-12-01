import torch
import torch.nn as nn
import torch.functional as F
import common

class FERB(nn.Module):
    def __init__(self, in_channels, k_size):
        super(FERB, self).__init__()
        self.block_0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, (1, k_size), stride=1, padding=(0,(k_size//2))),
            nn.Conv2d(64, 64, (k_size, 1), stride=1, padding=((k_size//2),0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (1, k_size), stride=1, padding=(0,(k_size//2))),
            nn.Conv2d(64, 64, (k_size, 1), stride=1, padding=((k_size//2),0)),
            nn.ReLU(inplace=True)
        )
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, (1, k_size), stride=1, padding=(0,(k_size//2))),
            nn.Conv2d(64, 64, (k_size, 1), stride=1, padding=((k_size//2),0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (1, k_size), stride=1, padding=(0,(k_size//2))),
            nn.Conv2d(64, 64, (k_size, 1), stride=1, padding=((k_size//2),0)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        residual_0 = self.block_0(x)
        residual_0 += x
        residual_1 = self.block_1(residual_0)
        residual_1 += residual_0
        residual_1 += x
        return residual_1

class Extraction_net(nn.Module):
    def __init__(self):
        super(Extraction_net, self).__init__()
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(255, rgb_mean, rgb_std)
        self.input = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.FERB_3 = FERB(64, 3)
        self.FERB_5 = FERB(64, 5)
        self.FERB_7 = FERB(64, 7)
        self.FERB_9 = FERB(64, 9)

    def forward(self, x):
        x = self.sub_mean(x)
        input = self.input(x)
        FERB_3 = self.FERB_3(input)
        FERB_5 = self.FERB_5(FERB_3)
        FERB_7 = self.FERB_7(FERB_5)
        FERB_9 = self.FERB_9(FERB_7)
        BottleNeck = torch.cat([input, FERB_3, FERB_5, FERB_7, FERB_9], 1)
        return BottleNeck

class Upscale_net(nn.Module):
    def __init__(self):
        super(Upscale_net, self).__init__()
        conv= common.default_conv
        self.input = nn.Sequential(
            nn.Conv2d(320, 64, 1, stride=1, padding=0),
            nn.Conv2d(64, 64, 3, stride=1, padding=1)
        )
        self.upscale = nn.ModuleList([
            common.Upsampler(conv, s, 64, act=False) for s in [2, 3, 4]
        ])
        self.output = nn.Conv2d(64, 3, 3, stride=1, padding=1)

    def forward(self, x, scale_idx):
        x = self.input(x)
        x = self.upscale[scale_idx](x)
        x = self.output(x)
        return x

class Refine_net(nn.Module):
    def __init__(self):
        super(Refine_net, self).__init__()
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.input = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.Asym_Block_1 = nn.Sequential(
            nn.Conv2d(16, 16, (1, 1), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, (1, 3), stride=1, padding=(0,1)),
            nn.Conv2d(16, 16, (3, 1), stride=1, padding=(1,0)),
            nn.ReLU(inplace=True)
        )
        self.Asym_Block_2 = nn.Sequential(
            nn.Conv2d(16, 16, (1, 3), stride=1, padding=(0,1)),
            nn.Conv2d(16, 16, (3, 1), stride=1, padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, (1, 5), stride=1, padding=(0,2)),
            nn.Conv2d(16, 16, (5, 1), stride=1, padding=(2,0)),
            nn.ReLU(inplace=True)
        )
        self.Asym_Block_3 = nn.Sequential(
            nn.Conv2d(16, 16, (1, 5), stride=1, padding=(0,2)),
            nn.Conv2d(16, 16, (5, 1), stride=1, padding=(2,0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, (1, 7), stride=1, padding=(0,3)),
            nn.Conv2d(16, 16, (7, 1), stride=1, padding=(3,0)),
            nn.ReLU(inplace=True)
        )
        self.Asym_Block_4 = nn.Sequential(
            nn.Conv2d(16, 16, (1, 7), stride=1, padding=(0,3)),
            nn.Conv2d(16, 16, (7, 1), stride=1, padding=(3,0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, (1, 9), stride=1, padding=(0,4)),
            nn.Conv2d(16, 16, (9, 1), stride=1, padding=(4,0)),
            nn.ReLU(inplace=True)
        )
        self.concat = nn.Conv2d(64, 3, 1, stride=1, padding=0)
        self.add_mean = common.MeanShift(255, rgb_mean, rgb_std, 1)

    def forward(self, HR):
        input = self.input(HR)
        Asym_1 = self.Asym_Block_1(input)
        Asym_2 = self.Asym_Block_2(input)
        Asym_3 = self.Asym_Block_3(input)
        Asym_4 = self.Asym_Block_4(input)
        residual = self.concat(torch.cat([Asym_1, Asym_2, Asym_3, Asym_4], 1))
        HR += residual
        HR = self.add_mean(HR)
        return HR

class MASRN_Net(nn.Module):
    def __init__(self):
        super(MASRN_Net, self).__init__()
        self.Extraction = nn.Extraction_net()
        self.Upscale = nn.Upscale_net()
        self.Refine = nn.Refine_net()

    def forward(self, x, scale):
        Extraction = self.Extraction(x)
        Upscale = self.Upscale(Extraction, self.scale_idx - 2)
        HR = self.Refine(Upscale)
        return HR

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
