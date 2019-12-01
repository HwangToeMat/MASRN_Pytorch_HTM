import torch
import torch.nn as nn
import torch.functional as F

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
    def __init__(self, in_channels):
        super(Extraction_net, self).__init__()
        self.input = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.FERB_3 = FERB(64, 3)
        self.FERB_5 = FERB(64, 5)
        self.FERB_7 = FERB(64, 7)
        self.FERB_9 = FERB(64, 9)

    def forward(self, x):
        input = self.input(x)
        FERB_3 = self.FERB_3(input)
        FERB_5 = self.FERB_5(FERB_3)
        FERB_7 = self.FERB_7(FERB_5)
        FERB_9 = self.FERB_9(FERB_7)
        BottleNeck = torch.cat([input, FERB_3, FERB_5, FERB_7, FERB_9], 1)
        return BottleNeck

class Upscale_net(nn.Module):
    

class Refine_net(nn.Module):
    def __init__(self, in_channels):
        super(Refine_net, self).__init__()
        self.input = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.Asym_Block_1 = nn.Sequential(
            nn.Conv2d(16, 16, (1, 1), stride=1, padding=0))
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

    def forward(self, HR):
        input = self.input(HR)
        Asym_1 = self.Asym_Block_1(input)
        Asym_2 = self.Asym_Block_2(input)
        Asym_3 = self.Asym_Block_3(input)
        Asym_4 = self.Asym_Block_4(input)
        residual = self.concat(torch.cat([Asym_1, Asym_2, Asym_3, Asym_4], 1))
        HR += residual
        return HR

class MASRN_Net(nn.Module):
    def __init__(self):
        super(MASRN_Net, self).__init__()
        self.input = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.ANRB1 = ANRB(64, 2)
        self.ANRB1 = ANRB(64, 2)
        self.ANRB1 = ANRB(64, 3)
        self.ANRB1 = ANRB(64, 3)
        self.ANRB1 = ANRB(64, 5)
        self.ANRB1 = ANRB(64, 8)
        self.ANRB1 = ANRB(64, 9)
        self.ANRB3 = ANRB(64)
        self.ANRB4 = ANRB(64)
        self.ANRB5 = ANRB(64)
        self.ANRB6 = ANRB(64)
        self.ANRB7 = ANRB(64)
        self.output = nn.Sequential(
            nn.Conv2d(64*(7+1), 64, 1, stride=1, padding=0),
            nn.Conv2d(64, 3, 3, stride=1, padding=1)
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        R_0 = self.input(x)
        R_1 = self.ANRB1(R_0)
        R_2 = self.ANRB2(R_1)
        R_3 = self.ANRB3(R_2)
        R_4 = self.ANRB4(R_3)
        R_5 = self.ANRB5(R_4)
        R_6 = self.ANRB6(R_5)
        R_7 = self.ANRB7(R_6)
        output = self.output(torch.cat([R_0, R_1, R_2, R_3, R_4, R_5, R_6, R_7], 1))
        output += x
        output = (self.tanh(output)+1)/2
        return output
