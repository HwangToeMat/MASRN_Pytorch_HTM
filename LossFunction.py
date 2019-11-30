import torch
from torch import nn
import torch.backends.cudnn as cudnn
import numpy as np

def PSNR(HR, fake_img):

    mse = nn.MSELoss()
    loss = mse(HR, fake_img)
    psnr = 10 * np.log10(1 / (loss.item() + 1e-10))
    return psnr
