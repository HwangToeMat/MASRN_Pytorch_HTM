import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from MASRN import MASRN_Net
from LossFunction import PSNR
from dataset_h5 import Read_dataset_h5
import pytorch_ssim

# Training settings
parser = argparse.ArgumentParser(description="PyTorch MASRN")
parser.add_argument("--batchSize", type=int, default=64)
parser.add_argument("--nEpochs", type=int, default=300)
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--threads", type=int, default=8)
parser.add_argument('--pretrained', default='', type=str)
parser.add_argument('--datapath_1', default='data/train_1.h5', type=str)
parser.add_argument('--datapath_2', default='data/train_2.h5', type=str)
parser.add_argument('--datapath_3', default='data/train_3.h5', type=str)
parser.add_argument("--gpus", default="0", type=str)

def main():
    global opt, model, optimizer
    epoch = 1
    opt = parser.parse_args() # opt < parser
    print(opt)

    print("===> Setting GPU")
    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus # set gpu
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed) # set seed
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True # find optimal algorithms for hardware

    print("===> Loading datasets")
    train_set_1 = Read_dataset_h5(opt.datapath_1)
    training_data_loader1 = DataLoader(dataset=train_set_1, num_workers=opt.threads,
        batch_size=opt.batchSize, shuffle=True) # read to DataLoader
    train_set_2 = Read_dataset_h5(opt.datapath_2)
    training_data_loader2 = DataLoader(dataset=train_set_2, num_workers=opt.threads,
        batch_size=opt.batchSize, shuffle=True) # read to DataLoader
    train_set_3 = Read_dataset_h5(opt.datapath_3)
    training_data_loader3 = DataLoader(dataset=train_set_3, num_workers=opt.threads,
        batch_size=opt.batchSize, shuffle=True) # read to DataLoader

    print("===> Building model")
    model = MASRN_Net()
    L1loss = nn.L1Loss()

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            checkpoint = torch.load(opt.pretrained)
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch'] + 1 # load model
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    if cuda:
        model = model.cuda()
        L1loss = L1loss.cuda()

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters())

    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("===> Setting Pretrained Optimizer")

    print("=> start epoch '{}'".format(epoch))
    print("===> Training")
    for epoch_ in range(epoch, opt.nEpochs + 1):
        print("===>  Start epoch {} #################################################################".format(epoch_))
        model.train()
        for data_loader in [training_data_loader1, training_data_loader2, training_data_loader3]:
            for _, (sub_ip_2, sub_ip_3, sub_ip_4, sub_la_2, sub_la_3, sub_la_4) in enumerate(data_loader):
                HR_list = [sub_la_2, sub_la_3, sub_la_4]
                LR_list = [sub_ip_2, sub_ip_3, sub_ip_4]
                for scale_ in [2,3,4]:
                    HR = Variable(HR_list[scale_-2])
                    LR = Variable(LR_list[scale_-2])
                    if torch.cuda.is_available():
                        HR = HR.cuda()
                        LR = LR.cuda()
                    # Train model
                    model.set_scale(scale_)
                    optimizer.zero_grad()
                    fake_img = model(LR)
                    loss = L1loss(fake_img, HR)
                    loss.backward()
                    optimizer.step()

            # Print Loss
            if _%5 == 0:
                print("===> Epoch[[{}]({}/{})]: L1loss : {:.5f}, PSNR : {:.5f}, SSIM : {:.5f}".format(epoch_, _*3, len(training_data_loader1)*3, loss.item(), PSNR(HR, fake_img), pytorch_ssim.ssim(HR, fake_img)))

        model_out_path = "checkpoint/" + "MASRN_epoch_{}.tar".format(epoch_)
        if not os.path.exists("checkpoint/"):
            os.makedirs("checkpoint/")
        torch.save({
                'epoch': epoch_,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, model_out_path)
        print("Checkpoint has been saved to the {}".format(model_out_path))

if __name__ == "__main__":
    main()
