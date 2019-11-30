import argparse, os
import glob
import h5py
import cv2
from PIL import Image
import numpy as np

# AUGMENT SETTINGS
parser = argparse.ArgumentParser(description="PyTorch MASRN")
parser.add_argument("--HRpath", type=str, default='data/DIV2K_train_HR')
parser.add_argument("--Savepath", type=str, default='data/train_x234.h5')
parser.add_argument("--Cropsize", type=int, default=48)
parser.add_argument("--Cropnum", type=int, default=100)

def data_aug():
    global opt
    opt = parser.parse_args()
    print(opt)
    sub_ip = []
    sub_la = []
    num = 1
    HRpath = load_img(opt.HRpath)
    for _ in HRpath:
        HR_img = read_img(_)
        sub_image = random_crop(HR_img, opt.Cropnum, opt.Cropsize)
        input = transfer_input(sub_image)
        label_x2 = img_downsize(sub_image, 2)
        label_x3 = img_downsize(sub_image, 3)
        label_x4 = img_downsize(sub_image, 4)
        sub_ip += input
        sub_la_2 += label_x2
        sub_la_3 += label_x3
        sub_la_4 += label_x4
        print('data no.',num)
        num += 1
    sub_ip = np.asarray(sub_ip)
    sub_la_2 = np.asarray(sub_la_2)
    sub_la_3 = np.asarray(sub_la_3)
    sub_la_4 = np.asarray(sub_la_4)
    print('input shape : ',sub_ip.shape)
    print('label shape : x2[',sub_la_2.shape,'], x3[',sub_la_3.shape,'], x4['sub_la_4.shape,']')
    save_h5(sub_ip, sub_la_2, sub_la_3, sub_la_4, opt.Savepath)
    print('---------save---------')

def load_img(file_path):
    dir_path = os.path.join(os.getcwd(), file_path)
    img_path = glob.glob(os.path.join(dir_path, '*.png'))
    return img_path

def read_img(img_path):
    # read image
    image = cv2.imread(img_path)
    return image

def mod_crop(image, scale):
    h = image.shape[0]
    w = image.shape[1]
    h = h - np.mod(h,scale)
    w = w - np.mod(w,scale)
    return image[0:h,0:w,:]

def random_crop(image, Cropnum, Cropsize):
    sub_img = []
    i = 0
    while i < Cropnum:
        h = np.random.randint(0, image.shape[0] - Cropsize)
        w = np.random.randint(0, image.shape[1] - Cropsize)
        sub_i = image[h:h+Cropsize,w:w+Cropsize]
        sub_i = mod_crop(sub_i, 4)
        sub_i = mod_crop(sub_i, 3)
        sub_img.append(sub_i)
        i += 1
    return sub_img

def transfer_input(img):
    img_list = []
    for _ in img:
        h = _.shape[0]
        w = _.shape[1]
        img_list.append(_.reshape(3, h, w))
    return img_list

def img_downsize(img, scale):
    dst_list = []
    for _ in img:
        h = _.shape[0]
        w = _.shape[1]
        dst = cv2.resize(_, dsize=(0, 0), fx=1/scale, fy=1/scale, interpolation=cv2.INTER_CUBIC)
        dst_list.append(dst.reshape(3, h/scale, w/scale))
    return dst_list

def save_h5(sub_ip, sub_la_2, sub_la_3, sub_la_4, savepath):
    path = os.path.join(os.getcwd(), savepath)
    with h5py.File(path, 'w') as hf:
        hf.create_dataset('input', data=sub_ip)
        hf.create_dataset('label_x2', data=sub_la_2)
        hf.create_dataset('label_x3', data=sub_la_3)
        hf.create_dataset('label_x4', data=sub_la_4)

if __name__ == '__main__':
    print('starting data augmentation...')
    data_aug()
