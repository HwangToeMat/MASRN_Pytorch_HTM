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
parser.add_argument("--LRsize", type=int, default=48)
parser.add_argument("--Cropnum", type=int, default=20)

def data_aug():
    global opt
    opt = parser.parse_args()
    print(opt)
    sub_ip_2 = []
    sub_la_2 = []
    sub_ip_3 = []
    sub_la_3 = []
    sub_ip_4 = []
    sub_la_4 = []
    num = 1
    HRpath = load_img(opt.HRpath)
    for _ in HRpath:
        HR_img = read_img(_)
        sub_image = random_crop(HR_img, opt.Cropnum, opt.LRsize * 2, 2)
        input, label = img_downsize(sub_image, 2)
        sub_ip_2 += input
        sub_la_2 += label
        sub_image = random_crop(HR_img, opt.Cropnum, opt.LRsize * 3, 3)
        input, label = img_downsize(sub_image, 3)
        sub_ip_3 += input
        sub_la_3 += label
        sub_image = random_crop(HR_img, opt.Cropnum, opt.LRsize * 4, 4)
        input, label = img_downsize(sub_image, 4)
        sub_ip_4 += input
        sub_la_4 += label
        print('data no.',num)
        num += 1
    sub_ip_2 = np.asarray(sub_ip_2)
    sub_ip_3 = np.asarray(sub_ip_3)
    sub_ip_4 = np.asarray(sub_ip_4)
    sub_la_2 = np.asarray(sub_la_2)
    sub_la_3 = np.asarray(sub_la_3)
    sub_la_4 = np.asarray(sub_la_4)
    print('input shape : x2[',sub_ip_2.shape,'], x3[',sub_ip_3.shape,'], x4[',sub_ip_4.shape,']')
    print('label shape : x2[',sub_la_2.shape,'], x3[',sub_la_3.shape,'], x4[',sub_la_4.shape,']')
    save_h5(sub_ip_2, sub_ip_3, sub_ip_4, sub_la_2, sub_la_3, sub_la_4, opt.Savepath)
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

def random_crop(image, Cropnum, Cropsize, scale):
    sub_img = []
    i = 0
    while i < Cropnum:
        h = np.random.randint(0, image.shape[0] - Cropsize)
        w = np.random.randint(0, image.shape[1] - Cropsize)
        sub_i = image[h:h+Cropsize,w:w+Cropsize]
        sub_i = mod_crop(sub_i, scale)
        sub_img.append(sub_i)
        i += 1
    return sub_img

def img_downsize(img, scale):
    dst_list = []
    img_list = []
    for _ in img:
        h = _.shape[0]
        w = _.shape[1]
        img_list.append(_.reshape(3, h, w))
        dst = cv2.resize(_, dsize=(0, 0), fx=1/scale, fy=1/scale, interpolation=cv2.INTER_CUBIC)
        dst_list.append(dst.reshape(3, int(h/scale), int(w/scale)))
    return dst_list, img_list

def save_h5(sub_ip_2, sub_ip_3, sub_ip_4, sub_la_2, sub_la_3, sub_la_4, savepath):
    path = os.path.join(os.getcwd(), savepath)
    with h5py.File(path, 'w') as hf:
        hf.create_dataset('input_x2', data=sub_ip_2)
        hf.create_dataset('input_x3', data=sub_ip_3)
        hf.create_dataset('input_x4', data=sub_ip_4)
        hf.create_dataset('label_x2', data=sub_la_2)
        hf.create_dataset('label_x3', data=sub_la_3)
        hf.create_dataset('label_x4', data=sub_la_4)

if __name__ == '__main__':
    print('starting data augmentation...')
    data_aug()
