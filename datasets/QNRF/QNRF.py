import glob

import numpy as np
import os
import random
from scipy import io as sio
import sys
import torch
from torch.utils import data
from PIL import Image, ImageOps
import tqdm
import cv2
import pandas as pd


class QNRF(data.Dataset):
    def __init__(self, data_path, mode='train', preload=False, main_transform=None, img_transform=None,
                 gt_transform=None):
        self.img_path = data_path
        self.gt_path = data_path
        # self.data_files = [filename for filename in os.listdir(self.img_path) \
        #                    if os.path.isfile(os.path.join(self.img_path, filename))]
        self.data_files = glob.glob(self.img_path + '/*.jpg')
        self.num_samples = len(self.data_files)
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform

        self.preload = preload
        if self.preload:
            self.names, self.imgs, self.dens = self.XWJ_read_image_and_gt_preload(self.data_files, self.gt_path, mode)

    def __getitem__(self, index):
        # fname = self.data_files[index]
        # img, den = self.read_image_and_gt(fname)
        # if self.main_transform is not None:
        #     img, den = self.main_transform(img, den)
        # if self.img_transform is not None:
        #     img = self.img_transform(img)
        # if self.gt_transform is not None:
        #     den = self.gt_transform(den)
        # return img, den

        name = ''
        if not self.preload:
            fname = self.data_files[index]
            name, img, den = self.XWJ_read_image_and_gt(fname)

        if self.preload:
            name, img, den = self.names[index], self.imgs[index], self.dens[index]

        if self.main_transform is not None:
            img, den = self.main_transform(img, den)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            den = self.gt_transform(den)

        return img, den

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self, fname):
        img = Image.open(os.path.join(self.img_path, fname))
        if img.mode == 'L':
            img = img.convert('RGB')

        # den = sio.loadmat(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.mat'))
        # den = den['map']
        den = pd.read_csv(os.path.join(self.gt_path, os.path.splitext(fname)[0] + '.csv'), sep=',', header=None).values

        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)
        return img, den

    def get_num_samples(self):
        return self.num_samples

    def XWJ_read_image_and_gt(self, fname, preload=False):

        img_temp = cv2.imread(os.path.join(self.img_path, fname))
        img_temp = img_temp[:, :, ::-1].copy()
        img = Image.fromarray(img_temp)
        # img = Image.open(os.path.join(self.img_path, fname))
        # if img.mode == 'L':
        # img = img.convert('RGB')
        imgname = fname.split('/')[-1]

        gtfile = self.gt_path + '/' + imgname.replace('.jpg', '.npy')
        den = np.load(gtfile)
        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)
        return imgname, img, den

    def XWJ_read_image_and_gt_preload(self, data_files, gtdir, mode):
        imgs = []
        dens = []
        names = []
        print('loading %s data into ram....' % mode)
        for file in tqdm(data_files):
            img_temp = cv2.imread(file)
            img_temp = img_temp[:, :, ::-1].copy()

            img = Image.fromarray(img_temp)

            # img = Image.open(file)
            # if img.mode == 'L':
            # img = img.convert('RGB')
            imgname = file.split('/')[-1]
            gtfile = gtdir + '/' + imgname.replace('.jpg', '.npy')
            den = np.load(gtfile)
            den = den.astype(np.float32, copy=False)
            den = Image.fromarray(den)
            names.append(imgname)
            imgs.append(img)
            dens.append(den)

        return names, imgs, dens
