from __future__ import print_function, division
import os
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import copy
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from scipy import misc
import numpy as np
import random
from random import randint
from SomeISP_operator_python.ISP_implement_fuction import ISP
from utils.denoising_utils import *


def create_augmentations(np_image):
    """
    convention: original, left, upside-down, right, rot1, rot2, rot3
    :param np_image:
    :return:
    """
    dtype = torch.cuda.FloatTensor
    aug = [np_image.copy(), np.rot90(np_image, 1, (1, 2)).copy(),
           np.rot90(np_image, 2, (1, 2)).copy(), np.rot90(np_image, 3, (1, 2)).copy()]
    flipped = np_image[:,::-1, :].copy()
    aug += [flipped.copy(), np.rot90(flipped, 1, (1, 2)).copy(), np.rot90(flipped, 2, (1, 2)).copy(), np.rot90(flipped, 3, (1, 2)).copy()]
    # aug_torch = [np_to_torch(np_image.copy()).type(dtype), np_to_torch(np.rot90(np_image, 1, (1, 2)).copy()).type(dtype),
    #              np_to_torch(np.rot90(np_image, 2, (1, 2)).copy()).type(dtype), np_to_torch(np.rot90(np_image, 3, (1, 2)).copy()).type(dtype)]
    # aug_torch += [np_to_torch(flipped.copy()).type(dtype), np_to_torch(np.rot90(flipped, 1, (1, 2)).copy()).type(dtype),
    #               np_to_torch(np.rot90(flipped, 2, (1, 2)).copy()).type(dtype), np_to_torch(np.rot90(flipped, 3, (1, 2)).copy()).type(dtype)]

    # return aug, aug_torch
    return aug


def create_augmentations_for_not_square(np_image):
    """
    convention: original, left, upside-down, right, rot1, rot2, rot3
    :param np_image:
    :return:
    """
    dtype = torch.cuda.FloatTensor
    flipped = np.rot90(np_image, 2, (1, 2)).copy()

    aug = [np_image.copy(), flipped.copy(),
           np.fliplr(np_image).copy(), np.fliplr(flipped).copy()]
    aug_torch = [np_to_torch(np_image.copy()).type(dtype), np_to_torch(flipped.copy()).type(dtype),
                 np_to_torch(np.fliplr(np_image).copy()).type(dtype),
                 np_to_torch(np.flipud(np_image).copy()).type(dtype)]

    return aug, aug_torch


def generate_reallike_noise_from_rgb(img_np_rgb, sigma_s='RAN', sigma_c='RAN'):
    # follow the pipline
    isp = ISP()
    gt, noise, sigma_s_r, sigma_c_r = isp.cbdnet_noise_generate_srgb(img_np_rgb.transpose(1,2,0), sigma_s, sigma_c)

    return gt.transpose(2, 0, 1), noise.transpose(2, 0, 1), sigma_s_r, sigma_c_r


def generate_pg_noise_level(sigma_s='RAN', sigma_c='RAN'):
    min_log = np.log([0.0001])
    if sigma_s == 'RAN':
        # sigma_s = min_log + np.random.rand(1) * (np.log([0.16]) - min_log)
        # sigma_s = min_log + np.random.rand(1) * (np.log([0.1]) - min_log)
        sigma_s = min_log + np.random.rand(1) * (np.log([0.16]) - min_log)

        sigma_s = np.exp(sigma_s)

    if sigma_c == 'RAN':
        # sigma_c = min_log + np.random.rand(1) * (np.log([0.06]) - min_log)
        sigma_c = min_log + np.random.rand(1) * (np.log([0.1]) - min_log)

        sigma_c = np.exp(sigma_c)

    # add noise
    #print('Generated Noise: sigma_s='+str(sigma_s*255)+' sigma_c='+str(sigma_c*255))
    return sigma_s, sigma_c


class MyDataset(Dataset):
    def __init__(self, gt_dir, img_dir, sigma,fixed_shape, transform=None, multilevel_noise=False, TEST=False):
        # set opt
        if TEST:
            multilevel_noise = False
        if multilevel_noise:
            min_log = np.log([0.0001])
            sigma_s = min_log + np.random.rand(1) * (np.log([sigma*2]) - min_log)
            sigma_s = np.exp(sigma_s)
            self.noisy_np = np.random.normal(0.0, 1.0, size= fixed_shape)*sigma_s
        else:
            self.noisy_np = np.random.normal(0.0, 1.0, size= fixed_shape)*sigma

        self.transform = transform
        self.fixed_shape = fixed_shape
        self.VGG_MEAN = [123.68, 116.78, 103.94]
        self.img_dir = img_dir # create a path '/path/to/data/trainA'
        self.gt_dir = gt_dir
        self.img_paths = []
        self.img_label =[]
        files = os.listdir(self.img_dir)
        for img_path in files:
            if img_path.endswith('jpg') or img_path.endswith('png'):
                self.img_paths.append(img_path)

        self.img_label = self.img_paths
        self.train_labels = copy.copy(self.img_paths)
        self.img_size = len(self.img_paths)  # get the size of dataset A


    def __getitem__(self, index):
        img_path = self.img_paths[index % self.img_size]  # make sure index is within then range
        # img_pil = Image.open(os.path.join(self.img_dir,img_path)).resize((self.fixed_shape[1], self.fixed_shape[2]))
        img_pil_ = Image.open(os.path.join(self.img_dir, img_path)).convert('RGB')
        gt_pil_ = Image.open(os.path.join(self.gt_dir, img_path)).convert('RGB')
        if img_pil_.width > 180 and img_pil_.height >180:
            w_star = randint(0, img_pil_.width-self.fixed_shape[1])
            h_star = randint(0, img_pil_.height-self.fixed_shape[2])
            img_pil = img_pil_.crop((w_star, h_star, w_star+self.fixed_shape[1], h_star+self.fixed_shape[2]))
            gt_pil = gt_pil_.crop((w_star, h_star, w_star+self.fixed_shape[1], h_star+self.fixed_shape[2]))

        else:
            img_pil = img_pil_.resize((self.fixed_shape[1], self.fixed_shape[2]))
            gt_pil = gt_pil_.resize((self.fixed_shape[1], self.fixed_shape[2]))


        img_np = pil_to_np(img_pil)
        gt_np = pil_to_np(gt_pil)
        # img_np = img_np.transpose(2,0,1).astype(np.float32)
        # img_torch = torch_to_np(img_np.astype(np.float32))
        img_aug = create_augmentations(img_np)
        gt_aug = create_augmentations(gt_np)
        rand_index = random.randint(0, 7)
        img = img_aug[rand_index]
        label = gt_aug[rand_index]
        # camera pipeline noise
        # noisey_img, noisey_noisey_img, sigma_s_r, sigma_c_r = generate_reallike_noise_from_rgb(img)

        # normal gassuian and poisson noise
        noisey_img = img
        sigma_g, sigma_p = generate_pg_noise_level(sigma_s='RAN', sigma_c='RAN')
        _, noisey_noisey_img = get_pg_noisy_image(img_np, sigma_p, sigma_g)
        # noisey_img = img + self.noisy_np
        # noisey_noisey_img = noisey_img + self.noisy_np
        # label_aug = create_augmentations(img_np)
        # label = label_aug[rand_index]
        output = (noisey_noisey_img, noisey_img, label)

        return output

    def __len__(self):
        return self.img_size

