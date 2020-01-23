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
from utils.denoising_utils import *
import random
from random import randint

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


class MyDataset(Dataset):
    def __init__(self, img_dir, img_gt, fixed_shape, transform=None, multilevel_noise=False, TEST=False):
        # set opt
        if TEST:
            multilevel_noise = False
        if multilevel_noise:
            min_log = np.log([0.0001])
            # sigma_s = min_log + np.random.rand(1) * (np.log([sigma*2]) - min_log)
            # sigma_s = np.exp(sigma_s)
            # self.noisy_np = np.random.normal(0.0, 1.0, size= fixed_shape)*sigma_s
        else:
            # self.noisy_np = np.random.normal(0.0, 1.0, size= fixed_shape)*sigma
            min_log = np.log([0.0001])

        self.transform = transform
        self.fixed_shape = fixed_shape
        self.VGG_MEAN = [123.68, 116.78, 103.94]
        self.img_dir = img_dir # create a path '/path/to/data/trainA'
        self.img_gt = img_gt
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
        img_gt_pil_ = Image.open(os.path.join(self.img_gt, img_path)).convert('RGB')
        if img_pil_.width > self.fixed_shape[1] and img_pil_.height > self.fixed_shape[2]:
            w_star = randint(0, img_pil_.width-self.fixed_shape[1])
            h_star = randint(0, img_pil_.height-self.fixed_shape[2])
            img_pil = img_pil_.crop((w_star, h_star, w_star+self.fixed_shape[1], h_star+self.fixed_shape[2]))
            img_gt_pil = img_gt_pil_.crop((w_star, h_star, w_star+self.fixed_shape[1], h_star+self.fixed_shape[2]))
        else:
            img_pil = img_pil_.resize((self.fixed_shape[1], self.fixed_shape[2]))
            img_gt_pil = img_gt_pil_.resize((self.fixed_shape[1], self.fixed_shape[2]))

        img_np = pil_to_np(img_pil)
        img_gt = pil_to_np(img_gt_pil)
        # img_np = img_np.transpose(2,0,1).astype(np.float32)
        # img_torch = torch_to_np(img_np.astype(np.float32))
        img_aug = create_augmentations(img_np)
        rand_index = random.randint(0, 7)
        img = img_aug[rand_index]
        noisey_img = img
        label_aug = create_augmentations(img_gt)
        label = label_aug[rand_index]
        output = (noisey_img, label)

        return output


    def __len__(self):
        return self.img_size

