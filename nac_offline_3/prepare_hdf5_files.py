import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from PIL import Image
from utils.denoising_utils import *
from random import randint
from tqdm import tqdm
from SomeISP_operator_python.ISP_implement_fuction import ISP
from real_world_noise_estimator import real_world_noise_estimator
from skimage.measure import compare_psnr
from matplotlib import pyplot as plt


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

    return aug


def normalize(data):
    return data/255.


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])



def prepare_data(data_path, gt_path, patch_size):
    # train
    print('process training data')
    # files = glob.glob(os.path.join(data_path, 'train', '*.png'))
    files = os.listdir(data_path)
    files.sort()
    h5f = h5py.File('train.h5', 'w')
    train_num = 0
    for i in tqdm(range(len(files))):
        img_pil_ = Image.open(os.path.join(data_path, files[i])).convert('RGB')
        ## dnd ###
        # img_name = files[i]
        # files_name_ = img_name.split('.')[0]
        # p1, p2 = files_name_.split('_')
        # gt_name = f'{p1}_{int(p2):02d}.png'

        ### sidd_val ###S
        gt_name = files[i]

        ### polyu ###
        # gt_name = files[i][:-8] + 'mean.JPG'

        gt_pil_ = Image.open(os.path.join(gt_path, gt_name)).convert('RGB')

        if img_pil_.width > patch_size and img_pil_.height > patch_size:
            w_star = randint(0, img_pil_.width - patch_size)
            h_star = randint(0, img_pil_.height - patch_size)
            img_pil = img_pil_.crop((w_star, h_star, w_star + patch_size, h_star + patch_size))
            gt_pil = gt_pil_.crop((w_star, h_star, w_star + patch_size, h_star + patch_size))

        else:
            img_pil = img_pil_.resize((patch_size, patch_size))
            gt_pil = gt_pil_.resize((patch_size, patch_size))

        img_np = pil_to_np(img_pil)
        gt_np = pil_to_np(gt_pil)

        img = img_np
        label = gt_np

        noisy_np = real_world_noise_estimator(img.transpose(1,2,0)).squeeze()

        noisey_img = img
        noisey_noisey_img = (img + noisy_np).clip(0,1)
        # p1 = compare_psnr((label+noisy_np).clip(0,1), noisey_img, data_range=1)
        # p2 = compare_psnr(label, (noisey_img-noisy_np).clip(0,1),data_range=1)
        # print('estimate p1: %.2f, p2: %.2f'%(p1,p2))

        output = (noisey_noisey_img, noisey_img, label)
        data = output
        h5f.create_dataset(str(train_num), data=data)
        train_num += 1

    h5f.close()
    print('training set, # samples %d\n' % train_num)


def prepare_test_data(data_path, gt_path):
    # val
    print('\nprocess validation data')
    # files = glob.glob(os.path.join(data_path, '*.png'))
    files = os.listdir(data_path)
    files.sort()
    h5f = h5py.File('val.h5', 'w')
    val_num = 0
    for i in tqdm(range(len(files))):
        # print("file: %s" % files[i])
        if files[i].endswith('.png') or files[i].endswith('.JPG'):
            # if img_path.endswith('JPG') or img_path.endswith('png'):
            img_pil = Image.open(os.path.join(data_path, files[i])).convert('RGB')
            ### dnd ###
            # img_name = files[i]
            # files_name_ = img_name.split('.')[0]
            # p1, p2 = files_name_.split('_')
            # gt_name = f'{p1}_{int(p2):02d}.png'

            ### sidd_val ###
            gt_name = files[i]

            ### polyu ###
            # gt_name = files[i][:-8]+'mean.JPG'

            gt_pil = Image.open(os.path.join(gt_path, gt_name)).convert('RGB')
            img = pil_to_np(img_pil)
            # img = np.float32(normalize(img))
            gt = pil_to_np(gt_pil)
            output = (img, gt)
            h5f.create_dataset(str(val_num), data=output)
            val_num += 1
    h5f.close()

    print('val set, # samples %d\n' % val_num)


class Dataset(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        self.train = train
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)
