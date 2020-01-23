#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import matplotlib.pyplot as plt
import os
import numpy as np
from models import *

import torch
import torch.optim
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
import tqdm
from examples.oracle_mydata import MyDataset
from utils.denoising_utils import *
from PIL import Image
from pathlib import Path
from real_world_noise_estimator import real_world_noise_estimator
np.random.seed(30)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor


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
    aug_torch = [np_to_torch(np_image.copy()).type(dtype), np_to_torch(np.rot90(np_image, 1, (1, 2)).copy()).type(dtype),
                 np_to_torch(np.rot90(np_image, 2, (1, 2)).copy()).type(dtype), np_to_torch(np.rot90(np_image, 3, (1, 2)).copy()).type(dtype)]
    aug_torch += [np_to_torch(flipped.copy()).type(dtype), np_to_torch(np.rot90(flipped, 1, (1, 2)).copy()).type(dtype),
                  np_to_torch(np.rot90(flipped, 2, (1, 2)).copy()).type(dtype), np_to_torch(np.rot90(flipped, 3, (1, 2)).copy()).type(dtype)]

    return aug, aug_torch


def save_image(name, image_np, output_path="./results/"):
    p = np_to_pil(image_np)
    p.save(output_path + "{}.png".format(name))


AUG_MEAN = True
# test
aug_num=8
sigma = 0
model_dir = './test_models/nac_offline_rcan_sidd_trainset_with_vdn/model.npz'
fixed_shape = [3, 256, 256]
TEST_ON_GRAY = False
test_img_root = '/raid/huangyuan/sidd/sidd_benchmarknoisyblockssrgb_png/'
dataset_name = 'sidd_benchmark'
output_dir = './test_output/test_results_rcan_nacpa_sidd_val/'
full_output_dir = Path(output_dir).resolve()

if not os.path.isdir(full_output_dir):
    os.mkdir(full_output_dir)

net = Rcan(3, n_resgroups=10, n_resblocks=10, n_feats=64, reduction=16)

s = sum([np.prod(list(p.size())) for p in net.parameters()]);
print('Number of params: %d' % s)

test_set = MyDataset(test_img_root, sigma, fixed_shape, multilevel_noise=False, TEST=True)

# load model
checkpoint = torch.load(model_dir)
net.load_state_dict(checkpoint['model_state_dict'])

net.eval()
with torch.no_grad():
    average_psnr = 0
    average_ssim = 0
    for idx_, img_name in enumerate(test_set.img_paths):
        print("NO. %d image %s"%(idx_, img_name))
        test_img_pil = Image.open(os.path.join(test_img_root, img_name)).convert('RGB')
        test_img_np = pil_to_np(test_img_pil)
        noise = real_world_noise_estimator(test_img_np.transpose(1, 2, 0))[0]
        noise2 = np.rot90(noise, 1, (1, 2))
        fake_img = test_img_np - noise

        test_img_aug, _ = create_augmentations(fake_img)
        test_out = []
        for idx, test_img_aug_ in enumerate(test_img_aug):
            if idx%2==0:
                test_noisy_img_torch = np_to_torch(test_img_aug_+noise).type(dtype)
            elif idx%2==1:
                test_noisy_img_torch = np_to_torch(test_img_aug_+noise2).type(dtype)

            out_effect_np_ = torch_to_np(net(test_noisy_img_torch))
            test_out.append(out_effect_np_)
        test_out[0] = test_out[0].transpose(1, 2, 0)
        for aug in range(1, 8):
            if aug < 4:
                test_out[aug] = np.rot90(test_out[aug].transpose(1, 2, 0), 4 - aug)
            else:
                test_out[aug] = np.flipud(np.rot90(test_out[aug].transpose(1, 2, 0), 8 - aug))

        if AUG_MEAN:
           final_reuslt = np.mean(test_out, 0)
        else:
            final_reuslt = test_out[0]

        tmp_name_p = img_name[:-4]
        save_image(tmp_name_p, np.clip(final_reuslt.transpose(2, 0, 1), 0, 1), output_path=output_dir)
        # print('%s saved'%(img_name))

print('I am finish testing now!')


