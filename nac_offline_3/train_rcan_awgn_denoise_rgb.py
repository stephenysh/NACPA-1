#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import matplotlib.pyplot as plt
import os
import numpy as np
from models import *
from pathlib import Path
import torch
import torch.optim
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

from examples.oracle_mydata import MyDataset
from utils.denoising_utils import *
from PIL import Image
import random
np.random.seed(30)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
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


def MSE(x, y):
    return np.square(x - y).mean()


def save_image(name, image_np, output_path="./results/"):
    p = np_to_pil(image_np)
    p.save(output_path + "{}.png".format(name))


def closure():

    global i, net_input, test_input, psnr_2_max, average_psnr, average_ssim, psnr_record_iter, psnr_max_record
    global noisy_np_norm, final_ssim, final_ssim_max, sigma, test_img_root, final_ssim_max_record
    global train_average_psnr, average_loss,temp_test_average_psnr
    average_loss = 0
    psnr_noisy = 0
    for idx_train, (noisy_img, label) in enumerate(net_input):
        noisy_img_torch = noisy_img.type(dtype)
        img_torch = label.type(dtype)
        out = net(noisy_img_torch)
        total_loss = mse(out, img_torch)
        average_loss += total_loss.item()
        total_loss.backward()
        # print("%s Iteration %05d"%(files_name, i))
        psnr_noisy_ = compare_psnr(label.detach().cpu().numpy(), out.detach().cpu().numpy())
        psnr_noisy += psnr_noisy_

    average_loss = average_loss/(idx_train+1)
    psnr_noisy = psnr_noisy/(idx_train+1)
    if train_average_psnr == 0:
        train_average_psnr = psnr_noisy
    elif train_average_psnr < psnr_noisy:
        train_average_psnr = psnr_noisy
    print('Iteration %05d lr: %f, Loss %f , PSNR_noisy: %f, PSNR_noisy_max: %f, current sigma: %f ' %
          (i, LR, average_loss, psnr_noisy, train_average_psnr, sigma*255))

    if i % 10 == 0: # test every 10 epoch on the test set
        net.eval()
        with torch.no_grad():
            psnr_1 = 0
            for idx_test, (noisy_img, label) in enumerate(net_input):
                test_noisy_img_torch = noisy_img.type(dtype)
                out_effect_np = torch_to_np(net(test_noisy_img_torch))
                psnr_1_ = compare_psnr(torch_to_np(label), np.clip(out_effect_np, 0, 1))
                psnr_1 += psnr_1_
            psnr_1 = psnr_1/(idx_test+1)
            if temp_test_average_psnr == 0:
                temp_test_average_psnr = psnr_1
            elif temp_test_average_psnr < psnr_1:
                temp_test_average_psnr = psnr_1

            print('Train set test %05d lr: %f, test psnr: %f , test max psnr: %f, current sigma: %f ' %
                  (i, LR, psnr_1, temp_test_average_psnr, sigma*255))
            average_psnr = 0
            average_ssim = 0
            for idx_, img_name in enumerate(test_set.img_paths):
                test_img_pil = Image.open(os.path.join(test_img_root, img_name)).convert('RGB')
                test_img_np = pil_to_np(test_img_pil)
                noisy_np_test = np.random.normal(0.0, 1.0, size=test_img_np.shape)*sigma

                test_noisy_img_torch = np_to_torch(test_img_np + noisy_np_test).type(dtype)
                test_out = torch_to_np(net(test_noisy_img_torch))
                test_out = test_out.transpose(1, 2, 0)
                final_reuslt = test_out

                psnr_2 = compare_psnr(test_img_np.transpose(1, 2, 0), np.clip(final_reuslt, 0, 1))
                final_ssim = compare_ssim(test_img_np.transpose(1, 2, 0), np.clip(final_reuslt, 0, 1), data_range=1, multichannel=True)
                average_psnr += psnr_2
                average_ssim += final_ssim

            average_psnr = average_psnr/(idx_+1)
            average_ssim = average_ssim/(idx_+1)
            if psnr_2_max==0:
                psnr_2_max = average_psnr
            elif psnr_2_max< average_psnr:
                psnr_2_max = average_psnr

            if final_ssim_max==0:
                final_ssim_max = average_ssim
            elif final_ssim_max<average_ssim:
                final_ssim_max = average_ssim

            print('real test average psnr: %f, average psnr max: %f, real test average ssim : %f, average ssim max: %f'
                  %(average_psnr, psnr_2_max, average_ssim, final_ssim_max))

    psnr_record_iter.append(i)
    psnr_max_record.append(psnr_2_max)
    final_ssim_max_record.append(final_ssim_max)
    i += 1

    return total_loss


train_average_psnr = 0
temp_test_average_psnr = 0

psnr_2_max = 0
final_ssim_max = 0

fixed_shape = [3, 50, 50]
save_every = 1
sigma = 5/255.
noisy_np_norm = np.random.normal(0.0, 1.0, size= fixed_shape)

img_root = './data/BSDS500/data/images/train_val/'

test_img_root = './data/denoising/kodim/'

psnr_record_iter = []
psnr_max_record = []
final_ssim_max_record = []
output_dir = 'results/rcan_rgb_sigma5/'
full_output_dir = Path(output_dir).resolve()
if not os.path.isdir(full_output_dir):
    os.mkdir(full_output_dir)

pad = 'replication'  # ['zero', 'replication', 'none']
OPT_OVER = 'net'  # 'net,input'
LR = 0.0001

OPTIMIZER = 'myadam'
show_every = 1
exp_weight = 0.99
num_iter_plan = 1000

net = Rcan(3, n_resgroups=10, n_resblocks=10, n_feats=64, reduction=16)

s = sum([np.prod(list(p.size())) for p in net.parameters()]);
print('Number of params: %d' % s)
# Loss
mse = torch.nn.MSELoss().type(dtype)

train_set = MyDataset(img_root, sigma, fixed_shape, multilevel_noise=False, TEST=False)
net_input = torch.utils.data.DataLoader(train_set, batch_size=20, shuffle=True, num_workers=0)
test_set = MyDataset(test_img_root, sigma, fixed_shape, multilevel_noise=False, TEST=True)
test_input = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)

i = 0
p = get_params(OPT_OVER, net, net_input)
optimize(net,OPTIMIZER, p, closure, LR, num_iter_plan, output_dir=output_dir, interval=10)

print('I am finish training now!')


