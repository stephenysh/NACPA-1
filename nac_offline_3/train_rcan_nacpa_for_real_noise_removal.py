#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import os
import numpy as np
from models import *

import torch
import torch.optim
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
from pathlib import Path

import tqdm
from prepare_hdf5_files import *
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
    best_epoch = 0

    idx_train = 0
    for batch_train in tqdm(net_input):
        noisey_noisey_img = batch_train[:,0,:,:,:]
        noisey_img = batch_train[:, 1, :, :, :]
        noisey_noisey_img_torch = noisey_noisey_img.type(dtype)
        out = net(noisey_noisey_img_torch)
        noisey_img_torch = noisey_img.type(dtype)
        total_loss = mse(out, noisey_img_torch)
        average_loss += total_loss.item()
        total_loss.backward()
        psrn_noisy_ = compare_psnr(np.clip(torch_to_np(noisey_img), 0, 1), out.detach().cpu().numpy()[0])
        psnr_noisy += psrn_noisy_
        idx_train += 1

    average_loss = average_loss/(idx_train)
    psnr_noisy = psnr_noisy/(idx_train)
    if train_average_psnr == 0:
        train_average_psnr = psnr_noisy
    elif train_average_psnr < psnr_noisy:
        train_average_psnr = psnr_noisy
    print('Iteration %05d lr: %f, Loss %f , PSNR_noisy: %f, PSNR_noisy_max: %f, current sigma: %f ' %
          (i, LR, average_loss, psnr_noisy, train_average_psnr, sigma*255))

    if i % 10 == 0:
        net.eval()
        with torch.no_grad():

            average_psnr = 0
            average_ssim = 0
            idx_ = 0
            # for idx_, img_name in enumerate(test_set.img_paths):
            for b_t in tqdm(test_input):
                test_img_np = b_t[:, 0, :, :, :].detach().cpu().numpy()[0]
                gt = b_t[:, 1, :, :, :].detach().cpu().numpy()[0]

                test_img_aug, _ = create_augmentations(test_img_np)
                # test_img_aug = create_augmentations(test_img_np)
                test_out = []
                for idx, test_img_aug_ in enumerate(test_img_aug):
                    test_noisy_img_torch = np_to_torch(test_img_aug_).type(dtype)
                    out_effect_np_ = torch_to_np(net(test_noisy_img_torch))
                    test_out.append(out_effect_np_)

                test_out[0] = test_out[0].transpose(1, 2, 0)
                test_out[1] = (np.rot90(test_out[1], 2, (1, 2))).transpose(1, 2, 0)
                test_out[2] = (np.fliplr(test_out[2])).transpose(1, 2, 0)
                test_out[3] = (np.fliplr(np.rot90(test_out[3], 2, (1, 2)))).transpose(1, 2, 0)
                # final_reuslt = np.median(out_effect_np, 0)
                final_reuslt = np.mean(test_out, 0)

                psnr_2 = compare_psnr(gt.transpose(1, 2, 0), np.clip(final_reuslt, 0, 1))
                final_ssim = compare_ssim(gt.transpose(1, 2, 0), np.clip(final_reuslt, 0, 1), data_range=1, multichannel=True)
                average_psnr += psnr_2
                average_ssim += final_ssim
                idx_ += 1

            average_psnr = average_psnr/(idx_)
            average_ssim = average_ssim/(idx_)
            if psnr_2_max==0:
                psnr_2_max = average_psnr
                best_epoch = i
            elif psnr_2_max< average_psnr:
                psnr_2_max = average_psnr
                best_epoch = i

            if final_ssim_max==0:
                final_ssim_max = average_ssim
            elif final_ssim_max<average_ssim:
                final_ssim_max = average_ssim

            print('real test average psnr: %f, average psnr max: %f, real test average ssim : %f, average ssim max: %f,best epoch: %d'
                  %(average_psnr, psnr_2_max, average_ssim, final_ssim_max, best_epoch))

    psnr_record_iter.append(i)
    psnr_max_record.append(psnr_2_max)
    final_ssim_max_record.append(final_ssim_max)
    i += 1

    return total_loss


train_average_psnr = 0
temp_test_average_psnr = 0

psnr_2_max = 0
final_ssim_max = 0

fixed_shape = [3, 64, 64]
imsize = -1
SAVE_DURING_TRAINING = True
save_every = 1
sigma = 20/255.
noisy_np_norm = np.random.normal(0.0, 1.0, size= fixed_shape)

img_root = '/raid/huangyuan/sidd/sidd_small/noisy_5000/'
gt_root = '/raid/huangyuan/sidd/sidd_small/gt_5000/'

test_img_root = './data/4rwn_test/'
test_gt_root = './data/4rwn_gt/'

PREPARE_FLAG = True
CONTINUE_TRAINING = False
model_dir = ''
psnr_record_iter = []
psnr_max_record = []
final_ssim_max_record = []
output_dir = 'results/nac_offline_rcan_4rwn_sidd_trainset_with_vdn/'
full_output_dir = Path(output_dir).resolve()
if not os.path.isdir(full_output_dir):
    os.mkdir(full_output_dir)

pad = 'replication'  # ['zero', 'replication', 'none']
OPT_OVER = 'net'  # 'net,input'
LR = 0.0001

OPTIMIZER = 'myadam'  # 'LBFGS'
show_every = 1
exp_weight = 0.99
num_iter_plan = 3000
figsize = 4

net = Rcan(3, n_resgroups=10, n_resblocks=10, n_feats=64, reduction=16)

if CONTINUE_TRAINING:
    checkpoint = torch.load(model_dir)
    net.load_state_dict(checkpoint['model_state_dict'])
    print("Loading from file %s"% model_dir)

s = sum([np.prod(list(p.size())) for p in net.parameters()]);
print('Number of params: %d' % s)
            # Loss
mse = torch.nn.MSELoss().type(dtype)

################# loading data #######################
# add_noise 2
if PREPARE_FLAG:
    prepare_data(img_root,gt_root, 64)
    prepare_test_data(test_img_root, test_gt_root)
    train_set = Dataset(train=True)
    test_set = Dataset(train=False)
    net_input = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True, num_workers=0)
    test_input = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
else:
    train_set = Dataset(train=True)
    test_set = Dataset(train=False)
    net_input = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True, num_workers=0)
    test_input = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
#####################################

i = 0
p = get_params(OPT_OVER, net, net_input)
optimize(net,OPTIMIZER, p, closure, LR, num_iter_plan, output_dir=output_dir, interval=10)

print('I am finish training now!')


