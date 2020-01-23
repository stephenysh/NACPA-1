# -*- coding:utf-8 -*-
import os
import glob
import cv2
import numpy as np
from random import randint

if __name__ == '__main__':
    data_dir = os.path.abspath("../test_sets/Urban100/")
    save_path = os.path.abspath("../test_sets/Urban100_20/")
    sigma = 20
    # fixed_shape = [256, 256]
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for image_file in glob.glob(os.path.join(data_dir, "*.jpg")):
        # img_dir = os.path.join(data_dir, image_file)
        img_dir = image_file
        print img_dir
        img = cv2.imread(img_dir)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # if img.shape[0] > fixed_shape[0] and img.shape[1] > fixed_shape[1]:
        #     w_star = randint(0, img.shape[0]-fixed_shape[0])
        #     h_star = randint(0, img.shape[1]-fixed_shape[1])
        #     img = img[w_star:w_star+fixed_shape[0],h_star: h_star+fixed_shape[1]]
        # else:
        #     img = cv2.resize(img, fixed_shape)
        img_name = img_dir.split('/')
        img_name = img_name[-1]
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(os.path.join(gray_dir, img_name), img)
        noise_np = sigma*np.random.normal(0.0, 1.0, size= img.shape)
        noisy_img = np.clip(img.astype('float32') + noise_np, 0, 255)
        # cv2.imwrite(os.path.join('../test_sets/Urban100_gray/', img_name), img)
        img_name = os.path.join(save_path, img_name)
        cv2.imwrite(img_name, noisy_img)

