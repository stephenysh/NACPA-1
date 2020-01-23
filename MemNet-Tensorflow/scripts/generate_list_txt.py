# -*- coding:utf-8 -*-
import os
import glob

if __name__ == '__main__':
    data_dir = os.path.abspath("../train_val_gray/")

    save_path = os.path.abspath("./")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file1 = open("train_val_gray_list.txt", "a")

    for image_file in glob.glob(os.path.join(data_dir, "*.jpg")):
        # img_dir = os.path.join(data_dir, image_file)
        img_dir = image_file
        print img_dir
        file1.write(img_dir+'\n')

    file1.close()