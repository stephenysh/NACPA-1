# -*- coding:utf-8 -*-
"""
An implementation of acGAN using TensorFlow (work in progress).
"""

import tensorflow as tf
import numpy as np
from model import MemNet_rgb
import os
import glob
import cv2



def create_augmentations(np_image):
    """
    convention: original, left, upside-down, right, rot1, rot2, rot3
    :param np_image:
    :return:
    """
    aug = [np_image.copy(), np.rot90(np_image, 1, (1, 2)).copy(),
           np.rot90(np_image, 2, (1, 2)).copy(), np.rot90(np_image, 3, (1, 2)).copy()]
    flipped = np_image[:,::-1, :].copy()
    aug += [flipped.copy(), np.rot90(flipped, 1, (1, 2)).copy(), np.rot90(flipped, 2, (1, 2)).copy(), np.rot90(flipped, 3, (1, 2)).copy()]

    return aug


def main(_):
    tf_flags = tf.app.flags.FLAGS
    # gpu config.
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    if tf_flags.phase == "train":
        with tf.Session(config=config) as sess: 
            # sess = tf.Session(config=config) # when use queue to load data, not use with to define sess
            train_model = MemNet_rgb.MemNet(sess, tf_flags)
            train_model.train(tf_flags.training_steps, tf_flags.summary_steps, 
                tf_flags.checkpoint_steps, tf_flags.save_steps)
    else:
        with tf.Session(config=config) as sess:
            test_model = MemNet_rgb.MemNet(sess, tf_flags)
            test_model.load(tf_flags.checkpoint)
            # test Set12
            # get psnr and ssim outside
            sigma =10/255.
            save_path = "./test_output/memnet_10_rgb_no_augs_urban100/"
            aug_mean = False
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for image_file in glob.glob(tf_flags.testing_set):
                print("testing {}...".format(image_file))
                # testing_set is path/*.jpg.
                # c_image = np.reshape(cv2.resize(cv2.imread(image_file, 0), (tf_flags.img_size, tf_flags.img_size)),
                #     (1, tf_flags.img_size, tf_flags.img_size, 1)) / 255.
                # c_image = np.reshape(cv2.resize(cv2.imread(image_file, 0), (256, 256)), (1, 256, 256, 1)) / 255.
                if aug_mean:
                    c_image = cv2.imread(image_file)/255.
                    c_image = c_image.transpose(2, 0, 1)
                    noise = np.random.normal(0.0, 1.0, size=c_image.shape)*sigma
                    noise2 = np.rot90(noise, 1, (1, 2))
                    c_image_aug = create_augmentations(c_image)
                    test_out = []
                    for aug_, c_image_aug_ in enumerate(c_image_aug):
                        if aug_%2==0:
                            c_image_aug_ = np.clip(c_image_aug_ + noise, 0, 1).transpose(1, 2, 0)
                            c_image_aug_ = c_image_aug_[np.newaxis, :, :, :]
                        elif aug_%2==1:
                            c_image_aug_ = np.clip(c_image_aug_ + noise2, 0, 1).transpose(1, 2, 0)
                            c_image_aug_ = c_image_aug_[np.newaxis, :, :, :]
                        recovery_image_ = test_model.test(c_image_aug_)
                        test_out.append(recovery_image_[0])
                    test_out[0] = test_out[0]
                    for aug in range(1, 8):
                        if aug < 4:
                            test_out[aug] = np.rot90(test_out[aug], 4 - aug)
                        else:
                            test_out[aug] = np.flipud(np.rot90(test_out[aug], 8 - aug))
                    recovery_image = np.mean(test_out, 0)
                    cv2.imwrite(os.path.join(save_path, image_file.split("/")[3]), np.uint8(recovery_image.clip(0., 1.) * 255.))
                else:
                    c_image = cv2.imread(image_file)/255.
                    noise = np.random.normal(0.0, 1.0, size=c_image.shape) * sigma
                    c_image = np.clip(c_image+noise,0, 1)
                    c_image = c_image[np.newaxis,:,:,:]

                    # In Caffe, Tensorflow, might must divide 255.?
                    recovery_image = test_model.test(c_image)

                    # save image
                    cv2.imwrite(os.path.join(save_path, image_file.split("/")[3]),
                        np.uint8(recovery_image[0, :].clip(0., 1.) * 255.))
                # recovery_image[0, :], 3D array.
            print("Testing done.")


if __name__ == '__main__':
    tf.app.flags.DEFINE_string("output_dir", "model_output", 
                               "checkpoint and summary directory.")
    tf.app.flags.DEFINE_string("phase", "train", 
                               "model phase: train/test.")
    tf.app.flags.DEFINE_string("training_set", "", 
                               "dataset path for training.")
    tf.app.flags.DEFINE_string("testing_set", "", 
                               "dataset path for testing.")
    tf.app.flags.DEFINE_integer("img_size", 256, 
                                "testing image size.")
    tf.app.flags.DEFINE_integer("batch_size", 64, 
                                "batch size for training.")
    tf.app.flags.DEFINE_integer("training_steps", 100000, 
                                "total training steps.")
    tf.app.flags.DEFINE_integer("summary_steps", 100, 
                                "summary period.")
    tf.app.flags.DEFINE_integer("checkpoint_steps", 1000, 
                                "checkpoint period.")
    tf.app.flags.DEFINE_integer("save_steps", 100, 
                                "checkpoint period.")
    tf.app.flags.DEFINE_string("checkpoint", None, 
                                "checkpoint name for restoring.")
    tf.app.run(main=main)