# -*- coding:utf-8 -*-
"""
Generate TFRecords file, for training.
1. Generate the list file, include VOC2007 and VOC2012 dataset. The list file format is:
path/image file.
"""
import os
import tensorflow as tf

if __name__ == '__main__':
    # data dir.
    data_VOC0712 = "../train_val_gray"
    data_VOC0712_Quality10 = "./train_val_20"
    sigma = 20/255.
    # TFRecordWriter, dump to tfrecords file
    writer = tf.python_io.TFRecordWriter(os.path.join("./bsd400_256_gray_20.tfrecords"))

    with open("./train_val_gray_list.txt", "r") as fo:
        # image_files = fo.readlines() # return list
        for line in fo:
            line = line.strip() # 去掉\n, 空格! Necessary! String.
            # line = line.split() # not need
            # line[2:], e.g. is 'VOC2012/2010_005111.jpg'
            names = line.split('/')
            image_VOC0712 = str(os.path.join(data_VOC0712, names[-1]))
            image_VOC0712_Quality10 = str(os.path.join(data_VOC0712_Quality10, names[-1]))
            print(image_VOC0712)
            print(image_VOC0712_Quality10)

            # Load image.
            image_clean = tf.gfile.FastGFile(image_VOC0712, 'rb').read()  
            # image data type is string. read and binary.
            image_noisy = tf.gfile.FastGFile(image_VOC0712_Quality10, 'rb').read()  

            # bytes write to Example proto buffer.
            example = tf.train.Example(features=tf.train.Features(feature={
                "image_clean": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_clean])),
                "image_noisy": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_noisy]))
                }))

            writer.write(example.SerializeToString()) # serialize to string.

    fo.close()
    writer.close()
