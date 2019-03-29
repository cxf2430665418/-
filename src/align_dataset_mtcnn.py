"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# python align_dataset_mtcnn.py \input_file \output_file --image_size 160 --margin 32 --random_order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import cv2
from skimage import draw
import matplotlib.pyplot as plt
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import detect_face
import random
from time import sleep


def main(args):
    sleep(random.random())
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 在日志目录的文本文件中存储一些Git修订信息 
    src_path, _ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = facenet.get_dataset(args.input_dir)

    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    minsize = 20  # 最小面尺寸
    threshold = [0.6, 0.7, 0.7]  # 三步门槛 
    factor = 0.709  # 比例因子

    # 在文件名中添加随机键以允许使用多个进程进行对齐
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        if args.random_order:
            random.shuffle(dataset)  # 随机排列数据集
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                if args.random_order:
                    random.shuffle(cls.image_paths)
            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename + '.png')
                print(image_path)   # 遍历一遍输入图片
                if not os.path.exists(output_filename):
                    try:
                        img = misc.imread(image_path)  # 将图片读取出来为array类型，即numpy类型
                        cxf = cv2.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim < 2:  # 数组维度
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                        img = img[:, :, 0:3]


                        bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                        # 第二个返回值没用到，所以用"，_"
                        # print(bounding_boxes)
                        print(points)
                        nrof_faces = bounding_boxes.shape[0]
                        # .shape[0]，数组最高维度的维数
                        print(nrof_faces)
                        if nrof_faces > 0:
                            det = bounding_boxes[:, 0:4]
                            img_size = np.asarray(img.shape)[0:2]
                            if nrof_faces > 1:
                                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                img_center = img_size / 2
                                offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)
                                # some extra weight on the centering
                                det = det[index, :]
                            det = np.squeeze(det)
                            # 降维 从数组的形状中删除单维条目，即把shape中为1的维度去掉
                            bb = np.zeros(4, dtype=np.int32)
                            # 生成4个整形元素的零矩阵
                            # print(det[0], det[1], det[2], det[3])
                            bb[0] = np.maximum(det[0] - args.margin / 2, 0)
                            bb[1] = np.maximum(det[1] - args.margin / 2, 0)
                            bb[2] = np.minimum(det[2] + args.margin / 2, img_size[1])
                            bb[3] = np.minimum(det[3] + args.margin / 2, img_size[0])
                            # print(bb)
                            # cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]

                            # scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')

                            rr, cc = draw.circle(points[5, 0], points[0, 0], 10)
                            draw.set_color(cxf, [rr, cc], [0, 255, 0])
                            rr, cc = draw.circle(points[6, 0], points[1, 0], 10)
                            draw.set_color(cxf, [rr, cc], [0, 255, 0])
                            rr, cc = draw.circle(points[7, 0], points[2, 0], 10)
                            draw.set_color(cxf, [rr, cc], [0, 255, 0])
                            rr, cc = draw.circle(points[8, 0], points[3, 0], 10)
                            draw.set_color(cxf, [rr, cc], [0, 255, 0])
                            rr, cc = draw.circle(points[9, 0], points[4, 0], 10)
                            draw.set_color(cxf, [rr, cc], [0, 255, 0])
                            plt.imshow(cxf)
                            plt.show()
                            cv2.rectangle(cxf, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 3)
                            cv2.imwrite(output_filename, cxf)
                            nrof_successfully_aligned += 1
                            # misc.imsave(output_filename, scaled)
                            text_file.write('%s %d %d %d %d\n' % (output_filename, bb[0], bb[1], bb[2], bb[3]))
                        else:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order',
                        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
