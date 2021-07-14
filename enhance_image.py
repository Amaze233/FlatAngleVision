"""
用于数据集增强的方法集合
使用时注意源数据集的备份
新图片将直接覆盖原图片
"""
__author__ = "洪易郅"
__status__ = "production"
__version__ = "0.0.3"
__date__ = "2021.7.14"

import glob
import os
import tensorflow as tf
import matplotlib as mat
import cv2
from PIL import Image
import random
from random import choice
import numpy as np
import copy

# target_path1 = '1.1.png'
# target_path2 = '1.2.png'
# target_path3 = '1.3.png'
image_list = []
jpeg_filenames = []
raw_pic_path = r'VOC2012\JPEGImages'
mask_pic_path = r'VOC2012\SegmentationObject'
global file_number
file_number = 0


# 随机增噪
def image_salt(image):
    image2salt = copy.deepcopy(image)
    repeat = int(image2salt.shape[0] * image2salt.shape[1] / 25)  # 噪点数
    maxwidth = max(image2salt.shape[0], image2salt.shape[1])
    for counter1 in range(repeat):
        # print(counter1 / repeat)
        black_white_salt = choice((0, 1))  # 1 for white, 0 for black
        i = random.randint(0, maxwidth) % image2salt.shape[0]
        j = random.randint(0, maxwidth) % image2salt.shape[1]
        if black_white_salt:
            image2salt[i, j] = 255
            image2salt[i, j] = 255
            image2salt[i, j] = 255
        else:
            image2salt[i, j] = 0
            image2salt[i, j] = 0
            image2salt[i, j] = 0
    return image2salt
    # cv2.imwrite(target_path1, image2salt)


# 图片旋转， mask对应改变
def image_rotate(image1, mask_image):
    image2rotate = image1
    rows, cols = image2rotate.shape[:2]
    rotate_core = (cols / 2, rows / 2)  # 旋转中心
    rotate_angle = [60, -60, 45, -45, 90, -90, 210, 240, -210, -240]
    paras = cv2.getRotationMatrix2D(rotate_core, choice(rotate_angle), 1)
    border_value = tuple(int(x) for x in choice(choice(image2rotate)))
    image_new = cv2.warpAffine(image2rotate, paras, (cols, rows), borderValue=border_value)
    mask_new = cv2.warpAffine(mask_image, paras, (cols, rows), borderValue=border_value)
    return image_new, mask_new
    # cv2.imwrite(target_path2, img_new)


# 图片平移,mask对应改变
def image_translation(image1, mask_image):
    image2translation = image1
    paras_wide = [[1, 0, 100], [1, 0, -100]]
    paras_height = [[0, 1, 100], [0, 1, -100]]
    rows, cols = image2translation.shape[:2]
    img_shift = np.float32([choice(paras_wide), choice(paras_height)])
    border_value = tuple(int(x) for x in choice(choice(image2translation)))
    image_new = cv2.warpAffine(image2translation, img_shift, (cols, rows), borderValue=border_value)
    mask_new = cv2.warpAffine(mask_image, img_shift, (cols, rows), borderValue=border_value)
    return image_new, mask_new
    # cv2.imwrite(target_path3, img_new)


# 对传入的图片列表做随机扩充
def image_enhance(image_list):
    counter = 0
    for image2enhance in image_list:
        print(counter / file_number)
        jpeg_filename = jpeg_filenames[counter]
        flag = choice((0, 1, 2))
        if flag == 0:
            cv2.imwrite(jpeg_filename, image_salt(image2enhance))
            counter += 1
        elif flag == 1:

            filename_tup = os.path.split(jpeg_filename)
            png_filename = filename_tup[1].replace('.jpg', '.png')
            mask_path = os.path.join(mask_pic_path, png_filename)
            mask2change = cv2.imread(mask_path)

            jpg_png = image_translation(image2enhance, mask2change)
            cv2.imwrite(jpeg_filename, jpg_png[0])
            cv2.imwrite(png_filename, jpg_png[1])

            counter += 1
        elif flag == 2:

            filename_tup = os.path.split(jpeg_filename)
            png_filename = filename_tup[1].replace('.jpg', '.png')
            mask_path = os.path.join(mask_pic_path, png_filename)
            mask2change = cv2.imread(mask_path)

            jpg_png = image_rotate(image2enhance, mask2change)
            cv2.imwrite(jpeg_filename, jpg_png[0])
            cv2.imwrite(png_filename, jpg_png[1])

            counter += 1


# def image_glob(raw_pic_path, jpeg_filenames):
#     jpeg_filenames = glob.glob(raw_pic_path)
#     for file_name in jpeg_filenames:
#         file = cv2.imread(file_name)
#         image_list.append(file)


def get_file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                global file_number
                file_number += 1
                jpeg_filenames.append(os.path.join(root, file))
                image2get = cv2.imread(os.path.join(root, file))
                image_list.append(image2get)


# img = cv2.imread('1.png')
# image_list.append(img)
# image_salt(image_list[0])
# image_rotate(image_list[0])
# image_translation(image_list[0])
get_file_name(raw_pic_path)
image_enhance(image_list)