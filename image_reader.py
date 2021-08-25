import os

import numpy as np

import tensorflow as tf

import cv2


# 读取图片的函数，接收六个参数

# 输入参数分别是图片名，图片路径，标签路径，图片格式，标签格式，需要调整的尺寸大小

def ImageReader(file_name, picture_path, label_path, picture_format=".tif", label_format=".tif", size=256):
    picture_name = picture_path + file_name + picture_format  # 得到图片名称和路径

    label_name = label_path + file_name + label_format  # 得到标签名称和路径

    picture = cv2.imread(picture_name, 1)  # 读取图片

    label = cv2.imread(label_name, 1)  # 读取标签


    height = label.shape[0]  # 得到图片的高

    width = label.shape[1]  # 得到图片的宽

    picture_resize_t = cv2.resize(picture, (size, size))  # 调整图片的尺寸，改变成网络输入的大小

##YUV颜色空间
    img_YUV = cv2.cvtColor(picture_resize_t, cv2.COLOR_BGR2HSV)
    y_image_H = (img_YUV[:, :, 0:1])
    y_image_S = (img_YUV[:, :, 1:2])
    y_image_V = (img_YUV[:, :, 2:3])


    picture_resize = y_image_V / 127.5 - 1.  # 归一化图片

    label_resize_t = cv2.resize(label, (size, size))  # 调整标签的尺寸，改变成网络输入的大小
    label_resize_t = label_resize_t[:, :, 0:1]   #单通道灰度图

    label_resize = label_resize_t / 127.5 - 1.  # 归一化标签

    return picture_resize, label_resize, height, width, y_image_H, y_image_S   # 返回网络输入的图片，标签，还有原图片和标签的长宽
