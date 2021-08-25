import tensorflow as tf
#import skimage.measure
import os
import cv2
import numpy as np
import math
from os.path import join as pjoin
from skimage.metrics import structural_similarity #ssim
from skimage.metrics import mean_squared_error #mse
from skimage.metrics import variation_of_information  #VI
from skimage.metrics import peak_signal_noise_ratio  #psnr
from skimage.metrics import normalized_root_mse  #nrmse

path1 = 'D:/hss/cGAN/train_out_color/'
path2 = 'D:/hss/cGAN/train_out_fusion/'
path3 = 'D:/hss/cGAN/train_out_gray/'
train1 = []
train2 = []
train3 = []

def image_reader(path,train):

    for filename in os.listdir(path):  # listdir的参数是文件夹的路径
        person_dir = pjoin(path, filename)
        im = cv2.imread(person_dir)
        im = np.array(im)
        train.append(im)  # 将list添加到已有的list中
    data = np.array(train)
    return data

image_color = image_reader(path1,train1)
image_fusion = image_reader(path2,train2)
image_gray = image_reader(path3,train3)

def mse(img1,img2):
    mse = mean_squared_error(img1, img2)
    return mse

def psnr(img1, img2):
    psnr = peak_signal_noise_ratio(img1,img2)
    return psnr

def vi(img1,img2):
    vi = variation_of_information(img1,img2)
    return vi

def nrmse(img1,img2):
    nrmse = normalized_root_mse(img1,img2)
    return nrmse

def SSIM(img1,img2):
    ss = np.mean(structural_similarity(img1, img2, multichannel=True))
    return ss

def pearsonr(x, y):
    return (((x-x.mean())/(x.std(ddof=0)))*((y-y.mean())/(y.std(ddof=0)))).mean()


mse1 = []
mse2 = []
nrmse1 = []
nrmse2 = []
psnr1 = []
psnr2 = []
vi1 = []
vi2 = []
#pearson = []
ssim1 = []
ssim2 = []

for idx_pd in range(len(image_fusion)):
    ssim1.append(SSIM(image_color[idx_pd],image_fusion[idx_pd]))
    ssim2.append(SSIM(image_gray[idx_pd],image_fusion[idx_pd]))
    psnr1.append(psnr(image_color[idx_pd],image_fusion[idx_pd]))
    psnr2.append(psnr(image_gray[idx_pd], image_fusion[idx_pd]))
    # vi1.append(vi(image_color[idx_pd], image_fusion[idx_pd]))
    # vi2.append(vi(image_gray[idx_pd], image_fusion[idx_pd]))
    mse1.append(mse(image_color[idx_pd], image_fusion[idx_pd]))
    mse2.append(mse(image_gray[idx_pd], image_fusion[idx_pd]))
    nrmse1.append(nrmse(image_color[idx_pd], image_fusion[idx_pd]))
    nrmse2.append(nrmse(image_gray[idx_pd], image_fusion[idx_pd]))

print("ssim1:",ssim1)
print("ssim2:",ssim2)
ssim = (np.mean(ssim1)+np.mean(ssim2))/2
psnr = (np.mean(psnr1)+np.mean(psnr2))/2
# vi = (np.mean(vi1)+np.mean(vi2))/2
mse = (np.mean(mse1)+np.mean(mse2))/2
nrmse = (np.mean(nrmse1)+np.mean(nrmse2))/2
print("ssim:",ssim)
print("psnr1:",psnr1)
print("psnr2:",psnr2)
print("psnr:",psnr)
# print("vi1:",vi1)
# print("vi2:",vi2)
# print("vi:",vi)
print("mse1:",mse1)
print("mse2:",mse2)
print("mse:",mse)
print("nrmse1:",nrmse1)
print("nrmse2:",nrmse2)
print("nrmse:",nrmse)







