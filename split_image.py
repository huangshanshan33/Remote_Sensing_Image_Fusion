import numpy as np
import cv2
import os.path
import glob

def split_image(img_path,output_path):
    #img_path = "D:/hss/cGAN/train_out1/out100.png"
    img = cv2.imread(img_path)
    img1 = img[0:256,0:256]
    img2 = img[0:256,256:512]
    img3 = img[0:256,512:768]
    outdir1 = os.path.join(output_path, os.path.basename(img_path))
    outdir2 = outdir1.split('.')
    cv2.imwrite(outdir2[0] + ".png", img1)

for jpgfile in glob.glob("D:/hss/cGAN/train_out1/*.png"):  # 对原图这个文件夹下的图片进行掩膜处理
    split_image(jpgfile, "D:/hss/cGAN/train_out1_color")  # 将掩膜后的图保存在原图mask文件夹下

