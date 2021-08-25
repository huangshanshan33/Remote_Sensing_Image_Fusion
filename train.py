from __future__ import print_function

import argparse

from random import shuffle

import random

import os

import sys

import math

import tensorflow as tf

import glob

import cv2

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from image_reader import *

from net import *

parser = argparse.ArgumentParser(description='')

parser.add_argument("--snapshot_dir", default='./snapshots_LSGAN_HSV_L1_10_psnr1_0.00002_sigmoid1', help="path of snapshots")  # 保存模型的路径

parser.add_argument("--out_dir", default='./train_out_LSGAN_HSV_psnr_L1_10_psnr1_0.00002_sigmoid1', help="path of train outputs")  # 训练时保存可视化输出的路径

parser.add_argument("--image_size", type=int, default=256, help="load image size")  # 网络输入的尺度

parser.add_argument("--random_seed", type=int, default=1234, help="random seed")  # 随机数种子

parser.add_argument('--base_lr', type=float, default=0.00002, help='initial learning rate for adam')  # 学习率

parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')  # 训练的epoch数量

parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')  # adam优化器的beta1参数

parser.add_argument("--summary_pred_every", type=int, default=10000,
                    help="times to summary.")  # 训练中每过多少step保存训练日志(记录一下loss值)

parser.add_argument("--write_pred_every", type=int, default=1000, help="times to write.")  # 训练中每过多少step保存可视化结果

parser.add_argument("--save_pred_every", type=int, default=50000, help="times to save.")  # 训练中每过多少step保存模型(可训练参数)

parser.add_argument("--lamda_l1_weight", type=float, default=10.0, help="L1 lamda")  # 训练中L1_Loss前的乘数

parser.add_argument("--lamda_gan_weight", type=float, default=1.0, help="GAN lamda")  # 训练中GAN_Loss前的乘数

parser.add_argument("--train_picture_format", default='.tif',
                    help="format of training datas.")  # 网络训练输入的图片的格式(图片在CGAN中被当做条件)

parser.add_argument("--train_label_format", default='.tif',
                    help="format of training labels.")  # 网络训练输入的标签的格式(标签在CGAN中被当做真样本)
#D:/LSGAN/CGAN_1/data
parser.add_argument("--train_picture_path", default='D:/LSGAN/CGAN_1/data/low_resolution_color/',help="path of training datas.")  # 网络训练输入的图片路径

parser.add_argument("--train_label_path", default='D:/LSGAN/CGAN_1/data/high_resolution_gray/',help="path of training labels.")  # 网络训练输入的标签路径

args = parser.parse_args()  # 用来解析命令行参数

EPS = 1e-12  # EPS用于保证log函数里面的参数大于零


def save(saver, sess, logdir, step):  # 保存模型的save函数

    model_name = 'model'  # 保存的模型名前缀

    checkpoint_path = os.path.join(logdir, model_name)  # 模型的保存路径与名称

    if not os.path.exists(logdir):  # 如果路径不存在即创建

        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)  # 保存模型

    print('The checkpoint has been created.')



def cv_inv_proc(image_H,image_S,img):  # cv_inv_proc函数将读取图片时归一化的图片还原成原图
    img = (img + 1.) * 127.5
    img_YUV = np.concatenate((image_H,image_S,img), axis=-1)

    return img_YUV

def cv_inv_proc_picture(img): #cv_inv_proc函数将读取图片时归一化的图片还原成原图
    img_rgb = (img + 1.) * 127.5
    return img_rgb.astype(np.float32) #返回bgr格式的图像，方便cv2写图像


def get_write_picture(picture,label,image_H,image_S,fake_y,width,height):  # get_write_picture函数得到训练过程中的可视化结果
    picture = cv_inv_proc(image_H,image_S,picture)  #L通道
    picture = picture.astype(np.uint8)  # 得到训练中可视化结果的第一行
    picture = cv2.cvtColor(picture, cv2.COLOR_HSV2BGR)

    label = cv_inv_proc_picture(label)  #灰度图
    label = np.concatenate((label,label,label), axis=-1)

    #inv_picture_image = cv2.resize(picture, (width,height))  # 将输入图像还原到原大小
    #inv_label_image = cv2.resize(label, (width, height)) #将标签还原到原大小L通道
    print(picture.shape)
    print(label.shape)
    print(fake_y.shape)

    fake_y = cv_inv_proc(image_H,image_S,fake_y[0])  # 还原生成的y域的图像
    output = fake_y.astype(np.uint8)  # 得到训练中可视化结果的第一行
    output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
    #output = cv2.resize(output, (width, height))  # 将生成图像还原到原大小
    print(output.shape)
    output = np.concatenate((picture, output, label), axis=1)  # 拼接得到输出结果

    return output


def l1_loss(src, dst):  # 定义l1_loss

    return tf.reduce_mean(tf.abs(src - dst))
##ls loss
def gan_loss(src, dst):  # 定义gan_loss，在这里用了二范数
    return tf.reduce_mean((src - dst) ** 2)
###psnr
def psnr2(im1,im2):
    #print("im1",im1)
    # # im1 = tf.image.convert_image_dtype(im1, tf.float32)
    # # im2 = tf.image.convert_image_dtype(im2, tf.float32)
    #psnr = tf.image.psnr(im1, im2, max_val=255.0)
    # # method 1
    diff = im1 - im2
    mse = tf.reduce_mean(tf.square(diff))
    psnr = 10 * (tf.log(1./mse)/tf.log(10.))
    #psnr = 10 * tf.log(1. / mse)

    return psnr

###SRGAN
# def PSNR(real, fake):
#     mse = tf.reduce_mean(tf.square(127.5*(real-fake)+127.5), axis=(-3, -2, -1))
#     psnr = tf.reduce_mean(10 * (tf.log(255*255 / tf.sqrt(mse)) / np.log(10)))
#     return psnr



def main():  # 训练程序的主函数

    if not os.path.exists(args.snapshot_dir):  # 如果保存模型参数的文件夹不存在则创建

        os.makedirs(args.snapshot_dir)

    if not os.path.exists(args.out_dir):  # 如果保存训练中可视化输出的文件夹不存在则创建

        os.makedirs(args.out_dir)

    train_picture_list = glob.glob(os.path.join(args.train_picture_path, "*"))  # 得到训练输入图像路径名称列表

    tf.set_random_seed(args.random_seed)  # 初始一下随机数

    train_picture = tf.placeholder(tf.float32, shape=[1, args.image_size, args.image_size, 1],
                                   name='train_picture')  # 输入的训练图像

    train_label = tf.placeholder(tf.float32, shape=[1, args.image_size, args.image_size, 1],
                                 name='train_label')  # 输入的与训练图像匹配的标签
    #image = tf.add(train_label,train_picture)
    #image = np.concatenate(train_label,train_picture)
    #gen_label表示L通道的融合结果
    gen_label = generator(image1=train_label ,image2 = train_picture, gf_dim=64, reuse=False, name='generator')  # 得到生成器的输出

####判断融合图像和L通道图像
    dis_real_1 = discriminator(image=train_picture, targets=train_label, df_dim=64, reuse=False,
                             name="discriminator_1")  # 判别器返回的对真实标签的判别结果
    # train_picture 为L通道图像  train_label为灰度图

    dis_fake_1 = discriminator(image=train_picture, targets=gen_label, df_dim=64, reuse=True,
                             name="discriminator_1")  # 判别器返回的对生成(虚假的)标签判别结果
#####添加一个鉴别器用于判断融合的图像和灰度图
    dis_real_2 = discriminator(image=train_label, targets=train_picture, df_dim=64, reuse=False,
                             name="discriminator_2")  # 判别器返回的对真实标签的判别结果

    dis_fake_2 = discriminator(image=train_label, targets=gen_label, df_dim=64, reuse=True,
                             name="discriminator_2")  # 判别器返回的对生成(虚假的)标签判别结果

    #gen_loss_GAN = tf.reduce_mean(-tf.log(dis_fake_1 + EPS))  # 计算生成器损失中的GAN_loss部分，最原始的GAN的损失函数（？）
    #此处修改生成器的损失函数为最小二乘损失
    gen_loss_GAN  = gan_loss(dis_fake_1, tf.ones_like(dis_fake_1))+gan_loss(dis_fake_2, tf.ones_like(dis_fake_2))
    gen_loss_L1 = 0.7*tf.reduce_mean(l1_loss(gen_label, train_label))+ 0.3*tf.reduce_mean(l1_loss(gen_label, train_picture))  # 计算生成器损失中的L1_loss部分，
    # 计算融合部分和灰度图之间的损失

    ###PSNR
    psnr_loss_1 = psnr2(gen_label, train_label)
    psnr_loss_2 = psnr2(gen_label, train_picture)
    psnr_loss = 0.7 * psnr_loss_1 + 0.3 * psnr_loss_2

    gen_loss = gen_loss_GAN * args.lamda_gan_weight + gen_loss_L1 * args.lamda_l1_weight +0.0001*psnr_loss # 计算生成器的loss

    d_loss_real = gan_loss(dis_real_1, tf.ones_like(dis_real_1)) \
                  + gan_loss(dis_real_2, tf.ones_like(dis_real_2)) # 计算判别器判别的真实灰度图像的loss
    d_loss_fusion = gan_loss(dis_fake_1, tf.zeros_like(dis_fake_1))+ \
                    gan_loss(dis_fake_2, tf.zeros_like(dis_fake_2))  # 计算判别器判别的生成融合图像的loss
    dis_loss = d_loss_real+d_loss_fusion
    #dis_loss = tf.reduce_mean(-(tf.log(dis_real_1 + EPS) + tf.log(1 - dis_fake_1 + EPS)))  # 计算判别器的loss

    gen_loss_sum = tf.summary.scalar("gen_loss", gen_loss)  # 记录生成器loss的日志

    dis_loss_sum = tf.summary.scalar("dis_loss", dis_loss)  # 记录判别器loss的日志

    summary_writer = tf.summary.FileWriter(args.snapshot_dir, graph=tf.get_default_graph())  # 日志记录器

    g_vars = [v for v in tf.trainable_variables() if 'generator' in v.name]  # 所有生成器的可训练参数

    d_vars = [v for v in tf.trainable_variables() if 'discriminator' in v.name]  # 所有判别器的可训练参数

    d_optim = tf.train.AdamOptimizer(args.base_lr, beta1=args.beta1)  # 判别器训练器

    g_optim = tf.train.AdamOptimizer(args.base_lr, beta1=args.beta1)  # 生成器训练器

    d_grads_and_vars = d_optim.compute_gradients(dis_loss, var_list=d_vars)  # 计算判别器参数梯度

    d_train = d_optim.apply_gradients(d_grads_and_vars)  # 更新判别器参数

    g_grads_and_vars = g_optim.compute_gradients(gen_loss, var_list=g_vars)  # 计算生成器参数梯度

    g_train = g_optim.apply_gradients(g_grads_and_vars)  # 更新生成器参数

    train_op = tf.group(d_train, g_train)  # train_op表示了参数更新操作

    config = tf.ConfigProto()

    config.gpu_options.allow_growth = True  # 设定显存不超量使用

    sess = tf.Session(config=config)  # 新建会话层

    init = tf.global_variables_initializer()  # 参数初始化器

    sess.run(init)  # 初始化所有可训练参数

    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=50)  # 模型保存器

    # # 加载模型，覆盖之前的参数，从上次训练的参数结果开始
    # # 先判断下有无模型
    #
    # # ------------------------------------------------------------------
    # # 模型的保存的路径
    # ckptpath = "./snapshots_LSGAN_HSV_L1_100_psnr_skip_0.00002/"
    # # 获得模型
    # ckpt = tf.train.get_checkpoint_state(ckptpath)
    # start = 0
    # # 判断是否要加载并且是否存在训练好的模型
    # if os.path.isfile(os.path.join(ckptpath, 'checkpoint')):
    #     # 重新加载模型
    #     # 读取最后一个模型的路径
    #     ckpt = tf.train.get_checkpoint_state(ckptpath)
    #     # 加载最后一个模型的数据
    #     saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
    #     saver.restore(sess, ckpt.model_checkpoint_path)  # 加载模型
    #     print("加载最后一个模型")
    #     model_path = ckpt.model_checkpoint_path
    #     # 获得最后一个模型的数字
    #     start = int(model_path.replace("./snapshots_LSGAN_HSV_L1_100_psnr_skip_0.00002/model", ""))
    #     print(start)
    # else:
    #     # 变量初始化
    #     sess.run(tf.global_variables_initializer())
    #     print("最后一个模型不存在！")
    #
    # # ------------------------------------------------------------------

    counter = 0  # counter记录训练步数

    for epoch in range(args.epoch):  # 训练epoch数

        shuffle(train_picture_list)  # 每训练一个epoch，就打乱一下输入的顺序

        for step in range(len(train_picture_list)):  # 每个训练epoch中的训练step数

            counter += 1

            picture_name, _ = os.path.splitext(os.path.basename(train_picture_list[step]))  # 获取不包含路径和格式的输入图片名称

            # 读取一张训练图片，一张训练标签，以及相应的高和宽

            picture_resize, label_resize, picture_height, picture_width, image_H, image_S = ImageReader(file_name=picture_name,
                                                                                      picture_path=args.train_picture_path,
                                                                                      label_path=args.train_label_path,
                                                                                      picture_format=args.train_picture_format,
                                                                                      label_format=args.train_label_format,
                                                                                      size=args.image_size)

            batch_picture = np.expand_dims(np.array(picture_resize).astype(np.float32), axis=0)  # 填充维度

            batch_label = np.expand_dims(np.array(label_resize).astype(np.float32), axis=0)  # 填充维度

            feed_dict = {train_picture: batch_picture, train_label: batch_label}  # 构造feed_dict

            gen_loss_value, dis_loss_value, _ = sess.run([gen_loss, dis_loss, train_op],
                                                         feed_dict=feed_dict)  # 得到每个step中的生成器和判别器loss

            if counter % args.save_pred_every == 0:  # 每过save_pred_every次保存模型

                save(saver, sess, args.snapshot_dir, counter)

            if counter % args.summary_pred_every == 0:  # 每过summary_pred_every次保存训练日志

                gen_loss_sum_value, discriminator_sum_value = sess.run([gen_loss_sum, dis_loss_sum],
                                                                       feed_dict=feed_dict)

                summary_writer.add_summary(gen_loss_sum_value, counter)

                summary_writer.add_summary(discriminator_sum_value, counter)

            if counter % args.write_pred_every == 0:  # 每过write_pred_every次写一下训练的可视化结果

                gen_label_value = sess.run(gen_label, feed_dict=feed_dict)  # run出生成器的输出
#picture,label,fake_y,image_U,image_V,width,height
                write_image = get_write_picture(picture_resize, label_resize,  image_H,image_S,gen_label_value,picture_width,picture_height)  # 得到训练的可视化结果

                write_image_name = args.out_dir + "/out" + str(counter) + ".png"  # 待保存的训练可视化结果路径与名称
                # write_image_name1 = args.out_dir + "/out_color" + str(counter) + ".png"  # 待保存的训练可视化结果路径与名称
                # write_image_name2 = args.out_dir + "/out_gray" + str(counter) + ".png"  # 待保存的训练可视化结果路径与名称

                cv2.imwrite(write_image_name, write_image)  # 保存训练的可视化结果
                # cv2.imwrite(write_image_name1, write_image1)  # 保存训练的可视化结果
                # cv2.imwrite(write_image_name2, write_image2)  # 保存训练的可视化结果

            print('epoch {:d} step {:d} \t gen_loss = {:.3f}, dis_loss = {:.3f}'.format(epoch, step, gen_loss_value,
                                                                                        dis_loss_value))


if __name__ == '__main__':
    main()
