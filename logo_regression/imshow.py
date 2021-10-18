# -*- coding: utf-8 -*-
# @Time    : 2021/7/5 9:57
# @Author  : guowei.yy@cai-inc.com
# @File    : imshow.py
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.image as mpimg
from PIL import Image



def case():
    img1 = cv2.imread('{}/no.0_zlogo_c4l2gd01r_1406.png'.format(base))
    img2 = cv2.imread('{}/no.1_zlogo_c4l2gd01r_1406.png'.format(base))
    img3 = cv2.imread('{}/no.2_zlogo_c4l2gd01r_1406.png'.format(base))
    img4 = cv2.imread('{}/no.3_zlogo_c4l2gd01r_1406.png'.format(base))

    # img1 = cv2.resize(img1, (800, 800))
    # img2 = cv2.resize(img2, (800, 800))
    # img3 = cv2.resize(img3, (800, 800))
    # img4 = cv2.resize(img4, (800, 800))

    fig = plt.figure()
    # img[:, :, ::-1]是将BGR转化为RGB
    plt.subplot(221)  # 要生成1行2列，这是第一个图plt.subplot('行','列','编号')
    plt.imshow(img1[:, :, ::-1])
    plt.title('dog-1')

    plt.subplot(222)
    plt.imshow(img2[:, :, ::-1])
    plt.title('dog-2')

    plt.subplot(223)
    plt.imshow(img3[:, :, ::-1])
    plt.title('dog-3')

    plt.subplot(224)
    plt.imshow(img4[:, :, ::-1])
    plt.title('cat-1')

    plt.show()


def idemo(imlist):
    # fig = plt.figure(figsize=[12, 9])  # k=8时，九宫格
    # fig = plt.figure(figsize=[9, 6])  # k=8时，九宫格
    fig, ax = plt.subplots(nrows=3, ncols=3)

    k = len(imlist)
    axs = []
    elm = int(np.sqrt(k))  # 边长
    for i in range(k):
        axs.append(fig.add_subplot(elm, elm, i + 1))

    for i, im in enumerate(imlist):
        # _img = mpimg.imread(base + "/" + imlist[i])  # 读取
        _img = Image.open(base + "/" + imlist[i]).convert('RGB')  # 读取
        _ax = axs[i]
        tt = "{} epoches".format((i + 1)*1000)
        _ax.set_title(tt)
        _ax.get_xaxis().set_visible(False)
        _ax.get_yaxis().set_visible(False)
        _ax.imshow(_img)

    plt.show()  # 图片的显示


def plotBlack(img, watermark):
    image = img.copy()  # 深拷贝，作为返回图像
    alpha = Image.new('L', watermark.size, 255)  # 根据水印size创建A通道, L表示8位灰度图
    if watermark.mode != 'RGBA':  # 如果水印图没有a通道，则加上纯1的a通道
        watermark = watermark.convert('RGB')  # 先转rgb
        watermark.putalpha(alpha)  # pil RGB 如何转 BGR

    # RGBA四通道图片能否调用paste函数.
    # if image.mode == 'RGBA':  # 如果水印图没有a通道，则加上纯1的a通道
    #     image = image.convert('RGB')  # 根据水印size创建A通道, L表示8位灰度图

    iw, ih = 0, 0
    a = 1  # 0.3
    paste_mask = watermark.split()[3].point(lambda i: i * a)  # 第四通道
    image.paste(watermark, (iw, ih), mask=paste_mask)  # pm是wm本身第四通道的像素值
    return image


def getShow(lgp):
    bg = Image.open('bgim.png')
    ibg = bg.resize((291, 87), Image.BICUBIC)
    wm = Image.open(lgp)
    rst = plotBlack(ibg, wm)
    # rst.show()
    return rst


def img_cvt():
    nsv = './zcy_show'
    for im in os.listdir(base):
        if im.startswith('no.'):
            imgp = '{}/{}'.format(base, im)
            simg = getShow(imgp).convert('RGB')
            simg.save('{}/{}'.format(nsv, im))
    print('over.')





if __name__ == '__main__':
    # base = 'D:/_Git_/watermark/logo_regression/trlog_1950'
    # base = 'D:/_Git_/watermark/logo_regression/zcy_show/rer'
    base = 'D:/_Git_/watermark/logo_regression/zcy_show/tr12'
    # base = 'D:/_Git_/watermark/logo_regression/zcy_show'
    # nsv = './zcy_show/rer'
    bimgs = os.listdir(base)
    # idemo(bimgs[:9])

    # getShow('zcy_logo.png')
    # img_cvt()

    # simg = getShow('zcy_logo.png').convert('RGB')
    simg = getShow('demo.jpg')  # .convert('RGB')
    # simg.show()
    simg.save('./demo_rst.png')
    print('over.')

    # 2,9,8,7,6,5,4,3




