# -*- coding:utf-8 -*-
# coding=utf-8
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
# from skimage.transform import resize
import numpy as np
import cv2

# 膨胀算法 Kernel
_DILATE_KERNEL = np.array([[0, 0, 1, 0, 0],
                           [0, 0, 1, 0, 0],
                           [1, 1, 1, 1, 1],
                           [0, 0, 1, 0, 0],
                           [0, 0, 1, 0, 0]], dtype=np.uint8)

class WatermarkRemover(object):
    """"
    去除图片中的水印(Remove Watermark)
    """
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.watermark_template_gray_img = None
        self.watermark_template_mask_img = None
        self.watermark_template_h = 0
        self.watermark_template_w = 0
        self.watermark_start_x = 0
        self.watermark_start_y = 0

    # 加载水印模板，以便后面批量处理去除水印
    def load_watermark_template(self, watermark_template_filename):
        self.generate_template_gray_and_mask(watermark_template_filename)

    # 对图片进行膨胀计算
    def dilate(self, img):
        dilated = cv2.dilate(img, _DILATE_KERNEL)
        return dilated

    # 处理水印模板，生成对应的检索位图和掩码位图
    def generate_template_gray_and_mask(self, watermark_template_filename):
        img = cv2.imread(watermark_template_filename)  # 水印模板原图
        # 灰度图、掩码图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        mask = self.dilate(mask)  # 使得掩码膨胀一圈，以免留下边缘没有被修复
        # mask = self.dilate(mask)  # 使得掩码膨胀一圈，以免留下边缘没有被修复

        # 水印模板原图去除非文字部分
        img = cv2.bitwise_and(img, img, mask=mask)

        # 后面修图时需要用到三个通道
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        self.watermark_template_gray_img = gray
        self.watermark_template_mask_img = mask

        self.watermark_template_h = img.shape[0]
        self.watermark_template_w = img.shape[1]

        cv2.imwrite('ww-gray.jpg', gray)
        cv2.imwrite('ww-mask.jpg', mask)

        return gray, mask

    # 从原图中寻找水印位置
    def find_watermark(self, filename):
        # Load the images in gray scale
        gray_img = cv2.imread(filename, 0)
        return self.find_watermark_from_gray(gray_img, self.watermark_template_gray_img)

    # 从灰度图中寻找水印位置
    def find_watermark_from_gray(self, gray_img, watermark_template_gray_img):
        # Load the images in gray scale
        method = cv2.TM_CCOEFF
        # Apply template Matching
        res = cv2.matchTemplate(gray_img, watermark_template_gray_img, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            x, y = min_loc
        else:
            x, y = max_loc

        return x, y, x + self.watermark_template_w, y + self.watermark_template_h

    # 去除图片中的水印
    def remove_watermark_raw(self, img, gray_mask):
        """
        :param img: 待去除水印图片位图
        :param watermark_template_gray_img: 水印模板的灰度图片位图，用于确定水印位置
        :param watermark_template_mask_img: 水印模板的掩码图片位图，用于修复原始图片
        :return: 去除水印后的图片位图
        """
        self.watermark_template_gray_img, self.watermark_template_mask_img = gray_mask

        # 寻找水印位置
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = self.find_watermark_from_gray(img_gray, self.watermark_template_gray_img)  # 水印模板的掩码图片
        self.watermark_start_x = x1
        self.watermark_start_y = y1
        # 制作原图的水印位置遮板
        mask = np.zeros(img.shape, np.uint8)
        # watermark_template_mask_img = cv2.cvtColor(watermark_template_gray_img, cv2.COLOR_GRAY2BGR)
        # mask[y1:y1 + self.watermark_template_h, x1:x1 + self.watermark_template_w] = watermark_template_mask_img
        mask[y1:y2, x1:x2] = self.watermark_template_mask_img
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # 用遮板进行图片修复，使用 TELEA 算法
        dst = cv2.inpaint(img, mask, 4, cv2.INPAINT_TELEA)
        # cv2.imwrite('dst.jpg', dst)

        return dst

    # 【不可直接调用】： 去除图片中的水印
    def remove_watermark(self, filename, output_filename=None):
        """
        :param filename: 待去除水印图片文件名称
        :param output_filename: 去除水印图片后的输出文件名称
        :return: 去除水印后的图片位图
        """
        # 读取原图
        img = cv2.imread(filename)
        dst = self.remove_watermark_raw(
            img, (self.watermark_template_gray_img, self.watermark_template_mask_img)
        )
        if output_filename is not None:
            cv2.imwrite(output_filename, dst)

        return dst


def get_loc(img_size, wm_size, mode='rb'):
    print(img_size, wm_size)
    x, y = img_size
    w, h = wm_size
    rst = (0, 0)
    if mode == 'rb':
        rst = (x - w, y - h)
    elif mode == 'md':  # 正中
        rst = ((x-w)//2, (y-h)//2)
    return rst


def wm1(img_src, wm_src, dest="out", loc=(50, 50), alpha=0.25):  # scale=5,
    fig = plt.figure()
    # watermark = plt.imread(wm_src)  # 读取水印
    watermark = np.array(plt.imread(wm_src))  # 读取水印
    # 调整水印大小
    # new_size = [int(watermark.shape[0]/scale), int(watermark.shape[1]/scale), watermark.shape[2]]
    # watermark = resize(watermark, new_size, mode='constant')
    watermark[:, :, -1] *= alpha  # 调整水印透明度
    plt.imshow(plt.imread(img_src))  # 读取图像

    plt.figimage(watermark, loc[0], loc[1], zorder=10)  # 添加水印
    plt.axis('off')  # 隐藏坐标轴
    plt.savefig(dest + "/wm1_rst.jpg", dpi=fig.dpi, bbox_inches='tight')  # 保存图像
    fig.show()
    return fig


def wm2(src, logo, out="out"):
    img = cv2.imread(src)
    logo = cv2.imread(logo)
    logo = cv2.resize(logo, (350, 50))  # (100, 717)

    inew = img - img
    locp = get_loc(img.shape[:2], logo.shape[:2], mode='md')  # 位置，大小
    inew[locp[0]: locp[0] + logo.shape[0], locp[1]:locp[1] + logo.shape[1]] = logo

    inew = cv2.addWeighted(img, 1, inew, 0.4, 0)  # m1 x alph + m2 x beta + 1
    cv2.imshow("hi", inew)
    cv2.waitKey()
    savep = out + '/wm-out.jpg'
    cv2.imwrite(savep, inew)


def i_water_maker(img):
    logo = cv2.imread("logo.jpg")
    # logo = cv2.resize(logo, (350, 50))  # (100, 717)
    inew = img - img
    locp = get_loc(img.shape[:2], logo.shape[:2], mode='md')  # 位置，大小
    inew[locp[0]: locp[0] + logo.shape[0], locp[1]:locp[1] + logo.shape[1]] = logo  # 赋值（粘贴）
    inew = cv2.addWeighted(img, 1, inew, 0.4, 0)  # 加权
    return inew


def wm3(src, logo):
    im = Image.open(src)
    mark = Image.open(logo)
    layer = Image.new('RGBA', im.size, (0, 0, 0, 0))
    layer.paste(mark, get_loc(im.size, mark.size, mode="md"))
    out = Image.composite(layer, im, layer)
    out.show()


if __name__ == '__main__':

    src = 'test_dewm/raw.png'
    logo = 'logos/jd_logo_.png'

    # wm1(src, logo)
    # wm2(src, logo)
    # wm3(src, logo)

    # """
    wr = WatermarkRemover()
    wr.remove_watermark_raw(img="wm-out.jpg", gray_mask=wr.generate_template_gray_and_mask("logos/jd-logo.jpg"))

    # """




