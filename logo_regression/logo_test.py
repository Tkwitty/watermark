# coding=utf-8
import cv2
import random
import numpy as np
from PIL import Image, ImageFilter
import os

def Aget_initpix(vsrc, valp, vlogo, prt=False):
    if valp == 1:
        rst = vlogo
    elif valp == 0:
        rst = vsrc
    else:
        rst = ((vsrc - valp * vlogo)) / (1 - valp)
    # 在纯背景处，希望过渡区 得到更大的像素值
    if rst < 0:
        rst = 0
    if rst > 255:
        rst = 255
    return rst

def get_logobypath(imgp, pngp):
    img = Image.open(imgp)
    image = img.copy()

    print("logo地址:", pngp)
    watermark = Image.open(pngp)  # 水印路径，加在下侧

    if watermark.mode != 'RGBA':
        alpha = Image.new('L', watermark.size, 255)  # 创建A通道  L表示8位灰度图
        watermark.putalpha(alpha)  # pil RGB 如何转 BGR

    # 获取正中位置
    bw, bh = img.size  # shape的宽高位置和pil的有区别！！！
    sw, sh = watermark.size
    iw = int((bw - sw) / 2)
    ih = int((bh - sh) / 2)

    TRANSPARENCY = 100
    paste_mask = watermark.split()[3].point(lambda i: i * TRANSPARENCY / 100.)  # 第四通道
    image.paste(watermark, (iw, ih + 60), mask=paste_mask)  # pm是wm本身第四通道的像素值
    return image, watermark, (iw, ih), paste_mask

def pix_logo_rm(src, wm, local, pmask):
    sw, sh = local
    src = np.array(src)  # 编程int类型，使得不会导致计算值溢出
    wm = np.array(wm)
    pmask = np.array(pmask)
    th, tw, tc = np.shape(src)

    timg = src.copy()  # 拷贝一个map
    h, w = pmask.shape  # 要遍历的宽高： 53, 371
    for i in range(h):
        for j in range(w):
            x, y = j+sw, i+sh  # 求操作图上的坐标
            if x < 0 or x >= tw or y < 0 or y >= th:
                continue
            for ci in range(3):
                vsrc = timg[y][x][ci]
                valp = pmask[i][j] / 255
                vlogo = wm[i][j][ci]
                # timg[y][x][ci] = int(get_initpix(vsrc, valp, vlogo))

                timg[y][x][ci] = int(Aget_initpix(vsrc, valp, vlogo, prt=True))

    # 至此，整个 timg 计算完毕
    timg_ = np.asarray(timg, dtype=np.uint8)
    ttimg = Image.fromarray(timg_).convert(mode="RGB")
    return np.array(ttimg)


def logo_clean_yy(srcp, pngp, savp):
    print("testing: ", srcp)
    ilogo = Image.open(srcp)  # 读入的图片

    # 2、采用逆向计算方式W， 先通过对原图与系统已有的logo模板，获取一些参数
    slogo, wm, (iw, ih), pmask = get_logobypath(srcp, pngp)  # 获取计算资源, 获取双水印
    dst_ = pix_logo_rm(slogo, wm, (iw, ih), pmask)  # 除原生水印  214 373
    dst = pix_logo_rm(Image.fromarray(dst_), wm, (iw, ih+60), pmask)  # 除下方60pix人工水印

    if not os.path.exists(savp):
        os.makedirs(savp)
    print("保存位置：", savp)

    # 保存结果图
    spic = srcp.rsplit('/', 1)[1].split('.')  # name + subfix

    w, h = ilogo.size
    out_image = np.zeros((h, w * 3, 3), dtype=np.uint8)
    out_image[:, :w] = ilogo  # 原图
    out_image[:, w:w * 2] = slogo  # 手工图 + logo 与原图对比
    out_image[:, w * 2:] = Image.fromarray(dst)

    # irsavep = savp + "/" + spic[0] + "_dlg." + spic[1]
    # slogo.save(irsavep)  # 保存双水印图

    dsavep = savp + "/dst/" + spic[0] + "_rm." + spic[1]
    idst = Image.fromarray(dst)
    idst.save(dsavep)

    rsavep = savp + "/" + spic[0] + "_comp." + spic[1]
    iout = Image.fromarray(out_image)
    iout.save(rsavep)
    iout.show()

def dir_logorm_test(dir, pngp):
    flist = os.listdir(dir)
    for pp in flist:
        if pp.endswith('.jpg'):
            logo_clean_yy(
                srcp=dir + pp,
                pngp=pngp,
                savp="test_out"
            )
    print("Over!!!")

if __name__ == '__main__':
    idir = "tests/"
    # dir_logorm_test(idir, pngp="mylogo_gd50w.png")
    dir_logorm_test(idir, pngp="ilogor.png")  # a微调
    # dir_logorm_test(idir, pngp="mylogo_c4gd50k_1124.png")  # 0.4 训练
    # dir_logorm_test(idir, pngp="logo_c4l2gd01w_2020.png")  # 0.4 训练


