# coding=utf-8
import cv2
import numpy as np
from PIL import Image
import random
import os

def get_src_logo_img(img):  # 传入pil-img
    image = img.copy()
    watermark = Image.open("opc/370x52.png")  # 水印路径
    TRANSPARENCY = 100
    # image = Image.fromarray(img)

    if watermark.mode != 'RGBA':
        alpha = Image.new('L', watermark.size, 255)  # 创建A通道  L表示8位灰度图
        watermark.putalpha(alpha)  # pil RGB 如何转 BGR

    # 获取正中位置
    # bh, bw, _ = img.shape  # shape的宽高位置和pil的有区别！！！
    bw, bh = img.size  # shape的宽高位置和pil的有区别！！！
    sw, sh = watermark.size
    iw = int((bw - sw) / 2)
    ih = int((bh - sh) / 2)
    print(bw, bh, sw, sh, "==>", iw, ih)
    paste_mask = watermark.split()[3].point(lambda i: i * TRANSPARENCY / 100.)  # 第四通道

    # wr, wg, wb, wa = watermark.split()
    # iwatermark = Image.merge('RGBA', (wg, wb, wr, wa))  # paste之前，水印红蓝反色

    image.paste(watermark, (iw, ih), mask=paste_mask)
    return image


def save_sameshape_logo_mask_pic(img, watermark):
    img = Image.open(img)
    watermark = Image.open(watermark)

    # """  # 准备空白，粘贴水印图，并转换为 dip的蒙版图
    bh, bw, c = np.array(img).shape
    sw, sh = watermark.size
    iw = int((bw - sw) / 2)
    ih = int((bh - sh) / 2)

    blk = Image.fromarray(np.zeros((bh, bw, c+1), dtype=np.uint8)).convert(mode="RGBA")
    _, bg, bb, ba = blk.split()  # 粘贴前先分离一下
    blk.paste(watermark, (iw, ih))  # 粘贴logo
    # 对blk作处理， 取其红色通道 , blk是img-size，而wm是logo-size, 需要在指定位置paste logo
    pr_, _, _, _ = blk.split()  # 粘贴后分离
    pr = np.array(pr_)  # 在阈值化之前，最好进行单通道的膨胀处理！

    # cv2.merge([pr, pr, pr])
    # bz_ = Image.fromarray(pr, mode='L')
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    dilated = cv2.dilate(pr, kernel)  # 膨胀图像
    pr = dilated  # 确认生成膨胀的图像
    # pr = pr  # 原图，不做膨胀

    pr[pr >= 1] = 255
    pri = Image.fromarray(pr)
    print(pri.size, bg.size)
    rblk = Image.merge("RGBA", (pri, bg, bb, pri))

    # 保存水印图 + 等分辨率 logo图
    ri = random.randint(0, 100)
    img.save("save/aha_" + str(ri) + ".png")
    rblk.save("save/aha_" + str(ri) + "_logo.png")  # , transparency=0
    # rblk.show()  # 查看蒙版

    # 以上
    return cv2.cvtColor(np.array(img).astype(np.uint8), cv2.COLOR_RGBA2BGR), \
           cv2.cvtColor(np.array(rblk).astype(np.uint8), cv2.COLOR_RGBA2GRAY)


def get_water(srcp, maskp, outp):
    # 黑底白字
    src = cv2.imread(srcp)  # 默认的彩色图(IMREAD_COLOR)方式读入原始图像
    # black.jpg
    mask = cv2.imread(maskp, cv2.IMREAD_GRAYSCALE)  # 灰度图(IMREAD_GRAYSCALE)方式读入水印蒙版图像
    # 参数：目标修复图像; 蒙版图（定位修复区域）; 选取邻域半径; 修复算法(包括INPAINT_TELEA/INPAINT_NS， 前者算法效果较好)
    # dst = cv2.inpaint(src, mask, 3, cv2.INPAINT_NS)

    dst = cv2.inpaint(src, mask, 5, cv2.INPAINT_TELEA)  # Unrecognized or unsupported array type in function 'cvGetMat'
    # cv2.imwrite(outp + '/rst1.jpg', dst)

    cv2.imshow("hi", src)
    cv2.imshow("fine", mask)
    cv2.imshow("hello", dst)
    dstp = "save/" + srcp.rsplit('/', 1)[1].split('.')[0] + "_inp.png"
    print(dstp)
    cv2.imwrite(dstp, dst)
    cv2.waitKey()


def logo_clean_show(srcp):
    tmplt = "opc/370x52.png"
    img, imask = save_sameshape_logo_mask_pic(srcp, tmplt)  # 根据任意原图 和一张logo模板，生成该图的对应size的蒙版(cv2)

    # dst = cv2.inpaint(img, imask, 3, cv2.INPAINT_TELEA)  # src, mask, 邻域半径, 修复方式
    # dst = cv2.inpaint(img, imask, 3, cv2.INPAINT_NS)  # src, mask, 邻域半径, 修复方式
    dst = cv2.inpaint(img, imask, 1, cv2.INPAINT_NS)  # src, mask, 邻域半径, 修复方式
    # dst = cv2.inpaint(img, imask, 5, cv2.INPAINT_TELEA)  # src, mask, 邻域半径, 修复方式
    # dst = cv2.inpaint(img, imask, 7, cv2.INPAINT_TELEA)  # src, mask, 邻域半径, 修复方式
    # dst = cv2.inpaint(img, imask, 9, cv2.INPAINT_TELEA)  # src, mask, 邻域半径, 修复方式
    # dst = cv2.inpaint(img, imask, 11, cv2.INPAINT_TELEA)  # src, mask, 邻域半径, 修复方式
    # dst = cv2.inpaint(img, imask, 15, cv2.INPAINT_TELEA)  # src, mask, 邻域半径, 修复方式

    # cv2.imshow("hi", img)
    # cv2.imshow("fine", imask)
    # cv2.imshow("hello", dst)
    # cv2.waitKey()

    spic = srcp.rsplit('/', 1)[1].split('.')  # name + subfix
    savep = "JD_out/" + spic[0] + "_rw." + spic[1]
    print(savep)
    cv2.imwrite(savep, dst)

    # idst = Image.fromarray(dst, mode="RGB")
    # idst.save(savep)

    # 保存 三位一体 对比图
    simg = Image.open(srcp)  # 原图
    slogo = get_src_logo_img(simg)  # jd输入
    w, h = simg.size
    out_image = np.zeros((h, w * 3, 3), dtype=np.uint8)
    out_image[:, :w] = simg
    out_image[:, w:w * 2] = slogo
    # out_image[:, w * 2:] = Image.fromarray(dst)  # 红蓝色反
    b, g, r = cv2.split(dst)
    idst = cv2.merge([r, g, b])
    out_image[:, w * 2:] = idst  # 红蓝反色
    rsavep = "JD_rst/" + spic[0] + "_comp." + spic[1]
    Image.fromarray(out_image).save(rsavep)


if __name__ == '__main__':
    # srcp = "clean_test/aha_13.png"
    # mb = "clean_test/aha_13_logo.png"
    # get_water(srcp, mb, "test_dewm")
    # srcp = "JD/jd_clean.png"

    flist = os.listdir("JD")
    for pp in flist:
        logo_clean_show("JD/" + pp)

    print("Over")




"""
需要生成水印图对应的 蒙版size图，再进行inpaint，重要参数：
logo膨胀参数， inpaint邻域半径， inpaint方式[INPAINT_TELEA/NS]
"""

