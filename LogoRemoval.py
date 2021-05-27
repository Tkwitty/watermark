# coding=utf-8
import cv2
import numpy as np
from PIL import Image, ImageFilter
import random
import os

def get_src_logo_img(img):  # 传入pil-img
    image = img.copy()
    watermark = Image.open("opc/370x52.png")  # 水印路径
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

    TRANSPARENCY = 100
    paste_mask = watermark.split()[3].point(lambda i: i * TRANSPARENCY / 100.)  # 第四通道

    # wr, wg, wb, wa = watermark.split()
    # iwatermark = Image.merge('RGBA', (wg, wb, wr, wa))  # paste之前，水印红蓝反色

    print("在下面一点的地方添加人工 logo")
    image.paste(watermark, (iw, ih + 60), mask=paste_mask)
    return image


def get_imask4dr(img, watermark, dradius):
    img = Image.open(img)
    watermark = Image.open(watermark)
    bh, bw, c = np.array(img).shape
    sw, sh = watermark.size
    iw = int((bw - sw) / 2)
    ih = int((bh - sh) / 2)
    blk = Image.fromarray(np.zeros((bh, bw, 4), dtype=np.uint8)).convert(mode="RGBA")  # 准备空白，粘贴水印图，并转换为 dip的蒙版图
    _, bg, bb, ba = blk.split()  # 粘贴前先分离一下
    blk.paste(watermark, (iw, ih))  # 粘贴logo
    # 对blk作处理， 取其红色通道 , blk是img-size，而wm是logo-size, 需要在指定位置paste logo
    pr_, _, _, _ = blk.split()  # 粘贴后分离
    pr = np.array(pr_)  # 在阈值化之前，最好进行单通道的膨胀处理！

    if dradius != 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dradius, dradius))  # 膨胀半径
        dilated = cv2.dilate(pr, kernel)  # 膨胀图像
        pr = dilated  # 确认生成膨胀的图像

    pr[pr >= 1] = 255
    pri = Image.fromarray(pr)
    rblk = Image.merge("RGBA", (pri, bg, bb, pri))
    return cv2.cvtColor(np.array(rblk).astype(np.uint8), cv2.COLOR_RGBA2GRAY)


def save_sameshape_logo_mask_pic(img, watermark, dradius):
    img = Image.open(img)
    watermark = Image.open(watermark)
    # """  # 准备空白，粘贴水印图，并转换为 dip的蒙版图
    bh, bw, c = np.array(img).shape
    # print(bh, bw, c)
    sw, sh = watermark.size
    iw = int((bw - sw) / 2)
    ih = int((bh - sh) / 2)
    blk = Image.fromarray(np.zeros((bh, bw, 4), dtype=np.uint8)).convert(mode="RGBA")
    # blk = Image.fromarray(np.zeros((bh, bw, c+1), dtype=np.uint8)).convert(mode="RGBA")
    _, bg, bb, ba = blk.split()  # 粘贴前先分离一下
    blk.paste(watermark, (iw, ih))  # 粘贴logo
    # 对blk作处理， 取其红色通道 , blk是img-size，而wm是logo-size, 需要在指定位置paste logo
    pr_, _, _, _ = blk.split()  # 粘贴后分离
    pr = np.array(pr_)  # 在阈值化之前，最好进行单通道的膨胀处理！

    # cv2.merge([pr, pr, pr])
    # bz_ = Image.fromarray(pr, mode='L')
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dradius, dradius))  # 膨胀半径
    dilated = cv2.dilate(pr, kernel)  # 膨胀图像
    pr = dilated  # 确认生成膨胀的图像
    pr = pr  # 原图，不做膨胀

    pr[pr >= 1] = 255
    pri = Image.fromarray(pr)
    print(pri.size, bg.size)
    rblk = Image.merge("RGBA", (pri, bg, bb, pri))

    # 保存水印图 + 等分辨率 logo图
    ri = random.randint(0, 100)
    img.save("save/aha_" + str(ri) + ".png")
    rblk.save("save/aha_" + str(ri) + "_logo.png")  # , transparency=0
    # rblk.show()  # 查看蒙版

    return cv2.cvtColor(np.array(img).astype(np.uint8), cv2.COLOR_RGBA2BGR), \
           cv2.cvtColor(np.array(rblk).astype(np.uint8), cv2.COLOR_RGBA2GRAY)


def save_sameshape_logo_mask_pic_4icomp(img, watermark, dradius):
    img = Image.open(img)
    watermark = Image.open(watermark)
    # """  # 准备空白，粘贴水印图，并转换为 dip的蒙版图
    bh, bw, c = np.array(img).shape
    # print(bh, bw, c)
    sw, sh = watermark.size
    iw = int((bw - sw) / 2)
    ih = int((bh - sh) / 2)
    blk = Image.fromarray(np.zeros((bh, bw, 4), dtype=np.uint8)).convert(mode="RGBA")
    # blk = Image.fromarray(np.zeros((bh, bw, c+1), dtype=np.uint8)).convert(mode="RGBA")
    _, bg, bb, ba = blk.split()  # 粘贴前先分离一下
    blk.paste(watermark, (iw, ih))  # 粘贴logo
    # 对blk作处理， 取其红色通道 , blk是img-size，而wm是logo-size, 需要在指定位置paste logo
    pr_, _, _, _ = blk.split()  # 粘贴后分离
    pr = np.array(pr_)  # 在阈值化之前，最好进行单通道的膨胀处理！

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dradius, dradius))  # 膨胀半径
    dilated = cv2.dilate(pr, kernel)  # 膨胀图像
    pr = dilated  # 确认生成膨胀的图像
    pr = pr  # 原图，不做膨胀

    pr[pr >= 1] = 255
    pri = Image.fromarray(pr)
    print(pri.size, bg.size)
    rblk = Image.merge("RGBA", (pri, bg, bb, pri))

    # 保存水印图 + 等分辨率 logo图
    ri = random.randint(0, 100)
    img.save("save/aha_" + str(ri) + ".png")
    rblk.save("save/aha_" + str(ri) + "_logo.png")  # , transparency=0

    return cv2.cvtColor(np.array(img).astype(np.uint8), cv2.COLOR_RGBA2BGR), \
           cv2.cvtColor(np.array(rblk).astype(np.uint8), cv2.COLOR_RGBA2GRAY)


from get_clk_local import *
def logo_clean_show(srcp, savp, dradius=3, iradius=3, imode=cv2.INPAINT_TELEA):
    # 1、采用impaint方式
    # tmplt = "opc/370x52.png"  # 水印图
    # img, imask = save_sameshape_logo_mask_pic(srcp, tmplt, dradius)  # 根据任意原图 和一张logo模板，生成该图的对应size的蒙版(cv2)
    # dst = cv2.inpaint(img, imask, iradius, imode)  # src, mask, 参考半径, 修复方式

    # 2、采用逆向计算方式W
    src, wm, (iw, ih), pmask = get_logobypath(srcp)
    dst = pix_logo_rm(src, wm, (iw, ih), pmask)

    # dst = cv2.inpaint(img, imask, 3, cv2.INPAINT_TELEA)  # src, mask, 邻域半径, 修复方式
    # dst = cv2.inpaint(img, imask, 3, cv2.INPAINT_NS)  # src, mask, 邻域半径, 修复方式
    # dst = cv2.inpaint(img, imask, 1, cv2.INPAINT_NS)  # src, mask, 邻域半径, 修复方式
    # dst = cv2.inpaint(img, imask, 5, cv2.INPAINT_TELEA)  # src, mask, 邻域半径, 修复方式
    # dst = cv2.inpaint(img, imask, 7, cv2.INPAINT_TELEA)  # src, mask, 邻域半径, 修复方式
    # dst = cv2.inpaint(img, imask, 9, cv2.INPAINT_TELEA)  # src, mask, 邻域半径, 修复方式
    # dst = cv2.inpaint(img, imask, 11, cv2.INPAINT_TELEA)  # src, mask, 邻域半径, 修复方式
    # dst = cv2.inpaint(img, imask, 15, cv2.INPAINT_TELEA)  # src, mask, 邻域半径, 修复方式

    # ms = "Mtelea" if imode == cv2.INPAINT_TELEA else "Mns"
    # savp = savp + "new_d" + str(dradius) + "_i" + str(iradius) + "_" + ms

    savp = savp + "yy_0"
    if not os.path.exists(savp):
        os.makedirs(savp)
    print("保存位置：", savp)

    # 保存结果图
    spic = srcp.rsplit('/', 1)[1].split('.')  # name + subfix
    # savep = savp + "/" + spic[0] + "_rm." + spic[1]
    # cv2.imwrite(savep, dst)

    # 保存对比图
    simg = Image.open(srcp)  # 原图
    simg = simg.convert(mode="RGB")
    slogo = get_src_logo_img(simg)  # 获取 logo 水印图

    w, h = simg.size
    out_image = np.zeros((h, w * 3, 3), dtype=np.uint8)
    out_image[:, :w] = simg  # 出问题
    # out_image[:, w:w * 2] = slogo
    out_image[:, w:w * 2] = src
    out_image[:, w * 2:] = Image.fromarray(dst)

    # b, g, r = cv2.split(dst)
    # idst = cv2.merge([r, g, b])
    # out_image[:, w * 2:] = idst  # 红蓝反色

    rsavep = savp + "/" + spic[0] + "_comp." + spic[1]
    Image.fromarray(out_image).save(rsavep)  # 保存

def logo_clean_yy(srcp, pngp, savp):
    print("testing: ", srcp)
    ilogo = Image.open(srcp)  # 读入的图片
    # ilogo.show()
    # print(ilogo.mode)
    # ilogo = ilogo.filter(ImageFilter.MedianFilter(5))  # 中值滤波，貌似没效果？
    # ilogo = ilogo.filter(ImageFilter.SMOOTH)  # 平滑滤波
    """好像不管怎么滤波都有一样，难道是 mask和wm的问题？？？"""

    # scipy 中值滤波
    # mfc = []
    # for ic in ilogo.split():
    #     ic_mf = signal.medfilt(ic, (5, 5))
    #     mfc.append(Image.fromarray(ic_mf.astype(np.uint8)))
    #     print("c")
    # ilogo = Image.merge('RGB', mfc)

    # 2、采用逆向计算方式W， 先通过对原图与系统已有的logo模板，获取一些参数
    slogo, wm, (iw, ih), pmask = get_logobypath(srcp, pngp)  # 获取计算资源, 获取双水印
    # dst_ = pix_logo_rm_jd(slogo, wm, (iw, ih), pmask)  # 除原生水印  214 373
    dst_ = pix_logo_rm(slogo, wm, (iw, ih), pmask)  # 除原生水印  214 373
    dst = pix_logo_rm(Image.fromarray(dst_), wm, (iw, ih+60), pmask)  # 除下方60pix人工水印
    # dst = pix_logo_rm(Image.fromarray(_dst), wm, (iw, ih+120), pmask)  # 对白色区域除水印

    # 由于自制水印的去除，几乎都恰巧计算会原来的值；
    # 而带有少许噪声的jd-logo，细微的差别会导致负数转移造成误差；

    # savp = savp + "yyjd"
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


def logo_clean_by_png(srcp, png, savp):
    ilogo = Image.open(srcp)  # 读入的图片

    # 2、采用逆向计算方式W， 先通过对原图与系统已有的logo模板，获取一些参数
    slogo, wm, (iw, ih), pmask = get_logobypath_png(srcp, png)  # 获取计算资源, 获取双水印
    dst_ = pix_logo_rm(slogo, wm, (iw, ih), pmask)  # 除原生水印  214 373
    dst = pix_logo_rm(Image.fromarray(dst_), wm, (iw, ih+60), pmask)  # 除下方60pix人工水印

    Image.fromarray(dst).show()

    # # savp = savp + "yyjd"
    # if not os.path.exists(savp):
    #     os.makedirs(savp)
    # # print("保存位置：", savp)

    # # 保存结果图
    # spic = srcp.rsplit('/', 1)[1].split('.')  # name + subfix

    # w, h = ilogo.size
    # out_image = np.zeros((h, w * 3, 3), dtype=np.uint8)
    # out_image[:, :w] = ilogo  # 原图
    # out_image[:, w:w * 2] = slogo  # 手工图 + logo 与原图对比
    # out_image[:, w * 2:] = Image.fromarray(dst)

    # rsavep = savp + "/" + spic[0] + "_comp." + spic[1]
    # iout = Image.fromarray(out_image)

    # iout.save(rsavep)
    # iout.show()


# 反推jd-logo函数
# 取字体点处 大量的值来 拟合计算 logo的像素值（r，g，b，a）
# 字体处的像素坐标： 230,380  235,405
def get_jd_logo(ilogo, local, logo_wh):
    il = np.array(ilogo)
    isrc = np.ones(np.shape(il), dtype=np.uint8) * 255

    lx, rx, ty, by = local[0], local[0] + logo_wh[0], local[1], local[1] + logo_wh[1]
    idst = isrc - il

    # idstlogo = idst[ty:by, lx:rx, :]
    # h, w, c = np.shape(idstlogo)  # 截取logo部分打印
    h, w, c = np.shape(il)  # 截取logo部分打印

    for i in range(ty, by):
        linel = ["(" + ','.join([str(pix[ci]) for ci in range(c)]) + ")" for pix in idst[i]]
        lstr = ' '.join(linel)
        print(lstr)  # 打印src与 加了logo的差值

    compLogo(il, (230, 380), (235, 405))
    idm = Image.fromarray(idst)  # .show()
    return idm
"""
    求一下这些像素点的 rgb 上的均值
    r,g,b,a = 最终可以通过一组等式求出  但是由于等式不足，需要遍历一个确定值
    将 a 从0~255个值进行遍历，每个a值将有一个确定的 rgb 值
    则 rgba 求解得到
    
    但是只得到字体核心处的（rgba），假设rgb恒定不变，字体到背景间变化的是a？
    再通过反减法 取出水印
"""

def compLogo(il, ptl, prb):
    lx, ty = ptl
    rx, by = prb
    ndarr = il[ty:by, lx:rx, :]
    h, w, c = np.shape(ndarr)

    # for ci in range(c):
    #     print("第", ci, "个通道")  # 各个通道求均值 [sum/counts]
    #     for i in range(h):
    #         sgc_list = [pix[ci] for pix in ndarr[i]]
    #         curh_s = np.sum(np.array(sgc_list))
    #         print(curh_s, " 求和后：", sgc_list)

    darr_r = ndarr[:, :, 0]
    darr_g = ndarr[:, :, 1]
    darr_b = ndarr[:, :, 2]

    print(darr_r)
    print(darr_g)
    print(darr_b)

    print(np.mean(darr_r))
    print(np.mean(darr_g))
    print(np.mean(darr_b))

    # (255, 255, 255)  白底处的像素
    # (237, 185, 188)  这是白底处加水印后的像素值

    r, g, b = 237, 185, 188


# 本方法主要用于 反推 jd-logo 的真实值
def logo_clean_yy_rsh(srcp, savp):
    ilogo = Image.open(srcp)  # 读入的图片

    # 2、采用逆向计算方式W， 先通过对原图与系统已有的logo模板，获取一些参数
    _, wm, (iw, ih), pmask = get_logobypath(srcp)  # 获取计算资源, 获取双水印
    dst = pix_logo_rm(ilogo, wm, (iw, ih), pmask)  # 除水印

    # 反推的 53x371 的京东水印图
    clogo = get_jd_logo(ilogo, (iw, ih), pmask.size)

    savp = savp + "yyjd"
    if not os.path.exists(savp):
        os.makedirs(savp)
    print("保存位置：", savp)

    # 保存结果图
    spic = srcp.rsplit('/', 1)[1].split('.')  # name + subfix

    w, h = ilogo.size
    out_image = np.zeros((h, w * 3, 3), dtype=np.uint8)
    out_image[:, :w] = ilogo  # 原图（0+logo）
    out_image[:, w:w * 2] = clogo  # 原图 - 0
    out_image[:, w * 2:] = Image.fromarray(dst)  # 除水印后

    iout = Image.fromarray(out_image)
    iout.show()

    # irsavep = savp + "/" + spic[0] + "_dlg." + spic[1]
    # slosgo.save(irsavep)  # 保存双水印图

    # dsavep = savp + "/" + spic[0] + "_rm." + spic[1]
    # idst = Image.fromarray(dst)
    # idst.save(dsavep)

    # rsavep = savp + "/" + spic[0] + "_comp." + spic[1]
    # iout.save(rsavep)


img_path = "JD/car2.jpg"
# img_path = "new/l1.pn g"
tmplt = "opc/370x52.png"
img = cv2.imread(img_path)
dradius = 0
iradius = 0
ipms = [cv2.INPAINT_TELEA, cv2.INPAINT_NS]
ipm = 0

def change_dradius(x):
    imask = get_imask4dr(img_path, tmplt, x)
    dst = cv2.inpaint(img, imask, iradius, ipms[ipm])  # src, mask, 参考半径, 修复方式
    cv2.imshow('myImg', dst)

def change_iradius(x):
    imask = get_imask4dr(img_path, tmplt, dradius)
    dst = cv2.inpaint(img, imask, x, ipms[ipm])  # src, mask, 参考半径, 修复方式
    cv2.imshow('myImg', dst)

def change_ipm(x):
    imask = get_imask4dr(img_path, tmplt, dradius)
    dst = cv2.inpaint(img, imask, iradius, ipms[x])  # src, mask, 参考半径, 修复方式
    cv2.imshow('myImg', dst)

def myslide():
    cv2.namedWindow('myImg')
    cv2.createTrackbar('dr', 'myImg', 0, 30, change_dradius)
    cv2.createTrackbar('ir', 'myImg', 0, 100, change_iradius)

    # 创建一个开关滑动条，只有两个值，起开关按钮作用
    switch = '0:T\n1:N'
    cv2.createTrackbar(switch, 'myImg', 0, 1, change_ipm)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        cv2.getTrackbarPos('dr', 'myImg')
        cv2.getTrackbarPos('ir', 'myImg')
        cv2.getTrackbarPos('ipm', 'myImg')

    cv2.destroyAllWindows()


def dir_test():
    flist = os.listdir("JD")
    for pp in flist:
        logo_clean_show(
            "JD/" + pp,
            savp="_solution_/",
            dradius=1,  # 膨胀半径 [无膨胀]
            iradius=0,  # inp半径 [0inp]
            imode=cv2.INPAINT_NS,
        )
    print("Over")


def get_logobypath(imgp, pngp):
    img = Image.open(imgp)
    image = img.copy()

    # pngp = "opc/370x52.png"
    # pngp = "uedx/25%.png"
    # pngp = "uedx/29%.png"
    # pngp = "uedx/27.png"
    # pngp = "uedx/28.png"
    # pngp = "uedx/29.png"
    # pngp = "opc/666.png"  #
    # pngp = "opc/666_33.png"  #
    # pngp = "opc/777.png"
    # pngp = "opc/888.png"
    # pngp = "opc/999.png"

    print("logo地址:", pngp)
    # watermark = Image.open("opc/666.png")  # 水印路径，加在下侧
    watermark = Image.open(pngp)  # 水印路径，加在下侧

    if watermark.mode != 'RGBA':
        alpha = Image.new('L', watermark.size, 255)  # 创建A通道  L表示8位灰度图
        watermark.putalpha(alpha)  # pil RGB 如何转 BGR

    # 获取正中位置
    # bh, bw, _ = img.shape  # shape的宽高位置和pil的有区别！！！
    bw, bh = img.size  # shape的宽高位置和pil的有区别！！！
    sw, sh = watermark.size
    iw = int((bw - sw) / 2)
    ih = int((bh - sh) / 2)

    # print(bw, bh, sw, sh, "==>", iw, ih)

    TRANSPARENCY = 100
    paste_mask = watermark.split()[3].point(lambda i: i * TRANSPARENCY / 100.)  # 第四通道

    image.paste(watermark, (iw, ih + 60), mask=paste_mask)  # pm是wm本身第四通道的像素值
    """
    模板图像的尺寸必须与变量image对应的图像尺寸一致。
    如果变量mask对应图像的值为255，则模板图像的值直接被拷贝过来；如果变量mask对应图像的值为0，则保持当前图像的原始值。

    alp => vmask / 255
    vpix = alp * logo + (1-alp) * src
    img = (src - alp * logo)/(1-alp)
    """
    return image, watermark, (iw, ih), paste_mask


def get_logobypath_png(imgp, png):
    img = Image.open(imgp)
    image = img.copy()
    watermark = png  # 水印 img

    if watermark.mode != 'RGBA':
        alpha = Image.new('L', watermark.size, 255)  # 创建A通道  L表示8位灰度图
        watermark.putalpha(alpha)  # pil RGB 如何转 BGR

    # 获取正中位置
    # bh, bw, _ = img.shape  # shape的宽高位置和pil的有区别！！！
    bw, bh = img.size  # shape的宽高位置和pil的有区别！！！
    sw, sh = watermark.size
    iw = int((bw - sw) / 2)
    ih = int((bh - sh) / 2)

    # print(bw, bh, sw, sh, "==>", iw, ih)
    TRANSPARENCY = 100
    paste_mask = watermark.split()[3].point(lambda i: i * TRANSPARENCY / 100.)  # 第四通道
    image.paste(watermark, (iw, ih + 60), mask=paste_mask)  # pm是wm本身第四通道的像素值
    """
    模板图像的尺寸必须与变量image对应的图像尺寸一致。
    如果变量mask对应图像的值为255，则模板图像的值直接被拷贝过来；如果变量mask对应图像的值为0，则保持当前图像的原始值。

    alp => vmask / 255
    vpix = alp * logo + (1-alp) * src
    img = (src - alp * logo)/(1-alp)
    """
    return image, watermark, (iw, ih), paste_mask

def dir_logorm_test(dir, pngp):
    flist = os.listdir(dir)
    for pp in flist:
        if pp.endswith('.jpg'):
            logo_clean_yy(
                srcp=dir + pp,
                pngp=pngp,
                # savp="xout/"
                savp="yout/"
            )
    print("Over!!!")

def dir_logorm_4_png(dir, png):
    flist = os.listdir(dir)
    for pp in flist:
        if pp.endswith('.jpg'):
            logo_clean_by_png(  # 根据png对象，去除水印
                srcp=dir + pp,
                png=png,
                savp="xout/"
            )
    # print("Over!!!")


if __name__ == '__main__':
    # dir_test()
    # myslide()

    # imgp = "_rev_/sy/jd0_logo_.jpg"
    # logo_clean_yy(srcp=imgp, savp="_solution_/")
    # logo_clean_yy_rsh(srcp=imgp, savp="_solution_/")
    # logo_clean_yy(srcp="new/l4.jpg", savp="_solution_/")
    # logo_clean_yy(srcp="_rev_/yy2_2014.jpg", savp="_solution_/")

    # logo_clean_yy(srcp="ijd/jdm.jpg", savp="_solution_/")
    # logo_clean_yy(srcp="ijd/l4.jpg", savp="_solution_/")

    # dir = "ulogo_crop/test/"
    dir = "_rev_/"
    # dir = "JD/"

    # dir_logorm_test(dir, pngp="opc/194_31_36_77.png")
    # dir_logorm_test(dir, pngp="opc/666.png")  # 191,15,36, 77
    # dir_logorm_test(dir, pngp="opc/190_26_36.png")  # 有红色
    # dir_logorm_test(dir, pngp="opc/192_26_36.png")  # 190_26_36  有红色

    # dir_logorm_test(dir, pngp="opc/199_16_34.png")
    # dir_logorm_test(dir, pngp="opc/199_16_34_a1.png")
    # dir_logorm_test(dir, pngp="opc/199_16_34_a2.png")
    # dir_logorm_test(dir, pngp="opc/199_16_34_a3.png")
    # dir_logorm_test(dir, pngp="opc/lkp_merge.png")  # 有细微红边
    # dir_logorm_test(dir, pngp="opc/lkp_tmd.png")  # 腐蚀一圈
    # dir_logorm_test(dir, pngp="opc/lkp_tmd2.png")  # 腐蚀两圈
    # dir_logorm_test(dir, pngp="opc/lkp_tmd3.png")  # 腐蚀两圈
    # dir_logorm_test(dir, pngp="opc/lkp_tmd6.png")  # 风翼对比后 【效果最好的图】

    # dir_logorm_test(dir, pngp="opc/lkp_tmd7.png")  # o操作
    # dir_logorm_test(dir, pngp="opc/lkp_tmd9.png")  # o操作

    # dir_logorm_test(dir, pngp="opc/lkp_tmd10.png")  # 回归 tmd6  亮绿色是因为  没有logo本身透明度不够
    # dir_logorm_test(dir, pngp="opc/lkp_tmd20.png")  # 只操作外圈时，内侧有亮边

    # dir_logorm_test(dir, pngp="opc/tmd_fine_11.png")  # 微调1
    # dir_logorm_test(dir, pngp="opc/tmd_fine_12.png")  # 回归 tmd6  亮绿色是因为  没有logo本身透明度不够
    # dir_logorm_test(dir, pngp="opc/tmd_fine_13.png")  #
    # dir_logorm_test(dir, pngp="opc/tmd_fine_14.png")  #
    dir_logorm_test(dir, pngp="opc/tmd_fine_15.png")  #

    # dir_logorm_test(dir, pngp="opc/tmd_fine_20.png")  # 精细微调 + 形态滤波
    # fine-15 最终方案，仍然有一些瑕疵，但是这些瑕疵算法修补成本高，抠图修补

    # 候补方案： 对边缘点采用  候补核滤波
    # 采用轮廓检测试试层次操作

    # 编写 rgb滑动 调色程序, 显示消去效果图
    # 188~194, 12~22, 30~40


"""
需要生成水印图对应的 蒙版size图，再进行inpaint，重要参数：
logo膨胀参数， inpaint邻域半径， inpaint方式[INPAINT_TELEA/NS]

10.20: 采用高斯拟合  学习logo图：   【方案的前提是：假设所有logo图都是同一个精确的logo】
    logo原图一定有一种标准值：Larr: (374,53,4) 矩阵的值
    在加logo图时，通过pil模块将larr向各种随机躁点图打水印（800x800）
    原图 -打水印-》 水印图 -消去模型-》原图
    noise2noise 模型：通过一系列的卷积参数，从海量的 cnn-w 去学习logo每个像素点的高斯分布
    
    模型中保存每个点位的 均值与高斯模型，g(c,w), 但是图像质量噪声有个偏差模型
    logo像素的区域先验，利用条件随机场进行一个分布特性的约束
    如何利用周围的点位进行 预测值的先验约束？所以用到卷积
    
    比如说对打上logo的图进行一次卷积操作，生成无水印图
    形态滤波
"""





