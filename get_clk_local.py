import cv2
import numpy as np
from PIL import Image, ImageFilter

def get_pixlocal(imgp):
    # 图片路径
    # img = cv2.imread('test.jpg')
    img = cv2.imread(imgp)
    a = []
    b = []
    print(np.shape(img))

    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            a.append(x)
            b.append(y)

            print(x, y, img.item(y, x, 0), img.item(y, x, 1), img.item(y, x, 2))

            cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
            cv2.putText(
                img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                1.0, (0, 0, 0), thickness=1
            )
            cv2.imshow("image", img)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    # print(a[0], b[0])


def look_pixv(imgp, plt, prb):
    img = Image.open(imgp)
    imgarr = np.array(img)
    print(imgarr.shape, np.shape(imgarr), img.size)

    lx, rx, ty, by = plt[0], prb[0], plt[1], prb[1]
    # lx, rx, ty, by = plt[0], prb[0], plt[1]+60, prb[1]+60
    # patch = imgarr[376:426, 216:236, :]  # w=20, h=50
    patch = imgarr[ty:by, lx:rx, :]  # w=20, h=50
    print(patch.shape)
    Image.fromarray(patch).show()
    # print(patch)

    h, w, c = patch.shape
    blk = Image.fromarray(np.ones((h, w, c), dtype=np.uint8)*255).convert(mode="RGBA")
    # blk = Image.fromarray(np.ones((h, w, c), dtype=np.uint8)*255).convert(mode="RGB")
    dst = blk - patch
    print(dst.shape)

    # for ci in range(c):
    #     print("第", ci, "个通道")
    #     for i in range(h):
    #         linel = [str(pix[ci]) for pix in patch[i]]
    #         lstr = ' '.join(linel)
    #         print(lstr)

    # for ci in range(c):
    #     print("第", ci, "个通道")
    #     for i in range(h):
    #         linel = [str(pix[ci]) for pix in dst[i]]
    #         lstr = ' '.join(linel)
    #         print(lstr)

    for i in range(h):
        # linel = ["(" + ','.join([str(pix[ci]) for ci in range(c)]) + ")" for pix in patch[i]]
        # linel = ["(" + ','.join([str(pix[ci]) for ci in range(3)]) + ")" for pix in patch[i]]
        linel = ["(" + ','.join([str(pix[ci]) for ci in range(3, 4)]) + ")" for pix in patch[i]]
        # linel = ["(" + ','.join([str(pix[ci][-1]) for ci in range(c)]) + ")" for pix in patch[i]]
        # linel = ["(" + ','.join([str(pix[ci]/255)[:5] for ci in range(3, 4)]) + ")" for pix in patch[i]]
        lstr = ' '.join(linel)
        print(lstr)

    # patch 是logo图
    # blk 是原图
    # dst 是差值图 【蓝绿色，黑背景】
    Image.fromarray(dst).show()

    blk = patch + dst  # 原图=logo图+差值图
    Image.fromarray(blk).show()

    # 如何求一张logo水印图的差值图

def dist_logopix_jds(imgap, imgbp, plt, prb):
    imga = Image.open(imgap).convert(mode="RGB")
    imgb = Image.open(imgbp).convert(mode="RGB")
    imgarr = np.array(imga)
    imgbrr = np.array(imgb)
    print(imgarr.shape, np.shape(imgarr), imga.size)
    print(imgbrr.shape, np.shape(imgbrr), imgb.size)

    lx, rx, ty, by = plt[0], prb[0], plt[1], prb[1]
    apatch = imgarr[ty:by, lx:rx, :]  # w=20, h=50
    bpatch = imgbrr[ty:by, lx:rx, :]  # w=20, h=50

    dis = apatch - bpatch
    # dis = apatch - apatch

    h, w, c = apatch.shape

    # for ci in range(c):
    #     print("第", ci, "个通道, logo字体像素差")
    #     for i in range(h):
    #         linel = [str(pix[ci]) for pix in dis[i]]
    #         lstr = ' '.join(linel)
    #         print(lstr)

    # blk = Image.fromarray(np.ones((h, w, c), dtype=np.uint8) * 255).convert(mode="RGB")
    #
    # adst = blk - apatch
    # for i in range(h):
    #     linel = ["(" + ','.join([str(pix[ci]) for ci in range(c)]) + ")" for pix in adst[i]]
    #     lstr = ' '.join(linel)
    #     print(lstr)
    #
    # bdst = blk - bpatch
    # for i in range(h):
    #     linel = ["(" + ','.join([str(pix[ci]) for ci in range(c)]) + ")" for pix in bdst[i]]
    #     lstr = ' '.join(linel)
    #     print(lstr)

    # bdst = blk - bpatch
    # 计算两个图像的像素差
    for i in range(h):
        linel = ["(" + ','.join([
            str(apix[ci] - bpix[ci]) if apix[ci] > bpix[ci] else str(bpix[ci] - apix[ci])
                for ci in range(c)
        ]) + ")" for apix, bpix in zip(apatch[i], bpatch[i])]
        lstr = ' '.join(linel)
        print(lstr)

    Image.fromarray(apatch).show()
    Image.fromarray(bpatch).show()




"""
找出jd水印的原生配方，根据配方来逆向计算：

"""

"""
分析： 透明是绝对要给透明度的，如果设置alp全为0，那么logo成分为0，看不到logo，不行
如果alp设置全为100， 那么将logo部分将全部取代原图像素，不合理

问题：透明度区还需要分字体和背景，在logo背景上透明度是0（没有logo成分），在logo字体上透明度为对应值

"""
from LogoRemoval import get_src_logo_img
def get_logobypath(imgp):
    img = Image.open(imgp)
    image = img.copy()
    # watermark = Image.open("opc/370x52.png")  # 水印路径，加在下侧
    # watermark = Image.open("opc/logo_a127.png")  # 水印路径，加在下侧
    # watermark = Image.open("opc/logo_a71.png")  # 水印路径，加在下侧
    # watermark = Image.open("opc/logo_X.png")  # 水印路径，加在下侧
    # watermark = Image.open("opc/111.png")  # 水印路径，加在下侧
    # watermark = Image.open("opc/222.png")  # 水印路径，加在下侧
    # watermark = Image.open("opc/333.png")  # 水印路径，加在下侧
    # watermark = Image.open("opc/444.png")  # 水印路径，加在下侧
    # watermark = Image.open("opc/555.png")  # 水印路径，加在下侧
    watermark = Image.open("opc/666.png")  # 水印路径，加在下侧

    # watermark = Image.open("_rev_/sy/a_logo/logo_X.png")  # 水印路径，加在下侧

    # watermark = Image.open("opc/_374x54.png")  # 水印路径，加在下侧， 还是不行，核心区残留颜色了

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

    # print(bw, bh, sw, sh, "==>", iw, ih)

    TRANSPARENCY = 100
    paste_mask = watermark.split()[3].point(lambda i: i * TRANSPARENCY / 100.)  # 第四通道
    # print(np.array(paste_mask))

    # print("pmask 全部255, 粘贴全部")
    # paste_mask = watermark.split()[3].point(lambda i: 255)  # 第四通道

    # print("pmask 全部0, 不粘贴")
    # paste_mask = watermark.split()[3].point(lambda i: 0)  # 第四通道

    # 字体处为1， 其余位0


    # wr, wg, wb, wa = watermark.split()
    # iwatermark = Image.merge('RGBA', (wg, wb, wr, wa))  # paste之前，水印红蓝反色

    image.paste(watermark, (iw, ih + 60), mask=paste_mask)  # pm是wm本身第四通道的像素值
    # print("透明通道的值：", paste_mask)  # 一个0~255的值的 单通道map
    # paste_mask.save("tpix/pmask.jpg")
    """
    模板图像的尺寸必须与变量image对应的图像尺寸一致。
    如果变量mask对应图像的值为255，则模板图像的值直接被拷贝过来；如果变量mask对应图像的值为0，则保持当前图像的原始值。
    
    alp => vmask / 255
    vpix = alp * logo + (1-alp) * src
    img = (src - alp * logo)/(1-alp)
    """
    return image, watermark, (iw, ih), paste_mask

# 原图 = logo图 + 差值图
# 要就清楚原纯logo 对应的差值图是多少
# 通过pix差值法，逆算出原图
"""
想要还原，需要的数据是：【最好都拥有一模一样的size】
    1、原图；
    2、纯logo图； 
    3、pmask图；

出现偏差的原因：
    1、logo色素值不纯、或给的独裁色素值不纯
    2、白色出现大反差，而有颜色的地方良好【因为白色区域权重】
"""
def get_initpix(vsrc, valp, vlogo):
    # valp = 0.0745 if valp > 0.0 else 0.0

    if valp == 1:
        return vlogo
    else:
        if valp != 0:
            # print(vsrc, valp, valp*255, vlogo)
            pass
        # return (vsrc - valp * vlogo) / (1 - valp)  # 因为是 uint，小减大时会算为 255-x
        return ((vsrc - valp * vlogo) if vsrc > valp * vlogo else (valp * vlogo - vsrc)) / (1 - valp)

    # return 255 - ((vsrc - valp * vlogo) / (1 - valp))


def Aget_initpix(vsrc, valp, vlogo, prt=False):
    rst = 0
    if valp == 1:
        rst = vlogo
    elif valp == 0:
        rst = vsrc
    else:
        # vsrc, vlogo = float(vsrc), float(vlogo)
        rst = ((vsrc - valp * vlogo)) / (1 - valp)
        # print("valp 为浮点数", vsrc, valp, vlogo, " ===> ", rst)

    # 在纯背景处，希望过渡区 得到更大的像素值
    if rst < 0:
        rst = 0
    if rst > 255:
        rst = 255

    # if prt:
    #     print(vsrc, valp, vlogo, " ===> ", rst)

    return rst


def sim2jd_logo(pix):
    flag = False
    # 237, 185, 188
    r, g, b = pix[0], pix[1], pix[2]
    dr = max(r, 237) - min(r, 237)
    dg = max(g, 185) - min(g, 185)
    db = max(b, 188) - min(b, 188)
    if max([dr, dg, db]) < 10 and np.mean([dr, dg, db]) < 6:
        flag = True
    return flag


def pix_logo_rm(src, wm, local, pmask):
    sw, sh = local
    # src.show()

    src = np.array(src)  # 编程int类型，使得不会导致计算值溢出
    wm = np.array(wm)
    pmask = np.array(pmask)
    # 从wh开始遍历每一个元素值
    # print("输入shape：",
    #     np.shape(src),  # (348, 500, 3)
    #     np.shape(wm),  # (53, 371, 4)
    #     np.shape(pmask),  # (53, 371)
    #     # pmask
    # )  # 输入shape： (348, 500, 3) (53, 371, 4) (53, 371)

    # 打印pmask
    # h, w = pmask.shape
    # for i in range(h):  # 每一行
    #     templ = [str(pmask[i][j]) if pmask[i][j] != 0 else '_' for j in range(w)]
    #     lstr = ' '.join(templ)
    #     print(lstr)
    # pass

    th, tw, tc = np.shape(src)
    # targ = Image.fromarray(np.zeros((th, tw, 3), dtype=np.uint8)).convert(mode="RGB")

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

            # 至此，一个像素点计算完毕
            # print(x, y, src[y][x], timg[y][x])

    # 至此，整个 timg 计算完毕
    # print(np.shape(timg))
    timg_ = np.asarray(timg, dtype=np.uint8)
    ttimg = Image.fromarray(timg_).convert(mode="RGB")

    # ttimg.show()
    # ttimg.save("tpix/yy___.jpg")  # 通过水印图得到原图
    # print("还原计算完毕！")
    return np.array(ttimg)


def pix_logo_rm_jd(src, wm, local, pmask):
    sw, sh = local
    # src.show()

    src = np.array(src)
    wm = np.array(wm)
    pmask = np.array(pmask)
    # 从wh开始遍历每一个元素值
    print("输入shape：",
        np.shape(src),  # (348, 500, 3)
        np.shape(wm),  # (53, 371, 4)
        np.shape(pmask),  # (53, 371)
        # pmask
    )  # 输入shape： (348, 500, 3) (53, 371, 4) (53, 371)

    # 打印pmask
    # h, w = pmask.shape
    # for i in range(h):  # 每一行
    #     templ = [str(pmask[i][j]) if pmask[i][j] != 0 else '_' for j in range(w)]
    #     lstr = ' '.join(templ)
    #     print(lstr)
    # pass

    th, tw, tc = np.shape(src)
    # targ = Image.fromarray(np.zeros((th, tw, 3), dtype=np.uint8)).convert(mode="RGB")

    timg = src.copy()  # 拷贝一个map
    # print(sw, sh)  # paste 的起始位置  64 147
    h, w = pmask.shape  # 要遍历的宽高： 53, 371
    for i in range(h):
        for j in range(w):
            x, y = j+sw, i+sh  # 求操作图上的坐标
            if x < 0 or x >= tw or y < 0 or y >= th:
                continue

            """  # 手工判断背景与非背景
            if not sim2jd_logo(timg[y][x]):  # 非背景， 思路：按照背景与非背景区别对待
                for ci in range(3):
                    vsrc = timg[y][x][ci]
                    # vsrc = timg[y][x][-ci]
                    valp = pmask[i][j] / 255
                    vlogo = wm[i][j][ci]

                    # print("采用logo的 逆通道， 处理黄背景问题（b通道为还原）")
                    # vlogo = wm[i][j][2-ci]  # 应该采用logo的 逆通道
                    timg[y][x][ci] = int(Aget_initpix(vsrc, valp, vlogo))
            else:  # 背景
                timg[y][x][:] = 255
            """
            # """
            for ci in range(3):
                vsrc = timg[y][x][ci]
                valp = pmask[i][j] / 255
                vlogo = wm[i][j][ci]

                # print("采用logo的 逆通道， 处理黄背景问题（b通道为还原）")
                # vlogo = wm[i][j][2-ci]  # 应该采用logo的 逆通道
                timg[y][x][ci] = int(Aget_initpix(vsrc, valp, vlogo))
            # """

            # 至此，一个像素点计算完毕
            # print(x, y, src[y][x], timg[y][x])

    # 至此，整个 timg 计算完毕
    # print(np.shape(timg))
    ttimg = Image.fromarray(timg).convert(mode="RGB")

    # print("高斯模糊")
    # ttimg = ttimg.filter(ImageFilter.GaussianBlur(radius=1))

    # ttimg.show()
    # ttimg.save("tpix/yy___.jpg")  # 通过水印图得到原图
    print("还原计算完毕！")
    return np.array(ttimg)


def contrast(imgp, yyop):
    # 还原的图 与原图 求差
    img = Image.open(imgp).convert(mode="RGB")
    yy = Image.open(yyop).convert(mode="RGB")
    imga = np.array(img)
    yya = np.array(yy)
    dst = imga - yya
    print(dst, np.max(dst), np.min(dst))


def showAlph(imgp):
    logo = Image.open(imgp).convert(mode="RGBA")
    ialph = logo.split()[3].convert(mode="L")
    ialph.show()


if __name__ == '__main__':
    # imgp = 'jd-logo.jpg'  # (800, 800, 3)
    # imgp = "JD/ej.jpg"  # (348, 500, 3)
    # imgp = "JD/car2.jpg"  # (348, 500, 3)
    # imgp = "ijd/l4.jpg"
    # imgp_ = "ijd/jdm.jpg"  # (348, 500, 3)
    # imgp = "ijd/jdm_dlg.jpg"  # (348, 500, 3)
    # imgp = "_solution_/yyjd/jdm_comp.jpg"  # 1813,376   1875,430
    # imgp = "_solution_/yyjd/l4_rm.jpg"  # 1813,376   1875,430
    # imgp = "_solution_/yyjd/yy8_2014_dlg.jpg"  # 1813,376   1875,430
    # imgp = "_solution_/yyjd/yy8_2014_comp.jpg"  # 1813,376   1875,430
    # imgp = "opc/370x52.png"
    # imgp = "opc/555.png"
    # imgp = "opc/666_33.png"  # 旁边颜色都成了 255
    # imgp = "opc/666.png"  # 旁边颜色都成了 255  [偏低 =》]
    # imgp = "opc/777.png"
    # imgp = "opc/888.png"
    # imgp = "opc/999.png"  # 字体颜色不太对  []
    # imgp = "uedx/29.png"
    # imgp = "uedx/30.png"
    # imgp = "uedx/27.png"
    # imgp = "uedx/30%.png"
    # imgp = "opc/194_31_36_77.png"
    # imgp = "opc/199_16_34_a2.png"
    # imgp = "opc/lkp_tmd5.png"
    imgp = "opc/ilogor.png"

    # imgp = "_rev_/sy/yy_rml.jpg"

    # imgp = "_rev_/sy/jd0_logo_.jpg"
    # imgp = "_rev_/sy/rmed.jpg"
    # imgp = "_rev_/sy/jd_sc.jpg"

    # imgp = "tpix/none.png"
    # src, wm, (iw, ih), pmask = get_logobypath(imgp)
    # src.save("tpix/yysrc.jpg")

    # limgp = "tpix/elogo.jpg"  # (348, 500, 3)
    # omgp = "tpix/ologo.jpg"  # (348, 500, 3)
    # get_pixlocal(imgp)

    # look_pixv(imgp, plt=(212, 377), prb=(237, 428))  # 212, 377; 237, 428
    # look_pixv(imgp, plt=(1813, 376), prb=(1875, 430))  # 212, 437; 237, 488
    # look_pixv(imgp, plt=(1813, 436), prb=(1875, 490))  # 212, 437; 237, 488

    # imgp = "_rev_/yy0_2014.jpg"
    # imgp_ = "_rev_/yy7_2014.jpg"
    # imgp_ = "_rev_/nologo/yy0_x.jpg"

    # dist_logopix_jds(imgp, imgp_, plt=(212, 377), prb=(237, 428))

    # look_pixv(imgp, plt=(212, 377), prb=(237, 428))
    look_pixv(imgp, plt=(0, 0), prb=(373, 54))
    # look_pixv(imgp, plt=(10, 8), prb=(28, 14))

    # 如果把透明度通道  当灰度值显示一下的话
    # showAlph(imgp)


    # 去水印
    # src, wm, (iw, ih), pmask = get_logobypath(imgp)
    # ttimg = pix_logo_rm(src, wm, (iw, ih), pmask)

    # yyo = "tpix/yy_rst.jpg"
    # contrast(imgp, yyo)

    pass
"""
明显 jd 原生水印 像素有毛边而且窄一点，内层呈现某种固定颜色
而 yy 手工水印 像素更宽
"""


"""
215 375
236 426

233,154
258,200

J字母处的pix值
62, 150
86, 200


(255,251,249) (241,231,230) (234,187,193) 

(238,185,191) (239,184,189) (238,186,190) (236,186,187) (236,186,189) (234,188,191) 

(248,219,221) (255,251,248)

"""

