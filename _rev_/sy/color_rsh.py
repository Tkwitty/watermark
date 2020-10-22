# coding=utf-8
import cv2
import numpy as np
from PIL import Image, ImageFilter
import random
import os


"""
当w3在调节时， 图像显示的是什么颜色 {取其中的类浅红色作为最终值}
    wtm 发现a可能为1
    也就是直接差值的
    
10.14：
    现在的问题是，自制jd-logo的外围轮廓稍微大一点，而且渐变区与字体区 参数有细微差别
    导致在取出jd时还能隐约看到 jd-logo
    
    那么通过纯底色的jd-logo将整个 原生jd-logo图给估计出来
    
    遍历logo区域的 显示图像像素P，而T像素全部为255，给定一个la，求出所有位置上的L像素
    求出之后将 L的rgba图给表示出来
    
    
"""

def getL_PT_pix(a, pixP, pixT):
    print(np.shape(pixP))
    print(np.shape(pixT))

    # 矩阵计算法
    la = a / 255
    Lrgb = np.asarray((pixP - pixT*(1-la)) / la, dtype=np.uint8)

    return np.array([Lrgb[0], Lrgb[1], Lrgb[2], a])

# 矩阵计算的 反推测试
# pp = np.array([240, 186, 186])  # [212,7,7,71]   [190   3  14  71]
# pt = np.array([255, 255, 255])
# print(getL_PT_pix(71, pp, pt))


# 根据a值， 接受两个 Image， 生成logo image("RGBA")
def getL_PT(a, Pimg, Timg):
    P = np.array(Pimg)
    T = np.array(Timg)
    # P, T = Pimg, Timg

    # 构造一个 矩阵
    print(np.shape(P))
    print(np.shape(T))

    # 矩阵计算法
    la = a / 255
    Lrgb = (P - T * (1 - la)) / la
    # 对于每个像素，进行计算 logo像素值， 假设背景纯白都是 255,255,255 情况下，同时la为常数
    # 感觉常数应该划分为 0与常数

    print(np.shape(Lrgb), Lrgb)
    h, w, c = np.shape(Lrgb)
    alpha = Image.new('L', (w, h), a)  # 创建A通道  L表示8位灰度图

    Lrgb[Lrgb > 255] = 255
    Lrgb[Lrgb < 0] = 0
    L = Image.fromarray(np.asarray(Lrgb, dtype=np.uint8))
    L.putalpha(alpha)  # pil RGB 如何转 BGR

    # return np.array(L)
    return L


# 单像素图像测试
iP = np.array([[[237, 185, 188]]])
iT = np.array([[[255, 255, 255]]])
print(getL_PT(71, iP, iT))

"""
    主要调节，jd图的 前背景 纯色像素值
    logo内部的 分类rgb像素值 

实验发现生成的 logo图 过宽，需要更细的字体图
"""
def getPij_class(pix):
    p_bk = np.asarray([255, 255, 255], dtype=np.float)  # 0
    p_ft = np.asarray([238, 182, 186], dtype=np.float)  # 1  周边有浅绿没去掉

    # p_ft = np.asarray([238, 185, 188], dtype=np.float)  # 1
    # p_ft = np.asarray([230, 186, 186], dtype=np.float)  # 1

    # 判断背景
    # lalpha = None  # 0, 1, 浮点数

    # 用差均值和差最值
    if np.mean(p_bk - pix)**2 < 65 and np.max((p_bk - pix)**2) < 120:  # 均值差12, 最值差
        lalpha = 0
    else:
        # 这里a是根据 一种线性计算方式 计算而来
        lalpha = 1
        # if np.mean((pix - p_ft)**2) <= 25 and np.max((pix - p_ft)**2) <= 25:
        if np.mean((pix - p_ft)**2) <= 65 and np.max((pix - p_ft)**2) <= 120:
            # print((pix - p_ft)**2, "被设置为字体ft 1")
            # lalpha = 1

            print("所有字体都从a值计算而来")
            lalpha = 1 - np.mean((pix[1:] - p_ft[1:]) / (p_bk[1:] - p_ft[1:]))  # 验证过，是1- 没错

        else:
            lalpha_ = 1 - np.mean((pix[1:] - p_ft[1:])/(p_bk[1:] - p_ft[1:]))  # 验证过，是1- 没错

            if lalpha_ > 1:
                print("！！！！！！！！！！！警告：", lalpha_, pix, p_ft, (pix[1:] - p_ft[1:]))
                # 专家建议，r梯度不管
            # print(lalpha_, pix, p_ft, (pix[1:] - p_ft[1:]))  # 专家建议，r梯度不管

            lalpha = lalpha_  # 整体呈蓝绿
            # lalpha = lalpha_**2  # 内延蓝绿，外延浅红
            # lalpha = 1 - (lalpha_ - 1)**2

            # 通过一个sinx函数的变种来实现 双曲增长曲线

            # if lalpha < 0.28:  # 排除极小a  # 但是这样外延出现了红色
            #     lalpha = 0

            # 如果alp在变化时需要相信的给出 其它rgb 值,那么此处需要额外的计算它

        lalpha = 1  # 测试：前景全为》？

    return lalpha


def get_Binclass(pix, th=7):
    p_bk = np.asarray([255, 255, 255], dtype=np.float)  # 差均值小于 6 就是背景

    # 用差均值和差最值
    if np.mean(p_bk - pix)**2 <= th**2:  # 差均值<7, 最值差
        lalpha = 0
    else:
        # 这里a是根据 一种线性计算方式 计算而来
        lalpha = 1
    return lalpha


def get_alph(img):
    w, h = img.size
    La = np.ones((h, w), dtype=np.uint8)  # 类别矩阵
    for i in range(h):
        for j in range(w):
            La[i][j] = get_Binclass(np.array(img)[i][j])  # lal 非0即1

    return La


def get_alph_blur(img):

    # 或者在处理之前，对jimg进行预处理
    # 还可以先 模糊后，阈值化，分离出更准确的字体区域  去椒盐
    # 字体领域太粗了，需要瘦一点 [可以通过ued-logo去， 也可通过]
    # 前后景问题通过 ued-logo 的阈值化得到 前后景阈值的mask
    # img = img.filter(ImageFilter.GaussianBlur(radius=1))
    # img = img.filter(ImageFilter.MedianFilter(5))

    w, h = img.size
    La = np.ones((h, w), dtype=np.uint8)  # 类别矩阵
    for i in range(h):
        for j in range(w):
            La[i][j] = get_Binclass(np.array(img)[i][j], th=6)  # lal 非0即1

    # 通过高斯模糊，去椒盐噪声
    # lgb = Image.fromarray(La*255).filter(ImageFilter.GaussianBlur(radius=1))
    # gb = np.array(lgb)
    # gb[gb > 127] = 255
    # gb[gb <= 127] = 0
    # Image.fromarray(gb).show()

    return La, img


def getMask_opc_logo(img):  # 通过ued-logo得到前后景阈值图
    opc = "_374x54.png"
    iopc = Image.open(opc)  # 对opc膨胀1个长度

    nopc = np.array(iopc)
    print(nopc)

    # 转cv2 做膨胀 【膨胀慎用，有可能会把 opc 】
    opc_cv = cv2.cvtColor(nopc, cv2.COLOR_RGBA2BGRA)
    kernel = np.ones((3, 3), np.uint8)
    # kernel = np.ones((1, 1), np.uint8)  # 半径1的【膨胀基本上没有膨胀效果】
    nopc_dl = cv2.dilate(nopc, kernel)  # 直接膨胀

    # 转array

    w, h = iopc.size
    La = np.ones((h, w), dtype=np.uint8)  # 类别矩阵
    for i in range(h):
        for j in range(w):
            # La[i][j] = get_Binclass(np.array(iopc)[i][j], th=6)  # lal 非0即1，差值判断
            # La[i][j] = 1 if nopc[i][j][3] > 0 else 0  # 根据透明度值判断，非0即1, 略小1码
            # La[i][j] = 1 if np.sum(nopc[i][j]) > 0 else 0  # 根据全部通道和值判断，非0即1
            # La[i][j] = 1 if np.sum(nopc_dl[i][j]) > 0 else 0  # 膨胀的mask，非0即1
            La[i][j] = nopc[i][j][3]  # 膨胀的mask，alpha 直接取opc的值 【 +膨胀 】

    return La, img


# 为防止 显示原本具有噪声， 此处分 背景0-[255,255,255]，前景71-[238,186,186]，过渡x
"""
按照 rgb 中 gb像素值进行划分 255-0， 186-71
在分类出，背景应该更加包纳一些
感觉上是
"""
def getL_PT_v1(a, Pimg, Timg):
    P = np.array(Pimg)
    T = np.array(Timg)
    # P, T = Pimg, Timg

    # 构造一个 矩阵
    print(np.shape(P))
    print(np.shape(T))

    # 矩阵计算法: 这里是统一按照 原输入来反向计算， 通过调节统一的a = 71来计算logo的ft像素值
    # 自适应算法： 分成三类：
    #       一类近似背景（255 255 255）， a=0
    #       二类近似前景（243 186 186）， a=71
    #       三类过渡区间（r~  g~  b~） ， a=~

    # 那么生成的L图 a>0的地方 全部为a=71情况下 的像素值，只是其a变化了
    # 于是第一件事先把 背景-0算000/非背景-71算Lrgb区分出来；在算a图层
    # 最终将L层与a图层结合得到最终

    # P显示图，T原图
    la = a / 255
    # Lrgb = (P - T * (1 - la)) / la  # 由于要分类讨论，矩阵法不凑效

    # 逐个像素进行计算 P, T
    h, w, c = np.shape(P)  # 要遍历的宽高： 53, 371
    Lrgba = np.ones((h, w, 4), dtype=np.uint8)
    for i in range(h):
        for j in range(w):

            # 下面对 P[i][j] 和 T[i][j] 进行计算 Lrgb
            # 然而 Lrgb 只需要对 P像素进行 分类 [0 背景, 1 前景, -1 过渡]
            lalpha = getPij_class(P[i][j])  # 前后景分类

            # lr, lg, lb = 204, 7, 7  # 边缘残红，中间残蓝绿
            # lr, lg, lb = 194, 7, 7  # 红变少
            # lr, lg, lb = 180, 7, 7  # 红变少
            # lr, lg, lb = 190, 4, 14  # 红变少  # gb过大的话，自提中心的 红色残留严重， 过小蓝绿残留
            # lr, lg, lb = 204, 4, 14  # 红变少  # gb过大的话，自提中心的 红色残留严重， 过小蓝绿残留
            # lr, lg, lb = 125, 4, 14  # 红变少  # gb过大的话，自提中心的 红色残留严重， 过小蓝绿残留
            # lr, lg, lb = 100, 4, 14  # 红变少  # gb过大的话，自提中心的 红色残留严重， 过小蓝绿残留
            # r设置太低，导致很多颜色上残留红色

            # lr, lg, lb = 212, 7, 7  # 手工logo色, 残留，蓝绿残留非常严重
            # lr, lg, lb = 194, 7, 7  # 手工logo色, 残留，蓝绿残留非常严重
            lr, lg, lb = 190, 4, 14  # 手工logo色, 残留，蓝绿残留非常严重
            if lalpha == 0:
                # Lrgba[i][j] = np.asarray([255, 255, 255, 0], dtype=np.uint8)
                Lrgba[i][j] = np.asarray([0, 0, 0, 0], dtype=np.uint8)
            elif lalpha == 1:
                Lrgba[i][j] = np.asarray([lr, lg, lb, 71], dtype=np.uint8)  # diy-logo: 212,7,7
            else:
                # 注意，在像素值变化的地方，rgb也应该做相应变化 【假设alp线性减小】
                lrgb = ((P[i][j] - T[i][j] * (1 - la)) / la)  # 根据a反算logo像素值
                Lrgba[i][j] = np.asarray([lrgb[0], lrgb[1], lrgb[2], lalpha*71], dtype=np.uint8)

            # 至此，一个像素点计算完毕
            if lalpha > 0:
                print(P[i][j], " ===> ", lalpha, ": ", Lrgba[i][j])

    # 至此，整个 timg 计算完毕
    # print(np.shape(timg))
    L = Image.fromarray(Lrgba).convert(mode="RGBA")

    # return np.array(L)
    return L


def getL_PT_v2(x, alp, Pimg, Timg):
    P = np.asarray(Pimg, dtype=int)
    T = np.asarray(Timg, dtype=int)
    # alp = np.asarray(alp, dtype=float)

    # 构造一个 矩阵
    print(np.shape(P))
    print(np.shape(T))

    # P显示图，T原图，alp为透明图层
    Lal = alp * x  # x=71 用于保存图层
    alf = Lal / 255  # 浮点数 用于计算
    print(Lal)
    # Lrgb = (P - T * (1 - la)) / la  # 由于要分类讨论，矩阵法不凑效

    # 逐个像素进行计算 P, T
    h, w, c = np.shape(P)  # 要遍历的宽高： 53, 371
    Lrgb = np.zeros((h, w, 3), dtype=np.int)  # 先采用有符号
    for i in range(h):
        for j in range(w):
            la = alf[i][j]
            if la > 0:
                Lrgb[i][j] = ((P[i][j] - T[i][j] * (1 - la)) / la)  # 出现了数据反转
            # else:  # 初始化为0的情况下 可以不操作
            #     Lrgb[i][j] = np.array([0, 0, 0])

    Lrgb[Lrgb < 0] = 0
    Lrgb[Lrgb > 255] = 255

    # 至此，整个 timg 计算完毕
    # print(np.shape(timg))
    L = Image.fromarray(Lrgb.astype(np.uint8)).convert(mode="RGB")
    alpha = Image.fromarray(Lal, mode='L')  # 创建A通道  L: 8位灰度图
    L.putalpha(alpha)  # 添加 A 图层

    print("显示前后景分离结果：")
    Image.fromarray(alp * 255, mode='L').show()  # 对于前景统统用 反算计算 [只有a用恒定值]

    # return np.array(L)
    return L


a = 100  # 0~255， 但需要70起步，此处设置为100
def get_rgb4a(a):
    # a为1~255
    alp = a/255
    r = int(255 - (18 / alp))
    g = int(255 - (70 / alp))
    b = int(255 - (67 / alp))
    print("调节结果：", r, g, b, a)
    return r, g, b, a


def get_initpix(vsrc, valp, vlogo, prt):
    rst = 0
    if valp == 1:
        rst = vlogo
    else:
        rst = ((vsrc - valp * vlogo) if vsrc > valp * vlogo else (valp * vlogo - vsrc)) / (1 - valp)

    if rst < 0:
        rst = 0
    elif rst > 255:
        rst = 255

    if prt:
        print(vsrc, valp, vlogo, " ===> ", rst)
    return rst


# def get_ipix()



# 反向消去
def get_jd_rmby_rgba(jimg, r, g, b, a):
    jarr = np.array(jimg)
    lgc = [r, g, b]

    h, w, c = np.shape(jarr)
    for i in range(h):
        # print(h, w, c, "正在处理：", i, "行...")
        for j in range(w):
            for ci in range(3):
                vsrc = jarr[i][j][ci]  # 输入
                valp = a / 255
                vlogo = lgc[ci]

                if i>20 and i<40 and j>23 and j<28:
                    prt = True
                else:
                    prt = False
                jarr[i][j][ci] = int(get_initpix(vsrc, valp, vlogo, prt))

    # 此时整个 jarr 已经操作完毕
    return jarr

# src, pmask 都出了问题
def pix_logo_rm(src_, wm_, local, pmask_):
    sw, sh = local

    src = np.array(src_)
    wm = np.array(wm_)
    pmask = np.array(pmask_)

    # 从wh开始遍历每一个元素值
    print("输入shape：",
        np.shape(src),  # (348, 500, 3)
        np.shape(wm),  # (53, 371, 4)
        np.shape(pmask),  # (53, 371)
        # pmask
    )  # 输入shape： (348, 500, 3) (53, 371, 4) (53, 371)

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

                # 23, 20 => 28, 40 之间的像素差
                # if i>20 and i<40 and j>23 and j<28:
                #     prt = True
                # else:
                #     prt = False
                timg[y][x][ci] = int(get_initpix(vsrc, valp, vlogo, prt=False))

    # 至此，整个 timg 计算完毕
    # print(np.shape(timg))
    ttimg = Image.fromarray(timg).convert(mode="RGB")

    # ttimg.show()
    # ttimg.save("tpix/yy___.jpg")  # 通过水印图得到原图
    print("还原计算完毕！")
    return np.array(ttimg)


def change_a(x):
    r, g, b, a = get_rgb4a(x)  # 获取理论的 jd-logo像素值  # 195 23 33 77
    # r, g, b, a = 212, 7, 7, 71  # 这里是 自制logo核心像素值
    # 显示结果正确【与原jd-logo去除颜色一致】

    # 根据rgba生成一张图  # 800, 800
    # h, w = 800, 800
    h, w = 71, 386
    ndarr = np.ones((h, w, 4), dtype=np.uint8)
    ndarr[:, :, 0] = r
    ndarr[:, :, 1] = g
    ndarr[:, :, 2] = b
    ndarr[:, :, 3] = a
    # ndarr[:, :, 3] = 255

    # 构造纯logo色的图像
    inlogo = Image.fromarray(ndarr).convert(mode="RGBA")
    paste_mask = inlogo.split()[3]
    print(inlogo.size, inlogo.mode, paste_mask.size, paste_mask.mode)  # (800, 800, 4) (800, 800)


    # 构造纯白底 + logo水印后图像
    inpic = Image.fromarray(((np.ones((h, w, 3), dtype=np.uint8))*255)).convert(mode="RGB")
    inpic.paste(inlogo, (0, 0), mask=paste_mask)  # 在纯白底上构造的 水印图
    # 【paste有问题】

    # 如果整张图当做 255 上加logo后的色彩
    # 这个东西没用，因为终极目标是还原，并不清楚logo的原色是怎样， 但a值可以确定一定大于70
    # 所以应该做：
    #       在不同的a值对 jd-logo字体处做消去，看看能否还原为纯白

    # print(r, g, b, a)
    # jarr = get_jd_rmby_rgba(jimg, r, g, b, a)  # 除水印，得到整张图

    # inpic.show()
    # jimg.show()
    # disarr = np.array(inpic, dtype=np.int)[20:40, 23:28,:] - np.array(jimg, dtype=np.int)[20:40, 23:28,:]
    # print(disarr)
    # 建议查看 23, 20 => 28, 40 之间的像素差

    # ypic = pix_logo_rm(inpic, inlogo, (0, 0), paste_mask)  # 除水印后
    ypic = pix_logo_rm(jimg, inlogo, (0, 0), paste_mask)  # 素材图按照计划除水印后

    # 为什么paste的可以还原，而jd的logo没法还原？

    # img = jarr
    img = cv2.cvtColor(ypic, cv2.COLOR_RGBA2BGRA)  # 显示logo 纯色图
    cv2.imshow("myImg", img)



def Achange_a(x):
    print("建议调节到 71，就别动了！")

    # 构造纯logo色的图像
    w, h = jimg.size
    timg = Image.fromarray(((np.ones((h, w, 3), dtype=np.uint8))*255)).convert(mode="RGB")  # 构造原图

    # 在计算整个logo之前，进行前后景分类，得到一个alpha通道
    # get_Binclass(jimg)
    # alp = get_alph(jimg)  # 01矩阵
    # alp, pimg = get_alph_blur(jimg)  # 01矩阵, 前后景区分预处理
    alp, pimg = getMask_opc_logo(jimg)  # 01矩阵, 前后景区分预处理

    # inlogo = getL_PT(x, jimg, timg)  # 获取logo（显示像素+原图像素） 【直接按逆配方反算，然而显示有噪声】
    # inlogo = getL_PT_v1(x, jimg, timg)  # 分类计算
    inlogo = getL_PT_v2(x, alp, pimg, timg)  # 分前后景0/1 + 固定a值

    # 整张图，先用LPT处理，字体上用

    paste_mask = inlogo.split()[3]
    print(inlogo.size, inlogo.mode, paste_mask.size, paste_mask.mode)

    # 保存之前做一下透明度处理： 先采用 极化透明
    # inlogo = opc_proc(inlogo)

    # 建议保存logo图
    al_savp = "./a_logo/logo_X.png"
    inlogo.save(al_savp)

    # jimg.show()  # 原图
    # inlogo.show()  # 生成的logo图

    # 构造原图 + logo后的图像 [发现手帖的 像素值过大，]
    inpic = Image.fromarray(((np.ones((h, w, 3), dtype=np.uint8)) * 255)).convert(mode="RGB")
    inpic.paste(inlogo, (0, 0), mask=paste_mask)  # 在纯白底上构造的 水印图
    inpic.show()  # 自构图

    ypic = pix_logo_rm(jimg, inlogo, (0, 0), paste_mask)
    Image.fromarray(ypic).convert("RGB").save("rmed.jpg")  # 查看蓝绿色 像素值
    # 素材图按照计划除水印后

    img = cv2.cvtColor(ypic, cv2.COLOR_RGBA2BGRA)  # 显示logo 纯色图
    cv2.imshow("myImg", img)

"""
根据一张纯白图+logo的 显示图像，反求水印图

测试证明， 透明度并非常数，而是确实有一个渐变过程:

那么要不  先不考虑渐变，采用一个 极化透明度：
    即 要么为0， 要么为a

"""
def test_slide():
    # jimg = Image.open('./yy_rml.jpg')
    jimg.show()

    cv2.namedWindow('myImg', cv2.WINDOW_NORMAL)
    # cv2.createTrackbar('a', 'myImg', 70, 255, change_a)
    cv2.createTrackbar('a', 'myImg', 70, 255, Achange_a)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        cv2.getTrackbarPos('a', 'myImg')

    cv2.destroyAllWindows()


if __name__ == '__main__':

    # jimg = Image.open('./yy_rml.jpg')
    # jimg = Image.open('./jd_pimg.jpg')  # size有误
    jimg = Image.open('./jd_sc.jpg')  # 纯logo补丁
    test_slide()


    pass




"""
关于毛边问题：
    1.采用蒙版处理；
    2.采用边缘

"""






