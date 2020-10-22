# 371x53 像素转换为 374x54
from PIL import Image
import numpy as np
import cv2

def otest():
    jpic = "370x52.png"
    jp = Image.open(jpic)
    znp = np.zeros((54, 374, 4), dtype=np.uint8)
    iz = Image.fromarray(znp)
    iz.paste(jp, (2, 1))
    iz.save("_374x54.png")
    iz.show()


# 那666的图，修改颜色值
# 新抠图的 alpha通道 [收益权]
def logo_prod(jpic):
    lg_lk = Image.open(jpic)
    nlg = np.asarray(lg_lk, dtype=np.uint8)
    w, h = lg_lk.size

    pnlogo = nlg.copy()
    for i in range(h):
        for j in range(w):  # 因为666的alpha通道没问题77（0.3），现在微调颜色值
            a = nlg[i][j][3]
            pnlogo[i][j] = np.array([199, 16, 34, a])  # 感觉还有点红色  # 192不行，
            # pnlogo[i][j][0] * 194
            # 191,16,36

    plogo = Image.fromarray(pnlogo)
    plogo.save("199_16_34.png")
    print("over！")


def edit_al_png(apng, al):
    print("根据新生alpha通道制作logo！")
    nlg = apng
    h, w, c = np.shape(apng)

    print(np.shape(nlg), np.shape(al))
    pnlogo = nlg.copy()
    for i in range(h):
        for j in range(w):  # 因为666的alpha通道没问题77（0.3），现在微调颜色值
            r, g, b = tuple(nlg[i][j][:3])
            a = al[i][j]
            pnlogo[i][j] = np.array([r, g, b, a])  # 感觉还有点红色  # 192不行，

    plogo = Image.fromarray(pnlogo)
    plogo.save("lkp_al_merge.png")


def edit_al_ft_png(apng, al, ft):
    print("根据新生alpha通道制作logo！")
    nlg = apng
    h, w, c = np.shape(apng)

    print(np.shape(nlg), np.shape(al))
    pnlogo = nlg.copy()
    for i in range(h):
        for j in range(w):  # 因为666的alpha通道没问题77（0.3），现在微调颜色值
            r, g, b = tuple(nlg[i][j][:3])
            a = al[i][j] if ft[i][j]>0 else 0
            pnlogo[i][j] = np.array([r, g, b, a])  # 感觉还有点红色  # 192不行，

    plogo = Image.fromarray(pnlogo)
    plogo.save("lkp_af_merge.png")


def prt_Np(arr, text="打印np-arr：", ):
    print(text)
    h, w = np.shape(arr)
    for i in range(h):
        linel = [str(pix) for pix in arr[i]]
        lstr = ' '.join(linel)
        print(lstr)


def a_outer_edt(slogo, adist, plogo, mode="newr"):
    print("修改外围a值：", np.shape(slogo), np.shape(plogo))
    # 按照显示像素值和 logo像素值，反求其a值
    new_logo = slogo.copy()
    h, w, c = np.shape(slogo)
    for i in range(h):
        for j in range(w):
            pr, pg, pb = tuple(plogo[i][j][:3])  # 白底 + logo = 显示图
            lr, lg, lb, la = tuple(slogo[i][j])  # 原始纯logo
            sr, sg, sb = 255, 255, 255  # 白底

            if adist[i][j] == 1:
                # new_ar = int((pr - sr) * 255/(lr - sr))  # 仅按照红色通道进行推算
                new_ag = int((pg - sg) * 255/(lg - sg))  # 仅按照g通道进行推算
                new_ab = int((pb - sb) * 255/(lb - sb))  # 仅按照b通道进行推算

                # print(pr, sr, lr, new_a_i)
                # if new_a_i > 77: new_a_i = 77
                # new_a = int(np.mean([new_ar, new_ag, new_ab]))
                new_a = int(np.mean([new_ag, new_ab]))
                # 感觉 r在变化：1。采用rgb均值，2。采用bg均值，3。采用gb均值+r变化

                # new_a =

                # 采用gb的a均值，并通过a计算r的变化
                new_r = lr
                if new_a > 0:
                    new_alp = new_a/255
                    new_r = int((pr - (1-new_alp)*sr)/new_alp)  # 反向计算alp
                    # if new_r > 255: new_r = 255
                    if new_r > 199: new_r = 199
                    # print(new_r)

                # 解燃眉之急，最后一行变成0 [因为c字母最下变有一行多余的框架]

                # print(lr, new_r, pr, new_a)
                # new_logo[i][j] = np.array([r, g, b, new_a])
                # new_logo[i][j] = np.array([new_r, lg, lb, new_a])
                # new_logo[i][j] = np.array([pr, lg, lb, new_a])  # r 高了会变绿，低了会留红， 中间是199：残留微红
                # new_logo[i][j] = np.array([lr+10, lg, lb, new_a])  # r 内层浅绿，外层浅红，lr不能变
                # new_logo[i][j] = np.array([lr, lg, lb, new_a])  # r 内层浅绿，外层浅红，lr不能变
                # new_logo[i][j] = np.array([lr, lg, lb, new_a])  # 有亮红色 全部191 【内圈】
                # new_logo[i][j] = np.array([new_r, lg, lb, new_a])  # 细微红边 [外圈]

                if mode == "newra":  # 外圈
                    new_logo[i][j] = np.array([new_r, lg, lb, new_a])  # 细微红边 [外圈]
                elif mode == "old":  # 内圈
                    # new_logo[i][j] = np.array([new_r, lg, lb, new_a])  # 内圈微微亮红
                    # new_logo[i][j] = np.array([new_r, lg, lb, la])  # 有亮红色【内圈，白底未去，残留亮红】
                    new_logo[i][j] = np.array([lr, lg, lb, la])  # 目前最佳效果（内圈未操作）：o字母内圈太亮
                    # new_logo[i][j] = np.array([lr, lg, lb, new_a])
                    # 针对o字母，内圈采用 newa+l
                elif mode == "newr":
                    new_logo[i][j] = np.array([new_r, lg, lb, la])
                elif mode == "newa":
                    new_logo[i][j] = np.array([lr, lg, lb, new_a])
                else:
                    print("模式不对！！！")

            # else:  # 非轮廓位
            #     if i == h - 1:
            #         new_a = 0
            #         new_logo[i][j] = np.array([lr, lg, lb, new_a])  # r 内层浅绿，外层浅红，外层a升高

    prt_Np(new_logo[:, :, 0])
    # prt_Np(new_logo[:, :, 0])
    # plogo = Image.fromarray(new_logo)
    # plogo.save("lkp_tmd3.png")  # o=字母不太行，那就腐蚀两圈

    return new_logo

def darw_ops_margin(npng, adist1, dcolor, spath):
    dr, dg, db = dcolor
    dmat = npng.copy()
    h, w, c = np.shape(npng)
    for i in range(h):
        for j in range(w):
            if adist1[i][j] > 0 :
                dmat[i][j] = np.array([dr, dg, db, 255])

    # Image.fromarray(dmat).save("margin_ops.jpg")
    Image.fromarray(dmat).save(spath)


# a3：只使用r的不行，既有红残留，又有绿残留    【还有一种，高位截止77】
# 采用rgb均值不行，没有红残留，但有绿残留，只是绿残留变浅
# 只采用gb均值，效果变好，还剩浅浅的一圈浅红色
# 采用gb均值，并且根据a计算 r的新值，发现还有一丢丢亮丽的红色残留


def erode_oterLayer(picp):
    # 采用cv2的图像处理 np-arr 腐蚀一下 alpha通道即可
    # 也可以采用轮廓图源【不可取】
    png = Image.open(picp)
    a_mat = np.array(png.split()[3])  # nlog 为
    # dst_logo = png.copy  # copy当做修改源

    # 全图rgb，外围操作数, 用ftp的外围，且不实用他的a层
    _, _, nops = deal_wpng2png("alogo.png")  # 根据alogo获取其外轮廓, 再腐蚀
    # 基于核心膨胀, 除去ft之外

    # a_erd = cv2.erode(RedThresh, kernel)
    """
    看了具体的图像发现，很多情况下不是多了一圈，而是边缘渐变的问题
    就是说，图像的粗细其实一样的，只是原本的边缘渐变开始的更早，而且最后一圈消失的更小
    轮廓不变，仅对图像骨架点位
    阈值化，腐蚀； 查表法实现轮廓细化骨架，针对骨架核心
    
    阈值化；
        腐蚀1层；将这一层的alpha =* 0.3（最外层腐蚀）
        再腐蚀一层，将这一层的alpha =* 0.6 
        再腐蚀一层，将这一层的alpha =* 0.8
    """
    prt_Np(a_mat)
    print("打印腐蚀后的像素值")
    kernel = np.ones((3, 3), np.uint8)
    a_erd_1 = cv2.erode(a_mat, kernel)  # 腐蚀
    prt_Np(a_erd_1)  # 打印边缘的 alpha 值

    a_erd_2 = cv2.erode(a_erd_1, kernel)  # 腐蚀, 腐蚀a层不对，应该腐蚀
    prt_Np(a_erd_2)  # 打印边缘的 alpha 值

    # 在两个图层， 值不相同的地方，采用中值（求和/2）
    # print("腐蚀+弱化的像素值")
    # a_e_mean = np.asarray((a_mat + a_erd_1)/2, dtype=np.uint8)
    # prt_Np(a_e_mean)

    """
    可以根据 标准的logo图，在已知rgb像素值的情况下，根据显示像素值，反求a值，
    只针对腐蚀消去的一圈进行a值的反向计算替换，其它位置不处理，轮廓外围按照大轮廓
    """

    # 取出要操作的边缘区域： 将原logo-a图，减去腐蚀后的a图，即边缘的点位集合
    # 对于这些点位集合分别根据 0l显示值 与像素值 反推其 a值，替换到 logo-a图的 a涂层上
    # 腐蚀层，即两图a值不想等的区域
    # adist = a_mat - a_erd_1  # 01 mat
    # adist[adist != 0] = 1  # 腐蚀的两圈

    a_mat[a_mat > 0] = 1
    a_erd_1[a_erd_1 > 0] = 1
    adist1 = a_mat - a_erd_1
    prt_Np(adist1)  # 腐蚀的一圈 (第一圈)

    a_erd_2[a_erd_2 > 0] = 1
    adist2 = a_mat - a_erd_2
    prt_Np(adist2)  # 腐蚀的两圈

    adist_2 = adist2 - adist1
    # adist_2[:, :118] = 0
    # adist_2[:, 170:] = 0
    # adist1[:, 118:170] = 0
    prt_Np(adist_2, "腐蚀的第2圈")  # 腐蚀的第2圈, 只有o字母

    # plogo = Image.open("0logo.jpg")
    plogo = Image.open("alogo.png")  # 经手工优化的 logo显示图
    npng = np.array(plogo)

    lprod1 = a_outer_edt(np.array(png), nops, npng, mode="newra")  # input png, edit points, display img
    lprod2 = a_outer_edt(lprod1, adist_2, npng, mode="old")  # input png, edit points, display img

    # 绘制一下，alogo哪些位置其作用了？
    darw_ops_margin(npng, nops, (0, 255, 0), "margin1_ops.png")
    darw_ops_margin(npng, nops, (0, 0, 255), "margin2_ops.png")

    plogo = Image.fromarray(lprod2)
    plogo.save("lkp_tmd9.png")  # 腐蚀两圈，外圈newr，内圈oldr

    # ath = a_mat.copy()
    # ath[ath > 0] = 1
    # image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 以上处理得到的 alpha层 进行png融合，保存图像
    # edit_al_png(np.array(png), a_e_mean)

def get_png_merge(ipic, apic):
    # 采用cv2的图像处理 np-arr 腐蚀一下 alpha通道即可
    # 也可以采用轮廓图源【不可取】
    apng = Image.open(apic)
    a_mat = np.array(apng.split()[3])  # nlog 为
    prt_Np(a_mat)

    ipng = Image.open(ipic)
    edit_al_png(np.array(ipng), a_mat)


def get_png_jpg_merge(ipic, apic, fpic):  # rgb, a, ft
    # 采用cv2的图像处理 np-arr 腐蚀一下 alpha通道即可
    # 也可以采用轮廓图源【不可取】
    apng = Image.open(apic)
    a_mat = np.array(apng.split()[3])  # nlog 为
    prt_Np(a_mat)

    fjpg = Image.open(fpic)
    f_mat = np.array(fjpg.split()[0])  # nlog 为
    prt_Np(f_mat)

    ipng = Image.open(ipic)
    edit_al_ft_png(np.array(ipng), a_mat, f_mat)


def prt_Npc(arr, text="打印np-arr：", ):
    print(text)
    # h, w = np.shape(arr)
    h, w, c = np.shape(arr)
    for i in range(h):
        # linel = [str(pix) for pix in arr[i]]
        linel = ["(" + ','.join([str(pix[ci]) for ci in range(c)]) + ")" for pix in arr[i]]
        lstr = ' '.join(linel)
        print(lstr)


def deal_wpng2png(wpic):
    wpng = Image.open(wpic)
    w_mat = np.array(wpng)  # nlog 为
    h,w,c = np.shape(w_mat)
    prt_Npc(w_mat)

    nblk = np.ones((h,w,c), dtype=np.uint8)*255
    wdst = nblk - w_mat
    wdstg = wdst[:, :, 1]
    wdstg[wdstg>0] = 255

    # cv2.imshow("hello", wdstr)
    # cv2.waitKey()
    prt_Np(wdstg)

    wnr = w_mat[:, :, 0]  # 小于等于238的为字体核心
    wng = w_mat[:, :, 1]  # 小于等于238的为字体核心
    wnb = w_mat[:, :, 2]  # 小于等于238的为字体核心
    wnr[wnr > 238] = 0
    wng[wng > 183] = 0
    wnb[wnb > 188] = 0
    wnk = wnr*wng*wnb  # 非0即>0
    wnk[wnk>0]=255
    prt_Np(wnk)

    # ft - ftk = 轮廓
    ilk = wdstg - wnk
    prt_Np(ilk)

    # 检测ft的轮廓
    # cloneimage, contours, hierarchy = cv2.findContours(wdstg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours(wdstg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # for i, contour in enumerate(contours):  # [pints]
    #     print(np.shape(contours))
    #     cv2.drawContours(w_mat, contours, i, (0, 255, 0), 1)
    """w_mat
    # 函数cv2.drawContours(image, contours, contourIdx, color, thickness, lineType, hierarchy, maxLevel, offset)
    # 第一个参数是一张图片，可以是原图或者其他。
    # 第二个参数是轮廓，也可以说是cv2.findContours()找出来的点集，一个列表。
    # 第三个参数是对轮廓（第二个参数）的索引，当需要绘制独立轮廓时很有用，若要全部绘制可设为-1。`
    # 接下来的参数是轮廓的颜色和厚度
    """
    # cv2.imshow("detect contours", w_mat)
    # cv2.waitKey()

    # 利用ft的外轮廓，a值，制作整张png原图； a：

    Image.fromarray(wdstg).save("ft.jpg")  # 全字体
    Image.fromarray(wnr).save("ftk.jpg")  # 下面把字体核心区分开【r通道都等于238，即小于等于238的为字体核心】
    Image.fromarray(ilk).save("ilk.jpg")  # 下面把字体核心区分开【r通道都等于238，即小于等于238的为字体核心】
    return wdstg, wnr, ilk   # 0/255


# jpic = "666.png"  # 渐变
# logo_prod(jpic)

# erode_oterLayer("199_16_34.png")
# # erode_oterLayer("lkp.png")  # 直接拿这张图的轮廓透明度

# deal_wpng2png("alogo.png")

# get_png_merge("199_16_34.png", "lkp.png")  # 直接拿这张图的轮廓透明度
# get_png_merge("199_16_34.png", "alogo.png")  # 处理一下无透明度再来混合
# get_png_jpg_merge("199_16_34.png", "lkp.png", "ft.jpg")  # 处理一下无透明度再来混合

# erode_oterLayer("lkp_o_merge.png")  # 直接拿这张图的轮廓透明度
# erode_oterLayer("lkp_af_merge.png")  # 直接拿这张图的轮廓透明度
erode_oterLayer("lkp_af_merge.png")  # 直接拿这张图的轮廓透明度


# l0 = cv2.imread("0logo.jpg")
# cv2.imshow("hello", l0)
# cv2.waitKey()
