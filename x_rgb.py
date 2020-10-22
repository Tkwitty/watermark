# coding=utf-8
import cv2
from LogoRemoval import dir_logorm_4_png
import numpy as np
from PIL import Image


def change_r(x):
    # 根据x生成一张 Img-png
    cl.r = r_ + x  # 永久修改g值
    r, g, b = cl.getRgb()
    for i in range(lh):
        for j in range(lw):  # 因为666的alpha通道没问题77（0.3），现在微调颜色值
            a = nlg[i][j][3]
            pnlogo[i][j] = np.array([r, g, b, a])
    print("调节的像素值 rgb：", r, g, b)
    plogo = Image.fromarray(pnlogo)
    dir_logorm_4_png(dir="_rev_/xrgb/", png=plogo)
    cv2.imshow("myImg", img)


def change_g(x):
    # 根据x生成一张 Img-png
    cl.g = g_ + x  # 永久修改g值
    r, g, b = cl.getRgb()
    for i in range(lh):
        for j in range(lw):  # 因为666的alpha通道没问题77（0.3），现在微调颜色值
            a = nlg[i][j][3]
            pnlogo[i][j] = np.array([r, g, b, a])
    print("调节的像素值 rgb：", r, g, b)
    plogo = Image.fromarray(pnlogo)
    dir_logorm_4_png(dir="_rev_/xrgb/", png=plogo)

    cv2.imshow("myImg", img)


def change_b(x):
    # 根据x生成一张 Img-png
    cl.b = b_ + x  # 永久修改g值
    r, g, b = cl.getRgb()
    for i in range(lh):
        for j in range(lw):  # 因为666的alpha通道没问题77（0.3），现在微调颜色值
            a = nlg[i][j][3]
            pnlogo[i][j] = np.array([r, g, b, a])
    print("调节的像素值 rgb：", r, g, b)
    plogo = Image.fromarray(pnlogo)
    dir_logorm_4_png(dir="_rev_/xrgb/", png=plogo)

    cv2.imshow("myImg", img)

img = cv2.imread('JD/bird.jpg')  # 有个载体

lg_lk = Image.open("opc/666.png")
nlg = np.asarray(lg_lk, dtype=np.uint8)
pnlogo = nlg.copy()
lw, lh = lg_lk.size

cv2.namedWindow('myImg')
cv2.namedWindow('myImg', cv2.WINDOW_NORMAL)
cv2.createTrackbar('r', 'myImg', 5, 20, change_r)  # 基于 185
cv2.createTrackbar('g', 'myImg', 5, 20, change_g)  # 基于 10
cv2.createTrackbar('b', 'myImg', 5, 20, change_b)  # 基于 30


# 191,16,36
# 199,16,34 √
class Color():
    def __init__(self):
        self.r = 185
        self.g = 10
        self.b = 30

    def getRgb(self):
        return self.r, self.g, self.b

cl = Color()
while True:
    r_, g_, b_ = 185, 10, 30

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    cv2.getTrackbarPos('r', 'myImg')
    cv2.getTrackbarPos('g', 'myImg')
    cv2.getTrackbarPos('b', 'myImg')

cv2.destroyAllWindows()


# 编写 rgb滑动 调色程序, 显示消去效果图
# 188~194, 12~22, 30~40
