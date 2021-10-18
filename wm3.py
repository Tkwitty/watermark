from PIL import Image, ImageDraw
import cv2
import numpy as np
import numpy

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
    cv2.waitKey()


def get_water2(srcp, maskp, outp):
    src = cv2.imread(srcp)
    mask = cv2.imread(maskp)

    save = numpy.zeros(src.shape, numpy.uint8)  # 创建一张空图像用于保存
    for row in range(src.shape[0]):  # r
        for col in range(src.shape[1]):  # g
            for channel in range(src.shape[2]):  # b
                if mask[row, col, channel] == 0:
                    val = 0
                else:
                    reverse_val = 255 - src[row, col, channel]
                    val = 255 - reverse_val * 256 / mask[row, col, channel]
                    if val < 0:
                        val = 0
                save[row, col, channel] = val

    # cv2.imwrite(outp + '/rst2.jpg', save)
    cv2.imshow("hi", src)
    cv2.imshow("fine", mask)
    cv2.imshow("hello", save)
    cv2.waitKey()


def get_loc(img_size, wm_size, mode='rb'):
    x, y = img_size
    w, h = wm_size
    rst = (0, 0)
    if mode == 'rb':
        rst = (x - w, y - h)
    elif mode == 'md':  # 正中
        rst = ((x-w)//2, (y-h)//2)
    return rst


def add_watermark_to_image(image, watermark, alph):
    rgba_image = image.convert('RGBA')
    rgba_watermark = watermark.convert('RGBA')
    image_x, image_y = rgba_image.size
    watermark_x, watermark_y = rgba_watermark.size

    # 缩放图片
    scale = 10
    watermark_scale = max(image_x / (scale * watermark_x), image_y / (scale * watermark_y))
    new_size = (int(watermark_x * watermark_scale), int(watermark_y * watermark_scale))
    rgba_watermark = rgba_watermark.resize(new_size, resample=Image.ANTIALIAS)

    # 透明度
    # rgba_watermark_mask = rgba_watermark.convert("L").point(lambda x: min(x, 180))
    rgba_watermark_mask = rgba_watermark.convert("L").point(lambda x: x*alph)
    rgba_watermark.putalpha(rgba_watermark_mask)
    watermark_x, watermark_y = rgba_watermark.size

    rgba_image.paste(
        rgba_watermark,
        # get_loc((image_x, image_y), (watermark_x, watermark_y), mode='rb'),
        get_loc((image_x, image_y), (watermark_x, watermark_y), mode='md'),
        rgba_watermark_mask
    )  # 水印位置

    return rgba_image

def test_lenawm(srcp, wmp, alph=0.3):
    im_before = Image.open(srcp)
    im_watermark = Image.open(wmp)
    im_after = add_watermark_to_image(im_before, im_watermark, alph)  # 加水印

    im_before.show()
    im_after.show()
    # im.save('im_after.jpg')


# get_water()
# get_water2()

# 加水印
# src = 'E:/__TestData__/cb14.jpg'
# # logo = 'jd-logo.jpg'
# logo = 'zcy_logo.png'
# test_lenawm(src, logo, alph=0.8)


def del_Wm(mode=1):
    if mode == 1:
        # 去水印 1
        # srcp = "test_dewm/raw.png"
        # mb = "test_dewm/wm_tmp.png"
        srcp = "clean_test/aha_0.png"
        mb = "clean_test/aha_0_logo.png"
        get_water(srcp, mb, "test_dewm")

    elif mode == 2:
        # 去水印 2
        srcp = "test_dewm/raw.png"
        mb = "test_dewm/gw2.png"
        get_water2(srcp, mb, "test_dewm")

    else:
        print("mode 不匹配！")


def draw_min_rect_circle(img, cnts):  # conts = contours
    img = np.copy(img)
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # blue

        # min_rect = cv2.minAreaRect(cnt)  # min_area_rectangle
        # min_rect = np.int0(cv2.boxPoints(min_rect))
        # cv2.drawContours(img, [min_rect], 0, (0, 255, 0), 2)  # green
        #
        # (x, y), radius = cv2.minEnclosingCircle(cnt)
        # center, radius = (int(x), int(y)), int(radius)  # for the minimum enclosing circle
        # img = cv2.circle(img, center, radius, (0, 0, 255), 2)  # red
    return img

def extractJD(jd_logo_raw):
    jd_raw = cv2.imread(jd_logo_raw)
    cv2.imshow("jdraw", jd_raw)
    cv2.waitKey()

    jd_gray = cv2.cvtColor(jd_raw, cv2.COLOR_BGR2GRAY)

    print(jd_raw.shape)
    # 把jd扣出
    # 把像素纯净化【可以先不做】
    # 把非logo部分 透明化
    # 制作一张新图，在logo的坐标位置内部，提取红色通道，red>125的保留且最大化255
    # 其他通道为0，其他像素值为透明
    b, g, r = cv2.split(jd_raw)
    # cv2.imshow("g", g)
    # cv2.imshow("b", b)
    cv2.imshow("r", r)  # 红色通道
    cv2.imshow("gray", jd_gray)  # 灰度通道
    cv2.waitKey()

    ret, jd_th = cv2.threshold(jd_gray, thresh=150, maxval=255, type=cv2.THRESH_BINARY)
    print(jd_th)

    contours, hierarchy = cv2.findContours(jd_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(jd_raw, contours, -1, (255, 0, 0), 1)

    jd_fc = draw_min_rect_circle(jd_raw, contours)

    merged = cv2.merge([b, g, r])
    cv2.imshow("mgd", jd_fc)
    cv2.waitKey()
    pass

# 做一个好的模版图
def tichun_jdlogo(jd_logo_):
    jd_raw = cv2.imread(jd_logo_)

    kernel_size = (3, 3)
    jd_rawg = cv2.GaussianBlur(jd_raw, kernel_size, 1.5)

    cv2.imshow("jdraw", jd_raw)
    b, g, r = cv2.split(jd_rawg)

    cv2.waitKey()
    # jd_gray = cv2.cvtColor(jd_raw, cv2.COLOR_BGR2GRAY)
    ret, jd_th = cv2.threshold(r, thresh=200, maxval=255, type=cv2.THRESH_BINARY)  # 对r阈值化
    # cv2.imshow("gary", jd_gray)
    cv2.imshow("thed", jd_th)

    alpha = np.ones(r.shape, dtype=r.dtype) * 100  # 0～255
    alpha[alpha > 127] = 255
    alpha[alpha < 127] = 0

    b = np.zeros(r.shape, dtype=r.dtype)
    g = np.zeros(r.shape, dtype=r.dtype)

    merged = cv2.merge([b, g, r, alpha])

    # kernel = np.array([[0, -1, 0],
    #                    [-1, 5, -1],
    #                    [0, -1, 0]])  # 定义卷积核
    # m_hanced = cv2.filter2D(merged, -1, kernel)  # 进行卷积运算



    merged_ = cv2.cvtColor(merged, cv2.COLOR_BGR2BGRA)

    cv2.imshow("merged", merged_)
    cv2.waitKey()

    cv2.imwrite("jdlg.png", merged_)
    pass

# 灰度图，阈值化，
if __name__ == '__main__':

    del_Wm(1)
    # extractJD("logos/jd_logo_raw.jpeg")
    # tichun_jdlogo("logos/jd_logo_.png")


# 用 wm1 可以反水印，制作jd-logo，
