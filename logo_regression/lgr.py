# coding=utf-8
import numpy as np
from PIL import Image
import os

def model_x2src_w(x, ws):  # 消去模型 预测
    alpha = ws[:, :, -1] / 255  # 最后一层透明层
    lrgb = ws[:, :, :3]  # logo的rbg值
    # x = y*(1-a) + l*a
    # y = (x-l*a)/(1-a)
    y = (x - lrgb * alpha) / (1 - alpha)
    return y

def model_learn(dety, ws, lrate):
    h, w, c = np.shape(dety)  # 推荐采用矩阵计算
    ws_ = ws.copy()
    for i in range(h):
        for j in range(w):  # 此处应该只有rgb的梯度，根据每个梯度计算ws更新
            rd, gd, bd = tuple(dety[i][j])  # 拆分梯度值
            wr, wg, wb, wa = tuple(ws[i][j])  # 拆分old 权重参数
            wr = wr + lrate * (rd * (wa-1) / wa)
            wg = wg + lrate * (gd * (wa-1) / wa)
            wb = wb + lrate * (bd * (wa-1) / wa)
            delt_a = (np.mean([rd, gd, bd]) - np.mean([wr, wg, wb])) / (1-0)
            wa = wa + lrate * ()
    return 0

"""内部计算全部以0～1为准"""
def learn2everycolor(xnys):
    ws = np.ones((373, 54), dtype=np.float) * 0.5  # 初始值是一张全部为0.5的logo图片
    lrate = 0.04  # 学习率 (1/255)
    for x, y in xnys:  # 遍历所有训练样本，训练参数
        # ws（x）= y' ～ y
        yp = model_x2src_w(x, ws)
        dety = y - yp  # 差值
        loss = np.mean(dety)
        # 参数区位： i，j，k  位置+通道
        ws = model_learn(dety, ws, lrate)  # 通过梯度图，原参数图  更新获取新参数图


"""
    针对800x800中对于 373x54
    bw, bh = img.size  # shape的宽高位置和pil的有区别！！！
    sw, sh = watermark.size
    iw = int((bw - sw) / 2)
    ih = int((bh - sh) / 2)
"""
def loadDataset(dpath):
    xpath = os.path.join(dpath, "tx")
    ypath = os.path.join(dpath, "ty")
    ximgs = os.listdir(xpath)
    yimgs = os.listdir(ypath)
    xdata = []
    ydata = []
    if len(ximgs) == len(yimgs):
        print("数据集大小：", len(ximgs))
    for file in ximgs:
        iname, iext = tuple(file.split('.'))
        xname = file
        yname = iname + "_y." + iext
        ximg = Image.open(os.path.join(xpath, xname))
        yimg = Image.open(os.path.join(ypath, yname))
        xdata.append(np.array(ximg))
        ydata.append(np.array(yimg))

    return np.array(xdata), np.array(ydata)


# 针对logo区域的数据集
def load_logoDataset(dpath, logo_size):
    # xpath = os.path.join(dpath, "tx")
    xpath = os.path.join(dpath, "zx")
    ypath = os.path.join(dpath, "ty")
    ximgs = os.listdir(xpath)
    yimgs = os.listdir(ypath)
    xdata = []
    ydata = []
    if len(ximgs) == len(yimgs):
        print("数据集大小：", len(ximgs))

    for file in ximgs:
        print("reading img ", file)
        iname, iext = tuple(file.split('.'))
        xname = file
        yname = iname + "_y." + iext
        ximg = Image.open(os.path.join(xpath, xname))
        yimg = Image.open(os.path.join(ypath, yname))
        w, h = ximg.size
        lw, lh = logo_size
        iw, ih = int((w - lw) / 2), int((h - lh) / 2)
        xarr = np.array(ximg)/255  # 采用0～1浮点值
        yarr = np.array(yimg)/255
        if np.shape(xarr) == np.shape(yarr):
            try:
                xdata.append(xarr[ih:ih+lh, iw:iw+lw, :3])  # 只取logo区域的rgb通道
                ydata.append(yarr[ih:ih+lh, iw:iw+lw, :3])
            except:
                print('添加异常，pass')
        else:
            print("shape 异常：", np.shape(xarr), np.shape(yarr))
        if len(xdata) == len(ydata):
            print('完成数据加载...')
    return np.array(xdata), np.array(ydata)


# xs, ys = load_logoDataset("datas", (373, 54))
# print(np.shape(xs), np.shape(ys))

# 针对logo区域的数据集
def load_gray_logoDataset(dpath, logo_size):
    xpath = os.path.join(dpath, "tx")
    ypath = os.path.join(dpath, "ty")
    ximgs = os.listdir(xpath)
    yimgs = os.listdir(ypath)
    xdata = []
    ydata = []
    if len(ximgs) == len(yimgs):
        print("数据集大小：", len(ximgs))

    for file in ximgs:
        print("reading img ", file)
        iname, iext = tuple(file.split('.'))
        xname = file
        yname = iname + "_y." + iext
        ximg = Image.open(os.path.join(xpath, xname))
        yimg = Image.open(os.path.join(ypath, yname))
        w, h = ximg.size
        lw, lh = logo_size
        iw, ih = int((w - lw) / 2), int((h - lh) / 2)
        # xarr = np.array(ximg.convert(mode="L"))/255  # 采用0～1浮点值
        # yarr = np.array(yimg.convert(mode="L"))/255
        # xarr = np.array(ximg.convert(mode="L"))
        # yarr = np.array(yimg.convert(mode="L"))
        xarr = np.array(ximg)
        yarr = np.array(yimg)
        if np.shape(xarr) == np.shape(yarr):
            xdata.append(xarr[ih:ih+lh, iw:iw+lw, :3])  # 只取logo区域的rgb通道
            ydata.append(yarr[ih:ih+lh, iw:iw+lw, :3])
            print("datas shape：", np.shape(xarr), np.shape(yarr))
        else:
            print("shape 异常：", np.shape(xarr), np.shape(yarr))
    return np.array(xdata), np.array(ydata)


# 从dir加载图片的logo位置的数据样本
def load_delogo_dataset(tpath, logo_size):
    timgs = os.listdir(tpath)
    tdata = []
    imgps = []
    print("测试集大小：", len(timgs))
    for file in timgs:
        print("reading img ", file)
        if not file.endswith('.jpg'):
            continue
        ximg = Image.open(os.path.join(tpath, file))
        w, h = ximg.size
        lw, lh = logo_size
        iw, ih = int((w - lw) / 2), int((h - lh) / 2)
        xarr = np.array(ximg)
        if np.shape(xarr)[2] > 2:
            tdata.append(xarr[ih:ih + lh, iw:iw + lw, :3])
            imgps.append(os.path.join(tpath, file))
        else:
            print("channel num exception !")
    return np.array(tdata), imgps


# 从dir加载图片的logo位置的数据样本
def load_delogo_data(ipath, logo_size):
    ximg = Image.open(ipath)
    w, h = ximg.size
    lw, lh = logo_size
    iw, ih = int((w - lw) / 2), int((h - lh) / 2)
    xarr = np.array(ximg)
    return xarr[ih:ih + lh, iw:iw + lw, :3], ximg

