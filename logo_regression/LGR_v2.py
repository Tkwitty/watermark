# coding=utf8
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import time
import matplotlib.pyplot as plt
from sklearn import svm
import joblib
import tornado.ioloop
import tornado.web
import io
import os
from urllib.request import urlopen

from lgr import loadDataset, load_logoDataset, load_gray_logoDataset, \
    load_delogo_dataset, load_delogo_data
from logo_test import logo_clean

# 提取图像hog特征向量  (54, 373, 3)
def hog_extractor(hog, img):  # 计算hog特征
    winStride = (8, 8)
    padding = (8, 8)
    # print("提取前矩阵维度：", np.shape(img))  # (54, 373, 3)
    hog_vector = hog.compute(img, winStride, padding).reshape((-1,))
    return hog_vector

def getTimeVersion():
    stime = time.strftime('%H%M', time.localtime(time.time()))
    return stime

def tensor_test():
    v = tf.truncated_normal([10, 10], mean=0.5, stddev=0.25)
    sess = tf.Session()
    print(sess.run(v))
    sess.close()

def plot_losscurve(lossdicts, sav):
    fig = plt.figure(figsize=(7, 5))  # figsize是图片的大小`
    ax1 = fig.add_subplot(1, 1, 1)  # ax1是子图的名字`
    ax1.set_title('loss curve figs')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('training loss')

    print("plot begain")
    for dic in lossdicts:  # epoch + val
        eps = dic['eps']
        lss = dic['loss']
        ax1.plot(eps, lss, label=dic['title'])  # 线型，颜色，标记，名称

    plt.legend()
    plt.savefig("loss_" + sav + ".png")
    plt.show()
    print("plot over")


def train_logo():
    # 准备一波样本，x为带水印的mat，y为不带水印的mat
    dpath = "nologo"  # 数据集 dir
    # xs, ys = loadDataset(dpath)  # 从图片dir加载数据集
    xs, ys = load_logoDataset(dpath, (373, 54))  # 从图片dir加载数据集
    print(np.shape(xs), np.shape(ys))

    X = tf.placeholder(tf.float32, name='X')
    Y = tf.placeholder(tf.float32, name='Y')

    # val_rgb = np.ones((54, 373, 3)) * 0.5
    # val_a = np.zeros((54, 373)) * 0.5
    # W_rgb = tf.Variable(tf.constant_initializer(val_rgb), name='weight_rgb')
    # W_a = tf.Variable(tf.constant_initializer(val_a), name='weight_a')

    # W_rgb = tf.Variable(tf.fill([54, 373, 3], 0.9), name='weight_rgb')  # 199,16,34, 77
    # W_a = tf.Variable(tf.fill([54, 373, 1], 0.9), name='weight_a')

    # l2 正则化 正确使用姿势 [在定义权重初始化 之前定义 正则化], 若用正则化则打开 wd
    initial_rgb = tf.fill([54, 373, 3], 0.4)
    initial_a = tf.fill([54, 373, 1], 0.4)
    weight_decay_rgb = tf.multiply(tf.nn.l2_loss(initial_rgb), 0.04, name='weight_loss_rgb')  # wd 惩罚因子 0.04
    weight_decay_a = tf.multiply(tf.nn.l2_loss(initial_a), 0.04, name='weight_loss_a')
    tf.add_to_collection('losses', weight_decay_rgb)
    tf.add_to_collection('losses', weight_decay_a)
    W_rgb = tf.Variable(initial_rgb, name='weight_rgb')
    W_a = tf.Variable(initial_a, name='weight_a')

    # 均值0.5,标差1分布   minval=0, maxval=1
    # W_rgb = tf.Variable(tf.truncated_normal([54, 373, 3], mean=0.4, stddev=0.25), name='weight_rgb')
    # W_a = tf.Variable(tf.truncated_normal([54, 373, 1], mean=0.4, stddev=0.25), name='weight_a')  # 非0则255

    # 定义后向运算：y = w（x）
    print("定义后向运算：y = w（x）")
    # print(W_rgb.shape, W_a.shape)
    # print(tf.multiply(tf.tile(W_a, (1, 1, 3)), W_rgb).shape)
    factor1 = tf.subtract(X, tf.multiply(tf.tile(W_a, (1, 1, 3)), W_rgb), name=None)   # 减法
    factor2 = tf.subtract(tf.ones([54, 373, 1], tf.float32), W_a, name=None)   # 减法
    Y_pred = tf.divide(factor1, factor2, name=None)

    # 定义损失计算
    print("定义损失计算与sgd优化")
    points = 54 * 373
    loss = tf.reduce_sum(tf.pow(Y_pred - Y, 2))/points  # 每个点位差值平方的 均值： 定义loss值
    # learning_rate = 0.001
    learning_rate = 0.1
    # learning_rate = 0.25
    # learning_rate = 0.5
    # learning_rate = 0.75
    # learning_rate = 1
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)  # 定义 SGD 随机梯度下降
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)  # adam 优化算法
    # optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)
    # 学习率越小，越能接近极值；学习率过大，找不到极值

    arg_version = "c4l2gd01w"
    n_samples = xs.shape[0]  # 训练样本总数
    init = tf.global_variables_initializer()  # 开始训练
    with tf.Session() as sess:
        sess.run(init)  # 初始化所有变量
        # 将搜集的变量写入事件文件，提供给Tensorboard使用
        writer = tf.summary.FileWriter('./graphs/logo_reg', sess.graph)

        # 训练模型: steps
        steps = 10000
        iloss = {'title': arg_version, 'eps': [], 'loss': [], 'tr_time': 0}
        t0 = time.perf_counter()
        for i in range(steps):
            total_loss = 0  # 设定总共的 损失初始值为0
            for x, y in zip(xs, ys):  # 遍历每一个样本，进行前后向计算
                _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
                total_loss += l  # 计算所有的损失值进行叠加  # 叠加样本集的损失值

            if i % 100 == 0 or i == steps-1:
                mse = total_loss/n_samples
                iloss['eps'].append(i)
                iloss['loss'].append(mse)
                print('Epoch {0}: {1}'.format(i, mse))

        t1 = time.perf_counter()
        print("train time waste: ", t1 - t0)
        writer.close()  # 关闭writer
        W_rgb, W_a = sess.run([W_rgb, W_a])  # 取出w值

    # ndarry 数据与格式转换
    rgb = np.asarray(W_rgb * 255, dtype=np.uint8)
    a_ = W_a[:, :, 0] * 255
    a = np.asarray(a_, dtype=np.uint8)
    irgb = Image.fromarray(rgb, mode="RGB")  # rgb通道
    alpha = Image.fromarray(a, mode="L")  # a通道
    irgb.putalpha(alpha)  # 转换为 rgba 整图
    irgb.show()

    tr_version = arg_version + "_" + getTimeVersion()
    irgb.save("logo_" + tr_version + ".png")

    # 绘制+保存 loss 曲线
    iloss['title'] = arg_version
    iloss['tr_time'] = t1 - t0
    plot_losscurve([iloss], tr_version)

    """
    优化方向：
        数据集扩充：补充更多的数据集；
        采用更好优化算法：sgd，adam；
        通过读入图像的初始化权重；（可以不必，纯学习效果不错）
        注意处理梯度消失问题，或者计算边界值问题 (麻点是如何产生的？)
            发现麻点跟w初始化值有关，当初始值设为0.4时麻点消失；
            也就是 a值为 0或1 都可以使得损失值的降低，就看a值的一个初始方向是啥？最后收敛就是啥
        正则化：
        mse loss优化 ==》
            SSIM（结构相似）损失
            PSNR(Peak Signal-to-Noise Ratio) 峰值信噪比
            MS-SSIM + L1 损失函数是最好的
        
        增加一个logo分类：
            构造训练的 x-patchs 和 y-1/0； 
            对所有的样本点进行一个hog特征点提取，在进行svm分类；
        
    r0.5 + i0.5 + gd + 1000ep => 0.0004, 0.0011
    r0.5 + ad + 10000ep => 0.22196987484182631
    0.0023
    """

def get_hog():
    # hog 提取参数  # 128, 64, 8, 16
    winSize = (32, 32)
    blockSize = (16, 16)
    blockStride = (4, 4)
    cellSize = (8, 8)
    nbins = 9
    # 定义对象hog，同时输入定义的参数，剩下的默认即可
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    return hog


def get_model_metric(clf, X, Y):
    from sklearn.metrics import confusion_matrix, roc_auc_score
    y_pred = clf.predict(X)
    tn, fp, fn, tp = confusion_matrix(Y, y_pred).ravel()  # 混淆矩阵
    n = tn + fp + fn + tp
    acc, prec = round((tn + tp) / n * 1.0, 8), round(tp / (tp + fn) * 1.0, 8)
    auc = round(roc_auc_score(Y, y_pred), 8)
    metrics = {"acc": acc, "prec": prec, "auc": auc}
    print(metrics)
    return metrics


def model2pkl(clf, mname):
    joblib.dump(clf, mname + ".pkl")
    print("测试模型持久化与加载。。。")
    time.sleep(5)
    iclf = joblib.load("logo_svm_cls.pkl")
    # print("testing svm logo 分类结果：", iclf.predict(X))


def hogtest():
    dpath = "datas/nologo"  # 数据集 dir
    xs1, xs0 = load_gray_logoDataset(dpath, (373, 54))  # 从图片dir加载数据集
    hog = get_hog()
    X = []
    Y = []
    for xm in xs1:  # 由于xs全是 wxhx3 通道的map，需要转成灰度图
        x_hogs = hog_extractor(hog, xm)
        X.append(x_hogs)
        Y.append(1)  # 有logo

    for xm in xs0:  # 由于xs全是 wxhx3 通道的map，需要转成灰度图
        x_hogs = hog_extractor(hog, xm)
        X.append(x_hogs)
        Y.append(0)  # 无logo

    print("logo分类数据集：", np.shape(X), np.shape(Y))  # 202500维向量
    # (112, 202500) (112,)

    isvm = svm.SVC(gamma='scale')
    isvm.fit(X, Y)
    """ # SVC参数:
    C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False
    """

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(X, Y)

    print('svm')
    get_model_metric(isvm, X, Y)
    print('lr')
    get_model_metric(lr, X, Y)
    print('lr预测结果：', lr.predict(X))

    model2pkl(lr, 'lr_logo_cls')
    print('over.')


# def logo_cls2del_test(testp):
#     ts = load_delogo_dataset(testp, )  # 从图片dir加载数据
#     hog = get_hog()
#     T = []

def hog_svm_logo_cls_rm():
    # tlogos, imgs = load_delogo_dataset("tests", (373, 54))  # 从图片dir加载数据
    tlogos, imgs = load_delogo_dataset("cls_rm_tests", (373, 54))  # 从图片dir加载数据
    hog = get_hog()
    svm = joblib.load("save/logo_svm_cls.pkl")
    pngp = "ilogor.png"
    for tlogo, img in zip(tlogos, imgs):
        hogs = hog_extractor(hog, tlogo)  # hog 特征提取
        cls = svm.predict([hogs])  # svm 分类器
        print("svm 分类：", cls)
        if cls[0] == 1:
            print("识别到logo水印，消去水印...")
            logo_clean(img, pngp, savp="xout")  # 去核心水印
        else:
            Image.open(img).show()
            print("没有logo水印，pass")
    print("over!")


def img_logo_rm(imgp, pname):
    pngp = "ilogor.png"
    ilogo, img = load_delogo_data(imgp, (373, 54))
    svm = joblib.load("save/logo_svm_cls.pkl")
    hogs = hog_extractor(get_hog(), ilogo)  # hog 特征提取
    cls = svm.predict([hogs])  # svm 分类器
    print("svm 分类：", cls)
    if cls[0] == 1:
        print("识别到logo水印，消去水印...")
        img_rm = logo_clean(imgp, pngp, savp="svc_temp", pname=pname)  # 去核心水印
    else:
        Image.open(imgp).show()
        print("没有logo水印，pass")
        img_rm = ""

    return img_rm


root_path = "/Users/zcy/PycharmProjects/_sy_/watermark/logo_regression/svc_temp"
class MainHandler(tornado.web.RequestHandler):

    def post(self):
        """post请求"""
        self.get()

    def get(self):
        """get请求"""
        t1 = time.perf_counter()
        pic = self.get_argument('picuri')
        pname = pic.rsplit('/', 1)[1]
        suffix = pname.split('.', 1)[1]
        if suffix in ['jpg', 'jpeg', 'png', 'bmp', 'gif'] and pic.startswith('http'):
            image_bytes = urlopen(pic).read()
            pic = io.BytesIO(image_bytes)
        img_rm = img_logo_rm(pic, pname)  # 分类 与 消去
        t2 = time.perf_counter()
        print("请求处理耗时: %s sec." % (t2 - t1))

        # "<img src='" + root_path + "/1622_comp.jpg'><br/>" +
        # rst = "<a href='/downfile?bpath=dst&fname=1622_rm.jpg'>消去图像下载</a>"
        rst = "<a href='/downfile?bpath=dst&fname=" + img_rm + "'>消去图像下载</a>"
        self.write(rst)


class DownloadFileHandler(tornado.web.RequestHandler):

    def get(self):
        print("demo: 127.0.0.1:10935/downfile?bpath=dst&fname=1622_rm.jpg")
        self.post()

    def post(self):
        """下载文件: 127.0.0.1:10934/downfile?filename=readme.txt"""
        bpath = ""
        if "bpath" in self.request.arguments:
            bpath = self.get_argument('bpath')
        fname = self.get_argument('fname')

        self.set_header('Content-Type', 'application/octet-stream')
        self.set_header('Content-Disposition', 'attachment; filename=' + fname)
        buf_size = 4096
        dpath = os.path.join(root_path + "/" + bpath, fname)
        print("下载地址：", dpath)
        with open(dpath, 'rb') as f:
            while True:
                data = f.read(buf_size)
                if not data:
                    break
                self.write(data)
        self.finish()


def logo_rm_websvc():

    application = tornado.web.Application([
        (r"/rmlogo", MainHandler),
        (r"/downfile", DownloadFileHandler),
    ])
    port = 10935
    application.listen(port)
    print(str(port) + "/rmlogo svc ready...")
    tornado.ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    # train_logo()

    hogtest()  # hog 特征提取 + svm分类器

    # tensor_test()
    # hog_svm_logo_cls_rm()

    # imgp = "cls_rm_tests/t8.jpg"
    # img_logo_rm(imgp)

    # loss 曲线绘制测试
    # tloss1 = {
    #     'title': 'my_5eps',
    #     'eps': [1, 2, 3, 4, 5],
    #     'loss': [0.1, 0.2, 0.3, 0.4, 0.5]
    # }
    # tloss2 = {
    #     'title': 'my_5eps',
    #     'eps': [1, 2, 3, 4, 5],
    #     'loss': [0.4, 0.3, 0.2, 0.6, 0.4]
    # }
    # plot_losscurve([tloss1], "test")

    # 去水印的 web 服务
    # logo_rm_websvc()

"""
127.0.0.1:10935/rmlogo?picuri=
https://img14.360buyimg.com/n0/jfs/t1/115934/38/18280/237070/5f661f3aE050bfd8c/5ddcac4e96da4828.jpg
"""



