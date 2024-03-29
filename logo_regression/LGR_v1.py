# coding=utf8
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import matplotlib.pyplot as plt
from lgr import loadDataset, load_logoDataset

def getTimeVersion():
    stime = time.strftime('%H%M', time.localtime(time.time()))
    return stime

def tensor_test():
    v = tf.truncated_normal([10, 10], mean=0.5, stddev=0.25)
    sess = tf.Session()
    print(sess.run(v))
    sess.close()

def plot_losscurve(lossdicts, sav):
    fig = plt.figure(figsize=(7, 5))  # figsize是图片的大小
    ax1 = fig.add_subplot(1, 1, 1)  # ax1是子图的名字
    ax1.set_title('loss curve figs')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('training loss')

    print("plot begain")
    for dic in lossdicts:  # epoch + val
        eps = dic['eps']
        lss = dic['loss']
        ax1.plot(eps, lss, label=dic['title'])  # 线型，颜色，标记，名称

    plt.legend()
    plt.savefig("save/loss_" + sav + ".png")
    plt.show()
    print("plot over")


def generate_png_w(wrgb, wa, name):
    rgb = np.asarray(wrgb * 255, dtype=np.uint8)
    a_ = wa[:, :, 0] * 255
    a = np.asarray(a_, dtype=np.uint8)
    irgb = Image.fromarray(rgb, mode="RGB")  # rgb通道
    alpha = Image.fromarray(a, mode="L")  # a通道
    irgb.putalpha(alpha)  # 转换为 rgba 整图
    irgb.save("save/logo_" + name + ".png")
    irgb.show()


def train_logo():
    # 准备一波样本，x为带水印的mat，y为不带水印的mat
    dpath = "datas/nologo"  # 数据集 dir
    # xs, ys = loadDataset(dpath)  # 从图片dir加载数据集
    xs, ys = load_logoDataset(dpath, (373, 54))  # 从图片dir加载数据集
    print(np.shape(xs), np.shape(ys))

    X = tf.placeholder(tf.float32, name='X')
    Y = tf.placeholder(tf.float32, name='Y')

    # val_rgb = np.ones((54, 373, 3)) * 0.5
    # val_a = np.zeros((54, 373)) * 0.5
    # W_rgb = tf.Variable(tf.constant_initializer(val_rgb), name='weight_rgb')
    # W_a = tf.Variable(tf.constant_initializer(val_a), name='weight_a')

    # W_rgb = tf.Variable(tf.fill([54, 373, 3], 0.4), name='weight_rgb')  # 199,16,34, 77
    # W_a = tf.Variable(tf.fill([54, 373, 1], 0.4), name='weight_a')

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
    # loss = tf.reduce_sum(tf.pow(Y_pred - Y, 2))/points  # 每个点位差值平方的 均值： 定义loss值  MSE - ls loss
    loss = tf.reduce_sum(tf.abs(Y_pred - Y))/points  # l1 loss
    # learning_rate = 0.001
    # learning_rate = 0.1
    # learning_rate = 0.25
    learning_rate = 0.5
    # learning_rate = 0.75
    # learning_rate = 1
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)  # 定义 SGD 随机梯度下降
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)  # adam 优化算法
    # optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)
    # 学习率越小，越能接近极值；学习率过大，找不到极值

    # arg_version = "c4l2gd50w"
    arg_version = "x_c4l1sl2gd50w"
    n_samples = xs.shape[0]  # 训练样本总数
    init = tf.global_variables_initializer()  # 开始训练
    with tf.Session() as sess:
        sess.run(init)  # 初始化所有变量
        # 将搜集的变量写入事件文件，提供给Tensorboard使用
        writer = tf.summary.FileWriter('./graphs/logo_reg', sess.graph)

        # 训练模型: steps
        steps = 1000
        iloss = {'title': arg_version, 'eps': [], 'loss': [], 'tr_time': 0}
        t0 = time.perf_counter()
        for i in range(steps):
            total_loss = 0  # 设定总共的 损失初始值为0
            for x, y in zip(xs, ys):  # 遍历每一个样本，进行前后向计算
                _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
                total_loss += l  # 计算所有的损失值进行叠加  # 叠加样本集的损失值

            if i % 100 == 0 or i == steps-1:
            # if i % 2 == 0 or i == steps-1:
                mse = total_loss/n_samples
                iloss['eps'].append(i)
                iloss['loss'].append(mse)
                print('Epoch {0}: {1}'.format(i, mse))

                wrgb, wa = sess.run([W_rgb, W_a])  # 取出w值
                generate_png_w(wrgb, wa, arg_version + "_" + str(i))  # 生成与显示 当前logo学习效果

        t1 = time.perf_counter()
        print("train time waste: ", t1 - t0)
        writer.close()  # 关闭writer
        W_rgb, W_a = sess.run([W_rgb, W_a])  # 取出w值

    # ndarry 数据与格式转换
    tr_version = arg_version + "_" + getTimeVersion()
    generate_png_w(W_rgb, W_a, tr_version)

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
            MS-SSIM+L1 损失函数是最好的
        
        增加一个logo分类：
            构造训练的 x-patchs 和 y-1/0； 
            对所有的样本点进行一个hog特征点提取，在进行svm分类；
            
        训练可视化：如何在训练的epoch过程中将参数获取
        
    r0.5 + i0.5 + gd + 1000ep => 0.0004, 0.0011
    r0.5 + ad + 10000ep => 0.22196987484182631
     0.0004211
    """

if __name__ == '__main__':
    train_logo()

    # tensor_test()

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

