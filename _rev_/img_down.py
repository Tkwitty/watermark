# coding=utf-8
from urllib.request import urlopen
import os
import urllib.request

#  n0/jfs => n1/s800x800_jfs
with open('imglist.txt', 'r') as f:
    urll = f.readlines()

def req_download(IMAGE_URL, SAV_PATH):
    import requests
    r = requests.get(IMAGE_URL, stream=True)
    with open(SAV_PATH, 'wb') as f:
        for chunk in r.iter_content(chunk_size=32):
            f.write(chunk)

import time
def getTimeVersion():
    stime = time.strftime('%H%M', time.localtime(time.time()))
    return stime

file_path = './'
for i, url in enumerate(urll):
    if len(url.strip()) > 10:
        url = " http://" + url.split('.jpg', 1)[0] + '.jpg'
    else:
        continue
    try:
        if not os.path.exists(file_path):
            os.makedirs(file_path)  # 如果没有这个path则直接创建
        file_suffix = os.path.splitext(url)[1]  # 文件拓展名
        fn = "yy" + str(i)
        filename = '{}nologo/{}_y{}'.format(file_path, fn, file_suffix)

        xurl = url.replace('n0/jfs', 'n1/s800x800_jfs')  # 下载大图
        req_download(xurl, filename)
        print("下载图片：", xurl, "\n", filename)

    except IOError as e:
        print(1, e)
    except Exception as e:
        print(2, e)  # unknown url type

print("over")








