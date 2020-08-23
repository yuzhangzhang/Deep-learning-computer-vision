import re
import pandas as pd
from os.path import dirname,abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
from os.path import join as pjoin
from sklearn.utils import shuffle
import cv2
import numpy as np
import torch
import pandas as pd
def make_list():
    path = 'D:\Python\CV\week11\week11\\0_annotation_train.txt'
    file = open(path,encoding='utf-8')
    source = file.read()
    print(source)
    label_list =[]
    image_list = []
    images = []

    ROOT = 'D:\Python\CV\week11\week11\week10_dataset'
    for line in source.split("\n"):
        label = re.findall('"name":"(\w+)"', line)
        label_list.append(label)
        image = re.findall('image/(\w+)/(\w+).(\w+)', line)
        images.append(image)
    label_list.pop(-1)
    for i in images:
        for j in i:
            a,b,c=j
            image_adr = ROOT+ '\\' + a + '\\' + b +'.' + c
            image_list.append(image_adr)
    df = pd.DataFrame({'image':image_list,'label':label_list})
    # df = shuffle(df)
    df.to_csv(pjoin(ROOT,'Train.csv'),index=False,encoding='utf-8_sig')
    print('Train.csv saved')

    return image_list,label_list

import matplotlib.pyplot as plt

def make_image(image_list,out_size=[224,224]):
    train_image = []
    for i in image_list:
        cv_img = cv2.imdecode(np.fromfile(i, dtype=np.uint8), -1)## 读取图像，解决imread不能读取中文路径的问题
        image = cv2.resize(cv_img, (out_size[0], out_size[1]), interpolation=cv2.INTER_LINEAR)# crop the image to discard useless parts
        # cv2.imshow('image',image)
        # cv2.waitKey(10)
        # plt.show()
        image = image[:, :, ::-1].transpose(2,0,1).astype(np.float32) / (255.0 / 2) - 1
        train_image.append(image)
    return train_image

def one_hot(gt):  #将label转换成0-1编码
    gt_vector = torch.ones(1, 13)
    gt_vector *= 0.0
    gt_vector[0, gt] = 1.0  #标签是几对应第几位编码是1，其余为0
    gt_vector = gt_vector.numpy()
    return gt_vector

def make_label(label_list):
    classifier = {'百岁山':1,'怡宝':2,'百事可乐':3,'景甜':4,'娃哈哈':5,'康师傅':6,'苏打水':7,
                  '天府可乐':8,'可口可乐':9,'农夫山泉':10,'恒大冰泉':11,'其它':12,'冰露':13}
    gt = []
    labels = []
    for i in label_list:
        if len(i)>1:
            i.pop(-1)
        for j in i:
            label = classifier[j]
            labels.append(label)
            gt_vector = one_hot(label)
            gt.append(gt_vector)
    return gt,labels

def train_image_gen(batch_size=10):
    image_list,label_list = make_list()
    images = make_image(image_list)
    gts, labels = make_label(label_list)

    batch_images = [images[i:i + batch_size] for i in range(0, len(images) - 1, batch_size)]
    batch_images = np.array(batch_images)

    batch_gts = [gts[i:i + batch_size] for i in range(0, len(gts) - 1, batch_size)]
    batch_gts = np.array(batch_gts)

    batch_labels = [labels[i:i + batch_size] for i in range(0, len(labels) - 1, batch_size)]
    # batch_labels = np.array(batch_labels)
    batch_labels = np.array(batch_labels, dtype=np.int64)

    yield torch.from_numpy(batch_images), torch.from_numpy(batch_gts), torch.from_numpy(batch_labels)

#
# from PIL import Image
# import os
# def PNG_JPG(PngPath):
#     img = cv2.imdecode(np.fromfile(PngPath, dtype=np.uint8), -1)
#
#     # img = cv.imread(PngPath, 0)
#     w, h = img.shape[::-1]
#     infile = PngPath
#     outfile = os.path.splitext(infile)[0] + ".jpg"
#     img = Image.open(infile)
#     img = img.resize((int(w / 2), int(h / 2)), Image.ANTIALIAS)
#     try:
#         if len(img.split()) == 4:
#             # prevent IOError: cannot write mode RGBA as BMP
#             r, g, b, a = img.split()
#             img = Image.merge("RGB", (r, g, b))
#             img.convert('RGB').save(outfile, quality=70)
#             os.remove(PngPath)
#         else:
#             img.convert('RGB').save(outfile, quality=70)
#             os.remove(PngPath)
#         return outfile
#     except Exception as e:
#         print("PNG转换JPG 错误", e)




#
# #
# image_list,label_list=make_list()
# print(len(image_list))
# gt,label = make_label(label_list)
# # print('gt{},label{}'.format(gt[120:150],label[120:150]))
# print(len(label))
# image=make_image(image_list)
# print(len(image))
# for i in range(120,147):
#     print('image{},i{}'.format(image[i].shape,i))
