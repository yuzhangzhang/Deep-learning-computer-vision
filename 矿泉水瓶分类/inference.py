import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import numpy as np
import pandas as pd
import data
import cv2


def test_acc(image_list,label_list,images,labels,start,end):
    model = torch.load('bottle_resnet18.pt')
    model.eval()# 把训练好的模型的参数冻结
    correct = 0
    for i in range(start, end):
        y = model(images[i])
        label = labels[i]
        pred = torch.argmax(y).item()  # y是非0即1的向量，1所在的位置对应它的标签，所以取（y-1）最小值，其所在位置即为标签
        if label == pred:
            correct += 1
        print('**********************************')
        print('类别：{}'.format(label_list[i]))
        print("label:{} ".format(label))
        print("pred:{} ".format(pred))
        cv_img = cv2.imdecode(np.fromfile(image_list[i], dtype=np.uint8), -1)## 读取图像，解决imread不能读取中文路径的问题
        cv_img = cv2.resize(cv_img, (224,224), interpolation=cv2.INTER_LINEAR)# crop the image to discard useless parts
        cv2.imshow('img',cv_img)
        cv2.waitKey(10000)

    correct = float(correct / float(end - start))
    print('test_acc=%s'%correct)
    plt.show()
    return correct

if __name__=='__main__':
    gen = data.train_image_gen(1)
    images, gts, labels = next(gen)
    PD = pd.read_csv('D:\Python\CV\week11\week11\week10_dataset\Train.csv')
    image_list = PD['image']
    label_list = PD['label']
    test_acc(image_list,label_list,images,labels,140,146)
