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
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms

# 得到图片路径、label路径、label名称
def make_list():
    path = 'D:\Python\CV\week12\week10_dataset\\0_annotation_train.txt' # 标注文件，含有图片名称、label、地址等信息
    file = open(path,encoding='utf-8')
    source = file.read()
    label_path =[]  # 存放label的路径
    label_name = [] # 存放label名称
    image_path = [] # 存放图片的路径
    images = [] # 存放解析出来的含有图片路径的信息（文件夹名称和图片名称）

    LABEL_ROOT = 'D:\Python\CV\week12\\anjiang2020_admin-week10-dataset-master\week10-dataset\cvf-week11-datasets-all-labled\\annotation\V006\\annotations'
    IMAGE_ROOT = 'D:\Python\CV\week12\week10_dataset'

    for line in source.split("\n"):
        label = re.findall('"name":"(\w+)"', line)
        label_name.append(label)
        image = re.findall('image/(\w+)/(\w+)', line)
        images.append(image)
    for i in images:
        for j in i:
            a,b=j   # a:文件夹名称 b:图片名称
            image_adr = IMAGE_ROOT+ '\\' + a + '\\' + b +'.jpg'
            image_path.append(image_adr)
            label_adr = LABEL_ROOT+'\\' + a + '\\' + b +'.xml'
            label_path.append(label_adr)
    label_name.pop(-1) # 最后一个为空
    df = pd.DataFrame({'image':image_path,'label':label_path,'class':label_name})#将图片路径，label路径和label名称写入csv文件
    df = shuffle(df)  # 乱序
    df.to_csv(pjoin(IMAGE_ROOT,'Train.csv'),index=False,encoding='utf-8_sig')#保存文件
    print('Train.csv saved')
    print(len(image_path),len(label_path),len(label_name))
    return image_path,label_path,label_name  # 虽然存放在不同列表中，但都是一一对应的

# 得到图片bbox的坐标信息（左上角坐标x1y1和右下角坐标x2y2）
def get_bbox(label_list,image_list):
    bboxes = []  # 存放所有label的bbox坐标信息
    # i = 0
    for xml_file,image_file in zip( label_list,image_list):  # 同时拿出一张图片的路径和对应label路径（图片路径和label路径是生成时对应好的）
        # i +=1
        cv_img = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), -1) # 读取图像，解决imread不能读取中文路径的问题
        h,w,c = cv_img.shape
        object = []  # 存放一张图片中的bbox坐标信息，可能有多个bbox
        tree = ET.parse(xml_file)  # 解析读取xml函数
        filename,_ = tree.find('filename').text.split('.')
        for size in tree.findall('size'):
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            for obj in tree.findall('object'):
                bbox = obj.find('bndbox')
                obj_struct = [w * int(bbox.find('xmin').text) / width,     # 列表形式存放一个bbox坐标信息
                              h * int(bbox.find('ymin').text) / height,
                              w * int(bbox.find('xmax').text) / width,
                              h * int(bbox.find('ymax').text) / height]
                object.append(obj_struct)
        # print(object[0])
        # draw_img_rects(cv_img, object,i)
        bboxes.append(object)
    return bboxes  # 与图片路径等信息也是一一对应的

# 在图片中画出bbox
def draw_img_rects(image, object,i):
    # for bbox in object:
    bbox = object[0]
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmin) + int(w), int(ymin) + int(h)), (0, 255, 0), 10) # 在图片中画矩形框
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', image)
    cv2.imwrite('image/'+str(i)+'.jpg',image)
    cv2.waitKey(10)  # 持续时长


# 将label名称对应到0~13个数字上
def make_label(label_name):
    classifier = {'百岁山':1,'怡宝':2,'百事可乐':3,'景甜':4,'娃哈哈':5,'康师傅':6,'苏打水':7,
                  '天府可乐':8,'可口可乐':9,'农夫山泉':10,'恒大冰泉':11,'其它':12,'冰露':13}
    labels_list = []
    for i in label_name: # label_name中存放的是分类的名称
        if len(i)>1:
            i.pop(-1)
        for j in i:
            label = classifier[j] # 根据名称对应到数字
            labels_list.append(label)
    return labels_list  # 对应到数字的label

# 计算图片的均值和方差
def get_mean_std(image_list):
    '''
	file_name: train,txt
	img_path: 'week10_dataset/image/'
	146
	normMean = [140.7888129  131.87180331 126.43424442]
	normStd = [53.84969082 54.91440049 56.4051085 ]
	normMean = [142.20270479 131.69554463 126.34654796]
    normStd = [53.93910735 55.32934892 56.66741758]
    '''
    means = [0, 0, 0]
    stdevs = [0, 0, 0]
    num_imgs = 0
    for image_path in image_list:
        num_imgs += 1 # 记录图片的个数
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)## 读取图像，解决imread不能读取中文路径的问题
        img = np.array(img).astype(np.float32)
        for i in range(3): # 对图片三个通道分别求均值和方差，累加所有图片
            means[i] += img[:, :, i].mean()
            stdevs[i] += img[:, :, i].std()
    means.reverse()
    stdevs.reverse()
    means = np.asarray(means) / num_imgs   # 所有图片的均值和方差之和除以图片总数
    stdevs = np.asarray(stdevs) / num_imgs
    # print("normMean = {}".format(means))
    # print("normStd = {}".format(stdevs))
    return means, stdevs


class myDataset(DataLoader):
    image_size = 448  # 图片期待缩小到的尺寸

    def __init__(self, image_path, labels_list, bboxes, train, transform):
        self.boxes = []  # tensor形式的所有坐标信息
        self.labels = [] # 所有数字label
        self.image = []  # 所有图片路径
        self.train = train
        self.transform = transform
        self.S = 14  # grid number 14*14 normally
        self.B = 2  # bounding box number in each grid
        self.classes_num = 14  # how many classes
        self.mean = (142, 132, 126)  # RGB
        for image,label,bbox in zip(image_path,labels_list,bboxes):
            self.boxes.append(torch.tensor(bbox))
            self.labels.append(label)
            self.image.append(image)
        self.num_samples = len(self.boxes)

    def __getitem__(self, idx):
        img = self.image[idx]
        img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), -1)## 读取图像，解决imread不能读取中文路径的问题
        box = self.boxes[idx]
        # print('box{}'.format(box))
        # box = box.unsqueeze(0)
        # print('box{}'.format(box))

        # print(box)
        labels = self.labels[idx]
        if self.train:
            # torch自带的transform会造成bbox的坐标,需要自己来定义数据增强
            pass
        h, w, _ = img.shape
        box /= torch.LongTensor([w, h, w, h]).expand_as(box)  # 坐标归一化处理，为了方便训练
        img = self.BGR2RGB(img)  # cv2读入的图片是BGR形式，pytorch pretrained model use RGB，所以要转换一下
        img = self.subMean(img, self.mean)  # 减去均值
        img = cv2.resize(img, (self.image_size, self.image_size))  # 将所有图片都resize到指定大小\
        target = self.encoder(box, labels)  # 将图片标签编码到14x14*24的向量

        for t in self.transform:
            img = t(img)

        return img, target

    def __len__(self):
        return self.num_samples

    def subMean(self, img, mean):
        mean = np.array(mean, dtype=np.float32)
        img = img - mean
        # img = img/(255.0/2)-1
        # img = np.array(img,dtype=np.float32)
        return img

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def encoder(self, boxes, labels):
        '''
        将图片标签编码到14x14*24的向量
        前10层为坐标信息，图片分成14*14个格子，bbox所在的格子的置信度为1，其他格子为0，
        后14层为分类信息，label对应的那一层为1，其他层为0
        boxes (tensor) [[x1,y1,x2,y2]]
        labels (tensor) [...]
        return 14x14x24
        '''
        # print(boxes)
        grid_num = 14
        yolo_output = 24  # 5*2+14
        target = torch.zeros((grid_num, grid_num, yolo_output))
        cell_size = 1. / grid_num  # 每个格子的大小
        # 右下坐标        左上坐标
        # x2,y2           x1,y1
        x2y2 = boxes[:,2:]  # 所有的bbox
        x1y1 = boxes[:,:2]
        wh = x2y2 - x1y1
        cxcy = (x2y2 + x1y1) / 2 # 中心点坐标

        for i in range(cxcy.size()[0]):   # cxcy.size()[0]代表 一张图像的物体总数，遍历一张图像的物体总数
            # 物体中心坐标
            cxcy_sample = cxcy[i]
            # 指示落在那网格，如[0,0]
            ij = (cxcy_sample / cell_size).ceil() - 1  # 中心点对应格子的坐标# ceil返回数字的上入整数
            # cxcy_sample为一个物体的中心点坐标，求该坐标位于14x14网格的哪个网格
            # cxcy_sample坐标在0-1之间  现在求它再0-14之间的值，故乘以14
            # ij长度为2，代表14x14框的某一个框 负责预测一个物体

            #    0 1    2 3   4      5 6   7 8   9
            # target[中心坐标,长宽,置信度,中心坐标,长宽,置信度, 14个类别] x 14x14  因为一个框预测两个物体

            # 每行的第4和第9的值设置为1，即每个网格提供的两个真实候选框 框住物体的概率是1.
            # xml中坐标理解：原图像左上角为原点，右边为x轴，下边为y轴。
            # 而二维矩阵（x，y）  x代表第几行，y代表第几列
            # 假设ij为（1,2） 代表x轴方向长度为1，y轴方向长度为2
            # 二维矩阵取（2,1） 从0开始，代表第2行，第1列的值

            # 第一个框的置信度
            target[int(ij[1]), int(ij[0]), 4] = 1
            # 第二个框的置信度
            target[int(ij[1]), int(ij[0]), 9] = 1
            # 类别，加9是因为前0-9为两个真实候选款的值。后10-20为20分类   将对应分类标为1
            target[int(ij[1]), int(ij[0]), int(labels) + 9] = 1
            # xy为归一化后网格的左上坐标---->相对整张图
            xy = ij * cell_size  # 匹配到的网格的左上角相对坐标
            # 物体中心相对左上的坐标 ---> 坐标x,y代表了预测的bounding
            # box的中心与栅格边界的相对值
            # delta_xy：真实框的中心点坐标相对于  位于该中心点所在网格的左上角   的相对坐标，此时可以将网格的左上角看做原点，你这点相对于原点的位置。取值在0-1，但是比1/14小
            delta_xy = (cxcy_sample - xy) / cell_size  # 其实就是offset
            # (1) 每个小格会对应B(2)个边界框，边界框的宽高范围为全图，表示以该小格为中心寻找物体的边界框位置。
            # (2) 每个边界框对应一个分值，代表该处是否有物体及定位准确度
            # (3) 每个小格会对应C个概率值，找出最大概率对应的类别P(Class|object)，并认为小格中包含该物体或者该物体的一部分。

            # 坐标w,h代表了预测的bounding box的width、height相对于整幅图像width,height的比例
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
            target[int(ij[1]), int(ij[0]), :2] = delta_xy
            # 每一个网格有两个边框
            target[int(ij[1]), int(ij[0]), 7:9] = wh[i] # 长宽
            # 中心坐标偏移
            # 由此可得其实返回的中心坐标其实是相对左上角顶点的偏移，因此在进行预测的时候还需要进行解码
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
            # print('target{}'.format(target))
        return target


if __name__ == '__main__':
    image_path, label_path,label_name = make_list()
    label_list = make_label(label_name)
    bboxes = get_bbox(label_path,image_path)
    print(bboxes)
    print(len(bboxes))
    # for box in bboxes:
    #     print(box)
    # print(len(bboxes))

    # image_path = 'D:\Python\CV\week12\week10_dataset\skypower\\6ae24597511ce627185bb26737a5a64.jpg'
    # img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)  ## 读取图像，解决imread不能读取中文路径的问题
    # cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    # cv2.imshow('image',img)
    # cv2.waitKey(0)