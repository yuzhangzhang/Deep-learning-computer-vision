import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import cv2
import numpy as np
import torchvision.transforms as transforms
from torchvision import models

import yolo_model
import data
import Loss
import ResNet


epochs = 10
learning_rate = 0.001
batch_size = 2
image_path, label_path,label_name = data.make_list() #
label_list = data.make_label(label_name)
bboxes = data.get_bbox(label_path,image_path)


#train dataset
train_dataset = data.myDataset(image_path=image_path[0:140], labels_list =label_list[0:140] ,bboxes=bboxes[0:140], train=True, transform=[transforms.ToTensor()])
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


# 定义模型
# yolov0_model = yolo_model.YOLOV0()
yolov0_model = torch.load('bottle_resnet18.pt')

# load pre_trained model 加载resnet18的预训练模型
# resnet = models.resnet18(pretrained=True)
# new_state_dict = resnet.state_dict()
#
# op = yolov0_model.state_dict()
# for k in new_state_dict.keys():  #把resnet18的预训练模型的参数加到yolov0模型中
#     # print(k)
#     if k in op.keys() and not k.startswith('fc'):  # startswith() 方法用于检查字符串是否是以指定子字符串开头，如果是则返回 True，否则返回 False
#         # print('yes')
#         op[k] = new_state_dict[k]
# yolov0_model.load_state_dict(op)

criterion = Loss.yoloLoss(14, 2, 5, 0.5)#权重参数，某些要占大的比重，某些占小的比重

#调整学习率
params = []
params_dict = dict(yolov0_model.named_parameters())

for key, value in params_dict.items():
    if key.startswith('features'):
        params += [{'params': [value], 'lr':learning_rate * 1}]
    else:
        params += [{'params': [value], 'lr':learning_rate}]


# 定义优化算法为sdg:随机梯度下降
# optimizer = optim.SGD(yolov0_model.detector.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
optimizer = optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer = optim.Adam(yolov0_model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


# train
def train():
    for epoch in range(epochs):
        learning_rate = 0.00005
        if 3 < epoch < 6 :
            learning_rate = 0.00001
        if epoch >= 6:
            learning_rate = 0.000005
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        loss = 0
        for i, (image, target) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = yolov0_model(image)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
        print("epoch{},loss: {}".format(epoch, loss.data.item()))

    torch.save(yolov0_model, 'bottle_resnet18.pt')
    print('bottle_resnet18.pt saved')

if __name__=='__main__':
    train()