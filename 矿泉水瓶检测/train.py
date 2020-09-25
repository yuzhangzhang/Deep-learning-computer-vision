import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import cv2
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
from gridmask import GridMask

import yolo_model
import data
import Loss
import ResNet


epochs = 30
learning_rate = 0.0001
batch_size = 2
image_path, label_path,label_name = data.make_list()
label_list = data.make_label(label_name)
bboxes = data.get_bbox(label_path,image_path)


#train dataset
train_dataset = data.myDataset(image_path=image_path[0:130], labels_list =label_list[0:130] ,bboxes=bboxes[0:130], train=True, transform=[transforms.ToTensor()])
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# network structure
# net = ResNet.resnet50()

# load pre_trained model
# resnet = models.resnet50(pretrained=True)
# new_state_dict = resnet.state_dict()
# #
#
# op = net.state_dict()
# for k in new_state_dict.keys():
#     print(k)
#     if k in op.keys() and not k.startswith('fc'):  # startswith() 方法用于检查字符串是否是以指定子字符串开头，如果是则返回 True，否则返回 False
#         print('yes')
#         op[k] = new_state_dict[k]
# net.load_state_dict(op)

# 定义模型
net = torch.load('bottle_resnet4.pt')

criterion = Loss.yoloLoss(14, 2, 5, 0.5)#权重参数，某些要占大的比重，某些占小的比重
net.train()

# #调整学习率
params = []
params_dict = dict(net.named_parameters())

for key, value in params_dict.items():
    if key.startswith('features'):
        params += [{'params': [value], 'lr':learning_rate * 1}]
    else:
        params += [{'params': [value], 'lr':learning_rate}]


# 定义优化算法为sdg:随机梯度下降
# optimizer = optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)

grid = GridMask(d1=96, d2=224, rotate=360, ratio=0.6, mode=1, prob=0.8)

# train

for epoch in range(epochs):
    total_loss = 0
    net.train()
    if epoch < 8:
        learning_rate = 0.001
    if 8 <epoch < 11:
        learning_rate = 0.0001
    else:
        learning_rate = 0.00001

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    print('\n\nStarting epoch %d / %d' % (epoch + 1, epochs))
    for i, (image, target) in enumerate(train_loader):
        # print(image.shape)
        grid.set_prob(epoch, 240)
        image = grid(image)
        outputs = net(image)
        loss = criterion(outputs, target)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch:{},loss:{}'.format(epoch,total_loss/65))
        # if (i + 1) % 5 == 0:
            # print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f'
            #       % (epoch + 1,epochs,i + 1, len(train_loader),loss.item(),total_loss / (i + 1)))

torch.save(net, 'bottle_resnet5.pt')
print('bottle_resnet5.pt saved')


