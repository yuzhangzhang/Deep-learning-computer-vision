# encoding:utf-8

import torch
from torch.autograd import Variable
import torch.nn as nn
import sys

import torchvision.transforms as transforms
import cv2
import numpy as np
import pdb
import os

classes_list = ["今麦郎", "冰露", "百岁山", "怡宝", "百事可乐", "景甜", "娃哈哈", "康师傅", "苏打水", "天府可乐",
                "可口可乐", "农夫山泉", "恒大冰泉", "其他"]


def nms(bboxes, scores, threshold=0.5):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    _, index = scores.sort(0, descending=True)
    # print(index)
    order_array = index.numpy()
    # print(order_array)
    index = order_array
    # print(index)
    index_len = len(index)
    # print(index_len)
    keep = []

    while index.size > 0:

        index_len = index_len - 1

        i = index[0]  # every time the first is the biggst, and add it directly

        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap

        y11 = np.maximum(y1[i], y1[index[1:]])

        x22 = np.minimum(x2[i], x2[index[1:]])

        y22 = np.minimum(y2[i], y2[index[1:]])

        w_tmp = np.maximum(0, x22 - x11 + 1)  # the weights of overlap

        h_tmp = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w_tmp * h_tmp  # 重叠面积/交集面积

        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)  # 并集面积

        idx = np.where(ious <= threshold)[0]

        index = index[idx + 1]  # because index start from 1

    return torch.LongTensor(keep)

def get_decoder(pred):
    '''
    pred (tensor)
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''

    grid_num = 14
    boxes = []
    cls_indexs = []
    probs = []
    cell_size = 1. / grid_num
    pred = pred.data
    pred = pred.squeeze(0)  # 将维，四维变成三维
    contain1 = pred[:, :, 4].unsqueeze(2)
    contain2 = pred[:, :, 9].unsqueeze(2)
    contain = torch.cat((contain1, contain2), 2)
    mask1 = contain > 0.1  # 大于阈值
    mask2 = (contain == contain.max())  # we always select the best contain_prob what ever it>0.9pred[:,:,4]
    mask = (mask1 + mask2).gt(0)
    # min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2): # 两个预测框
                if mask[i, j, b] == 1: #有目标物体的
                    # print("mask[%d,%d,%d]==1" % (i, j, b))
                    box = pred[i, j, b * 5:b * 5 + 4]
                    contain_prob = torch.FloatTensor([pred[i, j, b * 5 + 4]])
                    xy = torch.FloatTensor([j, i]) * cell_size  # cell左上角  up left of cell
                    box[:2] = box[:2] * cell_size + xy  # return cxcy relative to image
                    box_xy = torch.FloatTensor(box.size())  # 转换成xy形式    convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]
                    max_prob, cls_index = torch.max(pred[i, j, 10:], 0)  # 取出14个分类中最大的索引即为当前分类

                    print("get a box")
                    print(box_xy)
                    print("the box cls=")
                    print(cls_index)
                    print("the box max_prob")
                    print(max_prob)
                    print("the box contain_prob")
                    print(contain_prob)
                    print("float((contain_prob*max_prob)[0])")
                    print(float((contain_prob * max_prob)[0]))
                    ENABLE_VALUE = 0.1

                    if float((contain_prob * max_prob)[0]) > ENABLE_VALUE:
                        # print("find a box (%d %d %d)" % (i, j, b))
                        boxes.append(box_xy.view(1, 4))
                        tmp_list = []
                        tmp_int = cls_index.item()
                        tmp_list.append(tmp_int)
                        tmp_tensor = torch.tensor(tmp_list)
                        cls_indexs.append(tmp_tensor)
                        probs.append(contain_prob * max_prob)
                    else:
                        print("contain_prob*max_prob not > 0.1")
                        continue

    if len(boxes) == 0:
        boxes = torch.zeros((1, 4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes, 0)  # (n,4)
        probs = torch.cat(probs, 0)  # (n,)
        cls_indexs_len = len(cls_indexs)
        # print("cls_indexs_len=%d" % (cls_indexs_len))
        cls_indexs = torch.cat(cls_indexs, 0)  # (n,)

    keep = nms(boxes, probs) # 去除冗余框  keep[0]为最大值的框，取出
    return boxes[keep[0]], cls_indexs[keep[0]], probs[keep[0]]

if __name__ == '__main__':
    pass






