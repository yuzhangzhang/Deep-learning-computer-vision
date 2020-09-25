# encoding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

yolo_v1_output = 24  # 5*2 + 14

class yoloLoss(nn.Module):  # (14, 2, 5, 0.5)
    def __init__(self, S, B, l_coord, l_noobj):
        # 为了更重视8维的坐标预测，给这些算是前面赋予更大的loss weight
        # 对于有物体的记为λ,coord，在pascal VOC训练中取5，
        # 对于没有object的bbox的confidence loss，前面赋予更小的loss weight 记为 λ,noobj, 在pascal VOC训练中取0.5
        # 有object的bbox的confidence loss (上图红色框) 和类别的loss （上图紫色框）的loss weight正常取1
        super(yoloLoss, self).__init__()
        self.S = S  # 14代表将图像分为14x14的网格
        self.B = B  # 2代表一个网格预测两个框
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def compute_iou(self, box1, box2):#anchor有多个
        # iou的作用是，当一个物体有多个框时，选一个相比ground truth最大的执行度的为物体的预测，
        # 然后将剩下的框降序排列，如果后面的框中有与这个框的iou大于一定的阈值时则将这个框舍去
        # （这样就可以抑制一个物体有多个框的出现了），目标检测算法中都会用到这种思想。

        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        # print('iou{}'.format(iou))
        return iou

    def forward(self, pred_tensor, target_tensor):  # s-size=14 B-boxcount=2
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,30)
        本来有，预测无-->计算response loss响应损失
        本来有，预测有-->计算not response loss 未响应损失
        本来无，预测无-->无损失(不计算)
        本来无，预测有-->计算不包含obj损失  只计算第4,9位的有无物体概率的loss
        '''
        # print('pred_tensor.size(): ', pred_tensor.size())
        # print('target_tensor.size(): ', target_tensor.size())
        N = pred_tensor.size()[0]  # batch-size N=2
        coo_mask = target_tensor[:, :, :, 4] > 0  # 具有目标标签的索引
        noo_mask = target_tensor[:, :, :, 4] == 0  # 不具有目标的标签索引
        # 得到含物体的坐标等信息
        # unsqueeze(-1) 扩展最后一维，用0填充，使得形状与target_tensor一样
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        coo_mask = coo_mask.bool()
        # 得到不含物体的坐标等信息
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.bool()

        # coo_pred 取出预测结果中有物体的网格，并改变形状为（xxx,24）  xxx代表一个batch的图片上的存在物体的网格总数    24代表2*5+14
        coo_pred = pred_tensor[coo_mask].view(-1, yolo_v1_output)
        # contiguous将不连续的数组调整为连续的数组
        box_pred = coo_pred[:, :10].contiguous().view(-1, 5)  # box[x1,y1,w1,h1,c1]
        # 每个网格预测的类别  后14
        class_pred = coo_pred[:, 10:]  # [x2,y2,w2,h2,c2]

        # 对真实标签做同样操作
        coo_target = target_tensor[coo_mask].view(-1, yolo_v1_output)
        box_target = coo_target[:, :10].contiguous().view(-1, 5)
        class_target = coo_target[:, 10:]
        class_target = class_target * (1 - 0.1) + torch.ones_like(class_target)* 0.1 / 2


        # 1. compute not contain obj loss
        # 计算不包含obj损失  即本来无，预测有
        # 在预测结果中拿到真实无物体的网格，并改变形状
        noo_pred = pred_tensor[noo_mask].view(-1, yolo_v1_output)
        noo_target = target_tensor[noo_mask].view(-1, yolo_v1_output)
        noo_pred_mask = torch.ByteTensor(noo_pred.size())
        noo_pred_mask.zero_()
        noo_pred_mask[:, 4] = 1 # 将有物体的部分置1
        noo_pred_mask[:, 9] = 1
        noo_pred_mask = noo_pred_mask.bool()
        noo_pred_c = noo_pred[noo_pred_mask]  # noo pred只需要计算 c 的损失 size[-1,2]
        noo_target_c = noo_target[noo_pred_mask] # 拿到第4列和第9列里面的值  真值为0，真实无物体（即拿到真实无物体的网格中，这些网格有物体的概率值，为0）
        nooobj_loss = F.mse_loss(noo_pred_c, noo_target_c, size_average=None)  # 对应的位置做均方误差 如果 size_average = True，返回 loss.mean()。

        # compute contain obj loss
        #计算包含obj损失  即本来有，预测有  和  本来有，预测无
        coo_response_mask = torch.ByteTensor(box_target.size())
        coo_response_mask.zero_()
        coo_not_response_mask = torch.ByteTensor(box_target.size())
        coo_not_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size())

        # 预测值，有多个box的话那么就取一个最大的box，出来就可以了其他的不要啦
        for i in range(0, box_target.size()[0], 2):  # choose the best iou box
            # choose the best iou box ， box1 是预测的 box2 是我们提供的
            # print('box_target.size()[0]{}'.format(box_target.size()))
            box1 = box_pred[i:i + 2]  #batch size=2
            # print('box1{}'.format(box1))
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            box1_xyxy[:, :2] = box1[:, :2] / 14. - 0.5 * box1[:, 2:4]#cx,cy,h,w ->x1,y1,x2,y2
            box1_xyxy[:, 2:4] = box1[:, :2] / 14. + 0.5 * box1[:, 2:4]
            box2 = box_target[i].view(-1, 5)
            # print('box2{}'.format(box2))
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:, :2] = box2[:, :2] / 14. - 0.5 * box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2] / 14. + 0.5 * box2[:, 2:4]
            iou = self.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])  # [2,1]
            max_iou, max_index = iou.max(0)
            max_index = max_index.data

            coo_response_mask[i + max_index] = 1
            coo_response_mask = coo_response_mask.bool()
            coo_not_response_mask[i + 1 - max_index] = 1
            coo_not_response_mask = coo_not_response_mask.bool()
            box_target_iou[i + max_index, torch.LongTensor([4])] = (max_iou).data

        box_target_iou = Variable(box_target_iou)

        # 2.response loss
        # 2.response loss，iou符合的
        # 选择IOU最好的box来进行调整  负责检测出某物体
        box_pred_response = box_pred[coo_response_mask].view(-1, 5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1, 5)
        box_target_response = box_target[coo_response_mask].view(-1, 5)
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response_iou[:, 4], size_average=None)

        # 3.loc_loss
        center_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], size_average=None)
        # print("center_loss= ", center_loss.item())

        hw_loss = F.mse_loss(box_pred_response[:, 2:4], box_target_response[:, 2:4], size_average=None)
        # print("hw_loss= ", hw_loss.item())

        loc_loss = center_loss + hw_loss
        # print("loc_loss= ", loc_loss.item())

        # 4.not response loss
        # 3.not response loss iou不符合的
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1, 5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1, 5)
        box_target_not_response[:, 4] = 0
        # not_contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)
        # I believe this bug is simply a typo
        not_contain_loss = F.mse_loss(box_pred_not_response[:, 4], box_target_not_response[:, 4], size_average=None)

        # 5.class loss
        class_loss = F.mse_loss(class_pred, class_target, size_average=None)

        #
        # print("l_coord= ", self.l_coord)  # 5
        # print("loc_loss= ", loc_loss.item())  #
        # print("contain_loss= ", contain_loss.item())  #
        # print("not_contain_loss= ", not_contain_loss.item())  #
        # print("nooobj_loss= ", nooobj_loss.item())  # 342.2611
        # print("class_loss= ", class_loss.item())  # 3.4275

        # 除以N  即平均一张图的总损失
        all_loss = (self.l_coord * loc_loss + 2 * contain_loss + not_contain_loss + self.l_noobj * nooobj_loss + class_loss) / N
        #
        # print("all loss= ", all_loss.item())
        return all_loss










