# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



def conv3x3(in_planes, out_planes, stride=1, padding=1):# stride默认为1对应实线残差结构，输入为2时对应虚线残差结构，缩小高和宽
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)  # 使用BN层时加不加入bias效果一样，但是不加更节省参数

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)  # ? Why no bias: 如果卷积层之后是BN层，那么可以不用偏置参数，可以节省内存


#残差结构
class BasicBlock(nn.Module):
    expansion = 1  # 经过Block之后channel的变化量，对应残差结构主分支上卷积核的个数有没有发生变换，50层以下每个残差结构的主分支上的卷积核个数一样，所以系数为1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):# inplanes：输入特征矩阵深度 planes：输出特征矩阵深度，对应主分支上卷积核个数 stride：步长 downsample：下采样
        # downsample: 调整维度一致之后才能相加
        # norm_layer：batch normalization layer
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d  # 如果bn层没有自定义，就使用标准的bn层
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)#这里所输入的参数为输入特征矩阵的深度，对应的是卷积层1的输出特征矩阵深度
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  # 保存x，捷径分支上的输出值

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)#输出先加上捷径分支的输出在通过激活函数

        if self.downsample is not None:#如果没有输入下采样参数那么对应实线残差结构，如果输入了下采样参数那么将输入特征矩阵x输入到下采样函数，得到捷径分支输出
            identity = self.downsample(x)  # downsample调整x的维度，F(x)+x一致才能相加

        out += identity
        out = self.relu(out)  # 先相加再激活

        return out



#整个ResNet这个网络框架
class ResNet(nn.Module):
    def __init__(self, block, layers, num_class=24*14*14, norm_layer=None):#block：根据传入层结构不同对应不同残差结构-Bottleneck\BasicBlock;layers：对应所使用残差结构的数目;num_class：训练集的分类个数
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64#经过max pool之后所得到的特征矩阵深度

        # conv1 in ppt figure
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)#为了让输出特征矩阵的高和宽缩小一半padding为3;输入特征矩阵的深度为GRB为3
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#最大池化下采样操作
        self.layer1 = self._make_layer(block, 64, layers[0])#conv2，所有第一层都是64
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)#conv3
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)#conv4
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)#conv5
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)等于GAP，自适应平均池化下采样，不论之前高和宽是多少，都变成（1，1）
        self.fc = nn.Linear(512 * block.expansion, num_class)#全连接层，第一个参数展平后的节点个数

        for m in self.modules():#初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):#block:Bottleneck\BasicBlock;planes:残差结构中卷积层所使用卷积核个数，对应每个第一层
        # 生成不同的stage/layer
        # block: block type(basic block/bottle block)
        # blocks: blocks的数量
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:#对于大于50层的conv2的第一层也是虚线残差结构但是只需要调整特征矩阵的深度，但conv3-conv5所对应的虚线残差结构不仅调整深度，高和宽要缩小一半
            # 需要调整维度，生成下采样
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),  # 同时调整spatial(H x W))和channel两个方向
                norm_layer(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))  # 第一个block单独处理
        self.inplanes = planes * block.expansion  # 记录layerN的channel变化，具体请看ppt resnet表格
        for _ in range(1, blocks):  # 从1开始循环，因为第一个模块前面已经单独处理
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)  # 使用Sequential层组合blocks，形成stage。如果layers=[2,3,4]，那么*layers=2 3 4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x



def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):

    model = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model
